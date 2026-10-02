"""Execute translated plans against the Pinecone Documents API."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Protocol

from pinecone_opensearch.compiler.plan import PineconeCall, TranslationResult
from pinecone_opensearch.config import ClientConfig
from pinecone_opensearch.errors import raise_as_opensearch
from pinecone_opensearch.fusion import fuse_with_pipeline, sum_raw_scores
from pinecone_opensearch.mapping import MappingRegistry
from pinecone_opensearch import shape as shape_mod


class DocumentsAPI(Protocol):
    def search(self, *, namespace: str, score_by: Any, top_k: int,
               include_fields: Any = None, filter: Any = None) -> Any: ...
    def fetch(self, *, namespace: str, ids: Any = None, filter: Any = None,
              include_fields: Any = None, pagination_token: Any = None) -> Any: ...
    def upsert(self, *, namespace: str, documents: Any) -> Any: ...
    def update(self, *, namespace: str, documents: Any = None, filter: Any = None,
               set_fields: Any = None, remove_fields: Any = None) -> Any: ...
    def delete(self, *, namespace: str, ids: Any = None, filter: Any = None,
               delete_all: bool = False) -> Any: ...


class IndexHandle(Protocol):
    @property
    def documents(self) -> DocumentsAPI: ...
    def describe_index_stats(self, **kwargs: Any) -> Any: ...


class PineconeBackend:
    """Live Pinecone SDK backend. Constructed lazily; injectable for tests."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        pinecone_client: Any | None = None,
        index_factory: Any | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self._pc = pinecone_client
        self._index_factory = index_factory
        self._indexes: dict[str, Any] = {}

    @property
    def pc(self) -> Any:
        if self._pc is None:
            from pinecone import Pinecone

            if not self._api_key:
                raise RuntimeError(
                    "PINECONE_API_KEY is required for live calls "
                    "(or pass pinecone_api_key= / inject a backend)"
                )
            self._pc = Pinecone(api_key=self._api_key)
        return self._pc

    def index(self, name: str) -> Any:
        if name not in self._indexes:
            if self._index_factory is not None:
                self._indexes[name] = self._index_factory(name)
            else:
                self._indexes[name] = self.pc.index(name)
        return self._indexes[name]

    def documents(self, name: str) -> Any:
        return self.index(name).documents


def normalize_score_by(score_by: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Convert plan ``score_by`` dict → SDK list-of-typed-clauses."""
    if not score_by or "constant_score" in score_by:
        return None
    out: list[dict[str, Any]] = []
    for kind, payload in score_by.items():
        if kind == "_embed":
            continue
        if kind == "dense_vector":
            out.append({
                "type": "dense_vector",
                "field": payload["field"],
                "values": list(payload["values"]),
            })
        elif kind == "text":
            out.append({
                "type": "text",
                "query": payload["query"],
                "fields": list(payload.get("fields") or []),
            })
        elif kind == "query_string":
            q = str(payload.get("query", ""))
            fields = payload.get("fields") or []
            if fields and ":" not in q:
                parts = [f"{f}:({q})" for f in fields]
                q = " OR ".join(parts)
            out.append({"type": "query_string", "query": q})
        elif kind == "sparse_vector":
            sv = payload.get("sparse_values") or {}
            clause: dict[str, Any] = {
                "type": "sparse_vector",
                "field": payload["field"],
                "sparse_values": sv,
            }
            out.append(clause)
        else:
            raise ValueError(f"Unknown score_by kind {kind!r}")
    return out


def execute_plan(
    result: TranslationResult,
    *,
    backend: PineconeBackend,
    registry: MappingRegistry,
    config: ClientConfig,
    logical_index: str,
    source_filter: Any = True,
    default_fetch_score: float = 1.0,
    embedders: Any = None,
    pipelines: Any = None,
) -> dict[str, Any]:
    """Run a translation plan and return an OpenSearch-shaped search response."""
    started = time.perf_counter()
    warnings = list(result.warnings)
    read_units = 0
    embed_ms = 0.0

    if result.plan == "fan_out":
        return _execute_fan_out(
            result,
            backend=backend,
            registry=registry,
            config=config,
            logical_index=logical_index,
            source_filter=source_filter,
            started=started,
            warnings=warnings,
            embedders=embedders,
            pipelines=pipelines,
        )

    call = result.calls[0]
    try:
        if call.method == "documents.fetch":
            docs, units = _fetch_all(backend, call)
            read_units += units
            score_by = None
            hits = [
                shape_mod.document_to_hit(
                    d,
                    index=logical_index,
                    registry=registry,
                    score_by=score_by,
                    default_score=default_fetch_score,
                    source_filter=source_filter,
                )
                for d in docs
            ]
        else:
            docs, units, score_by, emb = _search_one(backend, call, embedders=embedders)
            read_units += units
            embed_ms += emb
            hits = [
                shape_mod.document_to_hit(
                    d,
                    index=logical_index,
                    registry=registry,
                    score_by=score_by,
                    source_filter=source_filter,
                )
                for d in docs
            ]
    except Exception as exc:  # noqa: BLE001
        raise_as_opensearch(exc)
        raise  # pragma: no cover

    pagination = result.pagination
    from_ = pagination.get("from", 0)
    size = pagination.get("size", 10)
    page = hits[from_ : from_ + size]
    relation = "gte" if len(hits) >= pagination.get("top_k", size) else "eq"
    max_score = max((h["_score"] for h in page), default=None)

    meta = None
    if config.include_pinecone_meta:
        meta = {
            "plan": result.plan,
            "calls": len(result.calls),
            "read_units": read_units,
            "warnings": warnings,
            "embed_ms": embed_ms,
        }

    return shape_mod.shape_search_hits(
        page,
        index=logical_index,
        took_ms=shape_mod.elapsed_ms(started),
        max_score=max_score,
        total_value=len(page),
        total_relation=relation,
        pinecone_meta=meta,
    )


def _materialize_embeds(
    score_by: dict[str, Any] | None,
    embedders: Any,
) -> tuple[dict[str, Any] | None, float]:
    """Resolve neural ``_embed`` placeholders into concrete vectors."""
    if not score_by or "_embed" not in score_by:
        return score_by, 0.0
    if embedders is None:
        raise RuntimeError("neural query requires embedders registry")
    meta = score_by["_embed"]
    vector, embed_ms = embedders.embed_text(meta["model_id"], meta["query_text"])
    out = {k: v for k, v in score_by.items() if k != "_embed"}
    if meta.get("sparse"):
        # vector is {indices, values}
        out["sparse_vector"] = {
            **out["sparse_vector"],
            "sparse_values": vector if isinstance(vector, dict) else {
                "indices": getattr(vector, "indices", []),
                "values": getattr(vector, "values", []),
            },
        }
        # drop top_k from inside sparse_vector if present — top_k is request-level
        out["sparse_vector"].pop("top_k", None)
    else:
        out["dense_vector"] = {
            **out["dense_vector"],
            "values": list(vector),
        }
        out["dense_vector"].pop("top_k", None)
    return out, embed_ms


def _search_one(
    backend: PineconeBackend,
    call: PineconeCall,
    *,
    embedders: Any = None,
) -> tuple[list[Any], int, dict[str, Any] | None, float]:
    req = call.request
    score_by = req.get("score_by")
    score_by, embed_ms = _materialize_embeds(score_by, embedders)
    sdk_score_by = normalize_score_by(score_by)
    if not sdk_score_by:
        raise ValueError("documents.search requires score_by")
    # Strip top_k from nested score_by leftovers for display
    clean_score_by = None
    if score_by:
        clean_score_by = {
            k: ({sk: sv for sk, sv in v.items() if sk != "top_k"} if isinstance(v, dict) else v)
            for k, v in score_by.items()
            if k != "_embed"
        }
    resp = backend.documents(call.index).search(
        namespace=call.namespace,
        score_by=sdk_score_by,
        top_k=int(req.get("top_k", 10)),
        include_fields=req.get("include_fields"),
        filter=req.get("filter"),
    )
    matches = list(getattr(resp, "matches", None) or [])
    usage = getattr(resp, "usage", None)
    units = int(getattr(usage, "read_units", 0) or 0) if usage else 0
    return matches, units, clean_score_by, embed_ms


def _fetch_all(backend: PineconeBackend, call: PineconeCall) -> tuple[list[Any], int]:
    req = call.request
    docs_api = backend.documents(call.index)
    collected: list[Any] = []
    units = 0
    token = None
    top_k = int(req.get("top_k", 10))
    ids = req.get("ids")

    while True:
        resp = docs_api.fetch(
            namespace=call.namespace,
            ids=ids,
            filter=None if ids else req.get("filter"),
            include_fields=req.get("include_fields"),
            pagination_token=token,
        )
        usage = getattr(resp, "usage", None)
        units += int(getattr(usage, "read_units", 0) or 0) if usage else 0

        documents = getattr(resp, "documents", None) or {}
        if isinstance(documents, dict):
            collected.extend(documents.values())
        else:
            collected.extend(list(documents))

        if ids or len(collected) >= top_k:
            break
        pagination = getattr(resp, "pagination", None)
        token = getattr(pagination, "next", None) if pagination else None
        if not token:
            break

    return collected[:top_k], units


def _execute_fan_out(
    result: TranslationResult,
    *,
    backend: PineconeBackend,
    registry: MappingRegistry,
    config: ClientConfig,
    logical_index: str,
    source_filter: Any,
    started: float,
    warnings: list[str],
    embedders: Any = None,
    pipelines: Any = None,
) -> dict[str, Any]:
    read_units = 0
    embed_ms = 0.0
    # Preserve call order for fusion weights
    ordered_hit_lists: list[list[dict[str, Any]] | None] = [None] * len(result.calls)

    with ThreadPoolExecutor(max_workers=min(8, len(result.calls))) as pool:
        future_map = {
            pool.submit(_search_one, backend, c, embedders=embedders): i
            for i, c in enumerate(result.calls)
        }
        for fut in as_completed(future_map):
            idx = future_map[fut]
            call = result.calls[idx]
            try:
                matches, units, score_by, emb = fut.result()
            except Exception as exc:  # noqa: BLE001
                raise_as_opensearch(exc)
                raise
            read_units += units
            embed_ms += emb
            hits = [
                shape_mod.document_to_hit(
                    doc,
                    index=logical_index,
                    registry=registry,
                    score_by=score_by,
                    source_filter=source_filter,
                )
                for doc in matches
            ]
            ordered_hit_lists[idx] = hits

    subquery_hits = [h or [] for h in ordered_hit_lists]
    fusion_meta = result.fusion or {}
    strategy = fusion_meta.get("strategy") or "pipeline"

    if strategy == "sum_raw_scores":
        warnings.append("Approximate fusion: summed raw subquery scores")
        ranked = sum_raw_scores(subquery_hits)
    else:
        pipeline_id = fusion_meta.get("search_pipeline")
        pipeline = None
        if pipelines is not None and pipeline_id:
            pipeline = pipelines.resolve(pipeline_id)
            if pipeline is None:
                warnings.append(f"search_pipeline {pipeline_id!r} not found; using default min_max+arithmetic_mean")
        elif pipeline_id and pipelines is None:
            warnings.append(f"search_pipeline {pipeline_id!r} requested but no registry; using default min_max+arithmetic_mean")
        ranked = fuse_with_pipeline(subquery_hits, pipeline)

    pagination = result.pagination
    from_ = pagination.get("from", 0)
    size = pagination.get("size", 10)
    page = ranked[from_ : from_ + size]
    relation = "gte" if len(ranked) >= pagination.get("top_k", size) else "eq"
    max_score = max((h["_score"] for h in page), default=None)

    meta = None
    if config.include_pinecone_meta:
        meta = {
            "plan": result.plan,
            "calls": len(result.calls),
            "read_units": read_units,
            "warnings": warnings,
            "embed_ms": embed_ms,
            "fusion": strategy,
        }
    return shape_mod.shape_search_hits(
        page,
        index=logical_index,
        took_ms=shape_mod.elapsed_ms(started),
        max_score=max_score,
        total_value=len(page),
        total_relation=relation,
        pinecone_meta=meta,
    )


def wait_for_fields(
    backend: PineconeBackend,
    *,
    pinecone_index: str,
    namespace: str,
    doc_id: str,
    expected: dict[str, Any],
    timeout_s: float = 90.0,
    poll_s: float = 0.25,
) -> float:
    """Poll fetch-by-id until ``doc_id`` reflects every patched field value. Returns wait seconds."""
    expected = {k: v for k, v in expected.items() if not k.startswith("_")}
    started = time.perf_counter()
    while True:
        try:
            resp = backend.documents(pinecone_index).fetch(namespace=namespace, ids=[doc_id])
        except Exception as exc:  # noqa: BLE001
            raise_as_opensearch(exc)
            raise
        documents = getattr(resp, "documents", None) or {}
        doc = documents.get(doc_id) if isinstance(documents, dict) else None
        if doc is not None:
            current = doc.to_dict() if hasattr(doc, "to_dict") else dict(doc)
            if all(_same_value(current.get(k), v) for k, v in expected.items()):
                return time.perf_counter() - started
        if time.perf_counter() - started >= timeout_s:
            raise TimeoutError(f"refresh wait_for timed out after {timeout_s}s waiting for update of {doc_id!r}")
        time.sleep(poll_s)


def _same_value(actual: Any, wanted: Any) -> bool:
    if isinstance(wanted, float) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        return abs(float(actual) - wanted) <= 1e-6 * max(1.0, abs(wanted))
    if isinstance(wanted, list) and isinstance(actual, list) and len(wanted) == len(actual):
        return all(_same_value(a, w) for a, w in zip(actual, wanted))
    return actual == wanted


def wait_for_ids(
    backend: PineconeBackend,
    *,
    pinecone_index: str,
    namespace: str,
    ids: list[str],
    timeout_s: float = 90.0,
    poll_s: float = 0.25,
) -> float:
    """Poll fetch-by-id until all IDs are visible or timeout. Returns wait seconds."""
    if not ids:
        return 0.0
    started = time.perf_counter()
    pending = set(ids)
    while pending:
        elapsed = time.perf_counter() - started
        if elapsed >= timeout_s:
            raise TimeoutError(
                f"refresh wait_for timed out after {timeout_s}s; still missing {sorted(pending)}"
            )
        try:
            resp = backend.documents(pinecone_index).fetch(
                namespace=namespace,
                ids=list(pending),
            )
        except Exception as exc:  # noqa: BLE001
            raise_as_opensearch(exc)
            raise
        documents = getattr(resp, "documents", None) or {}
        found = set(documents.keys()) if isinstance(documents, dict) else {getattr(d, "id", None) for d in documents}
        pending -= {i for i in pending if i in found}
        if pending:
            time.sleep(poll_s)
    return time.perf_counter() - started
