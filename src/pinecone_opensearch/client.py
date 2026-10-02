"""OpenSearch-compatible client facade (M0 translate + M1 execute)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

from opensearchpy.exceptions import ConflictError, NotFoundError, RequestError

from pinecone_opensearch.compiler import build_plan, parse_search_body
from pinecone_opensearch.config import ClientConfig, load_config
from pinecone_opensearch.embedders import EmbedderRegistry
from pinecone_opensearch.errors import raise_as_opensearch
from pinecone_opensearch.exceptions import UnsupportedQueryError
from pinecone_opensearch.executor import PineconeBackend, execute_plan, wait_for_fields, wait_for_ids
from pinecone_opensearch.mapping import MappingRegistry
from pinecone_opensearch.pipelines import PipelineRegistry
from pinecone_opensearch import shape as shape_mod
from pinecone_opensearch.writes import flatten_and_convert, generate_id, unflatten_source


class OpenSearch:
    """Drop-in stand-in for ``opensearchpy.OpenSearch`` backed by Pinecone Documents."""

    def __init__(
        self,
        pinecone_api_key: str | None = None,
        config: str | Path | ClientConfig | dict[str, Any] | None = "pinecone_opensearch.yaml",
        hosts: Any = None,
        *,
        backend: PineconeBackend | None = None,
        **kwargs: Any,
    ) -> None:
        self.pinecone_api_key = pinecone_api_key
        if isinstance(config, ClientConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = load_config(data=config)
        elif config is None:
            self.config = load_config(data={"mode": "strict", "indices": {}})
        else:
            self.config = load_config(config)
        self.registry = MappingRegistry.from_config(self.config)
        self._hosts = hosts
        self._kwargs = kwargs
        self._backend = backend or PineconeBackend(api_key=pinecone_api_key)
        self._last_written: dict[str, list[str]] = {}  # pinecone_index → ids
        self.pipelines = PipelineRegistry(self.config.search_pipelines)
        self.embedders = EmbedderRegistry(self.config.embedders, pc_factory=lambda: self._backend.pc)

    # --- resolve helpers -------------------------------------------------

    def _resolve(self, index: str, routing: str | None = None) -> tuple[str, Any, str]:
        logical, index_cfg = self.config.resolve_index(index)
        ns = index_cfg.default_namespace
        if routing and index_cfg.namespace_from_routing:
            ns = str(routing)
        return logical, index_cfg, ns

    def _record_writes(self, pinecone_index: str, ids: list[str]) -> None:
        bucket = self._last_written.setdefault(pinecone_index, [])
        bucket.extend(ids)
        # keep a bounded recent set for refresh
        if len(bucket) > 5000:
            del bucket[:-2000]

    # --- M0 --------------------------------------------------------------

    def translate(
        self,
        index: str,
        body: dict[str, Any] | None = None,
        *,
        routing: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Dry-run: return the Pinecone request plan and warnings without executing."""
        if body is None:
            body = kwargs.get("body") or {}
        logical, index_cfg, ns = self._resolve(index, routing)
        parsed = parse_search_body(body, index=logical, registry=self.registry)
        result = build_plan(
            parsed,
            logical_index=logical,
            index_cfg=index_cfg,
            registry=self.registry,
            config=self.config,
            namespace=ns if routing else None,
        )
        return result.to_dict()

    def _plan_search(
        self,
        index: str,
        body: dict[str, Any],
        *,
        routing: str | None = None,
        size: int | None = None,
        from_: int | None = None,
        _source: Any = None,
    ) -> tuple[Any, str, Any]:
        body = dict(body or {})
        if size is not None:
            body["size"] = size
        if from_ is not None:
            body["from"] = from_
        if _source is not None:
            body["_source"] = _source
        logical, index_cfg, ns = self._resolve(index, routing)
        parsed = parse_search_body(body, index=logical, registry=self.registry)
        result = build_plan(
            parsed,
            logical_index=logical,
            index_cfg=index_cfg,
            registry=self.registry,
            config=self.config,
            namespace=ns if routing else None,
        )
        # Ensure namespace on calls
        for call in result.calls:
            if routing and index_cfg.namespace_from_routing:
                call.namespace = ns
            elif not call.namespace:
                call.namespace = index_cfg.default_namespace
        return result, logical, body.get("_source", True)

    # --- reads -----------------------------------------------------------

    def search(
        self,
        index: str | None = None,
        body: dict[str, Any] | None = None,
        *,
        size: int | None = None,
        from_: int | None = None,
        _source: Any = None,
        search_pipeline: str | None = None,
        routing: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if body is None:
            body = kwargs.pop("body", None) or {}
        if index is None:
            index = kwargs.pop("index", None)
        if index is None:
            raise TypeError("search() missing required argument: 'index'")
        if search_pipeline:
            body = dict(body)
            body["search_pipeline"] = search_pipeline
        plan, logical, source_filter = self._plan_search(
            index, body, routing=routing, size=size, from_=from_, _source=_source
        )
        return execute_plan(
            plan,
            backend=self._backend,
            registry=self.registry,
            config=self.config,
            logical_index=logical,
            source_filter=source_filter,
            embedders=self.embedders,
            pipelines=self.pipelines,
        )

    def msearch(self, body: Any = None, **kwargs: Any) -> dict[str, Any]:
        """NDJSON-style or list-of-dicts msearch; runs searches concurrently."""
        entries = _parse_msearch_body(body if body is not None else kwargs.get("body"))
        responses: list[dict[str, Any] | None] = [None] * len(entries)

        def _one(i: int, header: dict[str, Any], search_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            idx = header.get("index")
            routing = header.get("routing")
            try:
                return i, self.search(index=idx, body=search_body, routing=routing)
            except Exception as exc:  # noqa: BLE001 — msearch isolates failures
                return i, {
                    "error": {
                        "type": type(exc).__name__,
                        "reason": str(exc),
                    },
                    "status": getattr(exc, "status_code", 500),
                }

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(entries)))) as pool:
            futs = [pool.submit(_one, i, h, b) for i, (h, b) in enumerate(entries)]
            for fut in as_completed(futs):
                i, resp = fut.result()
                responses[i] = resp
        return shape_mod.shape_msearch_response([r for r in responses if r is not None])

    def count(
        self,
        index: str | None = None,
        body: dict[str, Any] | None = None,
        *,
        routing: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if body is None:
            body = kwargs.get("body") or {}
        if index is None:
            index = kwargs.get("index")
        if index is None:
            raise TypeError("count() missing required argument: 'index'")
        logical, index_cfg, ns = self._resolve(index, routing)
        query = (body or {}).get("query") or {"match_all": {}}

        # match_all → namespace stats when possible
        if query == {"match_all": {}} or query.get("match_all") is not None and len(query) == 1:
            try:
                stats = self._backend.index(index_cfg.pinecone_index).describe_index_stats()
                namespaces = getattr(stats, "namespaces", None) or {}
                if isinstance(namespaces, dict) and ns in namespaces:
                    ns_stats = namespaces[ns]
                    count = int(getattr(ns_stats, "vector_count", None)
                                or getattr(ns_stats, "record_count", None)
                                or (ns_stats.get("vector_count") if isinstance(ns_stats, dict) else 0)
                                or 0)
                    return shape_mod.shape_count_response(count)
                total = getattr(stats, "total_vector_count", None)
                if total is not None:
                    return shape_mod.shape_count_response(int(total))
            except Exception:
                # fall through to fetch-based count
                pass

        # Filter-only / general: translate to fetch plan and page up to 10k
        search_body = {"size": 10_000, "query": query}
        plan, _logical, _src = self._plan_search(index, search_body, routing=routing)
        for call in plan.calls:
            call.request["include_fields"] = []
        # Force fetch-style when possible by executing and counting hits without paging slice issues
        try:
            # Use execute but with size 10000
            resp = execute_plan(
                plan,
                backend=self._backend,
                registry=self.registry,
                config=self.config,
                logical_index=logical,
                source_filter=False,
                embedders=self.embedders,
                pipelines=self.pipelines,
            )
            return shape_mod.shape_count_response(len(resp["hits"]["hits"]))
        except Exception as exc:  # noqa: BLE001
            raise_as_opensearch(exc)
            raise

    def get(
        self,
        index: str,
        id: str,
        *,
        routing: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        logical, index_cfg, ns = self._resolve(index, routing)
        try:
            resp = self._backend.documents(index_cfg.pinecone_index).fetch(
                namespace=ns, ids=[id]
            )
        except Exception as exc:  # noqa: BLE001
            raise_as_opensearch(exc)
            raise
        documents = getattr(resp, "documents", None) or {}
        doc = documents.get(id) if isinstance(documents, dict) else None
        if doc is None:
            raise NotFoundError(404, f"Document {id!r} not found", {"_id": id})
        raw = doc.to_dict() if hasattr(doc, "to_dict") else dict(doc)
        source = unflatten_source(
            {k: v for k, v in raw.items() if not k.startswith("_")}, self.registry.pinecone_types(logical)
        )
        source = shape_mod.apply_source_filter(source, _get_source_filter(kwargs))
        return shape_mod.shape_get_response(index=logical, doc_id=id, found=True, source=source)

    def mget(self, body: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        body = body or kwargs.get("body") or {}
        docs_req = body.get("docs") or []
        ids = body.get("ids")
        index = body.get("index") or kwargs.get("index")
        if ids is not None:
            docs_req = [{"_index": index, "_id": i} for i in ids]
        docs_out: list[dict[str, Any]] = []
        # Group by index+namespace
        groups: dict[tuple[str, str], list[str]] = {}
        meta: dict[str, tuple[str, str]] = {}
        for d in docs_req:
            idx = d.get("_index") or index
            doc_id = d.get("_id") or d.get("id")
            routing = d.get("routing") or kwargs.get("routing")
            logical, index_cfg, ns = self._resolve(idx, routing)
            groups.setdefault((index_cfg.pinecone_index, ns), []).append(doc_id)
            meta[doc_id] = (logical, ns)

        found: dict[str, Any] = {}
        for (pc_index, ns), id_list in groups.items():
            for chunk_start in range(0, len(id_list), 1000):
                chunk = id_list[chunk_start : chunk_start + 1000]
                try:
                    resp = self._backend.documents(pc_index).fetch(namespace=ns, ids=chunk)
                except Exception as exc:  # noqa: BLE001
                    raise_as_opensearch(exc)
                    raise
                documents = getattr(resp, "documents", None) or {}
                if isinstance(documents, dict):
                    found.update(documents)

        for d in docs_req:
            doc_id = d.get("_id") or d.get("id")
            logical = meta.get(doc_id, (d.get("_index") or index, ""))[0]
            doc = found.get(doc_id)
            if doc is None:
                docs_out.append(shape_mod.shape_get_response(index=logical, doc_id=doc_id, found=False))
            else:
                raw = doc.to_dict() if hasattr(doc, "to_dict") else dict(doc)
                source = unflatten_source(
                    {k: v for k, v in raw.items() if not k.startswith("_")}, self.registry.pinecone_types(logical)
                )
                source = shape_mod.apply_source_filter(source, _get_source_filter({**kwargs, **d}))
                docs_out.append(
                    shape_mod.shape_get_response(index=logical, doc_id=doc_id, found=True, source=source)
                )
        return {"docs": docs_out}

    # --- writes ----------------------------------------------------------

    def index(
        self,
        index: str,
        body: dict[str, Any] | None = None,
        *,
        id: str | None = None,
        routing: str | None = None,
        refresh: bool | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body = body if body is not None else kwargs.get("body") or {}
        logical, index_cfg, ns = self._resolve(index, routing)
        doc_id = id or generate_id()
        doc = flatten_and_convert(
            body,
            index=logical,
            registry=self.registry,
            mode=self.config.mode,
            doc_id=doc_id,
        )
        try:
            self._backend.documents(index_cfg.pinecone_index).upsert(
                namespace=ns, documents=[doc]
            )
        except Exception as exc:  # noqa: BLE001
            raise_as_opensearch(exc)
            raise
        self._record_writes(index_cfg.pinecone_index, [doc_id])
        if refresh in (True, "true", "wait_for"):
            wait_for_ids(
                self._backend,
                pinecone_index=index_cfg.pinecone_index,
                namespace=ns,
                ids=[doc_id],
                timeout_s=float(self.config.limits.get("refresh_timeout_s", 90)),
            )
        return shape_mod.shape_index_response(index=logical, doc_id=doc_id, result="created")

    def create(
        self,
        index: str,
        id: str,
        body: dict[str, Any] | None = None,
        *,
        routing: str | None = None,
        refresh: bool | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body = body if body is not None else kwargs.get("body") or {}
        logical, index_cfg, ns = self._resolve(index, routing)
        # Conflict if exists
        try:
            resp = self._backend.documents(index_cfg.pinecone_index).fetch(
                namespace=ns, ids=[id]
            )
            documents = getattr(resp, "documents", None) or {}
            if id in documents:
                raise ConflictError(409, f"Document {id!r} already exists", {"_id": id})
        except ConflictError:
            raise
        except Exception as exc:  # noqa: BLE001
            # 404 on empty fetch is fine
            if getattr(exc, "status", None) not in (404, None) and not isinstance(exc, NotFoundError):
                raise_as_opensearch(exc)

        doc = flatten_and_convert(
            body, index=logical, registry=self.registry, mode=self.config.mode, doc_id=id
        )
        try:
            self._backend.documents(index_cfg.pinecone_index).upsert(
                namespace=ns, documents=[doc]
            )
        except Exception as exc:  # noqa: BLE001
            raise_as_opensearch(exc)
            raise
        self._record_writes(index_cfg.pinecone_index, [id])
        if refresh in (True, "true", "wait_for"):
            wait_for_ids(
                self._backend,
                pinecone_index=index_cfg.pinecone_index,
                namespace=ns,
                ids=[id],
                timeout_s=float(self.config.limits.get("refresh_timeout_s", 90)),
            )
        return shape_mod.shape_index_response(index=logical, doc_id=id, result="created")

    def update(
        self,
        index: str,
        id: str,
        body: dict[str, Any] | None = None,
        *,
        routing: str | None = None,
        refresh: bool | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body = body if body is not None else kwargs.get("body") or {}
        if "script" in body:
            raise UnsupportedQueryError("script", "update script is not supported")
        logical, index_cfg, ns = self._resolve(index, routing)
        doc_as_upsert = bool(body.get("doc_as_upsert") or body.get("upsert"))
        patch = body.get("doc") or {}
        converted = flatten_and_convert(
            patch, index=logical, registry=self.registry, mode=self.config.mode
        )
        # update API expects _id on the document record
        update_doc = {"_id": id, **converted}

        exists = False
        try:
            resp = self._backend.documents(index_cfg.pinecone_index).fetch(
                namespace=ns, ids=[id]
            )
            documents = getattr(resp, "documents", None) or {}
            exists = id in documents
        except Exception:
            exists = False

        if not exists:
            if doc_as_upsert:
                full = flatten_and_convert(
                    patch, index=logical, registry=self.registry, mode=self.config.mode, doc_id=id
                )
                try:
                    self._backend.documents(index_cfg.pinecone_index).upsert(
                        namespace=ns, documents=[full]
                    )
                except Exception as exc:  # noqa: BLE001
                    raise_as_opensearch(exc)
                    raise
                result = "created"
            else:
                raise NotFoundError(404, f"Document {id!r} not found", {"_id": id})
        else:
            try:
                self._backend.documents(index_cfg.pinecone_index).update(
                    namespace=ns, documents=[update_doc]
                )
            except Exception as exc:  # noqa: BLE001
                raise_as_opensearch(exc)
                raise
            result = "updated"

        self._record_writes(index_cfg.pinecone_index, [id])
        if refresh in (True, "true", "wait_for"):
            timeout_s = float(self.config.limits.get("refresh_timeout_s", 90))
            if result == "updated":
                wait_for_fields(
                    self._backend,
                    pinecone_index=index_cfg.pinecone_index,
                    namespace=ns,
                    doc_id=id,
                    expected=converted,
                    timeout_s=timeout_s,
                )
            else:
                wait_for_ids(
                    self._backend,
                    pinecone_index=index_cfg.pinecone_index,
                    namespace=ns,
                    ids=[id],
                    timeout_s=timeout_s,
                )
        return shape_mod.shape_update_response(index=logical, doc_id=id, result=result)

    def delete(
        self,
        index: str,
        id: str,
        *,
        routing: str | None = None,
        refresh: bool | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        logical, index_cfg, ns = self._resolve(index, routing)
        try:
            self._backend.documents(index_cfg.pinecone_index).delete(
                namespace=ns, ids=[id]
            )
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
            if status == 404:
                return shape_mod.shape_delete_response(
                    index=logical, doc_id=id, result="not_found"
                )
            raise_as_opensearch(exc)
            raise
        if refresh in (True, "true", "wait_for"):
            # wait until gone
            wait_for_ids(
                self._backend,
                pinecone_index=index_cfg.pinecone_index,
                namespace=ns,
                ids=[],  # nothing to wait for visibility
                timeout_s=1,
            )
        return shape_mod.shape_delete_response(index=logical, doc_id=id, result="deleted")

    def delete_by_query(
        self,
        index: str,
        body: dict[str, Any] | None = None,
        *,
        routing: str | None = None,
        refresh: bool | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Delete documents matching a filter-only query (no text-match operators)."""
        body = body if body is not None else kwargs.get("body") or {}
        logical, index_cfg, ns = self._resolve(index, routing)
        # Force filter context by wrapping in bool.filter when needed
        query = body.get("query") or {"match_all": {}}
        # Translate via a filter-only search body
        search_body = {"size": 0, "query": {"bool": {"filter": [query]}}}
        # If already bool with only filter/must_not, keep structure
        if isinstance(query, dict) and "bool" in query:
            bq = query["bool"] or {}
            scoring_keys = [k for k in ("must", "should") if bq.get(k)]
            if scoring_keys:
                # Still attempt — plan will treat must as filter if we wrap; reject scoring types
                pass
        plan_dict = self.translate(index=index, body=search_body, routing=routing)
        # Collect filters from plan calls
        filt = None
        for call in plan_dict.get("calls") or []:
            req = call.get("request") or {}
            if req.get("filter") is not None:
                filt = req["filter"]
            if req.get("score_by") and any(
                k in (req.get("score_by") or {}) for k in ("text", "query_string", "dense_vector", "sparse_vector")
            ):
                # Shouldn't happen with filter wrap, but guard
                raise UnsupportedQueryError(
                    "query",
                    "delete_by_query does not allow scoring / text-match operators",
                )
        _assert_no_text_match_filter(filt)

        try:
            resp = self._backend.documents(index_cfg.pinecone_index).delete(
                namespace=ns,
                filter=filt,
                delete_all=(filt is None and _is_match_all(query)),
            )
        except Exception as exc:  # noqa: BLE001
            raise_as_opensearch(exc)
            raise

        deleted = getattr(resp, "matched_records", None)
        if deleted is None:
            deleted = 0
        if refresh in (True, "true", "wait_for"):
            # Best-effort: nothing specific to wait on for filter deletes
            pass
        return {
            "took": 0,
            "timed_out": False,
            "total": deleted,
            "deleted": deleted,
            "batches": 1,
            "version_conflicts": 0,
            "noops": 0,
            "retries": {"bulk": 0, "search": 0},
            "throttled_millis": 0,
            "requests_per_second": -1,
            "throttled_until_millis": 0,
            "failures": [],
        }

    def bulk(
        self,
        body: Any = None,
        *,
        index: str | None = None,
        routing: str | None = None,
        refresh: bool | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        actions = list(_iter_bulk_actions(body if body is not None else kwargs.get("body"), default_index=index))
        return _execute_bulk(self, actions, routing=routing, refresh=refresh)

    # --- cluster ---------------------------------------------------------

    def ping(self, **kwargs: Any) -> bool:
        try:
            if self.config.indices:
                name = next(iter(self.config.indices.values())).pinecone_index
                self._backend.pc.indexes.describe(name=name)
            else:
                list(self._backend.pc.indexes.list())
            return True
        except Exception:
            return False

    def info(self, **kwargs: Any) -> dict[str, Any]:
        return shape_mod.shape_info_response()

    @property
    def search_pipeline(self) -> Any:
        return _SearchPipelineClient(self)

    @property
    def indices(self) -> Any:
        from pinecone_opensearch import indices as indices_mod

        return indices_mod.IndicesClient(self)

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError(
            f"OpenSearch.{name} is not implemented by pinecone-opensearch"
        )


class _SearchPipelineClient:
    def __init__(self, client: OpenSearch) -> None:
        self._client = client

    def put(self, id: str, body: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        body = body if body is not None else kwargs.get("body") or {}
        return self._client.pipelines.put(id, body)

    def get(self, id: str | None = None, **kwargs: Any) -> dict[str, Any]:
        return self._client.pipelines.get(id)

    def delete(self, id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.pipelines.delete(id)


def _get_source_filter(params: dict[str, Any]) -> Any:
    """Translate get/mget ``_source`` / ``_source_includes`` / ``_source_excludes`` to a source filter."""

    def as_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return list(value or [])

    source = params.get("_source", True)
    if source is False or source == "false":
        return False
    includes = as_list(params.get("_source_includes"))
    excludes = as_list(params.get("_source_excludes"))
    if isinstance(source, (list, str)) and source not in ("true",):
        includes = includes or as_list(source)
    elif isinstance(source, dict):
        includes = includes or as_list(source.get("includes") or source.get("include"))
        excludes = excludes or as_list(source.get("excludes") or source.get("exclude"))
    if not includes and not excludes:
        return True
    return {"includes": includes, "excludes": excludes}


def _is_match_all(query: Any) -> bool:
    return isinstance(query, dict) and (query == {"match_all": {}} or set(query.keys()) == {"match_all"})


def _assert_no_text_match_filter(filt: Any, path: str = "filter") -> None:
    """Reject $match_any / $match_all / $match_phrase in delete_by_query filters."""
    if filt is None:
        return
    if isinstance(filt, dict):
        for key, val in filt.items():
            if key in {"$match_any", "$match_all", "$match_phrase"}:
                raise UnsupportedQueryError(
                    path,
                    f"delete_by_query rejects text-match operator {key}",
                )
            if key.startswith("$"):
                if isinstance(val, list):
                    for i, item in enumerate(val):
                        _assert_no_text_match_filter(item, f"{path}.{key}[{i}]")
                else:
                    _assert_no_text_match_filter(val, f"{path}.{key}")
            elif isinstance(val, dict):
                _assert_no_text_match_filter(val, f"{path}.{key}")
    elif isinstance(filt, list):
        for i, item in enumerate(filt):
            _assert_no_text_match_filter(item, f"{path}[{i}]")


def _parse_msearch_body(body: Any) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if body is None:
        return []
    if isinstance(body, list):
        # alternate header/body dicts
        pairs = []
        it = iter(body)
        for header in it:
            search_body = next(it, {})
            pairs.append((header or {}, search_body or {}))
        return pairs
    if isinstance(body, (str, bytes)):
        text = body.decode() if isinstance(body, bytes) else body
        lines = [ln for ln in text.splitlines() if ln.strip()]
        pairs = []
        for i in range(0, len(lines), 2):
            header = json.loads(lines[i])
            search_body = json.loads(lines[i + 1]) if i + 1 < len(lines) else {}
            pairs.append((header, search_body))
        return pairs
    raise TypeError("msearch body must be NDJSON string or list of header/body pairs")


def _iter_bulk_actions(body: Any, *, default_index: str | None) -> list[dict[str, Any]]:
    """Normalize bulk body into a list of action dicts with op/meta/source."""
    if body is None:
        return []
    lines: list[Any]
    if isinstance(body, (str, bytes)):
        text = body.decode() if isinstance(body, bytes) else body
        lines = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    elif isinstance(body, list):
        lines = body
    else:
        raise TypeError("bulk body must be NDJSON string or list")

    actions: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        meta_line = lines[i]
        i += 1
        if not isinstance(meta_line, dict) or not meta_line:
            continue
        op = next(iter(meta_line))
        meta = meta_line[op] or {}
        source = None
        if op in ("index", "create", "update"):
            source = lines[i] if i < len(lines) else {}
            i += 1
        actions.append({
            "op": op,
            "index": meta.get("_index") or default_index,
            "id": meta.get("_id"),
            "routing": meta.get("routing"),
            "source": source or {},
        })
    return actions


_MAX_UPSERT_DOCS = 1000
_MAX_UPSERT_BYTES = 2 * 1024 * 1024 - 64 * 1024


def _upsert_chunks(docs: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    """Split upserts under Pinecone's per-request limits (1000 docs, 2MB)."""
    chunk: list[dict[str, Any]] = []
    size = 2
    for doc in docs:
        doc_size = len(json.dumps(doc, separators=(",", ":"), default=str).encode("utf-8")) + 1
        if chunk and (len(chunk) >= _MAX_UPSERT_DOCS or size + doc_size > _MAX_UPSERT_BYTES):
            yield chunk
            chunk, size = [], 2
        chunk.append(doc)
        size += doc_size
    if chunk:
        yield chunk


def _execute_bulk(
    client: OpenSearch,
    actions: list[dict[str, Any]],
    *,
    routing: str | None,
    refresh: bool | str | None,
) -> dict[str, Any]:
    from pinecone_opensearch.writes import flatten_and_convert

    items: list[dict[str, Any]] = []
    errors = False
    # Batch upserts by (pc_index, namespace)
    upsert_batches: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    # Track action indices for batched upserts
    pending_meta: list[tuple[int, str, str, str]] = []  # item_idx placeholder unused

    def flush_upserts() -> None:
        nonlocal errors
        for (pc_index, ns, logical), docs in list(upsert_batches.items()):
            for chunk in _upsert_chunks(docs):
                try:
                    client._backend.documents(pc_index).upsert(namespace=ns, documents=chunk)
                    client._record_writes(pc_index, [d["_id"] for d in chunk])
                    for d in chunk:
                        items.append({
                            "index": {
                                "_index": logical,
                                "_id": d["_id"],
                                "status": 201,
                                "result": "created",
                            }
                        })
                except Exception as exc:  # noqa: BLE001
                    errors = True
                    for d in chunk:
                        items.append({
                            "index": {
                                "_index": logical,
                                "_id": d.get("_id"),
                                "status": 400,
                                "error": {"type": type(exc).__name__, "reason": str(exc)},
                            }
                        })
            upsert_batches.clear()

    for action in actions:
        op = action["op"]
        idx = action["index"]
        if not idx:
            errors = True
            items.append({op: {"status": 400, "error": {"reason": "missing index"}}})
            continue
        try:
            logical, index_cfg, ns = client._resolve(idx, action.get("routing") or routing)
        except Exception as exc:  # noqa: BLE001
            errors = True
            items.append({op: {"status": 400, "error": {"reason": str(exc)}}})
            continue

        doc_id = action.get("id") or (generate_id() if op in ("index", "create") else None)
        try:
            if op in ("index", "create"):
                if op == "create" and doc_id:
                    # check exists
                    resp = client._backend.documents(index_cfg.pinecone_index).fetch(
                        namespace=ns, ids=[doc_id]
                    )
                    documents = getattr(resp, "documents", None) or {}
                    if doc_id in documents:
                        errors = True
                        items.append({
                            "create": {
                                "_index": logical,
                                "_id": doc_id,
                                "status": 409,
                                "error": {"type": "version_conflict_engine_exception", "reason": "exists"},
                            }
                        })
                        continue
                source = action.get("source") or {}
                # update-style bulk uses {"doc": ...}
                if "doc" in source and op == "update":
                    source = source["doc"]
                doc = flatten_and_convert(
                    source,
                    index=logical,
                    registry=client.registry,
                    mode=client.config.mode,
                    doc_id=doc_id,
                )
                key = (index_cfg.pinecone_index, ns, logical)
                upsert_batches.setdefault(key, []).append(doc)
                if sum(len(v) for v in upsert_batches.values()) >= _MAX_UPSERT_DOCS:
                    flush_upserts()
            elif op == "update":
                flush_upserts()
                source = action.get("source") or {}
                if "script" in source:
                    raise UnsupportedQueryError("script", "update script is not supported")
                patch = source.get("doc") or source
                converted = flatten_and_convert(
                    patch, index=logical, registry=client.registry, mode=client.config.mode
                )
                client._backend.documents(index_cfg.pinecone_index).update(
                    namespace=ns, documents=[{"_id": doc_id, **converted}]
                )
                client._record_writes(index_cfg.pinecone_index, [doc_id])
                items.append({
                    "update": {"_index": logical, "_id": doc_id, "status": 200, "result": "updated"}
                })
            elif op == "delete":
                flush_upserts()
                client._backend.documents(index_cfg.pinecone_index).delete(
                    namespace=ns, ids=[doc_id]
                )
                items.append({
                    "delete": {"_index": logical, "_id": doc_id, "status": 200, "result": "deleted"}
                })
            else:
                errors = True
                items.append({op: {"status": 400, "error": {"reason": f"unsupported bulk op {op}"}}})
        except Exception as exc:  # noqa: BLE001
            errors = True
            items.append({
                op: {
                    "_index": logical,
                    "_id": doc_id,
                    "status": getattr(exc, "status_code", 400),
                    "error": {"type": type(exc).__name__, "reason": str(exc)},
                }
            })

    flush_upserts()

    if refresh in (True, "true", "wait_for"):
        # Best-effort: wait on all last-written ids across indexes
        for pc_index, ids in list(client._last_written.items()):
            # namespace unknown precisely for mixed bulk — skip if empty
            if not ids:
                continue
            # Use default namespace from first matching config
            ns = "__default__"
            for cfg in client.config.indices.values():
                if cfg.pinecone_index == pc_index:
                    ns = cfg.default_namespace
                    break
            try:
                wait_for_ids(
                    client._backend,
                    pinecone_index=pc_index,
                    namespace=ns,
                    ids=ids[-1000:],
                    timeout_s=float(client.config.limits.get("refresh_timeout_s", 90)),
                )
            except Exception:
                pass

    return shape_mod.shape_bulk_response(items, errors=errors)


__all__ = ["OpenSearch", "UnsupportedQueryError"]
