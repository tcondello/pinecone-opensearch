"""Reshape Pinecone Documents responses into OpenSearch-shaped dicts."""

from __future__ import annotations

import time
from typing import Any

from pinecone_opensearch.mapping import MappingRegistry
from pinecone_opensearch.scores import convert_hit_score, knn_metric_from_score_by
from pinecone_opensearch.writes import unflatten_source


def empty_search_response(
    *,
    index: str,
    took_ms: int = 0,
    pinecone_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return shape_search_hits(
        [],
        index=index,
        took_ms=took_ms,
        max_score=None,
        total_value=0,
        total_relation="eq",
        pinecone_meta=pinecone_meta,
    )


def shape_search_hits(
    hits: list[dict[str, Any]],
    *,
    index: str,
    took_ms: int,
    max_score: float | None,
    total_value: int,
    total_relation: str = "eq",
    pinecone_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resp: dict[str, Any] = {
        "took": took_ms,
        "timed_out": False,
        "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
        "hits": {
            "total": {"value": total_value, "relation": total_relation},
            "max_score": max_score,
            "hits": hits,
        },
    }
    if pinecone_meta is not None:
        resp["_pinecone"] = pinecone_meta
    return resp


def document_to_hit(
    doc: Any,
    *,
    index: str,
    registry: MappingRegistry,
    score_by: dict[str, Any] | None = None,
    default_score: float = 1.0,
    source_filter: Any = True,
) -> dict[str, Any]:
    """Convert a Pinecone Document (or dict-like) to an OpenSearch hit."""
    if hasattr(doc, "to_dict"):
        raw = doc.to_dict()
        doc_id = getattr(doc, "id", None) or raw.get("_id")
        score_raw = getattr(doc, "score", None)
        if score_raw is None:
            score_raw = raw.get("_score")
    elif isinstance(doc, dict):
        raw = dict(doc)
        doc_id = raw.get("_id") or raw.get("id")
        score_raw = raw.get("_score")
    else:
        raise TypeError(f"Unsupported document type: {type(doc)!r}")

    metric = knn_metric_from_score_by(score_by, index=index, registry=registry)
    score = convert_hit_score(score_raw, score_by=score_by, metric=metric, default=default_score)

    fields = {k: v for k, v in raw.items() if not k.startswith("_")}
    source = unflatten_source(fields, registry.pinecone_types(index))
    source = apply_source_filter(source, source_filter)

    return {
        "_index": index,
        "_id": doc_id,
        "_score": score,
        "_source": source,
    }


def apply_source_filter(source: dict[str, Any], source_filter: Any) -> dict[str, Any]:
    if source_filter is True or source_filter is None:
        return source
    if source_filter is False:
        return {}
    if isinstance(source_filter, list):
        return {k: source[k] for k in source_filter if k in source}
    if isinstance(source_filter, dict):
        includes = source_filter.get("includes") or source_filter.get("include")
        excludes = source_filter.get("excludes") or source_filter.get("exclude") or []
        if includes:
            out = {k: source[k] for k in includes if k in source}
        else:
            out = dict(source)
        for ex in excludes:
            if "*" in ex:
                # simple suffix/prefix wildcard
                import fnmatch

                for key in list(out):
                    if fnmatch.fnmatch(key, ex):
                        del out[key]
            else:
                out.pop(ex, None)
        return out
    return source


def shape_get_response(
    *,
    index: str,
    doc_id: str,
    found: bool,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resp: dict[str, Any] = {
        "_index": index,
        "_id": doc_id,
        "found": found,
    }
    if found:
        resp["_source"] = source or {}
        resp["_version"] = 1
    return resp


def shape_index_response(
    *,
    index: str,
    doc_id: str,
    result: str = "created",
) -> dict[str, Any]:
    return {
        "_index": index,
        "_id": doc_id,
        "_version": 1,
        "result": result,
        "_shards": {"total": 1, "successful": 1, "failed": 0},
    }


def shape_delete_response(
    *,
    index: str,
    doc_id: str,
    result: str = "deleted",
) -> dict[str, Any]:
    return {
        "_index": index,
        "_id": doc_id,
        "_version": 1,
        "result": result,
        "_shards": {"total": 1, "successful": 1, "failed": 0},
    }


def shape_update_response(
    *,
    index: str,
    doc_id: str,
    result: str = "updated",
) -> dict[str, Any]:
    return shape_index_response(index=index, doc_id=doc_id, result=result)


def shape_bulk_response(items: list[dict[str, Any]], *, errors: bool) -> dict[str, Any]:
    return {"took": 0, "errors": errors, "items": items}


def shape_count_response(count: int) -> dict[str, Any]:
    return {"count": count, "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0}}


def shape_msearch_response(responses: list[dict[str, Any]]) -> dict[str, Any]:
    return {"responses": responses}


def shape_info_response() -> dict[str, Any]:
    return {
        "name": "pinecone-opensearch",
        "cluster_name": "pinecone",
        "cluster_uuid": "pinecone",
        "version": {
            "distribution": "pinecone",
            "number": "2026-07",
            "build_flavor": "pinecone",
            "build_type": "python",
            "lucene_version": "n/a",
            "minimum_wire_compatibility_version": "n/a",
            "minimum_index_compatibility_version": "n/a",
        },
        "tagline": "OpenSearch drop-in for Pinecone Documents API",
    }


def elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))
