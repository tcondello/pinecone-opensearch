"""Parse OpenSearch search bodies into IR (query vs filter context)."""

from __future__ import annotations

from typing import Any

from pinecone_opensearch.compiler.ir import ParsedBody, ScoringClause
from pinecone_opensearch.mapping import MappingRegistry

UNSUPPORTED_TOP_LEVEL = frozenset({
    "aggs",
    "aggregations",
    "suggest",
    "highlight",
    "collapse",
    "search_after",
    "scroll",
    "pit",
    "explain",
    "profile",
    "rescore",
})

UNSUPPORTED_QUERY_TYPES = frozenset({
    "function_score",
    "script_score",
    "dis_max",
    "boosting",
    "nested",
    "geo_distance",
    "geo_bounding_box",
    "geo_shape",
    "percolate",
    "more_like_this",
    "script",
    "wrapper",
})


def parse_search_body(
    body: dict[str, Any],
    *,
    index: str,
    registry: MappingRegistry,
) -> ParsedBody:
    """Parse a search body into scoring clauses + filter tree IR."""
    parsed = ParsedBody()

    if not isinstance(body, dict):
        parsed.unsupported.append(("body", "Search body must be a JSON object"))
        return parsed

    parsed.size = int(body.get("size", 10))
    parsed.from_ = int(body.get("from", 0))
    parsed.source = body.get("_source", True)
    parsed.min_score = body.get("min_score")
    parsed.track_total_hits = body.get("track_total_hits")
    parsed.search_pipeline = body.get("search_pipeline")

    for key in ("timeout", "preference", "request_cache", "routing"):
        if key in body:
            parsed.warnings.append(f"Top-level {key!r} ignored")

    if "sort" in body:
        sort = body["sort"]
        if not _is_score_desc_only(sort):
            parsed.unsupported.append(("sort", "Only _score desc sort is supported"))

    for key in UNSUPPORTED_TOP_LEVEL:
        if key in body:
            parsed.unsupported.append((key, f"Top-level {key!r} is not supported"))

    query = body.get("query")
    if query is None:
        # match_all implied when only filters / empty
        return parsed

    _walk_query(query, context="query", path="query", parsed=parsed, index=index, registry=registry)
    return parsed


def _is_score_desc_only(sort: Any) -> bool:
    if sort in ("_score", ["_score"], [{"_score": "desc"}], [{"_score": {"order": "desc"}}]):
        return True
    if sort == [{"_score": "asc"}]:
        return False
    return False


def _single_key(clause: dict[str, Any]) -> tuple[str, Any]:
    if len(clause) != 1:
        keys = ", ".join(sorted(clause))
        raise ValueError(f"Expected single-key query clause, got: {keys}")
    key = next(iter(clause))
    return key, clause[key]


def _walk_query(
    node: Any,
    *,
    context: str,
    path: str,
    parsed: ParsedBody,
    index: str,
    registry: MappingRegistry,
) -> None:
    if node is None:
        return
    if not isinstance(node, dict):
        parsed.unsupported.append((path, f"Query clause must be an object, got {type(node).__name__}"))
        return

    # Empty object → match_all
    if not node:
        return

    try:
        qtype, payload = _single_key(node)
    except ValueError as exc:
        parsed.unsupported.append((path, str(exc)))
        return

    qpath = f"{path}.{qtype}"

    if qtype in UNSUPPORTED_QUERY_TYPES:
        parsed.unsupported.append((qpath, f"Query type {qtype!r} is not supported"))
        return

    handler = _HANDLERS.get(qtype)
    if handler is None:
        parsed.unsupported.append((qpath, f"Unknown or unsupported query type {qtype!r}"))
        return

    handler(
        payload,
        context=context,
        path=qpath,
        parsed=parsed,
        index=index,
        registry=registry,
    )


def _handle_match_all(payload, *, context, path, parsed, index, registry):
    # No scoring clause; filters alone → fetch plan
    return


def _walk_as_filter(node, *, path, parsed, index, registry):
    if not isinstance(node, dict) or not node:
        return
    try:
        qtype, payload = _single_key(node)
    except ValueError as exc:
        parsed.unsupported.append((path, str(exc)))
        return
    qpath = f"{path}.{qtype}"
    if qtype == "bool":
        _handle_bool_filter(payload, path=qpath, parsed=parsed, index=index, registry=registry)
        return
    if qtype in ("match_all",):
        return
    if qtype in UNSUPPORTED_QUERY_TYPES:
        parsed.unsupported.append((qpath, f"Query type {qtype!r} is not supported"))
        return
    # Store original OpenSearch clause; filters.py compiles it
    parsed.filter_clauses.append({qtype: payload, "_path": qpath})


def _handle_bool_filter(payload, *, path, parsed, index, registry):
    if not isinstance(payload, dict):
        parsed.unsupported.append((path, "bool payload must be an object"))
        return
    must = _as_list(payload.get("must"))
    filt = _as_list(payload.get("filter"))
    should = _as_list(payload.get("should"))
    must_not = _as_list(payload.get("must_not"))
    msm = payload.get("minimum_should_match")

    and_parts: list[Any] = []
    for i, clause in enumerate(must):
        sub = ParsedBody()
        _walk_as_filter(clause, path=f"{path}.must[{i}]", parsed=sub, index=index, registry=registry)
        _merge_parse(parsed, sub)
        and_parts.extend(sub.filter_clauses)
    for i, clause in enumerate(filt):
        sub = ParsedBody()
        _walk_as_filter(clause, path=f"{path}.filter[{i}]", parsed=sub, index=index, registry=registry)
        _merge_parse(parsed, sub)
        and_parts.extend(sub.filter_clauses)

    if should:
        if msm is not None and str(msm) not in ("1", "0") and int(msm) > 1:
            parsed.unsupported.append(
                (f"{path}.minimum_should_match", "minimum_should_match > 1 unsupported in filter context")
            )
        or_parts: list[Any] = []
        for i, clause in enumerate(should):
            sub = ParsedBody()
            _walk_as_filter(clause, path=f"{path}.should[{i}]", parsed=sub, index=index, registry=registry)
            _merge_parse(parsed, sub)
            or_parts.extend(sub.filter_clauses)
        if or_parts:
            and_parts.append({"bool": {"should": or_parts}})

    for i, clause in enumerate(must_not):
        sub = ParsedBody()
        _walk_as_filter(clause, path=f"{path}.must_not[{i}]", parsed=sub, index=index, registry=registry)
        _merge_parse(parsed, sub)
        for fc in sub.filter_clauses:
            and_parts.append({"bool": {"must_not": [fc]}})

    for part in and_parts:
        parsed.filter_clauses.append(part)


def _merge_parse(dst: ParsedBody, src: ParsedBody) -> None:
    dst.warnings.extend(src.warnings)
    dst.unsupported.extend(src.unsupported)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


# ---------------------------------------------------------------------------
# Query-context handlers (clean rewrite of bool)
# ---------------------------------------------------------------------------

def _handle_bool_query(payload, *, context, path, parsed, index, registry):
    if context == "filter":
        _handle_bool_filter(payload, path=path, parsed=parsed, index=index, registry=registry)
        return

    if not isinstance(payload, dict):
        parsed.unsupported.append((path, "bool payload must be an object"))
        return

    must = _as_list(payload.get("must"))
    should = _as_list(payload.get("should"))
    filt = _as_list(payload.get("filter"))
    must_not = _as_list(payload.get("must_not"))
    msm = payload.get("minimum_should_match")

    for i, clause in enumerate(must):
        _walk_query(clause, context="query", path=f"{path}.must[{i}]", parsed=parsed, index=index, registry=registry)
    should_filters: list[Any] = []
    should_scoring = 0
    for i, clause in enumerate(should):
        sub = ParsedBody()
        _walk_query(clause, context="query", path=f"{path}.should[{i}]", parsed=sub, index=index, registry=registry)
        _merge_parse(parsed, sub)
        parsed.scoring.extend(sub.scoring)
        should_scoring += len(sub.scoring)
        if len(sub.filter_clauses) == 1:
            should_filters.append(sub.filter_clauses[0])
        elif sub.filter_clauses:
            should_filters.append({"bool": {"filter": sub.filter_clauses}})

    required = 0 if (must or filt) else 1
    if msm is not None:
        try:
            required = 0 if str(msm) == "0%" else int(msm)
        except (TypeError, ValueError):
            parsed.unsupported.append((f"{path}.minimum_should_match", f"unsupported minimum_should_match {msm!r}"))
    if required > 1 and should:
        parsed.unsupported.append((f"{path}.minimum_should_match", "minimum_should_match > 1 is not supported"))
    elif should_filters:
        if required == 0:
            parsed.warnings.append(
                f"{path}.should: constant-score clauses only add score when must/filter is present; dropped"
            )
        elif should_scoring:
            parsed.warnings.append(
                f"{path}.should: mixes scoring and constant-score clauses; constant-score clauses dropped"
            )
        else:
            parsed.filter_clauses.append({"bool": {"should": should_filters}})

    for i, clause in enumerate(filt):
        _walk_as_filter(clause, path=f"{path}.filter[{i}]", parsed=parsed, index=index, registry=registry)
    for i, clause in enumerate(must_not):
        sub = ParsedBody()
        _walk_as_filter(clause, path=f"{path}.must_not[{i}]", parsed=sub, index=index, registry=registry)
        _merge_parse(parsed, sub)
        for fc in sub.filter_clauses:
            parsed.filter_clauses.append({"bool": {"must_not": [fc]}})

    if msm is not None and must and should:
        parsed.warnings.append(
            f"{path}.minimum_should_match ignored when must is present (OpenSearch should-only-adds-score semantics)"
        )


def _field_clause(payload) -> tuple[str, Any]:
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError("Expected single-field clause")
    field = next(iter(payload))
    return field, payload[field]


def _normalize_match_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {"query": value}


def _handle_match(payload, *, context, path, parsed, index, registry):
    try:
        field, raw = _field_clause(payload)
    except ValueError as exc:
        parsed.unsupported.append((path, str(exc)))
        return
    opts = _normalize_match_value(raw)
    fpath = f"{path}.{field}"

    if context == "filter":
        parsed.filter_clauses.append({"match": {field: opts}, "_path": fpath})
        return

    msm = opts.get("minimum_should_match")
    operator = (opts.get("operator") or "or").lower()
    fuzziness = opts.get("fuzziness")

    if msm is not None and str(msm) not in ("100%", "100"):
        if not (operator == "and" or str(msm) in ("100%", "100")):
            # allow and via operator
            if str(msm) not in ("100%", "100"):
                parsed.unsupported.append(
                    (f"{fpath}.minimum_should_match", "Only 100% / operator:and supported for match")
                )
                return

    params = {
        "field": field,
        "query": opts.get("query"),
        "operator": operator,
        "fuzziness": fuzziness,
        "boost": float(opts.get("boost", 1.0)),
        "minimum_should_match": msm,
    }
    if fuzziness is not None:
        parsed.scoring.append(ScoringClause(kind="match_fuzzy", path=fpath, params=params, boost=params["boost"]))
    else:
        parsed.scoring.append(ScoringClause(kind="match", path=fpath, params=params, boost=params["boost"]))


def _handle_match_phrase(payload, *, context, path, parsed, index, registry):
    try:
        field, raw = _field_clause(payload)
    except ValueError as exc:
        parsed.unsupported.append((path, str(exc)))
        return
    opts = _normalize_match_value(raw)
    fpath = f"{path}.{field}"
    if context == "filter":
        parsed.filter_clauses.append({"match_phrase": {field: opts}, "_path": fpath})
        return
    params = {
        "field": field,
        "query": opts.get("query"),
        "slop": int(opts.get("slop", 0)),
        "boost": float(opts.get("boost", 1.0)),
    }
    parsed.scoring.append(ScoringClause(kind="match_phrase", path=fpath, params=params, boost=params["boost"]))


def _handle_match_phrase_prefix(payload, *, context, path, parsed, index, registry):
    try:
        field, raw = _field_clause(payload)
    except ValueError as exc:
        parsed.unsupported.append((path, str(exc)))
        return
    opts = _normalize_match_value(raw)
    fpath = f"{path}.{field}"
    if context == "filter":
        parsed.unsupported.append((fpath, "match_phrase_prefix in filter context is not supported"))
        return
    params = {
        "field": field,
        "query": opts.get("query"),
        "boost": float(opts.get("boost", 1.0)),
    }
    parsed.scoring.append(
        ScoringClause(kind="match_phrase_prefix", path=fpath, params=params, boost=params["boost"])
    )


def _handle_multi_match(payload, *, context, path, parsed, index, registry):
    if not isinstance(payload, dict):
        parsed.unsupported.append((path, "multi_match payload must be an object"))
        return
    if context == "filter":
        parsed.unsupported.append((path, "multi_match in filter context is not supported"))
        return
    mm_type = payload.get("type", "best_fields")
    if mm_type not in ("best_fields", "most_fields", "phrase"):
        parsed.unsupported.append((f"{path}.type", f"multi_match type {mm_type!r} is not supported"))
        return
    if mm_type == "best_fields":
        parsed.warnings.append(
            f"{path}: Pinecone sums fields equally; OpenSearch best_fields takes max — rankings differ"
        )
    params = {
        "query": payload.get("query"),
        "fields": list(payload.get("fields") or []),
        "type": mm_type,
        "operator": (payload.get("operator") or "or").lower(),
        "boost": float(payload.get("boost", 1.0)),
    }
    kind = "multi_match_phrase" if mm_type == "phrase" else "multi_match"
    parsed.scoring.append(ScoringClause(kind=kind, path=path, params=params, boost=params["boost"]))


def _handle_query_string(payload, *, context, path, parsed, index, registry):
    if not isinstance(payload, dict):
        parsed.unsupported.append((path, "query_string payload must be an object"))
        return
    q = payload.get("query", "")
    if "[ TO " in q.upper() or " TO " in q and "[" in q:
        parsed.unsupported.append((path, "query_string range syntax [a TO b] is not supported"))
        return
    if "_exists_" in q:
        parsed.unsupported.append((path, "query_string _exists_ is not supported"))
        return
    # single-term prefix like auto* (not phrase prefix)
    import re
    if re.search(r"(?<![\"])\b\w+\*(?!\")", q) and '"' not in q:
        parsed.unsupported.append((path, "query_string single-term prefix is not supported"))
        return
    if context == "filter":
        parsed.unsupported.append((path, "query_string in filter context is not supported"))
        return
    params = {
        "query": q,
        "default_field": payload.get("default_field"),
        "fields": list(payload.get("fields") or []),
        "boost": float(payload.get("boost", 1.0)),
    }
    parsed.scoring.append(ScoringClause(kind="query_string", path=path, params=params, boost=params["boost"]))


def _handle_simple_query_string(payload, *, context, path, parsed, index, registry):
    if not isinstance(payload, dict):
        parsed.unsupported.append((path, "simple_query_string payload must be an object"))
        return
    if context == "filter":
        parsed.unsupported.append((path, "simple_query_string in filter context is not supported"))
        return
    q = str(payload.get("query", ""))
    # Translate | → OR, + → AND (approximate)
    translated = q.replace("|", " OR ").replace("+", " AND ")
    if translated != q:
        parsed.warnings.append(f"{path}: simple_query_string operators translated to query_string")
    params = {
        "query": translated,
        "fields": list(payload.get("fields") or []),
        "default_field": payload.get("default_field"),
        "boost": float(payload.get("boost", 1.0)),
    }
    parsed.scoring.append(ScoringClause(kind="query_string", path=path, params=params, boost=params["boost"]))


def _handle_knn(payload, *, context, path, parsed, index, registry):
    if not isinstance(payload, dict):
        parsed.unsupported.append((path, "knn payload must be an object"))
        return
    # OpenSearch knn: { field: { vector, k, filter?, ... } }
    try:
        field, opts = _field_clause(payload)
    except ValueError as exc:
        parsed.unsupported.append((path, str(exc)))
        return
    if not isinstance(opts, dict):
        parsed.unsupported.append((path, "knn field options must be an object"))
        return
    fpath = f"{path}.{field}"
    if context == "filter":
        parsed.unsupported.append((fpath, "knn in filter context is not supported"))
        return
    if "method_parameters" in opts:
        parsed.warnings.append(f"{fpath}.method_parameters ignored")
    if "rescore" in opts:
        parsed.warnings.append(f"{fpath}.rescore ignored")

    params = {
        "field": field,
        "vector": opts.get("vector"),
        "k": opts.get("k"),
        "min_score": opts.get("min_score"),
        "max_distance": opts.get("max_distance"),
        "boost": float(opts.get("boost", 1.0)),
    }
    if opts.get("filter") is not None:
        _walk_as_filter(opts["filter"], path=f"{fpath}.filter", parsed=parsed, index=index, registry=registry)

    kind = "knn_radial" if (opts.get("min_score") is not None or opts.get("max_distance") is not None) else "knn"
    parsed.scoring.append(ScoringClause(kind=kind, path=fpath, params=params, boost=params["boost"]))


def _handle_hybrid(payload, *, context, path, parsed, index, registry):
    if context == "filter":
        parsed.unsupported.append((path, "hybrid in filter context is not supported"))
        return
    if not isinstance(payload, dict):
        parsed.unsupported.append((path, "hybrid payload must be an object"))
        return
    queries = payload.get("queries") or []
    if not isinstance(queries, list) or not queries:
        parsed.unsupported.append((f"{path}.queries", "hybrid.queries must be a non-empty list"))
        return
    if len(queries) > 5:
        parsed.unsupported.append((f"{path}.queries", "hybrid.queries supports at most 5 subqueries"))
        return
    if payload.get("pagination_depth") is not None:
        parsed.hybrid_pagination_depth = int(payload["pagination_depth"])
    if payload.get("filter") is not None:
        _walk_as_filter(payload["filter"], path=f"{path}.filter", parsed=parsed, index=index, registry=registry)

    for i, sub in enumerate(queries):
        sub_parsed = ParsedBody()
        _walk_query(sub, context="query", path=f"{path}.queries[{i}]", parsed=sub_parsed, index=index, registry=registry)
        _merge_parse(parsed, sub_parsed)
        # Move sub scoring into parent; sub filters merge into shared filter
        for sc in sub_parsed.scoring:
            parsed.scoring.append(sc)
        for fc in sub_parsed.filter_clauses:
            parsed.filter_clauses.append(fc)


def _handle_term(payload, *, context, path, parsed, index, registry):
    parsed.filter_clauses.append({"term": payload, "_path": path})
    if context == "query":
        # term in query context still only filters in Pinecone (no score) — approximate
        parsed.warnings.append(f"{path}: term in query context treated as filter (constant score)")


def _handle_terms(payload, *, context, path, parsed, index, registry):
    parsed.filter_clauses.append({"terms": payload, "_path": path})
    if context == "query":
        parsed.warnings.append(f"{path}: terms in query context treated as filter (constant score)")


def _handle_range(payload, *, context, path, parsed, index, registry):
    parsed.filter_clauses.append({"range": payload, "_path": path})


def _handle_exists(payload, *, context, path, parsed, index, registry):
    parsed.filter_clauses.append({"exists": payload, "_path": path})


def _handle_ids(payload, *, context, path, parsed, index, registry):
    parsed.filter_clauses.append({"ids": payload, "_path": path})
    parsed.warnings.append(f"{path}: ids filter may require fetch∩intersect (open question 1)")


def _handle_constant_score(payload, *, context, path, parsed, index, registry):
    if not isinstance(payload, dict):
        parsed.unsupported.append((path, "constant_score payload must be an object"))
        return
    filt = payload.get("filter")
    boost = float(payload.get("boost", 1.0))
    if filt is not None:
        _walk_as_filter(filt, path=f"{path}.filter", parsed=parsed, index=index, registry=registry)
    # constant_score contributes no ANN/BM25 scoring — boost applied at fetch
    parsed.scoring.append(
        ScoringClause(kind="constant_score", path=path, params={"boost": boost}, boost=boost)
    )


def _handle_prefix(payload, *, context, path, parsed, index, registry):
    try:
        field, _ = _field_clause(payload)
    except ValueError:
        field = "?"
    ftype = registry.field_type(index, field)
    if ftype == "text" and context == "query":
        parsed.unsupported.append((path, "prefix on text in query context not yet mapped for M0"))
        return
    parsed.unsupported.append((path, "prefix on keyword fields is not supported"))


def _handle_wildcard(payload, *, context, path, parsed, index, registry):
    parsed.unsupported.append((path, "wildcard queries are not supported"))


def _handle_regexp(payload, *, context, path, parsed, index, registry):
    parsed.unsupported.append((path, "regexp queries are not supported"))


def _handle_fuzzy(payload, *, context, path, parsed, index, registry):
    parsed.unsupported.append((path, "fuzzy queries are not supported (use match fuzziness)"))


def _handle_neural(payload, *, context, path, parsed, index, registry):
    if context == "filter":
        parsed.unsupported.append((path, "neural in filter context is not supported"))
        return
    if not isinstance(payload, dict):
        parsed.unsupported.append((path, "neural payload must be an object"))
        return
    try:
        field, opts = _field_clause(payload)
    except ValueError as exc:
        parsed.unsupported.append((path, str(exc)))
        return
    if not isinstance(opts, dict):
        parsed.unsupported.append((path, "neural field options must be an object"))
        return
    fpath = f"{path}.{field}"
    if opts.get("filter") is not None:
        _walk_as_filter(opts["filter"], path=f"{fpath}.filter", parsed=parsed, index=index, registry=registry)
    params = {
        "field": field,
        "query_text": opts.get("query_text") or opts.get("query"),
        "model_id": opts.get("model_id"),
        "k": opts.get("k"),
        "boost": float(opts.get("boost", 1.0)),
    }
    if not params["query_text"]:
        parsed.unsupported.append((f"{fpath}.query_text", "neural.query_text is required"))
        return
    if not params["model_id"]:
        parsed.unsupported.append((f"{fpath}.model_id", "neural.model_id is required"))
        return
    parsed.scoring.append(ScoringClause(kind="neural", path=fpath, params=params, boost=params["boost"]))


def _handle_neural_sparse(payload, *, context, path, parsed, index, registry):
    if context == "filter":
        parsed.unsupported.append((path, "neural_sparse in filter context is not supported"))
        return
    if not isinstance(payload, dict):
        parsed.unsupported.append((path, "neural_sparse payload must be an object"))
        return
    try:
        field, opts = _field_clause(payload)
    except ValueError as exc:
        parsed.unsupported.append((path, str(exc)))
        return
    if not isinstance(opts, dict):
        parsed.unsupported.append((path, "neural_sparse field options must be an object"))
        return
    fpath = f"{path}.{field}"
    params = {
        "field": field,
        "query_text": opts.get("query_text") or opts.get("query"),
        "model_id": opts.get("model_id"),
        "k": opts.get("k"),
        "boost": float(opts.get("boost", 1.0)),
    }
    if not params["query_text"]:
        parsed.unsupported.append((f"{fpath}.query_text", "neural_sparse.query_text is required"))
        return
    if not params["model_id"]:
        parsed.unsupported.append((f"{fpath}.model_id", "neural_sparse.model_id is required"))
        return
    parsed.scoring.append(
        ScoringClause(kind="neural_sparse", path=fpath, params=params, boost=params["boost"])
    )

_HANDLERS = {
    "match_all": _handle_match_all,
    "bool": _handle_bool_query,
    "match": _handle_match,
    "match_phrase": _handle_match_phrase,
    "match_phrase_prefix": _handle_match_phrase_prefix,
    "multi_match": _handle_multi_match,
    "query_string": _handle_query_string,
    "simple_query_string": _handle_simple_query_string,
    "knn": _handle_knn,
    "hybrid": _handle_hybrid,
    "term": _handle_term,
    "terms": _handle_terms,
    "range": _handle_range,
    "exists": _handle_exists,
    "ids": _handle_ids,
    "constant_score": _handle_constant_score,
    "prefix": _handle_prefix,
    "wildcard": _handle_wildcard,
    "regexp": _handle_regexp,
    "fuzzy": _handle_fuzzy,
    "neural": _handle_neural,
    "neural_sparse": _handle_neural_sparse,
}
