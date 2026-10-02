"""Compile OpenSearch filter IR into Pinecone filter expressions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pinecone_opensearch.mapping import MappingRegistry

# Operators accepted on range clauses
_RANGE_OPS = ("gt", "gte", "lt", "lte")


class FilterCompileError(Exception):
    def __init__(self, path: str, message: str):
        super().__init__(message)
        self.path = path
        self.message = message


def compile_filters(
    clauses: list[Any],
    *,
    index: str,
    registry: MappingRegistry,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Compile a list of filter IR nodes to a single Pinecone filter (or None)."""
    warnings: list[str] = []
    compiled: list[dict[str, Any]] = []
    for clause in clauses:
        part = _compile_node(clause, index=index, registry=registry, warnings=warnings)
        if part is not None:
            compiled.append(part)
    if not compiled:
        return None, warnings
    if len(compiled) == 1:
        return compiled[0], warnings
    return {"$and": compiled}, warnings


def _strip_path(node: Any) -> tuple[Any, str]:
    if isinstance(node, dict) and "_path" in node:
        path = node["_path"]
        clean = {k: v for k, v in node.items() if k != "_path"}
        return clean, path
    return node, ""


def _compile_node(
    node: Any,
    *,
    index: str,
    registry: MappingRegistry,
    warnings: list[str],
) -> dict[str, Any] | None:
    if node is None:
        return None

    # Already-structured bool wrappers from parse
    if isinstance(node, dict) and set(node.keys()) == {"bool"}:
        return _compile_bool(node["bool"], index=index, registry=registry, warnings=warnings, path="bool")

    clean, path = _strip_path(node)
    if not isinstance(clean, dict) or not clean:
        return None

    if "bool" in clean and len(clean) == 1:
        return _compile_bool(clean["bool"], index=index, registry=registry, warnings=warnings, path=path or "bool")

    qtype = next(iter(clean))
    payload = clean[qtype]
    path = path or qtype

    if qtype == "term":
        return _compile_term(payload, index=index, registry=registry, path=path)
    if qtype == "terms":
        return _compile_terms(payload, index=index, registry=registry, path=path)
    if qtype == "range":
        return _compile_range(payload, index=index, registry=registry, path=path)
    if qtype == "exists":
        return _compile_exists(payload, index=index, registry=registry, path=path)
    if qtype == "ids":
        return _compile_ids(payload, path=path, warnings=warnings)
    if qtype == "match":
        return _compile_match_filter(payload, index=index, registry=registry, path=path)
    if qtype == "match_phrase":
        return _compile_match_phrase_filter(payload, index=index, registry=registry, path=path)

    raise FilterCompileError(path, f"Unsupported filter clause {qtype!r}")


def _compile_bool(
    payload: dict[str, Any],
    *,
    index: str,
    registry: MappingRegistry,
    warnings: list[str],
    path: str,
) -> dict[str, Any] | None:
    if "should" in payload and len(payload) == 1:
        parts = []
        for item in payload["should"]:
            c = _compile_node(item, index=index, registry=registry, warnings=warnings)
            if c is not None:
                parts.append(c)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return {"$or": parts}

    if "must_not" in payload and len(payload) == 1:
        parts = []
        for item in payload["must_not"]:
            c = _compile_node(item, index=index, registry=registry, warnings=warnings)
            if c is not None:
                parts.append(c)
        if not parts:
            return None
        if len(parts) == 1:
            return {"$not": parts[0]}
        return {"$not": {"$and": parts}}

    # Generic: treat remaining keys as $and of compiled children
    parts = []
    for key, value in payload.items():
        items = value if isinstance(value, list) else [value]
        for item in items:
            c = _compile_node(item, index=index, registry=registry, warnings=warnings)
            if c is not None:
                parts.append(c)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


def _field_and_value(payload: dict[str, Any]) -> tuple[str, Any]:
    if not isinstance(payload, dict) or len(payload) != 1:
        raise FilterCompileError("term", "Expected single-field clause")
    field = next(iter(payload))
    return field, payload[field]


def _compile_term(
    payload: dict[str, Any],
    *,
    index: str,
    registry: MappingRegistry,
    path: str,
) -> dict[str, Any]:
    field, value = _field_and_value(payload)
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    pc_field = registry.rewrite_field(index, field)
    ftype = registry.field_type(index, field)
    if ftype == "text":
        return {pc_field: {"$match_any": value}}
    return {pc_field: {"$eq": value}}


def _compile_terms(
    payload: dict[str, Any],
    *,
    index: str,
    registry: MappingRegistry,
    path: str,
) -> dict[str, Any]:
    # terms: { field: [v1, v2] } or { field: [...], boost: ... }
    field = None
    values = None
    for key, val in payload.items():
        if key in ("boost", "_name"):
            continue
        field, values = key, val
        break
    if field is None:
        raise FilterCompileError(path, "terms clause missing field")
    if not isinstance(values, list):
        raise FilterCompileError(path, "terms values must be a list")
    if len(values) > 10_000:
        raise FilterCompileError(path, "terms supports at most 10,000 values")
    pc_field = registry.rewrite_field(index, field)
    return {pc_field: {"$in": values}}


def _compile_range(
    payload: dict[str, Any],
    *,
    index: str,
    registry: MappingRegistry,
    path: str,
) -> dict[str, Any]:
    field, opts = _field_and_value(payload)
    if not isinstance(opts, dict):
        raise FilterCompileError(path, "range options must be an object")
    ftype = registry.field_type(index, field)
    if ftype == "keyword":
        raise FilterCompileError(path, "range on keyword is unsupported (Pinecone ranges are numbers-only)")
    # Unmapped string-looking ranges still attempted; keyword mapped rejected
    info = registry.get_field(index, field)
    pc_field = registry.rewrite_field(index, field)
    out: dict[str, Any] = {}
    for op in _RANGE_OPS:
        if op not in opts:
            continue
        val = opts[op]
        if ftype == "date" or (info and info.os_type == "date"):
            val = _to_epoch_ms(val, format=info.format if info else None, path=f"{path}.{op}")
        elif isinstance(val, str) and _looks_like_date(val):
            val = _to_epoch_ms(val, format=None, path=f"{path}.{op}")
        out[f"${op}"] = val
    if not out:
        raise FilterCompileError(path, "range clause has no gt/gte/lt/lte")
    return {pc_field: out}


def _looks_like_date(value: str) -> bool:
    return value.startswith("now") or "T" in value or value.count("-") >= 2


def _to_epoch_ms(value: Any, *, format: str | None, path: str) -> int | float:
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        raise FilterCompileError(path, f"Unsupported date value type {type(value).__name__}")
    if value == "now":
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    # Date math subset: now±N{y,M,w,d,h,m,s} and /d /M rounding — M0 minimal
    if value.startswith("now"):
        raise FilterCompileError(path, f"Date math {value!r} not fully implemented in M0")
    # ISO-ish parse
    try:
        text = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError as exc:
        raise FilterCompileError(path, f"Cannot parse date {value!r}") from exc


def _compile_exists(
    payload: dict[str, Any],
    *,
    index: str,
    registry: MappingRegistry,
    path: str,
) -> dict[str, Any]:
    field = payload.get("field") if isinstance(payload, dict) else None
    if not field:
        raise FilterCompileError(path, "exists.field is required")
    pc_field = registry.rewrite_field(index, field)
    return {pc_field: {"$exists": True}}


def _compile_ids(
    payload: dict[str, Any],
    *,
    path: str,
    warnings: list[str],
) -> dict[str, Any]:
    values = payload.get("values") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        raise FilterCompileError(path, "ids.values must be a list")
    warnings.append(f"{path}: compiling as _id $in (open question 1)")
    return {"_id": {"$in": values}}


def _match_opts(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    return {"query": raw}


def _compile_match_filter(
    payload: dict[str, Any],
    *,
    index: str,
    registry: MappingRegistry,
    path: str,
) -> dict[str, Any]:
    field, raw = _field_and_value(payload)
    opts = _match_opts(raw)
    pc_field = registry.rewrite_field(index, field)
    query = opts.get("query")
    operator = (opts.get("operator") or "or").lower()
    msm = opts.get("minimum_should_match")
    if operator == "and" or str(msm) in ("100%", "100"):
        return {pc_field: {"$match_all": query}}
    return {pc_field: {"$match_any": query}}


def _compile_match_phrase_filter(
    payload: dict[str, Any],
    *,
    index: str,
    registry: MappingRegistry,
    path: str,
) -> dict[str, Any]:
    field, raw = _field_and_value(payload)
    opts = _match_opts(raw)
    if opts.get("slop"):
        raise FilterCompileError(path, "match_phrase slop in filter context is not supported")
    pc_field = registry.rewrite_field(index, field)
    return {pc_field: {"$match_phrase": opts.get("query")}}
