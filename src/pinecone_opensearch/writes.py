"""Convert OpenSearch documents to Pinecone document records (and back)."""

from __future__ import annotations

import secrets
import string
from datetime import datetime, timezone
from typing import Any

from opensearchpy.exceptions import RequestError

from pinecone_opensearch.exceptions import UnsupportedQueryError
from pinecone_opensearch.mapping import MappingRegistry

_METADATA_BUDGET = 40 * 1024
_ID_ALPHABET = string.ascii_letters + string.digits + "-_"


def generate_id(length: int = 20) -> str:
    return "".join(secrets.choice(_ID_ALPHABET) for _ in range(length))


def unflatten_source(flat: dict[str, Any], types: dict[str, str] | None = None) -> dict[str, Any]:
    """Rebuild nested objects from ``address__city`` keys; drop internal keyword copies.

    ``types`` maps Pinecone field names to OpenSearch types so numbers and dates
    stored as floats / epoch-ms come back in their mapped shape.
    """
    out: dict[str, Any] = {}
    types = types or {}
    for key, value in flat.items():
        if key.startswith("_"):
            continue
        value = _restore_value(value, types.get(key))
        if key.endswith("__keyword"):
            # Internal metadata copy of text.keyword — omit from _source by default
            # unless there is no parent text field present
            base = key[: -len("__keyword")]
            # keep only if no FTS field with that base name in the same doc
            if base.replace("__", ".") in flat or base in flat:
                continue
        parts = key.split("__")
        cursor = out
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[parts[-1]] = value
    return out


_INTEGER_TYPES = frozenset({"integer", "long", "short", "byte"})


def _restore_value(value: Any, os_type: str | None) -> Any:
    if os_type in _INTEGER_TYPES and isinstance(value, float) and value.is_integer():
        return int(value)
    if os_type == "date" and isinstance(value, (int, float)) and not isinstance(value, bool):
        dt = datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
        if value % 86_400_000 == 0:
            return dt.strftime("%Y-%m-%d")
        return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return value


def flatten_and_convert(
    source: dict[str, Any],
    *,
    index: str,
    registry: MappingRegistry,
    mode: str = "strict",
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Flatten nested objects and convert field values for Pinecone upsert/update."""
    flat = _flatten(source)
    converted: dict[str, Any] = {}
    meta_sizes: dict[str, int] = {}

    for path, value in flat.items():
        if value is None:
            continue  # drop nulls
        info = registry.get_field(index, path)
        pc_name = registry.rewrite_field(index, path)
        os_type = info.os_type if info else _guess_type(value)

        if os_type == "text":
            converted[pc_name] = str(value)
            if info and info.has_keyword_subfield:
                converted[f"{pc_name}__keyword"] = str(value)
                meta_sizes[f"{pc_name}__keyword"] = _nbytes(str(value))
            # FTS fields don't count toward 40KB metadata budget
            continue

        if os_type == "knn_vector":
            if not isinstance(value, (list, tuple)):
                raise RequestError(400, f"knn_vector field {path!r} must be a list", {"path": path})
            converted[pc_name] = list(value)
            continue

        if os_type in ("sparse_vector", "rank_features"):
            converted[pc_name] = value
            continue

        if os_type == "date":
            epoch = _to_epoch_ms(value, format=info.format if info else None, path=path)
            converted[pc_name] = epoch
            meta_sizes[pc_name] = _nbytes(epoch)
            continue

        if os_type in ("integer", "long", "float", "double", "scaled_float"):
            num = float(value)
            if os_type == "long" and abs(num) > 2**53 and mode == "strict":
                raise UnsupportedQueryError(
                    path,
                    f"long value {value!r} exceeds float64 integer precision (2^53)",
                )
            converted[pc_name] = num
            meta_sizes[pc_name] = _nbytes(num)
            continue

        if os_type == "boolean":
            converted[pc_name] = bool(value)
            meta_sizes[pc_name] = 1
            continue

        # keyword / unmapped metadata
        if isinstance(value, list):
            if not all(isinstance(v, str) for v in value):
                if mode == "strict":
                    raise UnsupportedQueryError(
                        path,
                        "Arrays of numbers or objects are unsupported (metadata lists are strings only)",
                    )
            converted[pc_name] = [str(v) for v in value]
            meta_sizes[pc_name] = _nbytes(converted[pc_name])
            continue

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            converted[pc_name] = float(value)
            meta_sizes[pc_name] = _nbytes(converted[pc_name])
            continue

        if isinstance(value, bool):
            converted[pc_name] = value
            meta_sizes[pc_name] = 1
            continue

        converted[pc_name] = str(value)
        meta_sizes[pc_name] = _nbytes(converted[pc_name])

    total = sum(meta_sizes.values())
    if total > _METADATA_BUDGET:
        raise RequestError(
            400,
            f"Filterable metadata exceeds 40KB ({total} bytes)",
            {"field_sizes": meta_sizes},
        )

    if doc_id is not None:
        if len(doc_id) > 512:
            raise RequestError(400, "_id exceeds 512 characters", {"_id": doc_id})
        converted["_id"] = doc_id
    return converted


def _flatten(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def _guess_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "float"
    if isinstance(value, list) and value and isinstance(value[0], (int, float)) and not isinstance(value[0], bool):
        # could be dense vector — leave as keyword-ish list of numbers → rejected in strict via array rule
        return "keyword"
    return "keyword"


def _nbytes(value: Any) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, list):
        return sum(_nbytes(v) for v in value) + 2 * len(value)
    return len(str(value).encode("utf-8"))


def _to_epoch_ms(value: Any, *, format: str | None, path: str) -> int | float:
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        raise RequestError(400, f"Unsupported date value at {path}", {"path": path})
    if value == "now":
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    try:
        text = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError as exc:
        raise RequestError(400, f"Cannot parse date {value!r} at {path}", {"path": path}) from exc


def translate_os_mappings_to_pinecone_schema(
    mappings: dict[str, Any],
    *,
    knn_field: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Translate OpenSearch ``mappings`` to a Pinecone Documents ``schema``.

    Returns ``(schema, warnings)``. Only searchable fields are declared;
    metadata fields are omitted (auto-indexed on write).
    """
    warnings: list[str] = []
    props = (mappings.get("properties") if mappings else None) or {}
    fields: dict[str, Any] = {}
    knn_seen: list[str] = []

    def walk(prefix: str, properties: dict[str, Any]) -> None:
        for name, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            path = f"{prefix}.{name}" if prefix else name
            os_type = spec.get("type", "object")
            if os_type in ("nested", "geo_point", "geo_shape", "join", "percolator", "binary"):
                raise UnsupportedQueryError(path, f"Field type {os_type!r} is not supported")
            if os_type == "object" or ("properties" in spec and os_type == "object"):
                walk(path, spec.get("properties") or {})
                continue
            if "properties" in spec and os_type not in {
                "text", "keyword", "knn_vector", "sparse_vector", "rank_features",
            }:
                walk(path, spec.get("properties") or {})
                continue

            pc_name = path.replace(".", "__")
            if os_type == "knn_vector":
                knn_seen.append(path)
                space = (spec.get("space_type") or "cosinesimil").lower()
                if space in ("l1", "linf", "hamming"):
                    raise UnsupportedQueryError(path, f"space_type {space!r} is not supported")
                metric = {
                    "cosinesimil": "cosine",
                    "l2": "euclidean",
                    "innerproduct": "dotproduct",
                }.get(space)
                if metric is None:
                    raise UnsupportedQueryError(path, f"Unknown space_type {space!r}")
                dim = int(spec.get("dimension") or 0)
                if dim <= 0 or dim > 20_000:
                    raise UnsupportedQueryError(path, f"Invalid knn_vector dimension {dim}")
                fields[pc_name] = {"type": "dense_vector", "dimension": dim, "metric": metric}
            elif os_type == "text":
                analyzer = (spec.get("analyzer") or "standard").lower()
                fts: dict[str, Any] = {"language": "en"}
                if analyzer == "english":
                    fts["stemming"] = True
                    fts["remove_stop_words"] = True
                elif analyzer == "standard":
                    fts["stemming"] = False
                else:
                    fts["stemming"] = False
                    warnings.append(f"{path}: custom analyzer {analyzer!r} approximated")
                fields[pc_name] = {"type": "string", "full_text_search": fts}
            elif os_type in ("sparse_vector", "rank_features"):
                fields[pc_name] = {"type": "sparse_vector"}
            # keyword/number/boolean/date → metadata, not declared

    walk("", props)

    if len(knn_seen) > 1:
        if knn_field and knn_field in knn_seen:
            keep = knn_field.replace(".", "__")
            for path in knn_seen:
                pc = path.replace(".", "__")
                if pc != keep and pc in fields:
                    del fields[pc]
                    warnings.append(f"Dropped knn_vector {path!r}; keeping {knn_field!r}")
        else:
            raise UnsupportedQueryError(
                knn_seen[1],
                f"Multiple knn_vector fields {knn_seen}; set indices.*.knn_field in config",
            )

    return {"fields": fields}, warnings
