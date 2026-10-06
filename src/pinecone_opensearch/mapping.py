"""Local mapping registry: field types and OpenSearch → Pinecone path rewrite."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pinecone_opensearch.config import ClientConfig, IndexConfig


@dataclass
class FieldInfo:
    name: str
    os_type: str
    pinecone_name: str
    metric: str | None = None
    format: str | None = None
    has_keyword_subfield: bool = False
    dimension: int | None = None
    full_text_search: bool = False


@dataclass
class MappingRegistry:
    """Registry populated from config and ``indices.create`` (M1)."""

    _by_index: dict[str, dict[str, FieldInfo]] = field(default_factory=dict)
    _raw_mappings: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: ClientConfig) -> MappingRegistry:
        reg = cls()
        for name, idx in config.indices.items():
            reg.register_index(name, idx)
            if idx.mapping:
                reg.store_raw_mapping(name, idx.mapping)
        return reg

    def store_raw_mapping(self, logical_name: str, mapping: dict[str, Any]) -> None:
        self._raw_mappings[logical_name] = mapping

    def get_raw_mapping(self, logical_name: str) -> dict[str, Any] | None:
        return self._raw_mappings.get(logical_name)

    def drop_index(self, logical_name: str) -> None:
        self._by_index.pop(logical_name, None)
        self._raw_mappings.pop(logical_name, None)

    def register_index(self, logical_name: str, idx: IndexConfig) -> None:
        fields: dict[str, FieldInfo] = {}
        props = (idx.mapping.get("properties") if idx.mapping else None) or {}
        self._ingest_properties(fields, props, prefix="")
        # Convenience: knn_field from config without full mapping
        if idx.knn_field and idx.knn_field not in fields:
            fields[idx.knn_field] = FieldInfo(
                name=idx.knn_field,
                os_type="knn_vector",
                pinecone_name=idx.knn_field,
                metric="cosine",
            )
        self._by_index[logical_name] = fields

    def _ingest_properties(
        self,
        out: dict[str, FieldInfo],
        props: dict[str, Any],
        prefix: str,
    ) -> None:
        for name, spec in props.items():
            if not isinstance(spec, dict):
                continue
            path = f"{prefix}.{name}" if prefix else name
            os_type = spec.get("type", "object")
            if os_type == "object" or ("properties" in spec and os_type not in {
                "nested", "knn_vector", "text", "keyword", "integer", "long",
                "float", "double", "scaled_float", "boolean", "date",
                "sparse_vector", "rank_features",
            }):
                nested = spec.get("properties") or {}
                self._ingest_properties(out, nested, path)
                continue

            pinecone_name = self.flatten_path(path)
            info = FieldInfo(
                name=path,
                os_type=os_type,
                pinecone_name=pinecone_name,
                format=spec.get("format"),
            )
            if os_type == "knn_vector":
                info.dimension = int(spec["dimension"]) if spec.get("dimension") else None
                space = (spec.get("space_type") or "cosinesimil").lower()
                info.metric = {
                    "cosinesimil": "cosine",
                    "l2": "euclidean",
                    "innerproduct": "dotproduct",
                }.get(space)
            elif os_type == "text":
                info.full_text_search = True
                fields = spec.get("fields") or {}
                if "keyword" in fields:
                    info.has_keyword_subfield = True
                    kw_path = f"{path}.keyword"
                    out[kw_path] = FieldInfo(
                        name=kw_path,
                        os_type="keyword",
                        pinecone_name=f"{pinecone_name}__keyword",
                    )
            out[path] = info

    @staticmethod
    def flatten_path(path: str) -> str:
        """Rewrite object paths: address.city → address__city.

        Keyword subfields: title.keyword → title__keyword.
        """
        if path.endswith(".keyword"):
            base = path[: -len(".keyword")]
            return f"{base.replace('.', '__')}__keyword"
        return path.replace(".", "__")

    def rewrite_field(self, index: str, path: str) -> str:
        fields = self._by_index.get(index) or {}
        if path in fields:
            return fields[path].pinecone_name
        return self.flatten_path(path)

    def pinecone_types(self, index: str) -> dict[str, str]:
        """Pinecone field name → OpenSearch type for read-side value restoration."""
        return {f.pinecone_name: f.os_type for f in (self._by_index.get(index) or {}).values()}

    def dense_field(self, index: str) -> FieldInfo | None:
        """The index's knn_vector field when its dimension is known."""
        for info in (self._by_index.get(index) or {}).values():
            if info.os_type == "knn_vector" and info.dimension:
                return info
        return None

    def get_field(self, index: str, path: str) -> FieldInfo | None:
        return (self._by_index.get(index) or {}).get(path)

    def field_type(self, index: str, path: str) -> str | None:
        info = self.get_field(index, path)
        return info.os_type if info else None

    def is_text(self, index: str, path: str) -> bool:
        return self.field_type(index, path) == "text"

    def is_keyword(self, index: str, path: str) -> bool:
        t = self.field_type(index, path)
        if t == "keyword":
            return True
        if path.endswith(".keyword"):
            return True
        # Unmapped fields default to keyword metadata for term/terms filters
        return t is None
