"""Load and validate pinecone_opensearch.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_MODES = frozenset({"strict", "approx", "shadow"})


@dataclass
class IndexConfig:
    pinecone_index: str
    default_namespace: str = "__default__"
    namespace_from_routing: bool = True
    knn_field: str | None = None
    mapping: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientConfig:
    mode: str = "strict"
    indices: dict[str, IndexConfig] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    embedders: dict[str, Any] = field(default_factory=dict)
    search_pipelines: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)
    include_pinecone_meta: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def max_result_window(self) -> int:
        return int(self.limits.get("max_result_window", 10_000))

    def resolve_index(self, name: str) -> tuple[str, IndexConfig]:
        """Resolve alias → logical index name and return its config."""
        logical = self.aliases.get(name, name)
        if logical not in self.indices:
            raise KeyError(
                f"Unknown index {name!r} (resolved to {logical!r}). "
                f"Known: {sorted(self.indices)}"
            )
        return logical, self.indices[logical]


def _parse_index(name: str, raw: Any) -> IndexConfig:
    if isinstance(raw, str):
        return IndexConfig(pinecone_index=raw)
    if not isinstance(raw, dict):
        raise ValueError(f"indices.{name} must be a string or mapping")
    return IndexConfig(
        pinecone_index=raw.get("pinecone_index", name),
        default_namespace=raw.get("default_namespace", "__default__"),
        namespace_from_routing=bool(raw.get("namespace_from_routing", True)),
        knn_field=raw.get("knn_field"),
        mapping=dict(raw.get("mapping") or {}),
    )


def load_config(path: str | Path | None = None, data: dict[str, Any] | None = None) -> ClientConfig:
    """Load config from a YAML path and/or an in-memory dict."""
    raw: dict[str, Any] = {}
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Config root must be a mapping")
        raw.update(loaded)
    if data:
        raw.update(data)

    mode = str(raw.get("mode", "strict")).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode {mode!r}; expected one of {sorted(VALID_MODES)}")

    indices_raw = raw.get("indices") or {}
    indices = {name: _parse_index(name, cfg) for name, cfg in indices_raw.items()}

    return ClientConfig(
        mode=mode,
        indices=indices,
        aliases=dict(raw.get("aliases") or {}),
        embedders=dict(raw.get("embedders") or {}),
        search_pipelines=dict(raw.get("search_pipelines") or {}),
        limits=dict(raw.get("limits") or {}),
        include_pinecone_meta=bool(raw.get("include_pinecone_meta", True)),
        raw=raw,
    )
