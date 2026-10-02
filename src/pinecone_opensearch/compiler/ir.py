"""Intermediate representation for OpenSearch search bodies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Context = Literal["query", "filter"]


@dataclass
class ScoringClause:
    """A clause that contributes to ranking (becomes score_by)."""

    kind: str
    path: str
    params: dict[str, Any]
    boost: float = 1.0


@dataclass
class ParsedBody:
    """Parsed search request ready for planning."""

    scoring: list[ScoringClause] = field(default_factory=list)
    # Nested OpenSearch-shaped filter IR; compiled later to Pinecone filter dict
    filter_clauses: list[Any] = field(default_factory=list)
    size: int = 10
    from_: int = 0
    source: Any = True
    min_score: float | None = None
    track_total_hits: Any = None
    search_pipeline: str | None = None
    hybrid_pagination_depth: int | None = None
    warnings: list[str] = field(default_factory=list)
    unsupported: list[tuple[str, str]] = field(default_factory=list)
