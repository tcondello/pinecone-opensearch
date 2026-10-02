"""kNN / BM25 score conversion to OpenSearch scales."""

from __future__ import annotations

from typing import Any


def convert_knn_score(raw: float, *, metric: str | None) -> float:
    """Convert a Pinecone dense-vector score to OpenSearch kNN score scale.

    | space / metric   | Pinecone returns              | OpenSearch score              |
    | cosinesimil/cosine | cosine similarity ``c``      | ``(1 + c) / 2``               |
    | l2/euclidean       | squared Euclidean ``s``       | ``1 / (1 + s)``               |
    | innerproduct/dotproduct | dot product ``p``        | ``p+1`` if ``p>0`` else ``1/(1-p)`` |
    """
    m = (metric or "cosine").lower()
    if m in ("cosine", "cosinesimil"):
        return (1.0 + float(raw)) / 2.0
    if m in ("euclidean", "l2"):
        return 1.0 / (1.0 + float(raw))
    if m in ("dotproduct", "innerproduct"):
        p = float(raw)
        return p + 1.0 if p > 0 else 1.0 / (1.0 - p)
    return float(raw)


def convert_hit_score(
    raw: float | None,
    *,
    score_by: dict[str, Any] | None,
    metric: str | None,
    default: float = 1.0,
) -> float:
    """Convert one hit score based on the scoring method used."""
    if raw is None:
        return default
    if not score_by:
        return float(raw) if raw is not None else default
    if "dense_vector" in score_by:
        return convert_knn_score(float(raw), metric=metric)
    # BM25 / query_string / sparse: pass through
    return float(raw)


def knn_metric_from_score_by(
    score_by: dict[str, Any] | None,
    *,
    index: str,
    registry: Any,
) -> str | None:
    if not score_by or "dense_vector" not in score_by:
        return None
    field = score_by["dense_vector"].get("field")
    if not field:
        return None
    # Resolve reverse: pinecone name → FieldInfo
    fields = getattr(registry, "_by_index", {}).get(index) or {}
    for info in fields.values():
        if info.pinecone_name == field and info.metric:
            return info.metric
    return "cosine"
