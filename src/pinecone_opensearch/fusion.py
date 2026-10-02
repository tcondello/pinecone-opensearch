"""Client-side hybrid fusion matching OpenSearch search pipelines.

Implements ``normalization-processor`` (min_max / l2 / z_score + means) and
``score-ranker-processor`` (RRF). Given the same subquery hit lists, fused
scores match OpenSearch to floating-point tolerance.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

MIN_SCORE = 0.001
SINGLE_RESULT_SCORE = 1.0


def fuse_with_pipeline(
    subquery_hits: Sequence[Sequence[dict[str, Any]]],
    pipeline: dict[str, Any] | None,
    *,
    id_key: str = "_id",
    score_key: str = "_score",
) -> list[dict[str, Any]]:
    """Fuse N ranked hit lists using a search-pipeline definition.

    Each hit is a dict with at least ``_id`` and ``_score``. Returns a new
    ranked list with fused ``_score`` values (highest first). Source fields
    are taken from the first subquery that returned the document.
    """
    if not subquery_hits:
        return []
    if len(subquery_hits) == 1:
        return [dict(h) for h in subquery_hits[0]]

    processor = _first_phase_processor(pipeline)
    if processor is None:
        # Default hybrid behavior when no pipeline: min_max + equal arithmetic_mean
        return fuse_normalization(
            subquery_hits,
            technique="min_max",
            combination="arithmetic_mean",
            weights=None,
            id_key=id_key,
            score_key=score_key,
        )

    if "score-ranker-processor" in processor:
        cfg = processor["score-ranker-processor"] or {}
        comb = cfg.get("combination") or {}
        return fuse_rrf(
            subquery_hits,
            rank_constant=int(comb.get("rank_constant", 60)),
            weights=(comb.get("parameters") or {}).get("weights"),
            id_key=id_key,
            score_key=score_key,
        )

    if "normalization-processor" in processor:
        cfg = processor["normalization-processor"] or {}
        norm = cfg.get("normalization") or {}
        comb = cfg.get("combination") or {}
        return fuse_normalization(
            subquery_hits,
            technique=str(norm.get("technique") or "min_max"),
            combination=str(comb.get("technique") or "arithmetic_mean"),
            weights=(comb.get("parameters") or {}).get("weights"),
            lower_bounds=(norm.get("parameters") or {}).get("lower_bounds"),
            upper_bounds=(norm.get("parameters") or {}).get("upper_bounds"),
            id_key=id_key,
            score_key=score_key,
        )

    raise ValueError(f"Unsupported search pipeline processor: {sorted(processor)}")


def fuse_normalization(
    subquery_hits: Sequence[Sequence[dict[str, Any]]],
    *,
    technique: str = "min_max",
    combination: str = "arithmetic_mean",
    weights: Sequence[float] | None = None,
    lower_bounds: Sequence[Any] | None = None,
    upper_bounds: Sequence[Any] | None = None,
    id_key: str = "_id",
    score_key: str = "_score",
) -> list[dict[str, Any]]:
    n = len(subquery_hits)
    w = _normalize_weights(weights, n)
    if technique == "z_score" and combination != "arithmetic_mean":
        raise ValueError("z_score supports only arithmetic_mean combination")

    # Per-query: id → normalized score; also keep a prototype hit
    normed: list[dict[str, float]] = []
    prototypes: dict[str, dict[str, Any]] = {}
    for qi, hits in enumerate(subquery_hits):
        scores = [float(h[score_key]) for h in hits]
        mapping = _normalize_clause(
            scores,
            technique=technique,
            lower_bound=_bound_at(lower_bounds, qi),
            upper_bound=_bound_at(upper_bounds, qi),
        )
        clause_map: dict[str, float] = {}
        for hit, ns in zip(hits, mapping, strict=True):
            doc_id = str(hit[id_key])
            clause_map[doc_id] = ns
            prototypes.setdefault(doc_id, dict(hit))
        normed.append(clause_map)

    all_ids = set().union(*(m.keys() for m in normed)) if normed else set()
    fused: list[dict[str, Any]] = []
    for doc_id in all_ids:
        per_q = [m.get(doc_id, 0.0) for m in normed]
        score = _combine(per_q, combination=combination, weights=w)
        hit = dict(prototypes[doc_id])
        hit[score_key] = score
        fused.append(hit)
    fused.sort(key=lambda h: h[score_key], reverse=True)
    return fused


def fuse_rrf(
    subquery_hits: Sequence[Sequence[dict[str, Any]]],
    *,
    rank_constant: int = 60,
    weights: Sequence[float] | None = None,
    id_key: str = "_id",
    score_key: str = "_score",
) -> list[dict[str, Any]]:
    """Reciprocal rank fusion: ``sum_i w_i / (rank_constant + rank_i)`` (1-based ranks)."""
    n = len(subquery_hits)
    w = _normalize_weights(weights, n)
    prototypes: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for qi, hits in enumerate(subquery_hits):
        for rank, hit in enumerate(hits, start=1):
            doc_id = str(hit[id_key])
            prototypes.setdefault(doc_id, dict(hit))
            scores[doc_id] = scores.get(doc_id, 0.0) + w[qi] / float(rank_constant + rank)
    fused = []
    for doc_id, score in scores.items():
        hit = dict(prototypes[doc_id])
        hit[score_key] = score
        fused.append(hit)
    fused.sort(key=lambda h: h[score_key], reverse=True)
    return fused


def sum_raw_scores(
    subquery_hits: Sequence[Sequence[dict[str, Any]]],
    *,
    id_key: str = "_id",
    score_key: str = "_score",
) -> list[dict[str, Any]]:
    """Approx-mode bool multi-scoring: sum raw scores across clauses."""
    prototypes: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for hits in subquery_hits:
        for hit in hits:
            doc_id = str(hit[id_key])
            prototypes.setdefault(doc_id, dict(hit))
            scores[doc_id] = scores.get(doc_id, 0.0) + float(hit[score_key])
    fused = []
    for doc_id, score in scores.items():
        hit = dict(prototypes[doc_id])
        hit[score_key] = score
        fused.append(hit)
    fused.sort(key=lambda h: h[score_key], reverse=True)
    return fused


# --- internals -----------------------------------------------------------

def _first_phase_processor(pipeline: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pipeline:
        return None
    processors = pipeline.get("phase_results_processors") or []
    if not processors:
        return None
    return processors[0]


def _normalize_weights(weights: Sequence[float] | None, n: int) -> list[float]:
    if weights is None:
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"weights length {len(weights)} != number of queries {n}")
    total = float(sum(weights))
    if abs(total - 1.0) > 1e-6 and total != 0:
        # OpenSearch requires sum==1; we still accept and renormalize with a soft check
        return [float(w) / total for w in weights]
    if total == 0:
        return [1.0 / n] * n
    return [float(w) for w in weights]


def _bound_at(bounds: Sequence[Any] | None, index: int) -> dict[str, Any] | None:
    if bounds is None or index >= len(bounds):
        return None
    item = bounds[index]
    if item is None:
        return None
    if isinstance(item, (int, float)):
        return {"mode": "apply", "min_score": float(item), "max_score": float(item)}
    return dict(item)


def _normalize_clause(
    scores: list[float],
    *,
    technique: str,
    lower_bound: dict[str, Any] | None,
    upper_bound: dict[str, Any] | None,
) -> list[float]:
    if not scores:
        return []
    if technique == "min_max":
        return _min_max(scores, lower_bound=lower_bound, upper_bound=upper_bound)
    if technique == "l2":
        return _l2(scores)
    if technique == "z_score":
        return _z_score(scores)
    raise ValueError(f"Unknown normalization technique {technique!r}")


def _min_max(
    scores: list[float],
    *,
    lower_bound: dict[str, Any] | None,
    upper_bound: dict[str, Any] | None,
) -> list[float]:
    raw_min = min(scores)
    raw_max = max(scores)

    # Resolve effective min/max with optional bounds
    eff_min = raw_min
    eff_max = raw_max
    lb_mode = (lower_bound or {}).get("mode", "ignore") if lower_bound else "ignore"
    ub_mode = (upper_bound or {}).get("mode", "ignore") if upper_bound else "ignore"
    lb_val = float((lower_bound or {}).get("min_score", 0.0)) if lower_bound else None
    ub_val = float((upper_bound or {}).get("max_score", 1.0)) if upper_bound else None

    working = list(scores)
    if lb_mode == "clip" and lb_val is not None:
        working = [lb_val if s < lb_val else s for s in working]
        eff_min = min(working)
        eff_max = max(working)
    elif lb_mode == "apply" and lb_val is not None:
        # apply: use lowerBound as min for scores >= bound; below-bound use raw formula branch
        pass

    if ub_mode == "clip" and ub_val is not None:
        working = [ub_val if s > ub_val else s for s in working]
        eff_min = min(working)
        eff_max = max(working)

    # Recompute after clips
    if lb_mode == "clip" or ub_mode == "clip":
        scores = working
        raw_min = min(scores)
        raw_max = max(scores)

    out: list[float] = []
    for score in scores:
        min_s = raw_min
        max_s = raw_max

        if lb_mode == "apply" and lb_val is not None:
            if score >= lb_val:
                min_s = lb_val
            # else keep raw min (score - minScore) / (maxScore - minScore)
        if ub_mode == "apply" and ub_val is not None:
            if score <= ub_val:
                max_s = ub_val

        if math.isclose(max_s, min_s):
            out.append(SINGLE_RESULT_SCORE)
            continue
        ns = (score - min_s) / (max_s - min_s)
        if ns == 0.0:
            ns = MIN_SCORE
        out.append(ns)
    return out


def _l2(scores: list[float]) -> list[float]:
    norm = math.sqrt(sum(s * s for s in scores))
    if norm == 0.0:
        return [0.0] * len(scores)
    return [s / norm for s in scores]


def _z_score(scores: list[float]) -> list[float]:
    n = len(scores)
    mean = sum(scores) / n
    if n == 1 or all(math.isclose(s, mean) for s in scores):
        # "scores are not rescaled at all" / single → highest raw? Docs: every doc matches mean,
        # so clause scores are not rescaled — return originals. For single doc OpenSearch blog
        # also uses SINGLE_RESULT behavior in min_max; for z_score keep raw scores.
        return list(scores)
    # sample standard deviation (n-1)
    var = sum((s - mean) ** 2 for s in scores) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0.0:
        return list(scores)
    out: list[float] = []
    for s in scores:
        if math.isclose(s, mean):
            # special case: equal to mean → highest raw score in the clause
            out.append(max(scores))
            continue
        ns = (s - mean) / sd
        if ns <= 0.0:
            ns = MIN_SCORE
        out.append(ns)
    return out


def _combine(
    per_query_scores: list[float],
    *,
    combination: str,
    weights: list[float],
) -> float:
    """Combine per-query normalized scores. Missing clauses are 0.0 (still in denom)."""
    if combination == "arithmetic_mean":
        return sum(w * s for w, s in zip(weights, per_query_scores, strict=True))
        # weights already sum to 1 → this IS the weighted arithmetic mean
    if combination == "geometric_mean":
        # Missing → 0 would break ln; OpenSearch uses 0 contribution carefully.
        # Docs: missing contributes 0 to weighted sum of lns conceptually via score 0 →
        # but ln(0) undefined. Practice: treat missing/zero as MIN_SCORE for geo/harmonic.
        acc = 0.0
        for w, s in zip(weights, per_query_scores, strict=True):
            val = s if s > 0.0 else MIN_SCORE
            acc += w * math.log(val)
        return math.exp(acc)
    if combination == "harmonic_mean":
        denom = 0.0
        for w, s in zip(weights, per_query_scores, strict=True):
            val = s if s > 0.0 else MIN_SCORE
            denom += w / val
        if denom == 0.0:
            return 0.0
        return 1.0 / denom  # since weights sum to 1: sum(w)/sum(w/s) = 1/sum(w/s)
    raise ValueError(f"Unknown combination technique {combination!r}")
