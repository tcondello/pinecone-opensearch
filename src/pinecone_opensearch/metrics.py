"""Comparison metrics for shadow eval: overlap@k, RBO, filter agreement."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence


def body_hash(body: Any) -> str:
    """Stable SHA-256 of a search/write body for resumable logs."""
    payload = json.dumps(body, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def detect_query_type(body: dict[str, Any] | None) -> str:
    """Best-effort top-level query type label for report rollups."""
    if not body:
        return "empty"
    query = body.get("query")
    if query is None:
        return "filters_only" if any(k in body for k in ("post_filter",)) else "match_all"
    if not isinstance(query, dict) or not query:
        return "unknown"
    qtype = next(iter(query))
    if qtype == "bool":
        payload = query.get("bool") or {}
        must = payload.get("must") or []
        should = payload.get("should") or []
        scoring = must + should
        kinds = []
        for clause in scoring if isinstance(scoring, list) else [scoring]:
            if isinstance(clause, dict) and clause:
                kinds.append(next(iter(clause)))
        if not kinds:
            return "bool_filter"
        if len(set(kinds)) == 1:
            return f"bool_{kinds[0]}"
        return "bool_multi"
    return str(qtype)


def top_ids_and_scores(response: dict[str, Any] | None, *, k: int = 10) -> tuple[list[str], list[float | None]]:
    if not response or "hits" not in response:
        return [], []
    hits = (response.get("hits") or {}).get("hits") or []
    ids: list[str] = []
    scores: list[float | None] = []
    for hit in hits[:k]:
        ids.append(str(hit.get("_id")))
        scores.append(hit.get("_score"))
    return ids, scores


def overlap_at_k(ids_a: Sequence[str], ids_b: Sequence[str], *, k: int = 10) -> float:
    """Set overlap of the top-k ID lists.

    Denominator is ``min(k, max(|A|, |B|))`` so short identical result lists
    score 1.0 rather than ``1/k``.
    """
    if k <= 0:
        return 0.0
    a_list = list(ids_a[:k])
    b_list = list(ids_b[:k])
    if not a_list and not b_list:
        return 1.0
    denom = min(k, max(len(a_list), len(b_list)))
    return len(set(a_list) & set(b_list)) / float(denom)


def recall_at_k(predicted: Sequence[str], relevant: Sequence[str], *, k: int = 10) -> float:
    """Recall@k against qrels: |pred[:k] ∩ relevant| / |relevant| (0 if no qrels)."""
    if not relevant:
        return 0.0
    pred = set(predicted[:k])
    rel = set(relevant)
    return len(pred & rel) / float(len(rel))


def rbo(
    ids_a: Sequence[str],
    ids_b: Sequence[str],
    *,
    p: float = 0.9,
    k: int | None = None,
) -> float:
    """Rank-Biased Overlap with persistence ``p`` (default 0.9), extrapolated.

    Uses the finite-prefix sum plus a ``A_d * p^d`` residual so identical
    rankings score 1.0 (Webber et al. extrapolated RBO).
    """
    if k is not None:
        ids_a = list(ids_a[:k])
        ids_b = list(ids_b[:k])
    else:
        ids_a = list(ids_a)
        ids_b = list(ids_b)
    if not ids_a and not ids_b:
        return 1.0
    depth = max(len(ids_a), len(ids_b))
    if depth == 0:
        return 1.0

    seen_a: set[str] = set()
    seen_b: set[str] = set()
    agreement_sum = 0.0
    last_agreement = 0.0
    for d in range(1, depth + 1):
        if d <= len(ids_a):
            seen_a.add(ids_a[d - 1])
        if d <= len(ids_b):
            seen_b.add(ids_b[d - 1])
        last_agreement = len(seen_a & seen_b) / float(d)
        agreement_sum += last_agreement * (p ** (d - 1))
    return (1.0 - p) * agreement_sum + last_agreement * (p ** depth)

def filter_only_agreement(ids_a: Sequence[str], ids_b: Sequence[str]) -> float:
    """Unordered ID-set Jaccard for filter-only / match_all comparisons."""
    a = set(ids_a)
    b = set(ids_b)
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / float(len(union))


def is_filter_only_query(body: dict[str, Any] | None) -> bool:
    qtype = detect_query_type(body)
    return qtype in {"match_all", "bool_filter", "filters_only", "term", "terms", "range", "ids", "exists"}


def percentile(values: Iterable[float], pct: float) -> float | None:
    data = sorted(float(v) for v in values)
    if not data:
        return None
    if pct <= 0:
        return data[0]
    if pct >= 100:
        return data[-1]
    # Nearest-rank
    rank = int(round((pct / 100.0) * (len(data) - 1)))
    return data[rank]


def extract_embed_ms(response: dict[str, Any] | None) -> float:
    if not response:
        return 0.0
    meta = response.get("_pinecone") or {}
    return float(meta.get("embed_ms") or 0.0)
