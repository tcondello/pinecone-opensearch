"""M3 fusion math tests — OpenSearch normalization / RRF reference values."""

from __future__ import annotations

import pytest

from pinecone_opensearch.fusion import (
    MIN_SCORE,
    fuse_normalization,
    fuse_rrf,
    fuse_with_pipeline,
    sum_raw_scores,
)


def _hits(pairs: list[tuple[str, float]]) -> list[dict]:
    return [{"_id": i, "_score": s, "_source": {}} for i, s in pairs]


def test_min_max_basic_and_zero_substitution():
    # Docs example: scores 10000, 9, 8 → 1.0, ~0.00010008, 0.001
    hits = [_hits([("a", 10000.0), ("b", 9.0), ("c", 8.0)])]
    # Single clause normalize via fuse with one list — use internal by fusing with empty? 
    # Call fuse_normalization with one clause duplicated for combo, or test via two equal
    out = fuse_normalization(
        [_hits([("a", 10000.0), ("b", 9.0), ("c", 8.0)]), _hits([("a", 1.0)])],
        technique="min_max",
        combination="arithmetic_mean",
        weights=[1.0, 0.0],  # only first clause matters
    )
    by_id = {h["_id"]: h["_score"] for h in out}
    assert by_id["a"] == pytest.approx(1.0)
    assert by_id["c"] == pytest.approx(MIN_SCORE)
    assert by_id["b"] == pytest.approx((9 - 8) / (10000 - 8), rel=1e-6)


def test_min_max_single_score_is_one():
    out = fuse_normalization(
        [_hits([("only", 42.0)]), _hits([("only", 7.0)])],
        technique="min_max",
        combination="arithmetic_mean",
        weights=[0.5, 0.5],
    )
    assert out[0]["_id"] == "only"
    assert out[0]["_score"] == pytest.approx(1.0)  # 0.5*1 + 0.5*1


def test_min_max_arithmetic_mean_worked_example():
    # From OpenSearch docs comparison table intuition:
    # keyword: doc1=high→1.0, doc2=low→0.001; vector: doc2=1.0
    # equal weights → doc2 = 0.5005, doc1 = 0.5
    kw = _hits([("1", 10.0), ("2", 1.0), ("3", 5.0)])
    # Make 1 max, 2 min for keyword
    kw = _hits([("1", 10.0), ("3", 5.0), ("2", 1.0)])
    vec = _hits([("2", 9.0), ("1", 3.0), ("4", 1.0)])
    out = fuse_normalization(
        [kw, vec],
        technique="min_max",
        combination="arithmetic_mean",
        weights=[0.5, 0.5],
    )
    by_id = {h["_id"]: h["_score"] for h in out}
    # kw: 1→1.0, 3→(5-1)/(10-1)=0.444..., 2→0.001
    # vec: 2→1.0, 1→(3-1)/(9-1)=0.25, 4→0.001
    assert by_id["2"] == pytest.approx(0.5 * MIN_SCORE + 0.5 * 1.0, abs=1e-6)
    assert by_id["1"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.25, abs=1e-6)


def test_l2_preserves_ratios():
    clause = _hits([("a", 3.0), ("b", 4.0)])  # norm = 5
    out = fuse_normalization(
        [clause, _hits([("a", 1.0)])],
        technique="l2",
        combination="arithmetic_mean",
        weights=[1.0, 0.0],
    )
    by_id = {h["_id"]: h["_score"] for h in out}
    assert by_id["a"] == pytest.approx(3 / 5, abs=1e-6)
    assert by_id["b"] == pytest.approx(4 / 5, abs=1e-6)


def test_z_score_below_mean_collapses():
    # scores 10, 1, 1 → mean=4, below mean → 0.001
    clause = _hits([("hi", 10.0), ("lo1", 1.0), ("lo2", 1.0)])
    out = fuse_normalization(
        [clause, _hits([("hi", 1.0)])],
        technique="z_score",
        combination="arithmetic_mean",
        weights=[1.0, 0.0],
    )
    by_id = {h["_id"]: h["_score"] for h in out}
    assert by_id["lo1"] == pytest.approx(MIN_SCORE)
    assert by_id["lo2"] == pytest.approx(MIN_SCORE)
    assert by_id["hi"] > 1.0  # positive z or special


def test_geometric_and_harmonic_means():
    a = _hits([("d", 1.0)])
    b = _hits([("d", 1.0)])
    geo = fuse_normalization([a, b], technique="min_max", combination="geometric_mean", weights=[0.5, 0.5])
    har = fuse_normalization([a, b], technique="min_max", combination="harmonic_mean", weights=[0.5, 0.5])
    assert geo[0]["_score"] == pytest.approx(1.0, abs=1e-6)
    assert har[0]["_score"] == pytest.approx(1.0, abs=1e-6)

    # Known values: scores 1.0 and 0.25 equal weight geo = sqrt(1*0.25)=0.5
    a = _hits([("d", 10.0), ("x", 0.0)])  # after min_max: d=1, x=0.001 if two docs...
    # Simpler: two clauses each single hit score 1 after min_max already
    a = _hits([("d", 4.0)])
    b = _hits([("d", 1.0)])
    # each single → 1.0 after min_max
    geo = fuse_normalization([a, b], technique="min_max", combination="geometric_mean", weights=[0.5, 0.5])
    assert geo[0]["_score"] == pytest.approx(1.0, abs=1e-6)


def test_rrf_default_rank_constant_60():
    # doc A rank1 in both; B rank2 in first only
    q1 = _hits([("A", 10.0), ("B", 5.0)])
    q2 = _hits([("A", 9.0), ("C", 8.0)])
    out = fuse_rrf([q1, q2], rank_constant=60)
    by_id = {h["_id"]: h["_score"] for h in out}
    # equal weights 0.5 each
    assert by_id["A"] == pytest.approx(0.5 / 61 + 0.5 / 61, abs=1e-9)
    assert by_id["B"] == pytest.approx(0.5 / 62, abs=1e-9)
    assert out[0]["_id"] == "A"


def test_pipeline_score_ranker_and_normalization():
    q1 = _hits([("A", 10.0), ("B", 5.0)])
    q2 = _hits([("B", 9.0), ("A", 1.0)])
    rrf_pipe = {
        "phase_results_processors": [
            {"score-ranker-processor": {"combination": {"technique": "rrf", "rank_constant": 60}}}
        ]
    }
    out = fuse_with_pipeline([q1, q2], rrf_pipe)
    assert out[0]["_id"] in {"A", "B"}

    norm_pipe = {
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [0.3, 0.7]},
                    },
                }
            }
        ]
    }
    out2 = fuse_with_pipeline([q1, q2], norm_pipe)
    by_id = {h["_id"]: h["_score"] for h in out2}
    # q1 min_max: A=1.0, B=0.001; q2: B=1.0, A=0.001
    assert by_id["A"] == pytest.approx(0.3 * 1.0 + 0.7 * MIN_SCORE, abs=1e-6)
    assert by_id["B"] == pytest.approx(0.3 * MIN_SCORE + 0.7 * 1.0, abs=1e-6)
    assert out2[0]["_id"] == "B"


def test_sum_raw_scores_approx():
    out = sum_raw_scores([_hits([("a", 1.0), ("b", 2.0)]), _hits([("a", 3.0)])])
    by_id = {h["_id"]: h["_score"] for h in out}
    assert by_id["a"] == 4.0
    assert by_id["b"] == 2.0
