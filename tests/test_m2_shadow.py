"""M2 unit tests: metrics, ShadowClient, replay, report (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinecone_opensearch import OpenSearch, ShadowClient
from pinecone_opensearch.metrics import (
    body_hash,
    filter_only_agreement,
    overlap_at_k,
    rbo,
    recall_at_k,
)
from pinecone_opensearch.replay import iter_query_log, replay
from pinecone_opensearch.report import build_report, build_report_from_path
from .fakes import make_backend
from .inmemory_os import InMemoryOpenSearch

PRODUCTS_CFG = {
    "mode": "strict",
    "indices": {
        "products": {
            "pinecone_index": "products-pinecone",
            "knn_field": "embedding",
            "mapping": {
                "properties": {
                    "title": {"type": "text"},
                    "brand": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "price": {"type": "float"},
                    "embedding": {"type": "knn_vector", "dimension": 3, "space_type": "cosinesimil"},
                }
            },
        }
    },
    "include_pinecone_meta": True,
}


def test_overlap_and_rbo():
    a = ["1", "2", "3", "4"]
    b = ["1", "3", "5", "4"]
    assert overlap_at_k(a, b, k=4) == 0.75
    assert rbo(a, a, p=0.9) == pytest.approx(1.0)
    assert 0.0 < rbo(a, b, p=0.9) < 1.0
    assert filter_only_agreement(["a", "b"], ["b", "a"]) == 1.0
    assert recall_at_k(["1", "2", "9"], ["1", "2", "3"], k=10) == pytest.approx(2 / 3)


def test_shadow_search_logs_jsonl(tmp_path: Path):
    backend, _store, created = make_backend()
    created.add("products-pinecone")
    secondary = OpenSearch(pinecone_api_key="test", config=PRODUCTS_CFG, backend=backend)
    primary = InMemoryOpenSearch()

    doc = {
        "title": "Trail runner",
        "brand": "Acme",
        "category": "shoes",
        "price": 129,
        "embedding": [0.1, 0.2, 0.3],
    }
    primary.index(index="products", id="sku-1", body=doc)
    secondary.index(index="products", id="sku-1", body=doc)

    log_path = tmp_path / "shadow.jsonl"
    shadow = ShadowClient(primary, secondary, log_path, warm_up=0, alternate_order=True, k=10)

    body = {
        "size": 10,
        "query": {
            "bool": {
                "must": [{"match": {"title": "trail"}}],
                "filter": [{"term": {"brand": "Acme"}}],
            }
        },
    }
    resp = shadow.search(index="products", body=body)
    assert resp["hits"]["hits"][0]["_id"] == "sku-1"

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["method"] == "search"
    assert rec["status"] in {"exact", "approx"}
    assert rec["warmup"] is False
    assert rec["body_hash"] == body_hash(body)
    assert rec["metrics"]["overlap@k"] == 1.0
    assert rec["metrics"]["rbo"] == pytest.approx(1.0)
    assert rec["call_order"] in (["primary", "secondary"], ["secondary", "primary"])
    assert "primary" in rec["latency_ms"] and "secondary" in rec["latency_ms"]


def test_shadow_writes_both_and_warmup(tmp_path: Path):
    backend, _store, created = make_backend()
    created.add("products-pinecone")
    secondary = OpenSearch(pinecone_api_key="test", config=PRODUCTS_CFG, backend=backend)
    primary = InMemoryOpenSearch()
    log_path = tmp_path / "shadow.jsonl"
    shadow = ShadowClient(primary, secondary, log_path, warm_up=2, alternate_order=False)

    shadow.index(index="products", id="w1", body={"title": "A", "brand": "B", "embedding": [1, 0, 0]})
    shadow.index(index="products", id="w2", body={"title": "C", "brand": "D", "embedding": [0, 1, 0]})
    shadow.search(index="products", body={"query": {"match_all": {}}})

    # Both engines have the docs
    assert primary.get(index="products", id="w1")["found"] is True
    assert secondary.get(index="products", id="w1")["found"] is True

    records = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
    assert records[0]["warmup"] is True
    assert records[1]["warmup"] is True
    assert records[2]["warmup"] is False
    assert records[0]["call_order"] == ["primary", "secondary"]  # alternate_order=False


def test_shadow_unsupported_still_returns_primary(tmp_path: Path):
    backend, _store, created = make_backend()
    created.add("products-pinecone")
    secondary = OpenSearch(pinecone_api_key="test", config=PRODUCTS_CFG, backend=backend)
    primary = InMemoryOpenSearch()
    primary.index(index="products", id="1", body={"title": "x"})
    log_path = tmp_path / "shadow.jsonl"
    shadow = ShadowClient(primary, secondary, log_path, warm_up=0)

    # aggs unsupported on secondary; primary returns results
    resp = shadow.search(
        index="products",
        body={"query": {"match_all": {}}, "aggs": {"x": {"terms": {"field": "brand"}}}},
    )
    assert "hits" in resp
    rec = json.loads(log_path.read_text().strip().splitlines()[0])
    assert rec["status"] == "unsupported"
    assert rec["secondary"]["error"]["type"] == "UnsupportedQueryError"


def test_replay_and_report(tmp_path: Path):
    backend, _store, created = make_backend()
    created.add("products-pinecone")
    secondary = OpenSearch(pinecone_api_key="test", config=PRODUCTS_CFG, backend=backend)
    primary = InMemoryOpenSearch()
    doc = {"title": "Trail runner", "brand": "Acme", "embedding": [0.1, 0.2, 0.3]}
    primary.index(index="products", id="sku-1", body=doc)
    secondary.index(index="products", id="sku-1", body=doc)

    queries_path = tmp_path / "queries.jsonl"
    body = {"size": 5, "query": {"match": {"title": "trail"}}}
    queries_path.write_text(
        json.dumps({"index": "products", "body": body}) + "\n"
        + json.dumps({"index": "products", "body": {"query": {"match_all": {}}}}) + "\n",
        encoding="utf-8",
    )

    shadow_log = tmp_path / "out.jsonl"
    shadow = ShadowClient(primary, secondary, shadow_log, warm_up=0)
    summary = replay(shadow, iter_query_log(queries_path), qps=1000)
    assert summary["n"] == 2
    assert summary["errors"] == 0

    qrels_path = tmp_path / "qrels.jsonl"
    qrels_path.write_text(
        json.dumps({"body_hash": body_hash(body), "docs": ["sku-1"]}) + "\n",
        encoding="utf-8",
    )
    report = build_report_from_path(shadow_log, qrels_path=qrels_path)
    assert report["n_evaluated"] == 2
    assert "match" in report["by_query_type"] or "match_all" in report["by_query_type"]
    assert report["overall"]["recall@10"]["n"] >= 1
    assert "poc_charter" in report
    assert "checks" in report["poc_charter"]


def test_report_excludes_warmup():
    records = [
        {"warmup": True, "query_type": "match", "method": "search", "status": "exact",
         "latency_ms": {"primary": 1, "secondary": 1}, "metrics": {"overlap@k": 1.0, "rbo": 1.0}},
        {"warmup": False, "query_type": "match", "method": "search", "status": "exact",
         "latency_ms": {"primary": 10, "secondary": 20}, "metrics": {"overlap@k": 0.5, "rbo": 0.5}},
    ]
    report = build_report(records, warm_up_exclude=True)
    assert report["n_evaluated"] == 1
    assert report["overall"]["overlap@k"]["mean"] == 0.5


def test_cli_report(tmp_path: Path):
    from pinecone_opensearch.cli import main

    log = tmp_path / "s.jsonl"
    log.write_text(
        json.dumps({
            "warmup": False,
            "query_type": "match",
            "method": "search",
            "status": "exact",
            "latency_ms": {"primary": 5, "secondary": 8, "embed_ms": 0},
            "metrics": {"overlap@k": 1.0, "rbo": 1.0, "k": 10},
            "primary": {"ids": ["a"]},
            "secondary": {"ids": ["a"], "warnings": []},
        })
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    rc = main(["report", "--log", str(log), "--out", str(out), "--pretty"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["n_evaluated"] == 1
