"""M3 unit tests: neural, hybrid fusion execution, delete_by_query, pipelines."""

from __future__ import annotations

import pytest

from pinecone_opensearch import OpenSearch, UnsupportedQueryError
from .fakes import make_backend

CFG = {
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
                    "embedding": {"type": "knn_vector", "dimension": 3, "space_type": "cosinesimil"},
                }
            },
        }
    },
    "embedders": {
        "demo-model": {
            "type": "callable",
            "fn": lambda texts, **kw: [[0.1, 0.2, 0.3] for _ in texts],
        }
    },
    "search_pipelines": {
        "norm-pipeline": {
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {"weights": [0.5, 0.5]},
                        },
                    }
                }
            ]
        }
    },
    "include_pinecone_meta": True,
}


@pytest.fixture
def client() -> OpenSearch:
    backend, _store, created = make_backend()
    created.add("products-pinecone")
    c = OpenSearch(pinecone_api_key="test", config=CFG, backend=backend)
    # Register callable embedder from config (dict with fn)
    c.embedders.register("demo-model", CFG["embedders"]["demo-model"])
    return c


def _seed(client: OpenSearch) -> None:
    client.index(
        index="products",
        id="sku-1",
        body={"title": "Trail runner", "brand": "Acme", "category": "shoes", "embedding": [0.1, 0.2, 0.3]},
    )
    client.index(
        index="products",
        id="sku-2",
        body={"title": "Office chair", "brand": "Acme", "category": "furniture", "embedding": [0.9, 0.1, 0.0]},
    )


def test_neural_search_reports_embed_ms(client: OpenSearch):
    _seed(client)
    resp = client.search(
        index="products",
        body={
            "size": 5,
            "query": {
                "neural": {
                    "embedding": {
                        "query_text": "running shoes",
                        "model_id": "demo-model",
                        "k": 5,
                    }
                }
            },
        },
    )
    assert resp["hits"]["hits"]
    assert resp["_pinecone"]["plan"] == "single_search"
    assert resp["_pinecone"]["embed_ms"] >= 0


def test_hybrid_search_uses_pipeline_fusion(client: OpenSearch):
    _seed(client)
    resp = client.search(
        index="products",
        body={
            "size": 10,
            "search_pipeline": "norm-pipeline",
            "query": {
                "hybrid": {
                    "queries": [
                        {"match": {"title": "trail"}},
                        {"knn": {"embedding": {"vector": [0.1, 0.2, 0.3], "k": 10}}},
                    ]
                }
            },
        },
    )
    assert resp["_pinecone"]["plan"] == "fan_out"
    assert resp["_pinecone"]["fusion"] == "pipeline"
    assert resp["hits"]["hits"]
    assert resp["hits"]["hits"][0]["_id"] == "sku-1"


def test_search_pipeline_put_get(client: OpenSearch):
    client.search_pipeline.put(
        "rrf-pipeline",
        body={
            "phase_results_processors": [
                {"score-ranker-processor": {"combination": {"technique": "rrf", "rank_constant": 60}}}
            ]
        },
    )
    got = client.search_pipeline.get("rrf-pipeline")
    assert "rrf-pipeline" in got
    assert client.pipelines.resolve("rrf-pipeline") is not None


def test_delete_by_query_filter_only(client: OpenSearch):
    _seed(client)
    resp = client.delete_by_query(
        index="products",
        body={"query": {"term": {"brand": "Acme"}}},
    )
    assert resp["deleted"] == 2
    with pytest.raises(Exception):
        client.get(index="products", id="sku-1")


def test_delete_by_query_rejects_match(client: OpenSearch):
    _seed(client)
    with pytest.raises(UnsupportedQueryError):
        client.delete_by_query(
            index="products",
            body={"query": {"match": {"title": "trail"}}},
        )


def test_approx_bool_multi_sum(client: OpenSearch):
    backend, _store, created = make_backend()
    created.add("products-pinecone")
    cfg = dict(CFG)
    cfg["mode"] = "approx"
    c = OpenSearch(pinecone_api_key="test", config=cfg, backend=backend)
    c.index(index="products", id="sku-1", body={"title": "Trail runner", "brand": "Acme", "embedding": [0.1, 0.2, 0.3]})
    resp = c.search(
        index="products",
        body={
            "size": 5,
            "query": {
                "bool": {
                    "must": [
                        {"match": {"title": "trail"}},
                        {"knn": {"embedding": {"vector": [0.1, 0.2, 0.3], "k": 5}}},
                    ]
                }
            },
        },
    )
    assert resp["_pinecone"]["plan"] == "fan_out"
    assert resp["_pinecone"]["fusion"] == "sum_raw_scores"
    assert any("approximate" in w.lower() or "summed" in w.lower() for w in resp["_pinecone"]["warnings"])
