"""Golden M0 translate() tests — no network."""

from __future__ import annotations

import pytest

from pinecone_opensearch import OpenSearch, UnsupportedQueryError
from pinecone_opensearch.config import load_config


VECTOR = [0.1, 0.2, 0.3]


def test_knn_with_filters(client: OpenSearch):
    """Worked example: kNN + knn.filter → single_search dense_vector + filter."""
    body = {
        "size": 10,
        "query": {
            "knn": {
                "embedding": {
                    "vector": VECTOR,
                    "k": 10,
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"category": "shoes"}},
                                {"range": {"price": {"gte": 50, "lte": 200}}},
                            ]
                        }
                    },
                }
            }
        },
    }
    plan = client.translate(index="products", body=body)

    assert plan["plan"] == "single_search"
    assert plan["pagination"] == {"from": 0, "size": 10, "top_k": 10}
    assert len(plan["calls"]) == 1
    call = plan["calls"][0]
    assert call["method"] == "documents.search"
    assert call["index"] == "products-pinecone"
    assert call["namespace"] == "__default__"
    assert call["request"]["score_by"] == {
        "dense_vector": {
            "field": "embedding",
            "values": VECTOR,
            "top_k": 10,
        }
    }
    filt = call["request"]["filter"]
    # $and of category $eq and price range
    assert filt == {
        "$and": [
            {"category": {"$eq": "shoes"}},
            {"price": {"$gte": 50, "$lte": 200}},
        ]
    }


def test_bm25_with_filters(client: OpenSearch):
    """Worked example: match + bool.filter → single_search text + filter."""
    body = {
        "size": 20,
        "from": 0,
        "query": {
            "bool": {
                "must": [{"match": {"title": "trail runner"}}],
                "filter": [
                    {"term": {"brand": "Acme"}},
                    {"range": {"price": {"lt": 150}}},
                ],
            }
        },
    }
    plan = client.translate(index="products", body=body)

    assert plan["plan"] == "single_search"
    assert plan["pagination"]["top_k"] == 20
    call = plan["calls"][0]
    assert call["method"] == "documents.search"
    assert call["request"]["score_by"] == {
        "text": {"fields": ["title"], "query": "trail runner"}
    }
    assert call["request"]["filter"] == {
        "$and": [
            {"brand": {"$eq": "Acme"}},
            {"price": {"$lt": 150}},
        ]
    }


def test_hybrid_fan_out_plan_shape(client: OpenSearch):
    """Worked example: hybrid.queries → fan_out with shared filter."""
    body = {
        "size": 10,
        "query": {
            "hybrid": {
                "queries": [
                    {"match": {"title": "running shoes"}},
                    {"knn": {"embedding": {"vector": VECTOR, "k": 10}}},
                ],
                "filter": {"term": {"in_stock": True}},
                "pagination_depth": 50,
            }
        },
        "search_pipeline": "norm-pipeline",
    }
    plan = client.translate(index="products", body=body)

    assert plan["plan"] == "fan_out"
    assert len(plan["calls"]) == 2
    assert plan["fusion"]["search_pipeline"] == "norm-pipeline"
    assert plan["fusion"]["strategy"] == "pipeline"

    text_call, knn_call = plan["calls"]
    assert text_call["request"]["score_by"] == {
        "text": {"fields": ["title"], "query": "running shoes"}
    }
    assert knn_call["request"]["score_by"] == {
        "dense_vector": {
            "field": "embedding",
            "values": VECTOR,
            "top_k": 50,  # pagination_depth wins for hybrid subqueries' top_k helper
        }
    }
    # Shared filter on both
    for call in plan["calls"]:
        assert call["request"]["filter"] == {"in_stock": {"$eq": True}}
        assert call["request"]["top_k"] == 50


def test_field_path_rewrite_object_and_keyword(client: OpenSearch):
    body = {
        "size": 5,
        "query": {
            "bool": {
                "must": [{"match": {"title": "runner"}}],
                "filter": [
                    {"term": {"address.city": "Austin"}},
                    {"term": {"title.keyword": "Trail Runner"}},
                ],
            }
        },
    }
    plan = client.translate(index="products", body=body)
    filt = plan["calls"][0]["request"]["filter"]
    assert filt == {
        "$and": [
            {"address__city": {"$eq": "Austin"}},
            {"title__keyword": {"$eq": "Trail Runner"}},
        ]
    }


def test_alias_resolution(client: OpenSearch):
    body = {"size": 1, "query": {"match_all": {}}}
    plan = client.translate(index="products_alias", body=body)
    assert plan["logical_index"] == "products"
    assert plan["plan"] == "fetch"
    assert plan["calls"][0]["method"] == "documents.fetch"


def test_strict_aggs_unsupported(client: OpenSearch):
    body = {"size": 10, "query": {"match_all": {}}, "aggs": {"by_brand": {"terms": {"field": "brand"}}}}
    with pytest.raises(UnsupportedQueryError) as ei:
        client.translate(index="products", body=body)
    assert ei.value.path == "aggs"


def test_strict_script_score_unsupported(client: OpenSearch):
    body = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {"source": "1"},
            }
        }
    }
    with pytest.raises(UnsupportedQueryError) as ei:
        client.translate(index="products", body=body)
    assert "script_score" in ei.value.path


def test_strict_keyword_range_unsupported(client: OpenSearch):
    body = {
        "query": {
            "bool": {
                "filter": [{"range": {"brand": {"gte": "A", "lte": "M"}}}],
            }
        }
    }
    with pytest.raises(UnsupportedQueryError) as ei:
        client.translate(index="products", body=body)
    assert "range" in ei.value.path or "keyword" in str(ei.value).lower()


def test_strict_multi_scoring_bool(client: OpenSearch):
    body = {
        "size": 10,
        "query": {
            "bool": {
                "must": [
                    {"match": {"title": "shoes"}},
                    {"knn": {"embedding": {"vector": VECTOR, "k": 10}}},
                ]
            }
        },
    }
    with pytest.raises(UnsupportedQueryError) as ei:
        client.translate(index="products", body=body)
    assert "scoring" in str(ei.value).lower() or "knn" in ei.value.path


def test_approx_multi_scoring_bool_fans_out():
    cfg = load_config(
        data={
            "mode": "approx",
            "indices": {
                "products": {
                    "pinecone_index": "products-pinecone",
                    "knn_field": "embedding",
                    "mapping": {
                        "properties": {
                            "title": {"type": "text"},
                            "embedding": {"type": "knn_vector", "space_type": "cosinesimil"},
                        }
                    },
                }
            },
        }
    )
    client = OpenSearch(config=cfg)
    body = {
        "size": 10,
        "query": {
            "bool": {
                "must": [
                    {"match": {"title": "shoes"}},
                    {"knn": {"embedding": {"vector": VECTOR, "k": 10}}},
                ]
            }
        },
    }
    plan = client.translate(index="products", body=body)
    assert plan["plan"] == "fan_out"
    assert len(plan["calls"]) == 2
    assert any("approximate" in w.lower() or "sum" in w.lower() for w in plan["warnings"])


def test_match_all_fetch(client: OpenSearch):
    plan = client.translate(index="products", body={"query": {"match_all": {}}, "size": 10})
    assert plan["plan"] == "fetch"
    assert plan["calls"][0]["request"]["top_k"] == 10
