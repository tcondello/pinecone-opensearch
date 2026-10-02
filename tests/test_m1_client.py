"""M1 unit tests with a fake Pinecone backend (no network)."""

from __future__ import annotations

import pytest
from opensearchpy.exceptions import ConflictError, NotFoundError

from pinecone_opensearch import OpenSearch, UnsupportedQueryError, helpers
from pinecone_opensearch.scores import convert_knn_score
from pinecone_opensearch.writes import (
    flatten_and_convert,
    translate_os_mappings_to_pinecone_schema,
    unflatten_source,
)
from .fakes import make_backend

PRODUCTS_CFG = {
    "mode": "strict",
    "indices": {
        "products": {
            "pinecone_index": "products-pinecone",
            "knn_field": "embedding",
            "mapping": {
                "properties": {
                    "title": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "brand": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "price": {"type": "float"},
                    "in_stock": {"type": "boolean"},
                    "address": {
                        "type": "object",
                        "properties": {"city": {"type": "keyword"}},
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 3,
                        "space_type": "cosinesimil",
                    },
                }
            },
        }
    },
    "aliases": {"products_alias": "products"},
    "include_pinecone_meta": True,
}


@pytest.fixture
def client() -> OpenSearch:
    backend, _store, created = make_backend()
    created.add("products-pinecone")
    return OpenSearch(pinecone_api_key="test", config=PRODUCTS_CFG, backend=backend)


def _seed(client: OpenSearch) -> None:
    client.index(
        index="products",
        id="sku-1",
        body={
            "title": "Trail runner",
            "brand": "Acme",
            "category": "shoes",
            "price": 129,
            "in_stock": True,
            "address": {"city": "Austin"},
            "embedding": [0.1, 0.2, 0.3],
        },
    )
    client.index(
        index="products",
        id="sku-2",
        body={
            "title": "Office chair",
            "brand": "Acme",
            "category": "furniture",
            "price": 250,
            "in_stock": False,
            "address": {"city": "Dallas"},
            "embedding": [0.9, 0.1, 0.0],
        },
    )


def test_search_bm25_response_shape(client: OpenSearch):
    _seed(client)
    resp = client.search(
        index="products",
        body={
            "size": 10,
            "query": {
                "bool": {
                    "must": [{"match": {"title": "trail runner"}}],
                    "filter": [{"term": {"brand": "Acme"}}],
                }
            },
        },
    )
    assert "hits" in resp
    assert "took" in resp
    assert resp["_shards"]["successful"] == 1
    assert resp["hits"]["hits"]
    hit = resp["hits"]["hits"][0]
    assert hit["_index"] == "products"
    assert hit["_id"] == "sku-1"
    assert "_score" in hit
    assert hit["_source"]["title"] == "Trail runner"
    assert hit["_source"]["address"]["city"] == "Austin"
    assert resp["_pinecone"]["plan"] == "single_search"


def test_search_knn_score_conversion(client: OpenSearch):
    _seed(client)
    resp = client.search(
        index="products",
        body={
            "size": 5,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": [0.1, 0.2, 0.3],
                        "k": 5,
                        "filter": {"term": {"category": "shoes"}},
                    }
                }
            },
        },
    )
    hit = resp["hits"]["hits"][0]
    assert hit["_id"] == "sku-1"
    # Fake raw score = dot(query, embedding) = 0.14 → OpenSearch cosine (1+c)/2
    assert hit["_score"] == pytest.approx(convert_knn_score(0.14, metric="cosine"))



def test_search_fetch_match_all(client: OpenSearch):
    _seed(client)
    resp = client.search(index="products", body={"query": {"match_all": {}}, "size": 10})
    assert resp["_pinecone"]["plan"] == "fetch"
    assert len(resp["hits"]["hits"]) == 2
    assert all(h["_score"] == 1.0 for h in resp["hits"]["hits"])


def test_get_and_mget(client: OpenSearch):
    _seed(client)
    doc = client.get(index="products", id="sku-1")
    assert doc["found"] is True
    assert doc["_source"]["brand"] == "Acme"

    with pytest.raises(NotFoundError):
        client.get(index="products", id="missing")

    m = client.mget(body={"docs": [{"_index": "products", "_id": "sku-1"}, {"_index": "products", "_id": "nope"}]})
    assert m["docs"][0]["found"] is True
    assert m["docs"][1]["found"] is False


def test_count_match_all_uses_stats(client: OpenSearch):
    _seed(client)
    resp = client.count(index="products", body={"query": {"match_all": {}}})
    assert resp["count"] == 2


def test_create_conflict(client: OpenSearch):
    _seed(client)
    with pytest.raises(ConflictError):
        client.create(index="products", id="sku-1", body={"title": "dup"})


def test_update_and_delete(client: OpenSearch):
    _seed(client)
    upd = client.update(index="products", id="sku-1", body={"doc": {"price": 99}})
    assert upd["result"] == "updated"
    assert client.get(index="products", id="sku-1")["_source"]["price"] == 99.0

    deleted = client.delete(index="products", id="sku-2")
    assert deleted["result"] == "deleted"
    with pytest.raises(NotFoundError):
        client.get(index="products", id="sku-2")


def test_bulk_and_helpers(client: OpenSearch):
    body = [
        {"index": {"_index": "products", "_id": "b1"}},
        {"title": "Bulk one", "brand": "Z", "price": 10, "embedding": [0.0, 1.0, 0.0]},
        {"index": {"_index": "products", "_id": "b2"}},
        {"title": "Bulk two", "brand": "Z", "price": 20, "embedding": [0.0, 0.0, 1.0]},
        {"delete": {"_index": "products", "_id": "b2"}},
    ]
    resp = client.bulk(body=body)
    assert resp["errors"] is False
    assert client.get(index="products", id="b1")["found"] is True

    success, errors = helpers.bulk(
        client,
        [
            {"_op_type": "index", "_index": "products", "_id": "h1", "title": "Helper", "brand": "H"},
        ],
    )
    assert success == 1
    assert errors == []


def test_msearch(client: OpenSearch):
    _seed(client)
    resp = client.msearch(
        body=[
            {"index": "products"},
            {"query": {"match": {"title": "trail"}}},
            {"index": "products"},
            {"query": {"term": {"brand": "Acme"}}},
        ]
    )
    assert len(resp["responses"]) == 2
    assert "hits" in resp["responses"][0]


def test_indices_create_mapping_refresh_ping_info(client: OpenSearch):
    schema, warnings = translate_os_mappings_to_pinecone_schema(
        {
            "properties": {
                "title": {"type": "text", "analyzer": "english"},
                "embedding": {"type": "knn_vector", "dimension": 8, "space_type": "l2"},
                "tag": {"type": "keyword"},
            }
        }
    )
    assert "title" in schema["fields"]
    assert schema["fields"]["title"]["type"] == "string"
    assert schema["fields"]["embedding"]["metric"] == "euclidean"
    assert "tag" not in schema["fields"]  # metadata omitted

    created = client.indices.create(
        index="products",
        body={
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "embedding": {"type": "knn_vector", "dimension": 3, "space_type": "cosinesimil"},
                }
            }
        },
    )
    assert created["acknowledged"] is True
    assert client.indices.exists(index="products") is True
    mapping = client.indices.get_mapping(index="products")
    assert "products" in mapping

    _seed(client)
    refreshed = client.indices.refresh(index="products")
    assert "_shards" in refreshed

    assert client.ping() is True
    info = client.info()
    assert info["version"]["distribution"] == "pinecone"


def test_write_field_flatten_and_unflatten(client: OpenSearch):
    flat = flatten_and_convert(
        {"title": "X", "address": {"city": "Austin"}, "price": 10},
        index="products",
        registry=client.registry,
        doc_id="t1",
    )
    assert flat["address__city"] == "Austin"
    assert flat["title__keyword"] == "X"
    assert flat["_id"] == "t1"
    nested = unflatten_source({k: v for k, v in flat.items() if k != "_id"})
    assert nested["address"]["city"] == "Austin"


def test_translate_still_works(client: OpenSearch):
    plan = client.translate(
        index="products",
        body={"query": {"match": {"title": "trail"}}, "size": 5},
    )
    assert plan["plan"] == "single_search"


def test_nested_field_rejected_on_indices_create(client: OpenSearch):
    with pytest.raises(UnsupportedQueryError):
        client.indices.create(
            index="bad",
            body={"mappings": {"properties": {"items": {"type": "nested"}}}},
        )
