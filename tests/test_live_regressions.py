"""Regressions found in the live AWS OpenSearch vs Pinecone run."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from pinecone_opensearch import OpenSearch, UnsupportedQueryError, helpers
from .fakes import make_backend

CFG = {
    "mode": "strict",
    "indices": {
        "docs": {
            "pinecone_index": "docs-pinecone",
            "knn_field": "embedding",
            "mapping": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {"type": "knn_vector", "dimension": 3, "space_type": "cosinesimil"},
                }
            },
        }
    },
    "embedders": {
        "llama-text-embed-v2": {
            "type": "pinecone",
            "model": "llama-text-embed-v2",
            "parameters": {"input_type": "query"},
            "dimension": 3,
        }
    },
}


class StrictInference:
    """Mirrors pinecone 10 ``Inference.embed(model, inputs, parameters=None)``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def embed(
        self,
        model: str,
        inputs: str | Sequence[str],
        parameters: Mapping[str, Any] | None = None,
    ) -> Any:
        self.calls.append({"model": model, "inputs": list(inputs), "parameters": dict(parameters or {})})
        return SimpleNamespace(data=[SimpleNamespace(values=[0.1, 0.2, 0.3]) for _ in inputs])


def test_neural_query_uses_config_embedder_with_parameters():
    backend, _store, created = make_backend()
    created.add("docs-pinecone")
    backend.pc.inference = StrictInference()
    client = OpenSearch(config=CFG, backend=backend)
    client.index(index="docs", id="a", body={"text": "vitamin d", "embedding": [0.1, 0.2, 0.3]})

    resp = client.search(index="docs", body={
        "query": {"neural": {"embedding": {"query_text": "bone density", "model_id": "llama-text-embed-v2", "k": 5}}}
    })

    assert [h["_id"] for h in resp["hits"]["hits"]] == ["a"]
    assert backend.pc.inference.calls == [{
        "model": "llama-text-embed-v2",
        "inputs": ["bone density"],
        "parameters": {"input_type": "query", "dimension": 3},
    }]


def test_bulk_splits_upserts_under_2mb():
    backend, store, created = make_backend()
    created.add("docs-pinecone")
    sizes: list[int] = []
    docs_api = backend.documents("docs-pinecone")
    original = docs_api.upsert

    def recording_upsert(*, namespace: str, documents: list[Any], **kwargs: Any) -> Any:
        sizes.append(len(json.dumps(documents).encode()))
        return original(namespace=namespace, documents=documents, **kwargs)

    docs_api.upsert = recording_upsert
    backend._indexes["docs-pinecone"].documents = docs_api
    client = OpenSearch(config=CFG, backend=backend)
    big = "x" * 50_000
    actions = [{"_index": "docs", "_id": str(i), "_source": {"text": big, "embedding": [0.1, 0.2, 0.3]}} for i in range(120)]

    success, errors = helpers.bulk(client, actions, chunk_size=500)

    assert (success, errors) == (120, [])
    assert len(sizes) > 1
    assert max(sizes) < 2 * 1024 * 1024
    assert len(store["docs-pinecone"]["__default__"]) == 120


META_CFG = {
    "mode": "strict",
    "indices": {
        "meta": {
            "pinecone_index": "meta-pinecone",
            "mapping": {
                "properties": {
                    "title": {"type": "text"},
                    "year": {"type": "integer"},
                    "score": {"type": "float"},
                    "published": {"type": "date"},
                    "seen_at": {"type": "date"},
                    "tags": {"type": "keyword"},
                }
            },
        }
    },
}

META_DOC = {
    "title": "probe",
    "year": 2008,
    "score": 2.5,
    "published": "2008-12-13",
    "seen_at": "2008-12-13T10:30:00.250Z",
    "tags": ["a", "b"],
}


def _meta_client() -> tuple[OpenSearch, Any]:
    backend, _store, created = make_backend()
    created.add("meta-pinecone")
    client = OpenSearch(config=META_CFG, backend=backend)
    client.index(index="meta", id="d1", body=META_DOC)
    return client, backend


def test_source_restores_mapped_types_on_get_mget_and_search():
    client, _ = _meta_client()

    assert client.get(index="meta", id="d1")["_source"] == META_DOC
    assert client.mget(index="meta", body={"ids": ["d1"]})["docs"][0]["_source"] == META_DOC
    hit = client.search(index="meta", body={"query": {"term": {"year": 2008}}})["hits"]["hits"][0]
    assert hit["_source"] == META_DOC


def test_get_and_mget_honor_source_params():
    client, _ = _meta_client()

    assert client.get(index="meta", id="d1", _source_includes=["year", "tags"])["_source"] == {"year": 2008, "tags": ["a", "b"]}
    assert client.get(index="meta", id="d1", _source_includes="year,title")["_source"] == {"year": 2008, "title": "probe"}
    assert "title" not in client.get(index="meta", id="d1", _source_excludes=["title"])["_source"]
    assert client.get(index="meta", id="d1", _source=False)["_source"] == {}
    docs = client.mget(index="meta", body={"ids": ["d1"]}, _source_includes=["year"])["docs"]
    assert docs[0]["_source"] == {"year": 2008}


def test_update_refresh_waits_for_patched_values():
    client, backend = _meta_client()
    docs_api = backend.documents("meta-pinecone")
    real_update, real_fetch = docs_api.update, docs_api.fetch
    pending: list[dict[str, Any]] = []
    fetches_until_visible = 3

    def lagging_update(**kwargs: Any) -> Any:
        pending.append(kwargs)
        return SimpleNamespace()

    def lagging_fetch(**kwargs: Any) -> Any:
        nonlocal fetches_until_visible
        if pending:
            fetches_until_visible -= 1
            if fetches_until_visible <= 0:
                real_update(**pending.pop())
        return real_fetch(**kwargs)

    docs_api.update, docs_api.fetch = lagging_update, lagging_fetch
    backend._indexes["meta-pinecone"].documents = docs_api

    client.update(index="meta", id="d1", body={"doc": {"year": 2020}}, refresh="wait_for")

    assert client.get(index="meta", id="d1")["_source"]["year"] == 2020


COUNT_CFG = {
    "mode": "strict",
    "indices": {
        "c": {
            "pinecone_index": "c-pinecone",
            "knn_field": "embedding",
            "mapping": {
                "properties": {
                    "category": {"type": "keyword"},
                    "open": {"type": "boolean"},
                    "embedding": {"type": "knn_vector", "dimension": 3, "space_type": "cosinesimil"},
                }
            },
        }
    },
}

SHOULD_ONLY = {"bool": {"should": [{"term": {"category": "a"}}, {"term": {"open": False}}], "minimum_should_match": 1}}


def _count_client(n: int = 12) -> tuple[OpenSearch, Any, list[str]]:
    backend, _store, created = make_backend()
    created.add("c-pinecone")
    client = OpenSearch(config=COUNT_CFG, backend=backend)
    for i in range(n):
        client.index(index="c", id=str(i), body={"category": "abc"[i % 3], "open": i % 2 == 0, "embedding": [0.1, 0.2, 0.3]})
    docs_api = backend.documents("c-pinecone")
    calls: list[str] = []
    for name in ("search", "fetch", "list"):
        original = getattr(docs_api, name, None)
        if original is None:
            continue

        def wrapper(*args: Any, _name: str = name, _original: Any = original, **kwargs: Any) -> Any:
            calls.append(_name)
            return _original(*args, **kwargs)

        setattr(docs_api, name, wrapper)
    backend._indexes["c-pinecone"].documents = docs_api
    return client, backend, calls


def test_should_only_term_clauses_are_ored():
    client, _, _ = _count_client()
    plan = client.translate(index="c", body={"query": SHOULD_ONLY})
    assert plan["calls"][0]["request"]["filter"] == {"$or": [{"category": {"$eq": "a"}}, {"open": {"$eq": False}}]}
    assert client.count(index="c", body={"query": SHOULD_ONLY})["count"] == 8


def test_should_constant_clauses_with_filter_are_dropped_with_warning():
    client, _, _ = _count_client()
    body = {"query": {"bool": {"filter": [{"term": {"open": True}}], "should": [{"term": {"category": "a"}}]}}}
    plan = client.translate(index="c", body=body)
    assert plan["calls"][0]["request"]["filter"] == {"open": {"$eq": True}}
    assert any("should" in w and "dropped" in w for w in plan["warnings"])


def test_should_minimum_should_match_above_one_is_unsupported():
    client, _, _ = _count_client()
    body = {"query": {"bool": {"should": [{"term": {"category": "a"}}, {"term": {"open": True}}], "minimum_should_match": 2}}}
    with pytest.raises(UnsupportedQueryError):
        client.translate(index="c", body=body)


def test_filtered_count_uses_one_ids_only_search():
    client, _, calls = _count_client()
    assert client.count(index="c", body={"query": {"term": {"category": "b"}}})["count"] == 4
    assert calls == ["search"]


def test_filtered_count_falls_back_to_fetch_when_search_saturates(monkeypatch):
    import pinecone_opensearch.client as client_mod

    monkeypatch.setattr(client_mod, "_COUNT_SEARCH_TOP_K", 3)
    client, _, calls = _count_client()
    assert client.count(index="c", body={"query": {"term": {"open": True}}})["count"] == 6
    assert calls == ["search", "fetch"]
