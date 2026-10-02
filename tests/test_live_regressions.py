"""Regressions found in the live AWS OpenSearch vs Pinecone run."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from pinecone_opensearch import OpenSearch, helpers
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
