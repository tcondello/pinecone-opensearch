"""Create the SciFact index on OpenSearch (AWS) and Pinecone (via pinecone_opensearch) and load both.

Idempotent: skips index creation if present and only loads when doc counts are short.
"""

from __future__ import annotations

import sys
import time

from opensearchpy import helpers as os_helpers

from common import CONFIG, INDEX, corpus_docs, live_config, os_client
from pinecone_opensearch import OpenSearch, helpers as pc_helpers

CHUNK = 50


def progress(label: str, done: int, total: int, started: float) -> None:
    rate = done / max(time.time() - started, 1e-6)
    print(f"  {label} {done}/{total} ({100 * done / total:.1f}%) {rate:.0f} docs/s", flush=True)


def load_opensearch(docs: list[dict]) -> None:
    cfg = live_config()
    mapping = cfg["indices"][INDEX]["mapping"]
    os_mapping = {"properties": dict(mapping["properties"])}
    os_mapping["properties"]["embedding"] = {
        "type": "knn_vector",
        "dimension": 1024,
        "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"},
    }
    es = os_client()
    if not es.indices.exists(index=INDEX):
        es.indices.create(index=INDEX, body={
            "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": os_mapping,
        })
        print("opensearch: created index", flush=True)
    es.transport.perform_request("PUT", "/_search/pipeline/hybrid-norm", body=cfg["search_pipelines"]["hybrid-norm"])
    es.indices.refresh(index=INDEX)
    have = es.count(index=INDEX)["count"]
    if have >= len(docs):
        print(f"opensearch: already has {have} docs", flush=True)
        return
    started = time.time()
    done = 0
    for b in range(0, len(docs), 200):
        batch = docs[b : b + 200]
        actions = [{"_index": INDEX, "_id": d["_id"], "_source": {k: v for k, v in d.items() if k != "_id"}} for d in batch]
        os_helpers.bulk(es, actions)
        done += len(batch)
        progress("opensearch", done, len(docs), started)
    es.indices.refresh(index=INDEX)
    print(f"opensearch: count={es.count(index=INDEX)['count']}", flush=True)


def load_pinecone(docs: list[dict]) -> None:
    client = OpenSearch(config=CONFIG)
    cfg = live_config()
    pc_name = cfg["indices"][INDEX]["pinecone_index"]
    pc = client._backend.pc
    if not pc.indexes.exists(pc_name):
        resp = client.indices.create(index=INDEX, body={"mappings": cfg["indices"][INDEX]["mapping"]})
        print(f"pinecone: created index {pc_name}: {resp}", flush=True)
    stats = pc.index(name=pc_name).describe_index_stats()
    have = int(getattr(stats, "total_vector_count", 0) or 0)
    if have >= len(docs):
        print(f"pinecone: already has {have} docs", flush=True)
        return
    started = time.time()
    done = 0
    for b in range(0, len(docs), CHUNK * 10):
        batch = docs[b : b + CHUNK * 10]
        actions = [{"_index": INDEX, "_id": d["_id"], "_source": {k: v for k, v in d.items() if k != "_id"}} for d in batch]
        ok, errors = pc_helpers.bulk(client, actions, chunk_size=CHUNK, raise_on_error=False)
        if errors:
            print(f"pinecone: {len(errors)} errors, first: {errors[0]}", flush=True)
            sys.exit(1)
        done += len(batch)
        progress("pinecone", done, len(docs), started)


def main() -> None:
    docs = corpus_docs()
    print(f"corpus docs: {len(docs)}", flush=True)
    which = sys.argv[1:] or ["opensearch", "pinecone"]
    if "opensearch" in which:
        load_opensearch(docs)
    if "pinecone" in which:
        load_pinecone(docs)


if __name__ == "__main__":
    main()
