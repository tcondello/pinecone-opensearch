"""Compare count() on Pinecone (via pinecone_opensearch) with OpenSearch across filters, with latency."""

from __future__ import annotations

import time

from common import CONFIG, os_client
from pinecone_opensearch import OpenSearch

INDEX = "scifact-meta"
QUERIES = [
    {"exists": {"field": "cited_by"}},
    {"bool": {"must_not": [{"term": {"category": "biology"}}]}},
    {"term": {"open_access": True}},
    {"terms": {"tags": ["mouse", "human"]}},
    {"range": {"year": {"gte": 2003, "lt": 2021}}},
    {"range": {"published": {"gte": "2010-01-01", "lte": "2014-12-31"}}},
    {"term": {"category": "medicine"}},
    {"range": {"cited_by": {"gte": 450}}},
    {"bool": {"filter": [{"term": {"category": "genetics"}}, {"term": {"year": 2012}}]}},
    {"bool": {"should": [{"term": {"category": "biology"}}, {"term": {"open_access": False}}], "minimum_should_match": 1}},
    {"bool": {"filter": [{"terms": {"tags": ["review"]}}], "must_not": [{"range": {"year": {"lt": 2010}}}]}},
]


def timed(fn, reps: int = 5) -> tuple[int, float]:
    fn()
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        n = fn()["count"]
        samples.append((time.perf_counter() - t) * 1000)
    return n, sorted(samples)[len(samples) // 2]


def main() -> None:
    es, pc = os_client(), OpenSearch(config=CONFIG)
    bad = 0
    for q in QUERIES:
        body = {"query": q}
        os_n, os_ms = timed(lambda: es.count(index=INDEX, body=body))
        pc_n, pc_ms = timed(lambda: pc.count(index=INDEX, body=body))
        ok = os_n == pc_n
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} OS={os_n:5d} {os_ms:6.1f}ms | PC={pc_n:5d} {pc_ms:6.1f}ms | {str(q)[:80]}", flush=True)
    print(f"{len(QUERIES) - bad}/{len(QUERIES)} counts match")


if __name__ == "__main__":
    main()
