"""Live filters / writes / point-lookup / neural differential on the scifact-meta index.

  uv run filters_writes.py load    # create + load scifact-meta on both engines (idempotent)
  uv run filters_writes.py run     # run the differential, write out/fw_shadow.jsonl + print summary
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict

from opensearchpy import helpers as os_helpers

from common import CONFIG, DATA, OUT, corpus_docs, live_config, os_client, read_jsonl
from pinecone_opensearch import OpenSearch, ShadowClient, helpers as pc_helpers

INDEX = "scifact-meta"
CATEGORIES = ["biology", "medicine", "chemistry", "genetics", "neuroscience"]
TAGS = ["clinical", "in-vitro", "mouse", "human", "review", "meta-analysis", "cohort"]


def meta_for(doc_id: str) -> dict:
    h = int(hashlib.sha256(doc_id.encode()).hexdigest(), 16)
    year = 2000 + h % 25
    return {
        "category": CATEGORIES[h % 5],
        "tags": sorted({TAGS[(h >> 8) % 7], TAGS[(h >> 16) % 7]}),
        "year": year,
        "cited_by": (h >> 24) % 500,
        "open_access": bool((h >> 32) % 2),
        "published": f"{year}-{1 + (h >> 40) % 12:02d}-{1 + (h >> 48) % 28:02d}",
    }


def docs_with_meta() -> list[dict]:
    return [{**d, **meta_for(d["_id"])} for d in corpus_docs()]


def actions(docs: list[dict]) -> list[dict]:
    return [{"_index": INDEX, "_id": d["_id"], "_source": {k: v for k, v in d.items() if k != "_id"}} for d in docs]


def cmd_load() -> None:
    docs = docs_with_meta()
    cfg = live_config()
    mapping = cfg["indices"][INDEX]["mapping"]

    es = os_client()
    if not es.indices.exists(index=INDEX):
        os_mapping = {"properties": dict(mapping["properties"])}
        os_mapping["properties"]["embedding"] = {
            "type": "knn_vector", "dimension": 1024,
            "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"},
        }
        es.indices.create(index=INDEX, body={
            "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": os_mapping,
        })
    es.indices.refresh(index=INDEX)
    if es.count(index=INDEX)["count"] < len(docs):
        started = time.time()
        for b in range(0, len(docs), 200):
            os_helpers.bulk(es, actions(docs[b : b + 200]))
            print(f"  opensearch {min(b + 200, len(docs))}/{len(docs)} {time.time() - started:.0f}s", flush=True)
        es.indices.refresh(index=INDEX)
    print(f"opensearch: count={es.count(index=INDEX)['count']}", flush=True)

    client = OpenSearch(config=CONFIG)
    pc_name = cfg["indices"][INDEX]["pinecone_index"]
    pc = client._backend.pc
    if not pc.indexes.exists(pc_name):
        print(f"pinecone: create {client.indices.create(index=INDEX, body={'mappings': mapping})}", flush=True)
    have = int(pc.index(name=pc_name).describe_index_stats().total_vector_count or 0)
    if have < len(docs):
        started = time.time()
        for b in range(0, len(docs), 1000):
            ok, errors = pc_helpers.bulk(client, actions(docs[b : b + 1000]), raise_on_error=False)
            if errors:
                sys.exit(f"pinecone bulk errors: {errors[:2]}")
            print(f"  pinecone {min(b + 1000, len(docs))}/{len(docs)} {time.time() - started:.0f}s", flush=True)
    deadline = time.time() + 300
    while time.time() < deadline:
        n = int(pc.index(name=pc_name).describe_index_stats().total_vector_count or 0)
        print(f"pinecone: count={n}", flush=True)
        if n >= len(docs):
            break
        time.sleep(10)


def search_cases(texts: dict[str, str], vecs: dict[str, list[float]]) -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = []
    qids = list(vecs)[:40]
    filters = [
        ("term", {"term": {"category": "medicine"}}),
        ("terms", {"terms": {"tags": ["mouse", "human"]}}),
        ("range_int", {"range": {"year": {"gte": 2015, "lt": 2020}}}),
        ("range_date", {"range": {"published": {"gte": "2010-01-01", "lte": "2014-12-31"}}}),
        ("bool_flag", {"term": {"open_access": True}}),
        ("must_not", {"bool": {"must_not": [{"term": {"category": "biology"}}]}}),
        ("exists", {"exists": {"field": "cited_by"}}),
    ]
    for i, qid in enumerate(qids):
        fname, f = filters[i % len(filters)]
        match = {"match": {"text": texts[qid]}}
        cases.append((f"bool_match+{fname}", {"size": 10, "_source": ["title"], "query": {"bool": {"must": [match], "filter": [f]}}}))
        cases.append((f"knn_filter+{fname}", {"size": 10, "_source": ["title"], "query": {"knn": {"embedding": {"vector": vecs[qid], "k": 10, "filter": f}}}}))
    combos = [
        {"bool": {"filter": [{"term": {"category": "genetics"}}, {"term": {"year": 2012}}]}},
        {"bool": {"filter": [{"terms": {"tags": ["review"]}}, {"range": {"cited_by": {"gte": 450}}}]}},
        {"bool": {"filter": [{"term": {"open_access": False}}, {"range": {"published": {"gte": "2024-06-01"}}}]}},
        {"bool": {"filter": [{"term": {"category": "neuroscience"}}], "must_not": [{"terms": {"tags": ["mouse"]}}], "should": [], "minimum_should_match": 0}},
    ]
    for q in combos:
        cases.append(("filter_only", {"size": 1000, "_source": ["title"], "query": q}))
    return cases


def cmd_run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "fw_shadow.jsonl"
    log.unlink(missing_ok=True)
    es = os_client()
    pc = OpenSearch(config=CONFIG)
    shadow = ShadowClient(es, pc, log, warm_up=0)
    texts = {q["_id"]: q["text"] for q in read_jsonl(DATA / "scifact" / "queries.jsonl")}
    vecs = {r["_id"]: r["values"] for r in read_jsonl(DATA / "queries.emb.jsonl")}
    failures: dict[str, list[str]] = defaultdict(list)

    def step(label: str, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            failures[label].append(f"{type(exc).__name__}: {str(exc)[:300]}")
            return None

    cases = search_cases(texts, vecs)
    for i, (label, body) in enumerate(cases, 1):
        step(label, lambda: shadow.search(index=INDEX, body=body))
        if i % 20 == 0 or i == len(cases):
            print(f"  search {i}/{len(cases)}", flush=True)

    for label, body in cases:
        if label == "filter_only":
            step("count", lambda: shadow.count(index=INDEX, body={"query": body["query"]}))
    print("  counts done", flush=True)

    ids = [d["_id"] for d in read_jsonl(DATA / "scifact" / "corpus.jsonl")[:25]]
    for doc_id in ids[:10]:
        step("get", lambda: shadow.get(index=INDEX, id=doc_id, _source_includes=["title", "category", "year", "tags", "published"]))
    step("mget", lambda: shadow.mget(index=INDEX, body={"ids": ids}, _source_includes=["title", "category"]))
    fields = ["title", "category", "year", "cited_by", "open_access", "published", "tags"]
    source_mismatches = []
    for doc_id in ids:
        o = step("get_compare", lambda: es.get(index=INDEX, id=doc_id, _source_includes=fields))
        p_ = step("get_compare", lambda: pc.get(index=INDEX, id=doc_id, _source_includes=fields))
        if o and p_ and o["_source"] != p_["_source"]:
            source_mismatches.append({"id": doc_id, "opensearch": o["_source"], "pinecone": p_["_source"]})
    print("  point lookups done", flush=True)

    write_rows = []
    for n in range(10):
        doc_id = f"live-write-{n}"
        src = {"title": f"Live write probe {n}", "text": "zymurgic probe document for write visibility",
               "category": "chemistry", "tags": ["probe"], "year": 2030, "cited_by": n,
               "open_access": True, "published": "2030-01-15", "embedding": vecs[next(iter(vecs))]}
        step("index", lambda: shadow.index(index=INDEX, id=doc_id, body=src, refresh="wait_for"))
        t0 = time.perf_counter()
        visible = {}
        for side, client in (("opensearch", es), ("pinecone", pc)):
            r = step("get_after_index", lambda: client.get(index=INDEX, id=doc_id))
            visible[side] = bool(r and r.get("found"))
        step("update", lambda: shadow.update(index=INDEX, id=doc_id, body={"doc": {"cited_by": 999, "category": "genetics"}}, refresh="wait_for"))
        after = {}
        for side, client in (("opensearch", es), ("pinecone", pc)):
            r = step("get_after_update", lambda: client.get(index=INDEX, id=doc_id))
            after[side] = (r or {}).get("_source", {}).get("cited_by")
        write_rows.append({"id": doc_id, "visible_after_index": visible, "cited_by_after_update": after,
                           "elapsed_ms": (time.perf_counter() - t0) * 1000})
    probe_q = {"query": {"bool": {"filter": [{"term": {"tags": "probe"}}, {"term": {"category": "genetics"}}]}}}
    counts_before = {s: step("count_probe", lambda: c.count(index=INDEX, body=probe_q)) for s, c in (("opensearch", es), ("pinecone", pc))}
    for n in range(5):
        step("delete", lambda: shadow.delete(index=INDEX, id=f"live-write-{n}", refresh="wait_for"))
    step("delete_by_query", lambda: shadow.delete_by_query(index=INDEX, body={"query": {"term": {"tags": "probe"}}}, refresh=True))
    time.sleep(5)
    es.indices.refresh(index=INDEX)
    counts_after = {s: step("count_probe", lambda: c.count(index=INDEX, body={"query": {"term": {"tags": "probe"}}})) for s, c in (("opensearch", es), ("pinecone", pc))}
    print("  writes done", flush=True)

    neural = []
    for qid in list(vecs)[:20]:
        knn = step("neural_ref", lambda: pc.search(index=INDEX, body={"size": 10, "_source": False, "query": {"knn": {"embedding": {"vector": vecs[qid], "k": 10}}}}))
        neu = step("neural", lambda: pc.search(index=INDEX, body={"size": 10, "_source": False, "query": {"neural": {"embedding": {"query_text": texts[qid], "model_id": "llama-text-embed-v2", "k": 10}}}}))
        if knn and neu:
            a = [h["_id"] for h in knn["hits"]["hits"]]
            b = [h["_id"] for h in neu["hits"]["hits"]]
            neural.append({"overlap": len(set(a) & set(b)) / 10, "embed_ms": neu["_pinecone"]["embed_ms"]})
    print("  neural done", flush=True)

    summary = {
        "failures": {k: {"n": len(v), "first": v[0]} for k, v in failures.items()},
        "get_source_mismatches": {"n_checked": len(ids), "mismatches": source_mismatches[:3], "n": len(source_mismatches)},
        "writes": write_rows,
        "probe_counts_before_delete": counts_before,
        "probe_counts_after_delete": counts_after,
        "neural_vs_knn": {"n": len(neural), "mean_overlap": sum(x["overlap"] for x in neural) / max(len(neural), 1),
                           "mean_embed_ms": sum(x["embed_ms"] for x in neural) / max(len(neural), 1)},
    }
    (OUT / "fw_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["failures"], indent=2))


if __name__ == "__main__":
    {"load": cmd_load, "run": cmd_run}[sys.argv[1]]()
