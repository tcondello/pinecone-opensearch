"""Split Pinecone-side latency into client overhead vs raw SDK documents.search, per query type."""

from __future__ import annotations

import random
import time

from common import CONFIG, OUT, read_jsonl
from pinecone_opensearch import OpenSearch
from pinecone_opensearch.executor import normalize_score_by


def pct(v: list[float], q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def main() -> None:
    client = OpenSearch(config=CONFIG)
    rows = read_jsonl(OUT / "queries.jsonl")
    random.Random(3).shuffle(rows)
    docs = client._backend.documents("pc-os-scifact")
    stats: dict[str, dict[str, list[float]]] = {}
    for i, row in enumerate(rows[:600], 1):
        body = row["body"]
        s = stats.setdefault(row["qtype"], {"translate": [], "package": [], "sdk": []})
        t = time.perf_counter()
        plan = client.translate(index="scifact", body=body)
        s["translate"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        client.search(index="scifact", body=body)
        s["package"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        for call in plan["calls"]:
            req = call["request"]
            docs.search(namespace=call["namespace"], score_by=normalize_score_by(req["score_by"]),
                        top_k=int(req.get("top_k", 10)), include_fields=req.get("include_fields"), filter=req.get("filter"))
        s["sdk"].append((time.perf_counter() - t) * 1000)
        if i % 100 == 0:
            print(f"  {i}/600", flush=True)
    print(f"{'type':7} {'what':10} {'p50':>7} {'p95':>7} {'p99':>7} ms")
    for qtype, s in sorted(stats.items()):
        for what in ("translate", "package", "sdk"):
            v = s[what][10:]
            print(f"{qtype:7} {what:10} {pct(v, .5):7.1f} {pct(v, .95):7.1f} {pct(v, .99):7.1f}")


if __name__ == "__main__":
    main()
