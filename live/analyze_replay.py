"""Per-query-type recall@10 (both engines), top-10 overlap and latency percentiles from a replay shadow log."""

from __future__ import annotations

import json
import sys
from collections import defaultdict

from common import OUT, read_jsonl


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main() -> None:
    log = OUT / (sys.argv[1] if len(sys.argv) > 1 else "shadow.jsonl")
    qrels = {q["body_hash"]: q for q in read_jsonl(OUT / "qrels.jsonl")}
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in read_jsonl(log):
        if rec["warmup"]:
            continue
        q = qrels[rec["body_hash"]]
        rel = set(q["docs"])
        a = agg[q["qtype"]]
        for side in ("primary", "secondary"):
            ids = rec[side].get("ids") or []
            a[f"{side}_recall"].append(len(rel & set(ids[:10])) / len(rel))
            a[f"{side}_lat"].append(rec["latency_ms"][side])
        a["overlap"].append(rec["metrics"]["overlap@k"])
        a["errors"].append(1.0 if rec["secondary"].get("error") else 0.0)
    print(f"{'type':7} {'n':>4} {'recall OS':>9} {'recall PC':>9} {'overlap':>7} | {'p50 OS':>6} {'p50 PC':>6} {'p95 OS':>6} {'p95 PC':>6} {'p99 OS':>6} {'p99 PC':>6} ms | errors")
    for qtype, a in sorted(agg.items()):
        n = len(a["overlap"])
        print(
            f"{qtype:7} {n:4d} {sum(a['primary_recall']) / n:9.3f} {sum(a['secondary_recall']) / n:9.3f} {sum(a['overlap']) / n:7.3f} | "
            + " ".join(f"{pct(a[f'{s}_lat'], q):6.1f}" for q in (0.5, 0.95, 0.99) for s in ("primary", "secondary"))
            + f" ms | {int(sum(a['errors']))}"
        )


if __name__ == "__main__":
    main()
