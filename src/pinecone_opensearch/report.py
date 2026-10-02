"""Build shadow JSONL reports and POC charter pass/fail."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pinecone_opensearch.metrics import percentile, recall_at_k


# POC charter thresholds (spec)
P95_LATENCY_MS = 50.0
RECALL_AT_10_MIN = 0.90
WRITE_TO_READ_MS_MAX = 100.0


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_qrels(path: str | Path | None) -> dict[str, list[str]]:
    """Load qrels as ``body_hash -> [doc_id, ...]`` or ``query_id -> [...]``.

    Accepts TSV ``query_id\\tdoc_id`` or JSONL ``{"id"|"body_hash": ..., "docs": [...]}``.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    out: dict[str, list[str]] = defaultdict(list)
    if p.suffix in {".jsonl", ".json"}:
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("body_hash") or row.get("id") or row.get("query_id"))
            docs = row.get("docs") or row.get("relevant") or []
            if isinstance(docs, str):
                docs = [docs]
            out[key].extend(str(d) for d in docs)
        return dict(out)
    # TSV
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            # qrels formats: qid\tdocid  or  qid\t0\tdocid\trel
            if len(parts) >= 4:
                qid, doc_id = parts[0], parts[2]
            else:
                qid, doc_id = parts[0], parts[1]
            out[qid].append(doc_id)
    return dict(out)


def build_report(
    records: list[dict[str, Any]],
    *,
    qrels: dict[str, list[str]] | None = None,
    warm_up_exclude: bool = True,
    k: int = 10,
) -> dict[str, Any]:
    qrels = qrels or {}
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evaluated: list[dict[str, Any]] = []

    for rec in records:
        if warm_up_exclude and rec.get("warmup"):
            continue
        evaluated.append(rec)
        by_type[str(rec.get("query_type") or "unknown")].append(rec)

    per_type = {qt: _summarize_group(rows, qrels=qrels, k=k) for qt, rows in sorted(by_type.items())}
    overall = _summarize_group(evaluated, qrels=qrels, k=k)

    charter = _poc_charter(overall)
    return {
        "n_records_total": len(records),
        "n_evaluated": len(evaluated),
        "overall": overall,
        "by_query_type": per_type,
        "poc_charter": charter,
    }


def build_report_from_path(
    log_path: str | Path,
    *,
    qrels_path: str | Path | None = None,
    warm_up_exclude: bool = True,
    k: int = 10,
) -> dict[str, Any]:
    records = load_jsonl(log_path)
    qrels = load_qrels(qrels_path)
    # Fix n_records when we already materialized
    report = build_report(records, qrels=qrels, warm_up_exclude=warm_up_exclude, k=k)
    report["n_records_total"] = len(records)
    return report


def _summarize_group(
    rows: list[dict[str, Any]],
    *,
    qrels: dict[str, list[str]],
    k: int,
) -> dict[str, Any]:
    lat_pri = [float((r.get("latency_ms") or {}).get("primary") or 0.0) for r in rows]
    lat_sec = [float((r.get("latency_ms") or {}).get("secondary") or 0.0) for r in rows]
    overlaps = []
    rbos = []
    filter_agree = []
    recalls = []
    write_to_read = []
    status_counts: dict[str, int] = defaultdict(int)

    for r in rows:
        status_counts[str(r.get("status") or "exact")] += 1
        metrics = r.get("metrics") or {}
        if metrics.get("overlap@k") is not None:
            overlaps.append(float(metrics["overlap@k"]))
        if metrics.get("rbo") is not None:
            rbos.append(float(metrics["rbo"]))
        if metrics.get("filter_only_agreement") is not None:
            filter_agree.append(float(metrics["filter_only_agreement"]))
        if metrics.get("write_to_read_ms") is not None:
            write_to_read.append(float(metrics["write_to_read_ms"]))

        # Recall@k from qrels keyed by body_hash
        bh = r.get("body_hash")
        if bh and bh in qrels and r.get("method") == "search":
            pred = (r.get("secondary") or {}).get("ids") or []
            recalls.append(recall_at_k(pred, qrels[bh], k=k))

    return {
        "n": len(rows),
        "status_counts": dict(status_counts),
        "latency_ms": {
            "primary": {
                "p50": percentile(lat_pri, 50),
                "p95": percentile(lat_pri, 95),
                "mean": (sum(lat_pri) / len(lat_pri)) if lat_pri else None,
            },
            "secondary": {
                "p50": percentile(lat_sec, 50),
                "p95": percentile(lat_sec, 95),
                "mean": (sum(lat_sec) / len(lat_sec)) if lat_sec else None,
            },
        },
        "overlap@k": {
            "mean": (sum(overlaps) / len(overlaps)) if overlaps else None,
            "n": len(overlaps),
        },
        "rbo": {
            "mean": (sum(rbos) / len(rbos)) if rbos else None,
            "n": len(rbos),
        },
        "filter_only_agreement": {
            "mean": (sum(filter_agree) / len(filter_agree)) if filter_agree else None,
            "n": len(filter_agree),
        },
        "recall@10": {
            "mean": (sum(recalls) / len(recalls)) if recalls else None,
            "n": len(recalls),
        },
        "write_to_read_ms": {
            "p95": percentile(write_to_read, 95),
            "mean": (sum(write_to_read) / len(write_to_read)) if write_to_read else None,
            "n": len(write_to_read),
        },
    }


def _poc_charter(overall: dict[str, Any]) -> dict[str, Any]:
    sec_p95 = (overall.get("latency_ms") or {}).get("secondary", {}).get("p95")
    recall_mean = (overall.get("recall@10") or {}).get("mean")
    recall_n = (overall.get("recall@10") or {}).get("n") or 0
    wtr_p95 = (overall.get("write_to_read_ms") or {}).get("p95")

    checks = {
        "p95_latency_ms_lt_50": {
            "threshold": P95_LATENCY_MS,
            "value": sec_p95,
            "pass": sec_p95 is not None and sec_p95 < P95_LATENCY_MS,
            "skipped": sec_p95 is None,
        },
        "recall_at_10_ge_0_90": {
            "threshold": RECALL_AT_10_MIN,
            "value": recall_mean,
            "pass": recall_mean is not None and recall_mean >= RECALL_AT_10_MIN,
            "skipped": recall_n == 0,
        },
        "write_to_read_ms_lt_100": {
            "threshold": WRITE_TO_READ_MS_MAX,
            "value": wtr_p95,
            "pass": wtr_p95 is not None and wtr_p95 < WRITE_TO_READ_MS_MAX,
            "skipped": wtr_p95 is None,
        },
    }
    evaluated = [c for c in checks.values() if not c["skipped"]]
    passed = all(c["pass"] for c in evaluated) if evaluated else False
    return {
        "pass": passed,
        "checks": checks,
        "note": "Warm-up records excluded; secondary latency excludes embed_ms when logged",
    }
