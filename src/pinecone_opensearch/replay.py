"""Replay query logs against a ShadowClient at a target QPS."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterator

from pinecone_opensearch.shadow import ShadowClient

_SLOWLOG_SOURCE = re.compile(r"source\[(.*)\]\s*$")


def iter_query_log(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield ``{index, body, routing?}`` from JSONL or OpenSearch slowlog-ish text."""
    text = Path(path).read_text(encoding="utf-8")
    # Heuristic: JSONL if first non-empty line starts with {
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return
    if lines[0].lstrip().startswith("{"):
        for line in lines:
            row = json.loads(line)
            if "body" in row:
                yield {
                    "index": row.get("index"),
                    "body": row["body"],
                    "routing": row.get("routing"),
                }
            elif "query" in row or "size" in row:
                # bare search body with optional index field
                body = {k: v for k, v in row.items() if k not in {"index", "routing"}}
                yield {"index": row.get("index"), "body": body, "routing": row.get("routing")}
            else:
                yield row
        return

    # Slowlog: extract source[...] JSON and index from the line when present
    for line in lines:
        index = None
        m_idx = re.search(r"\bindex\[([^\]]+)\]", line)
        if m_idx:
            index = m_idx.group(1)
        m = _SLOWLOG_SOURCE.search(line)
        if not m:
            continue
        raw = m.group(1)
        # slowlog may escape; try json
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            continue
        yield {"index": index, "body": body, "routing": None}


def replay(
    shadow: ShadowClient,
    queries: Iterator[dict[str, Any]] | list[dict[str, Any]],
    *,
    default_index: str | None = None,
    qps: float | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Replay searches through ``shadow.search``. Returns a small run summary."""
    interval = (1.0 / qps) if qps and qps > 0 else 0.0
    n = 0
    errors = 0
    started = time.perf_counter()
    for item in queries:
        if limit is not None and n >= limit:
            break
        tick = time.perf_counter()
        index = item.get("index") or default_index
        body = item.get("body") or {}
        routing = item.get("routing")
        if not index:
            errors += 1
            n += 1
            continue
        try:
            kwargs: dict[str, Any] = {"index": index, "body": body}
            if routing is not None:
                kwargs["routing"] = routing
            shadow.search(**kwargs)
        except Exception:  # noqa: BLE001 — replay continues
            errors += 1
        n += 1
        if interval > 0:
            elapsed = time.perf_counter() - tick
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    return {
        "n": n,
        "errors": errors,
        "elapsed_s": time.perf_counter() - started,
        "log_path": str(shadow.log_path),
    }
