"""ShadowClient: dual-run OpenSearch vs Pinecone with JSONL comparison logs."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Literal

from pinecone_opensearch.exceptions import UnsupportedQueryError
from pinecone_opensearch.metrics import (
    body_hash,
    detect_query_type,
    extract_embed_ms,
    filter_only_agreement,
    is_filter_only_query,
    overlap_at_k,
    rbo,
    top_ids_and_scores,
)

Status = Literal["exact", "approx", "unsupported"]


class ShadowClient:
    """Send every call to both engines; return the primary response; append JSONL.

    Parameters
    ----------
    primary:
        Client whose response is returned to the caller (usually real OpenSearch).
    secondary:
        Comparison client (usually ``pinecone_opensearch.OpenSearch``).
    log_path:
        Append-only JSONL path (resumable — existing lines kept).
    warm_up:
        First N calls are flagged ``warmup: true`` and excluded from report aggregates.
    alternate_order:
        Flip which engine is invoked first on each call (fairness).
    exclude_embed_ms:
        Subtract ``_pinecone.embed_ms`` from secondary latency when present.
    k:
        Top-k used for overlap / RBO in search comparisons.
    """

    def __init__(
        self,
        primary: Any,
        secondary: Any,
        log_path: str | Path,
        *,
        warm_up: int = 100,
        alternate_order: bool = True,
        exclude_embed_ms: bool = True,
        k: int = 10,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.warm_up = int(warm_up)
        self.alternate_order = bool(alternate_order)
        self.exclude_embed_ms = bool(exclude_embed_ms)
        self.k = int(k)
        self._seq = 0
        self._lock = threading.Lock()
        self._call_count = 0
        # Resume seq from existing log if present
        if self.log_path.exists():
            with self.log_path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        self._seq += 1
                        self._call_count += 1

    # --- public API (subset mirrored from OpenSearch) --------------------

    def search(self, *args: Any, **kwargs: Any) -> Any:
        return self._dual("search", args, kwargs, compare="search")

    def msearch(self, *args: Any, **kwargs: Any) -> Any:
        return self._dual("msearch", args, kwargs, compare="msearch")

    def count(self, *args: Any, **kwargs: Any) -> Any:
        return self._dual("count", args, kwargs, compare="count")

    def get(self, *args: Any, **kwargs: Any) -> Any:
        return self._dual("get", args, kwargs, compare="get")

    def mget(self, *args: Any, **kwargs: Any) -> Any:
        return self._dual("mget", args, kwargs, compare="mget")

    def index(self, *args: Any, **kwargs: Any) -> Any:
        return self._dual("index", args, kwargs, compare="write")

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._dual("create", args, kwargs, compare="write")

    def update(self, *args: Any, **kwargs: Any) -> Any:
        return self._dual("update", args, kwargs, compare="write")

    def delete(self, *args: Any, **kwargs: Any) -> Any:
        return self._dual("delete", args, kwargs, compare="write")

    def bulk(self, *args: Any, **kwargs: Any) -> Any:
        return self._dual("bulk", args, kwargs, compare="write")

    def delete_by_query(self, *args: Any, **kwargs: Any) -> Any:
        return self._dual("delete_by_query", args, kwargs, compare="write")

    def ping(self, *args: Any, **kwargs: Any) -> Any:
        return self._dual("ping", args, kwargs, compare="none")

    def info(self, *args: Any, **kwargs: Any) -> Any:
        return self._dual("info", args, kwargs, compare="none")

    def translate(self, *args: Any, **kwargs: Any) -> Any:
        # Dry-run only on secondary (pinecone client)
        if hasattr(self.secondary, "translate"):
            return self.secondary.translate(*args, **kwargs)
        raise NotImplementedError("secondary client has no translate()")

    @property
    def indices(self) -> Any:
        return _ShadowIndices(self)

    def __getattr__(self, name: str) -> Any:
        # Prefer primary attribute for unknown APIs; still dual if callable
        attr = getattr(self.primary, name, None)
        if callable(attr):

            def _wrapper(*args: Any, **kwargs: Any) -> Any:
                return self._dual(name, args, kwargs, compare="none")

            return _wrapper
        if attr is not None:
            return attr
        raise AttributeError(name)

    # --- core dual-call --------------------------------------------------

    def _dual(
        self,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        compare: str,
    ) -> Any:
        with self._lock:
            self._call_count += 1
            seq = self._seq + 1
            self._seq = seq
            call_count = self._call_count
        warmup = call_count <= self.warm_up

        primary_first = True
        if self.alternate_order:
            primary_first = (call_count % 2) == 1

        order = ["primary", "secondary"] if primary_first else ["secondary", "primary"]

        results: dict[str, dict[str, Any]] = {}
        for side in order:
            client = self.primary if side == "primary" else self.secondary
            results[side] = _invoke(client, method, args, kwargs)

        # Fairness: exclude embed time from secondary latency
        sec_latency = float(results["secondary"]["latency_ms"])
        embed_ms = 0.0
        if self.exclude_embed_ms and results["secondary"]["error"] is None:
            embed_ms = extract_embed_ms(results["secondary"].get("response"))
            sec_latency = max(0.0, sec_latency - embed_ms)

        index, body = _extract_index_body(method, args, kwargs)
        plan = None
        status: Status = "exact"
        warnings: list[str] = []

        if results["secondary"]["error"] is not None:
            err = results["secondary"]["error"]
            if err.get("type") == "UnsupportedQueryError":
                status = "unsupported"
            else:
                status = "approx"
        else:
            sec_resp = results["secondary"].get("response")
            if isinstance(sec_resp, dict):
                meta = sec_resp.get("_pinecone") or {}
                plan = meta.get("plan")
                warnings = list(meta.get("warnings") or [])
                if warnings:
                    status = "approx"

        # If secondary unsupported but primary ok — still return primary (shadow contract)
        metrics: dict[str, Any] | None = None
        if compare == "search" and status != "unsupported":
            metrics = _search_metrics(
                results["primary"].get("response"),
                results["secondary"].get("response"),
                body=body if isinstance(body, dict) else None,
                k=self.k,
            )
        elif compare == "count":
            metrics = _count_metrics(
                results["primary"].get("response"),
                results["secondary"].get("response"),
            )
        elif compare == "write":
            metrics = {
                "write_to_read_ms": results["secondary"].get("write_to_read_ms"),
            }

        record = {
            "ts": time.time(),
            "seq": seq,
            "method": method,
            "index": index,
            "body_hash": body_hash(body) if body is not None else None,
            "query_type": detect_query_type(body) if isinstance(body, dict) else method,
            "plan": plan,
            "status": status,
            "warmup": warmup,
            "call_order": order,
            "latency_ms": {
                "primary": results["primary"]["latency_ms"],
                "secondary": sec_latency,
                "embed_ms": embed_ms,
            },
            "primary": _side_payload(results["primary"], k=self.k, compare=compare),
            "secondary": {
                **_side_payload(results["secondary"], k=self.k, compare=compare),
                "warnings": warnings,
            },
            "metrics": metrics,
        }
        self._append(record)

        if results["primary"]["error"] is not None:
            # Re-raise primary failure
            exc = results["primary"].get("exception")
            if exc is not None:
                raise exc
            raise RuntimeError(results["primary"]["error"])
        return results["primary"]["response"]

    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def _invoke(client: Any, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    write_to_read_ms = None
    try:
        fn = getattr(client, method)
        # Capture write-to-read when refresh=wait_for on pinecone client
        refresh = kwargs.get("refresh")
        response = fn(*args, **kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if refresh in (True, "true", "wait_for") and isinstance(response, dict):
            # Prefer measured wait if client exposed it
            meta = response.get("_pinecone") or {}
            if "write_to_read_ms" in meta:
                write_to_read_ms = float(meta["write_to_read_ms"])
            else:
                write_to_read_ms = latency_ms
        return {
            "response": response,
            "latency_ms": latency_ms,
            "error": None,
            "exception": None,
            "write_to_read_ms": write_to_read_ms,
        }
    except UnsupportedQueryError as exc:
        return {
            "response": None,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "error": {"type": "UnsupportedQueryError", "path": exc.path, "message": str(exc)},
            "exception": exc,
            "write_to_read_ms": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "response": None,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "exception": exc,
            "write_to_read_ms": None,
        }


def _extract_index_body(
    method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[str | None, Any]:
    index = kwargs.get("index")
    body = kwargs.get("body")
    if index is None and args:
        # search(index, body) or get(index, id) etc.
        if method in {"search", "count", "index", "create", "update", "delete", "get"}:
            index = args[0] if args else None
            if method in {"search", "count", "index"} and len(args) > 1:
                body = body if body is not None else args[1]
            if method in {"create", "update"} and len(args) > 2:
                body = body if body is not None else args[2]
    return (str(index) if index is not None else None), body


def _side_payload(result: dict[str, Any], *, k: int, compare: str) -> dict[str, Any]:
    out: dict[str, Any] = {"error": result.get("error")}
    resp = result.get("response")
    if compare == "search" and isinstance(resp, dict):
        ids, scores = top_ids_and_scores(resp, k=k)
        out["ids"] = ids
        out["scores"] = scores
    elif compare == "count" and isinstance(resp, dict):
        out["count"] = resp.get("count")
    elif compare == "get" and isinstance(resp, dict):
        out["id"] = resp.get("_id")
        out["found"] = resp.get("found")
    elif compare == "write" and isinstance(resp, dict):
        out["result"] = resp.get("result")
        out["id"] = resp.get("_id")
    return out


def _search_metrics(
    primary_resp: dict[str, Any] | None,
    secondary_resp: dict[str, Any] | None,
    *,
    body: dict[str, Any] | None,
    k: int,
) -> dict[str, Any]:
    p_ids, _ = top_ids_and_scores(primary_resp, k=k)
    s_ids, _ = top_ids_and_scores(secondary_resp, k=k)
    metrics: dict[str, Any] = {
        "k": k,
        "overlap@k": overlap_at_k(p_ids, s_ids, k=k),
        "rbo": rbo(p_ids, s_ids, p=0.9, k=k),
        "filter_only_agreement": None,
    }
    if is_filter_only_query(body):
        # Use fuller lists when available
        p_all, _ = top_ids_and_scores(primary_resp, k=10_000)
        s_all, _ = top_ids_and_scores(secondary_resp, k=10_000)
        metrics["filter_only_agreement"] = filter_only_agreement(p_all, s_all)
    return metrics


def _count_metrics(primary_resp: Any, secondary_resp: Any) -> dict[str, Any]:
    p = (primary_resp or {}).get("count") if isinstance(primary_resp, dict) else None
    s = (secondary_resp or {}).get("count") if isinstance(secondary_resp, dict) else None
    agree = p == s if p is not None and s is not None else None
    return {"primary_count": p, "secondary_count": s, "agreement": agree}


class _ShadowIndices:
    def __init__(self, shadow: ShadowClient) -> None:
        self._shadow = shadow

    def __getattr__(self, name: str) -> Any:
        def _indices_dual(*args: Any, **kwargs: Any) -> Any:
            method = name
            with self._shadow._lock:
                self._shadow._call_count += 1
                seq = self._shadow._seq + 1
                self._shadow._seq = seq
                call_count = self._shadow._call_count
            warmup = call_count <= self._shadow.warm_up
            primary_first = True
            if self._shadow.alternate_order:
                primary_first = (call_count % 2) == 1
            order = ["primary", "secondary"] if primary_first else ["secondary", "primary"]
            results: dict[str, dict[str, Any]] = {}
            for side in order:
                client = self._shadow.primary if side == "primary" else self._shadow.secondary
                indices = getattr(client, "indices", None)
                started = time.perf_counter()
                try:
                    fn = getattr(indices, method)
                    response = fn(*args, **kwargs)
                    results[side] = {
                        "response": response,
                        "latency_ms": (time.perf_counter() - started) * 1000.0,
                        "error": None,
                        "exception": None,
                    }
                except Exception as exc:  # noqa: BLE001
                    results[side] = {
                        "response": None,
                        "latency_ms": (time.perf_counter() - started) * 1000.0,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "exception": exc,
                    }
            record = {
                "ts": time.time(),
                "seq": seq,
                "method": f"indices.{method}",
                "index": args[0] if args else kwargs.get("index"),
                "body_hash": None,
                "query_type": f"indices.{method}",
                "plan": None,
                "status": "unsupported" if results["secondary"]["error"] else "exact",
                "warmup": warmup,
                "call_order": order,
                "latency_ms": {
                    "primary": results["primary"]["latency_ms"],
                    "secondary": results["secondary"]["latency_ms"],
                    "embed_ms": 0.0,
                },
                "primary": {"error": results["primary"]["error"]},
                "secondary": {"error": results["secondary"]["error"], "warnings": []},
                "metrics": None,
            }
            self._shadow._append(record)
            if results["primary"]["error"] is not None and results["primary"]["exception"] is not None:
                raise results["primary"]["exception"]
            return results["primary"]["response"]

        return _indices_dual
