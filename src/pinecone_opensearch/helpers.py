"""helpers.bulk / streaming_bulk / parallel_bulk (M1)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, Iterator


def bulk(
    client: Any,
    actions: Iterable[dict[str, Any]],
    *,
    stats_only: bool = False,
    raise_on_error: bool = True,
    chunk_size: int = 500,
    **kwargs: Any,
) -> tuple[int, list[Any] | int]:
    """Index/update/delete helpers — mirrors ``opensearchpy.helpers.bulk`` return shape.

    Returns ``(success_count, errors)`` or ``(success_count, error_count)`` when
    ``stats_only=True``.
    """
    success = 0
    errors: list[Any] = []
    chunk: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal success
        if not chunk:
            return
        ndjson: list[dict[str, Any]] = []
        for action in chunk:
            op_type = action.get("_op_type", "index")
            meta = {
                k[1:]: v
                for k, v in action.items()
                if k.startswith("_") and k not in ("_op_type", "_source")
            }
            # helpers use _index, _id
            meta_out = {}
            if "_index" in action:
                meta_out["_index"] = action["_index"]
            if "_id" in action:
                meta_out["_id"] = action["_id"]
            if action.get("routing"):
                meta_out["routing"] = action["routing"]
            ndjson.append({op_type: meta_out})
            if op_type in ("index", "create", "update"):
                source = action.get("_source", {k: v for k, v in action.items() if not k.startswith("_")})
                if op_type == "update" and "doc" not in source:
                    ndjson.append({"doc": source})
                else:
                    ndjson.append(source)
        resp = client.bulk(body=ndjson, **{k: v for k, v in kwargs.items() if k in ("refresh", "routing")})
        for item in resp.get("items", []):
            op, detail = next(iter(item.items()))
            status = detail.get("status", 200)
            if status >= 400 or "error" in detail:
                errors.append(item)
                if raise_on_error and not stats_only:
                    raise_exc = detail.get("error") or detail
                    raise Exception(f"bulk helper error: {raise_exc}")
            else:
                success += 1
        chunk.clear()

    for action in actions:
        chunk.append(action)
        if len(chunk) >= chunk_size:
            flush()
    flush()

    if stats_only:
        return success, len(errors)
    return success, errors


def streaming_bulk(
    client: Any,
    actions: Iterable[dict[str, Any]],
    *,
    chunk_size: int = 500,
    raise_on_error: bool = True,
    **kwargs: Any,
) -> Iterator[tuple[bool, Any]]:
    """Yield ``(ok, item)`` per action, checkpointing each chunk."""
    chunk: list[dict[str, Any]] = []

    def flush() -> Iterator[tuple[bool, Any]]:
        if not chunk:
            return
        # Reuse bulk() then map results
        ok_count, errors = bulk(
            client,
            chunk,
            stats_only=False,
            raise_on_error=False,
            chunk_size=len(chunk),
            **kwargs,
        )
        error_ids = set()
        for err in errors:
            _op, detail = next(iter(err.items()))
            error_ids.add(detail.get("_id"))
            yield False, err
        # Yield successes for actions not in errors (best-effort)
        for action in chunk:
            if action.get("_id") not in error_ids:
                yield True, action
        chunk.clear()

    for action in actions:
        chunk.append(action)
        if len(chunk) >= chunk_size:
            yield from flush()
    yield from flush()


def parallel_bulk(
    client: Any,
    actions: Iterable[dict[str, Any]],
    *,
    thread_count: int = 4,
    chunk_size: int = 500,
    **kwargs: Any,
) -> Iterator[tuple[bool, Any]]:
    """Parallel variant of :func:`streaming_bulk` with checkpointed chunks."""
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for action in actions:
        current.append(action)
        if len(current) >= chunk_size:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)

    def run_chunk(chunk: list[dict[str, Any]]) -> list[tuple[bool, Any]]:
        return list(
            streaming_bulk(client, chunk, chunk_size=len(chunk), raise_on_error=False, **kwargs)
        )

    with ThreadPoolExecutor(max_workers=max(1, thread_count)) as pool:
        futs = [pool.submit(run_chunk, c) for c in chunks]
        for fut in as_completed(futs):
            yield from fut.result()
