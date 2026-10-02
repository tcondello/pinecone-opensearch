"""Minimal in-memory OpenSearch-shaped client for shadow unit tests."""

from __future__ import annotations

from typing import Any


class InMemoryOpenSearch:
    """Stores docs per index and answers a tiny subset of the OpenSearch API."""

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, dict[str, Any]]] = {}

    def index(
        self,
        index: str,
        body: dict[str, Any] | None = None,
        *,
        id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body = body or {}
        doc_id = id or body.get("_id") or f"auto-{len(self._docs.get(index, {}))}"
        self._docs.setdefault(index, {})[doc_id] = dict(body)
        return {"_index": index, "_id": doc_id, "result": "created", "_version": 1}

    def get(self, index: str, id: str, **kwargs: Any) -> dict[str, Any]:
        doc = self._docs.get(index, {}).get(id)
        if doc is None:
            from opensearchpy.exceptions import NotFoundError

            raise NotFoundError(404, "not_found", {"_id": id})
        return {"_index": index, "_id": id, "found": True, "_source": doc}

    def search(self, index: str | None = None, body: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        body = body or {}
        index = index or kwargs.get("index")
        size = int(body.get("size", 10))
        docs = self._docs.get(index or "", {})
        query = body.get("query") or {"match_all": {}}
        hits = []
        for doc_id, src in docs.items():
            score = _score_doc(src, query)
            if score is None:
                continue
            hits.append({"_index": index, "_id": doc_id, "_score": score, "_source": src})
        hits.sort(key=lambda h: h["_score"], reverse=True)
        page = hits[:size]
        return {
            "took": 1,
            "timed_out": False,
            "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
            "hits": {
                "total": {"value": len(page), "relation": "eq"},
                "max_score": page[0]["_score"] if page else None,
                "hits": page,
            },
        }

    def count(self, index: str | None = None, body: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        resp = self.search(index=index, body={**(body or {}), "size": 10_000})
        return {"count": len(resp["hits"]["hits"]), "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0}}

    def delete(self, index: str, id: str, **kwargs: Any) -> dict[str, Any]:
        bucket = self._docs.get(index, {})
        if id in bucket:
            del bucket[id]
            return {"_index": index, "_id": id, "result": "deleted"}
        return {"_index": index, "_id": id, "result": "not_found"}

    def bulk(self, body: Any = None, **kwargs: Any) -> dict[str, Any]:
        return {"took": 0, "errors": False, "items": []}

    def ping(self, **kwargs: Any) -> bool:
        return True

    def info(self, **kwargs: Any) -> dict[str, Any]:
        return {"version": {"distribution": "opensearch", "number": "2.x"}}


def _score_doc(src: dict[str, Any], query: dict[str, Any]) -> float | None:
    if not query:
        return 1.0
    qtype = next(iter(query))
    payload = query[qtype]
    if qtype == "match_all":
        return 1.0
    if qtype == "match":
        field = next(iter(payload))
        q = str(payload[field] if not isinstance(payload[field], dict) else payload[field].get("query", "")).lower()
        blob = str(src.get(field, "")).lower()
        return float(sum(1 for t in q.split() if t in blob)) or None
    if qtype == "term":
        field = next(iter(payload))
        val = payload[field]["value"] if isinstance(payload[field], dict) else payload[field]
        return 1.0 if src.get(field) == val else None
    if qtype == "bool":
        filt = payload.get("filter") or []
        if not isinstance(filt, list):
            filt = [filt]
        for clause in filt:
            if _score_doc(src, clause) is None:
                return None
        must = payload.get("must") or []
        if not isinstance(must, list):
            must = [must]
        score = 0.0
        for clause in must:
            s = _score_doc(src, clause)
            if s is None:
                return None
            score += s
        if not must:
            return 1.0
        return score
    if qtype == "knn":
        field = next(iter(payload))
        opts = payload[field]
        vec = opts.get("vector") or []
        emb = src.get(field) or []
        if opts.get("filter") and _score_doc(src, opts["filter"]) is None:
            return None
        return sum(a * b for a, b in zip(vec, emb, strict=False))
    return 1.0
