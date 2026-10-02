"""Minimal fake Pinecone Documents backend for unit tests (no network)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeUsage:
    read_units: int = 1


@dataclass
class FakePagination:
    next: str | None = None


@dataclass
class FakeDoc:
    _data: dict[str, Any]

    @property
    def id(self) -> str:
        return self._data["_id"]

    @property
    def score(self) -> float | None:
        return self._data.get("_score")

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


@dataclass
class FakeSearchResponse:
    matches: list[FakeDoc] = field(default_factory=list)
    namespace: str = "__default__"
    usage: FakeUsage = field(default_factory=FakeUsage)


@dataclass
class FakeFetchResponse:
    documents: dict[str, FakeDoc] = field(default_factory=dict)
    namespace: str = "__default__"
    usage: FakeUsage = field(default_factory=FakeUsage)
    pagination: FakePagination | None = None


@dataclass
class FakeUpsertResponse:
    upserted_count: int = 0


@dataclass
class FakeUpdateResponse:
    matched_records: int | None = 1


@dataclass
class FakeDeleteResponse:
    matched_records: int | None = 1


@dataclass
class FakeNsStats:
    vector_count: int = 0


@dataclass
class FakeStats:
    namespaces: dict[str, FakeNsStats] = field(default_factory=dict)
    total_vector_count: int = 0


class FakeDocuments:
    def __init__(self, store: dict[str, dict[str, dict[str, Any]]], index_name: str):
        self._store = store
        self._index = index_name

    def _ns(self, namespace: str) -> dict[str, dict[str, Any]]:
        return self._store.setdefault(self._index, {}).setdefault(namespace, {})

    def upsert(self, *, namespace: str, documents: list[Any], **kwargs: Any) -> FakeUpsertResponse:
        bucket = self._ns(namespace)
        for doc in documents:
            d = dict(doc)
            bucket[d["_id"]] = d
        return FakeUpsertResponse(upserted_count=len(documents))

    def update(self, *, namespace: str, documents: list[Any] | None = None, **kwargs: Any) -> FakeUpdateResponse:
        bucket = self._ns(namespace)
        for doc in documents or []:
            d = dict(doc)
            doc_id = d.pop("_id")
            if doc_id in bucket:
                bucket[doc_id].update(d)
        return FakeUpdateResponse(matched_records=len(documents or []))

    def delete(self, *, namespace: str, ids: list[str] | None = None, filter: dict[str, Any] | None = None, delete_all: bool = False, **kwargs: Any) -> FakeDeleteResponse:
        bucket = self._ns(namespace)
        if delete_all:
            n = len(bucket)
            bucket.clear()
            return FakeDeleteResponse(matched_records=n)
        if ids is not None:
            for doc_id in ids:
                bucket.pop(doc_id, None)
            return FakeDeleteResponse(matched_records=len(ids))
        if filter is not None:
            to_del = [i for i, d in list(bucket.items()) if _match_filter(d, filter)]
            for i in to_del:
                bucket.pop(i, None)
            return FakeDeleteResponse(matched_records=len(to_del))
        return FakeDeleteResponse(matched_records=0)

    def fetch(
        self,
        *,
        namespace: str,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        include_fields: Any = None,
        pagination_token: str | None = None,
        **kwargs: Any,
    ) -> FakeFetchResponse:
        bucket = self._ns(namespace)
        if ids is not None:
            docs = {i: FakeDoc(dict(bucket[i])) for i in ids if i in bucket}
            return FakeFetchResponse(documents=docs, namespace=namespace)
        # filter / match_all: return all (tests use small stores)
        docs = {i: FakeDoc(dict(d)) for i, d in bucket.items()}
        if filter:
            docs = {i: d for i, d in docs.items() if _match_filter(d.to_dict(), filter)}
        return FakeFetchResponse(documents=docs, namespace=namespace)

    def search(
        self,
        *,
        namespace: str,
        score_by: Any,
        top_k: int,
        include_fields: Any = None,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> FakeSearchResponse:
        bucket = self._ns(namespace)
        matches: list[FakeDoc] = []
        for doc_id, data in bucket.items():
            if filter and not _match_filter(data, filter):
                continue
            scored = dict(data)
            scored["_score"] = _score(data, score_by)
            matches.append(FakeDoc(scored))
        matches.sort(key=lambda d: d.score or 0.0, reverse=True)
        return FakeSearchResponse(matches=matches[:top_k], namespace=namespace)


def _score(data: dict[str, Any], score_by: Any) -> float:
    # Deterministic toy scores for tests
    clauses = score_by if isinstance(score_by, list) else [score_by]
    total = 0.0
    for clause in clauses:
        if not isinstance(clause, dict):
            continue
        ctype = clause.get("type")
        if ctype == "dense_vector":
            # cosine-ish: use first dim agreement
            values = clause.get("values") or []
            emb = data.get(clause.get("field")) or []
            if values and emb:
                total += sum(a * b for a, b in zip(values, emb, strict=False))
            else:
                total += 0.5
        elif ctype == "text":
            q = str(clause.get("query", "")).lower().split()
            fields = clause.get("fields") or []
            blob = " ".join(str(data.get(f, "")).lower() for f in fields)
            total += sum(1.0 for t in q if t in blob)
        elif ctype == "query_string":
            total += 1.0
    return total


def _match_filter(doc: dict[str, Any], filt: dict[str, Any]) -> bool:
    if "$and" in filt:
        return all(_match_filter(doc, c) for c in filt["$and"])
    if "$or" in filt:
        return any(_match_filter(doc, c) for c in filt["$or"])
    if "$not" in filt:
        return not _match_filter(doc, filt["$not"])
    for field, cond in filt.items():
        val = doc.get(field)
        if isinstance(cond, dict):
            if "$eq" in cond and val != cond["$eq"]:
                return False
            if "$in" in cond and val not in cond["$in"]:
                return False
            if "$gte" in cond and not (val is not None and val >= cond["$gte"]):
                return False
            if "$lte" in cond and not (val is not None and val <= cond["$lte"]):
                return False
            if "$lt" in cond and not (val is not None and val < cond["$lt"]):
                return False
            if "$gt" in cond and not (val is not None and val > cond["$gt"]):
                return False
            if "$exists" in cond:
                exists = field in doc
                if bool(cond["$exists"]) != exists:
                    return False
            if "$match_any" in cond:
                token = str(cond["$match_any"]).lower()
                if token not in str(val).lower():
                    return False
            if "$match_all" in cond:
                tokens = str(cond["$match_all"]).lower().split()
                blob = str(val).lower()
                if not all(t in blob for t in tokens):
                    return False
        else:
            if val != cond:
                return False
    return True


class FakeIndex:
    def __init__(self, name: str, store: dict[str, dict[str, dict[str, Any]]]):
        self.name = name
        self.documents = FakeDocuments(store, name)
        self._store = store

    def describe_index_stats(self, **kwargs: Any) -> FakeStats:
        ns_map = self._store.get(self.name, {})
        namespaces = {ns: FakeNsStats(vector_count=len(docs)) for ns, docs in ns_map.items()}
        total = sum(n.vector_count for n in namespaces.values())
        return FakeStats(namespaces=namespaces, total_vector_count=total)


class FakeIndexesAPI:
    def __init__(self, created: set[str]):
        self._created = created

    def create(self, *, name: str, schema: Any = None, deployment: Any = None, **kwargs: Any) -> Any:
        self._created.add(name)
        return {"name": name}

    def delete(self, *, name: str, **kwargs: Any) -> Any:
        self._created.discard(name)
        return {"acknowledged": True}

    def describe(self, *, name: str, **kwargs: Any) -> Any:
        if name not in self._created:
            err = Exception(f"index {name} not found")
            err.status = 404  # type: ignore[attr-defined]
            raise err
        return {"name": name}

    def list(self) -> list[Any]:
        return [{"name": n} for n in self._created]


class FakePinecone:
    def __init__(self, store: dict[str, dict[str, dict[str, Any]]], created: set[str]):
        self._store = store
        self.indexes = FakeIndexesAPI(created)

    def index(self, name: str) -> FakeIndex:
        return FakeIndex(name, self._store)


def make_backend() -> tuple[Any, dict[str, dict[str, dict[str, Any]]], set[str]]:
    from pinecone_opensearch.executor import PineconeBackend

    store: dict[str, dict[str, dict[str, Any]]] = {}
    created: set[str] = set()
    pc = FakePinecone(store, created)
    backend = PineconeBackend(api_key="test-key", pinecone_client=pc, index_factory=pc.index)
    return backend, store, created
