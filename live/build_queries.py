"""Write live/out/queries.jsonl (match / knn / hybrid per SciFact test query) and qrels.jsonl keyed by body_hash."""

from __future__ import annotations

import json
import random
from collections import defaultdict

from common import DATA, INDEX, OUT, read_jsonl
from pinecone_opensearch.metrics import body_hash


def bodies(text: str, vec: list[float]) -> dict[str, dict]:
    match = {"match": {"text": text}}
    knn = {"knn": {"embedding": {"vector": vec, "k": 10}}}
    base = {"size": 10, "_source": ["title"]}
    return {
        "match": {**base, "query": match},
        "knn": {**base, "query": knn},
        "hybrid": {**base, "search_pipeline": "hybrid-norm", "query": {"hybrid": {"queries": [match, knn]}}},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    texts = {q["_id"]: q["text"] for q in read_jsonl(DATA / "scifact" / "queries.jsonl")}
    vecs = {r["_id"]: r["values"] for r in read_jsonl(DATA / "queries.emb.jsonl")}
    rel: dict[str, list[str]] = defaultdict(list)
    for line in (DATA / "scifact" / "qrels" / "test.tsv").read_text().splitlines()[1:]:
        qid, doc_id, score = line.split("\t")
        if int(score) > 0:
            rel[qid].append(doc_id)

    rows, qrels = [], []
    for qid, vec in vecs.items():
        for qtype, body in bodies(texts[qid], vec).items():
            rows.append({"index": INDEX, "qid": qid, "qtype": qtype, "body": body})
            qrels.append({"body_hash": body_hash(body), "qid": qid, "qtype": qtype, "docs": rel[qid]})
    random.Random(7).shuffle(rows)

    with (OUT / "queries.jsonl").open("w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    with (OUT / "qrels.jsonl").open("w") as fh:
        fh.writelines(json.dumps(q) + "\n" for q in qrels)
    print(f"wrote {len(rows)} queries, {len(qrels)} qrels rows")


if __name__ == "__main__":
    main()
