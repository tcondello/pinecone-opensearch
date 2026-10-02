import json
from collections import defaultdict
from common import DATA, OUT, read_jsonl
from filters_writes import search_cases
from pinecone_opensearch.metrics import body_hash

texts = {q["_id"]: q["text"] for q in read_jsonl(DATA / "scifact" / "queries.jsonl")}
vecs = {r["_id"]: r["values"] for r in read_jsonl(DATA / "queries.emb.jsonl")}
labels = {body_hash(b): l for l, b in search_cases(texts, vecs)}
recs = read_jsonl(OUT / "fw_shadow.jsonl")
agg = defaultdict(lambda: defaultdict(list))
for r in recs:
    if r["method"] == "search":
        lab = labels.get(r["body_hash"], "?")
        lab = lab.split("+")[0] + "+" + lab.split("+")[1] if "+" in lab else lab
    else:
        lab = r["method"]
    a = agg[lab]
    a["status"].append(r["status"])
    for side in ("primary", "secondary"):
        if r[side].get("error"):
            a[side + "_err"].append(r[side]["error"])
    m = r.get("metrics") or {}
    for k, v in m.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            a[k].append(v)
        elif isinstance(v, bool):
            a[k].append(1.0 if v else 0.0)
    a["lat_p"].append(r["latency_ms"]["primary"]); a["lat_s"].append(r["latency_ms"]["secondary"])
    a["_rec"].append(r)

def mean(v): return sum(v) / len(v) if v else None
for lab in sorted(agg):
    a = agg[lab]
    st = {s: a["status"].count(s) for s in set(a["status"])}
    mets = {k: round(mean(v), 3) for k, v in a.items() if k not in ("status", "lat_p", "lat_s", "_rec") and not k.endswith("_err")}
    print(f"{lab:26} n={len(a['status']):3} status={st} {mets} lat OS/PC={mean(a['lat_p']):.0f}/{mean(a['lat_s']):.0f}ms")
    for side in ("primary", "secondary"):
        if a[side + "_err"]:
            print(f"    {side} errors x{len(a[side+'_err'])}: {str(a[side+'_err'][0])[:300]}")
