"""Download BEIR SciFact and embed corpus + test queries with Pinecone Inference.

Resumable: embeddings append to live/data/*.emb.jsonl and already-embedded ids are skipped.
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

DATA = Path(__file__).parent / "data"
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
MODEL = "llama-text-embed-v2"
DIM = 1024
BATCH = 96


def download() -> Path:
    root = DATA / "scifact"
    if (root / "corpus.jsonl").exists():
        return root
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}", flush=True)
    with urllib.request.urlopen(URL) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(DATA)
    return root


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def embed_all(pc: Pinecone, items: list[tuple[str, str]], out: Path, input_type: str) -> None:
    done = {json.loads(line)["_id"] for line in out.read_text().splitlines()} if out.exists() else set()
    todo = [(i, t) for i, t in items if i not in done]
    total = len(items)
    print(f"{out.name}: {len(done)}/{total} already embedded, {len(todo)} to go", flush=True)
    started = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for b in range(0, len(todo), BATCH):
            batch = todo[b : b + BATCH]
            for attempt in range(6):
                try:
                    resp = pc.inference.embed(
                        model=MODEL,
                        inputs=[t for _, t in batch],
                        parameters={"input_type": input_type, "dimension": DIM, "truncate": "END"},
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    wait = min(2**attempt, 30)
                    print(f"  embed error ({exc.__class__.__name__}), retry in {wait}s", flush=True)
                    time.sleep(wait)
            else:
                sys.exit("embedding failed after retries")
            for (doc_id, _), item in zip(batch, resp.data):
                fh.write(json.dumps({"_id": doc_id, "values": list(item.values)}) + "\n")
            fh.flush()
            n = len(done) + b + len(batch)
            rate = (b + len(batch)) / max(time.time() - started, 1e-6)
            eta = (total - n) / rate if rate else 0
            print(f"  {n}/{total} ({100 * n / total:.1f}%) {rate:.0f}/s eta {eta:.0f}s", flush=True)


def main() -> None:
    load_dotenv(Path(__file__).parent.parent / ".env")
    root = download()
    corpus = read_jsonl(root / "corpus.jsonl")
    queries = {q["_id"]: q["text"] for q in read_jsonl(root / "queries.jsonl")}
    test_qids = []
    for line in (root / "qrels" / "test.tsv").read_text().splitlines()[1:]:
        qid = line.split("\t")[0]
        if qid not in test_qids:
            test_qids.append(qid)
    print(f"corpus={len(corpus)} test_queries={len(test_qids)}", flush=True)

    pc = Pinecone()
    embed_all(
        pc,
        [(d["_id"], f"{d.get('title', '')}\n{d['text']}".strip()) for d in corpus],
        DATA / "corpus.emb.jsonl",
        "passage",
    )
    embed_all(pc, [(q, queries[q]) for q in test_qids], DATA / "queries.emb.jsonl", "query")


if __name__ == "__main__":
    main()
