from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml
from dotenv import load_dotenv
from opensearchpy import OpenSearch as OsClient

ROOT = Path(__file__).parent.parent
LIVE = Path(__file__).parent
DATA = LIVE / "data"
OUT = LIVE / "out"
CONFIG = LIVE / "pinecone_opensearch.live.yaml"
INDEX = "scifact"

load_dotenv(ROOT / ".env")


def os_hosts() -> list[dict]:
    cached = LIVE / "os_hosts.json"
    if cached.exists():
        return json.loads(cached.read_text())
    out = subprocess.run(
        [str(ROOT / "scripts/aws/opensearch_domain.sh"), "endpoint"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return json.loads(out[1])


def os_client() -> OsClient:
    return OsClient(hosts=os_hosts(), timeout=60)


def live_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def corpus_docs() -> list[dict]:
    vectors = {r["_id"]: r["values"] for r in read_jsonl(DATA / "corpus.emb.jsonl")}
    docs = []
    for d in read_jsonl(DATA / "scifact" / "corpus.jsonl"):
        docs.append({"_id": d["_id"], "title": d.get("title", ""), "text": d["text"], "embedding": vectors[d["_id"]]})
    return docs
