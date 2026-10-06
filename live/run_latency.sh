#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
HOSTS="$(cat live/os_hosts.json 2>/dev/null || scripts/aws/opensearch_domain.sh endpoint | tail -1)"
mkdir -p live/out
echo "== host: $(hostname) $(date -u +%FT%TZ)"
(cd live && uv run build_queries.py)
echo "== replay: 900 queries (match / knn / hybrid), 30 warm-up"
rm -f live/out/shadow.jsonl
uv run pinecone-opensearch replay --log live/out/queries.jsonl --out live/out/shadow.jsonl \
  --config live/pinecone_opensearch.live.yaml --opensearch-hosts "$HOSTS" --warm-up 30 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  printf '   %s/900\n' "$(wc -l < live/out/shadow.jsonl 2>/dev/null || echo 0)"; sleep 10
done
wait "$pid"
uv run pinecone-opensearch report --log live/out/shadow.jsonl --qrels live/out/qrels.jsonl --out live/out/report.json >/dev/null
(cd live && uv run analyze_replay.py)
echo "== filters / writes / lookups / neural"
(cd live && uv run filters_writes.py run && uv run analyze_fw.py)
