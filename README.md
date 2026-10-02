# OpenSearch Drop-in Client for Pinecone

`pinecone-opensearch` is a Python package whose `OpenSearch` class mirrors `opensearch-py` methods, arguments, and response shapes, but executes against the **Pinecone Documents API** (`2026-07`, `pinecone>=10`).

Change one import and one constructor; keep your query bodies.

```python
from pinecone_opensearch import OpenSearch   # was: from opensearchpy import OpenSearch

client = OpenSearch(
    pinecone_api_key=os.environ["PINECONE_API_KEY"],
    config="pinecone_opensearch.yaml",
)
resp = client.search(index="products", body=body)
```

## Install

```bash
pip install pinecone-opensearch
# or from source
uv sync --group dev
```

Requires Python 3.10+.

## Config

```yaml
mode: strict   # strict | approx | shadow
indices:
  products:
    pinecone_index: products-pinecone
    knn_field: embedding
    mapping:
      properties:
        title: { type: text, fields: { keyword: { type: keyword } } }
        brand: { type: keyword }
        embedding: { type: knn_vector, dimension: 1536, space_type: cosinesimil }
aliases:
  products_alias: products
embedders:
  my-model:
    type: pinecone
    model: multilingual-e5-large
search_pipelines:
  norm-pipeline:
    phase_results_processors:
      - normalization-processor:
          normalization: { technique: min_max }
          combination:
            technique: arithmetic_mean
            parameters: { weights: [0.3, 0.7] }
deployment:
  deployment_type: managed
  cloud: aws
  region: us-east-1
```

## Drop-in surface

| Method | Notes |
| --- | --- |
| `search` / `msearch` / `count` / `get` / `mget` | Documents search/fetch |
| `index` / `create` / `update` / `delete` / `bulk` | Upsert/update/delete |
| `helpers.bulk` / `streaming_bulk` / `parallel_bulk` | Checkpointed batches |
| `delete_by_query` | Filter-only (no text-match ops) |
| `indices.create` / `delete` / `exists` / `get_mapping` / `refresh` | Schema + mapping registry |
| `search_pipeline.put` / `get` | Local pipeline registry |
| `translate` | Dry-run Pinecone plan |
| `ping` / `info` | `distribution: "pinecone"` |

## Translate (dry-run)

```python
plan = client.translate(index="products", body=body)
# plan["plan"] ∈ {fetch, single_search, fan_out}
```

```bash
uv run pinecone-opensearch translate -c pinecone_opensearch.yaml -i products -b query.json --pretty
```

## Shadow eval

```python
from pinecone_opensearch import ShadowClient
from opensearchpy import OpenSearch as OsClient

shadow = ShadowClient(OsClient(...), client, "shadow.jsonl", warm_up=100)
shadow.search(index="products", body=body)  # returns primary; logs both
```

```bash
uv run pinecone-opensearch replay --log queries.jsonl --out shadow.jsonl \
  --config pinecone_opensearch.yaml --opensearch-hosts '[...]' --qps 10
uv run pinecone-opensearch report --log shadow.jsonl --qrels qrels.jsonl --pretty
```

## Hybrid / neural

- Hybrid subqueries run concurrently; fusion follows the named `search_pipeline` (`normalization-processor` or RRF `score-ranker-processor`).
- `neural` / `neural_sparse` embed via config `embedders` (Pinecone Inference or a callable). Embed time is reported in `_pinecone.embed_ms`, excluded from fair latency comparisons.

## Limitations & unsupported

**Not identical rankings.** ANN, BM25 stats, and analyzers differ; the client reports approximations instead of hiding them.

**Unsupported (strict mode raises `UnsupportedQueryError`):** aggregations; nested/join; function_score/script_score/dis_max/boosting/rescore; field sorts & cursors; prefix/wildcard/regexp/fuzzy/range on keyword; geo & percolate; highlight/suggest/collapse/explain/profile; l1/linf/hamming; multiple `knn_vector` fields without `knn_field` config.

**Out of scope:** cluster/snapshot/security/ISM/ingest APIs; non-Python REST proxy.

## Tests

```bash
uv sync --group dev
uv run pytest
```

All unit tests use fake backends — **no live Pinecone or OpenSearch credentials required**.

### Untested here (needs live differential)

- End-to-end latency/recall vs OpenSearch 2.x Docker + real Pinecone index (BEIR / prospect logs)
- Pinecone Inference embedder live calls
- Analyzer / BM25 parameter parity
- Write-to-read visibility timing under production load

## License

Apache-2.0 — see [LICENSE](LICENSE).
