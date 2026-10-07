# OpenSearch Drop-in Client for Pinecone

`pinecone-opensearch` is a Python package whose `OpenSearch` class mirrors `opensearch-py` methods, arguments, and response shapes, but executes against the **Pinecone Documents API** (`2026-07`, `pinecone>=10`).

Change one import and one constructor; keep your query bodies.

> Community project, not an official Pinecone product. Alpha: the surface below has been validated against live OpenSearch and Pinecone (see [Validation](#validation)), but expect rough edges.

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
pip install "git+https://github.com/tcondello/pinecone-opensearch"
# or from a clone
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

`tests/test_live_regressions.py` pins every bug the live runs below turned up.

## Validation

The client was run side by side against Amazon OpenSearch Service 2.19 and a Pinecone serverless index (both us-east-1) on [BEIR SciFact](https://github.com/beir-cellar/beir): 5,183 abstracts and 300 judged queries, embedded with `llama-text-embed-v2` (1024 dims). Latency was measured from an EC2 `c7i.large` in the same region.

| Query | Recall@10 OpenSearch / Pinecone | Top-10 overlap | p50 ms OS / PC | p95 ms OS / PC |
| --- | --- | --- | --- | --- |
| `match` | 0.770 / 0.770 | 86% | 11 / 18 | 24 / 80 |
| `knn` | 0.849 / 0.896 | 90% | 12 / 16 | 24 / 60 |
| `hybrid` (normalization pipeline) | 0.896 / 0.904 | 87% | 18 / 24 | 44 / 92 |

- **Filters:** `term`, `terms`, `range` (numeric and date), `exists`, `must_not`, `bool.should` and boolean flags all translate exactly, both inside `bool` with `match` and inside `knn.filter`. Filtered `count` matches OpenSearch exactly.
- **Writes and lookups:** `index`, `update`, `delete`, `delete_by_query`, `get` and `mget` all behave like OpenSearch, and `_source` comes back with the mapped types.
- **Client overhead:** translation takes about 0.1 ms, and searches through the client take the same time as raw Pinecone SDK calls.

### Known gaps

- `refresh="wait_for"` waits until a write can be read by id. Filtered search can briefly lag behind that.
- Filtered `count` is one search request: about 10–100 ms vs about 5 ms on OpenSearch. It falls back to paging at 10,000 or more matches.
- Validation so far covers one 5k-document corpus. Larger corpora, analyzer and BM25 parity, and production write loads are still open.

### Reproducing

`live/` holds the harness. The AWS scripts use your default AWS credentials (`AWS_PROFILE`, `AWS_REGION`, default `us-east-1`).

```bash
echo 'PINECONE_API_KEY=...' > .env
scripts/aws/opensearch_domain.sh create && scripts/aws/opensearch_domain.sh wait
cd live && uv run prepare.py && uv run load.py && uv run filters_writes.py load && cd ..
scripts/aws/ec2_runner.sh create && scripts/aws/ec2_runner.sh setup
scripts/aws/ec2_runner.sh run live/run_latency.sh
scripts/aws/ec2_runner.sh terminate && scripts/aws/opensearch_domain.sh delete
```

The domain costs about $2 a day and the runner about $0.09 an hour, so tear both down when you're done.

## Contributing

Issues and PRs are welcome. Run `uv run pytest` before opening a PR. A bug found against a live cluster should come with a fake-backend regression test in `tests/test_live_regressions.py`.

## License

Apache-2.0 — see [LICENSE](LICENSE).
