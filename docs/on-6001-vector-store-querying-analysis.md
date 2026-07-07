# ON-6001 Vector Store Querying Analysis

## Summary

ON-6001 asked for better vector-store querying, accepted/rejected distance logging, a switch from L2 distance to cosine similarity, recalibration of distance-dependent logic, and analysis of how to improve retrieved chunk and markdown-excerpt relevance.

The implementation now supports configurable Chroma distance metrics (`l2`, `cosine`, `ip`), records the selected metric in manifests and run snapshots, logs accepted/rejected markdown distance distributions, and separates the embedding provider configuration from the general LLM provider.

The recommended default is:

```yaml
vector_store:
  distance_metric: "cosine"
  retrieval_max_distance: 0.55
```

This recommendation is based on interpretability and threshold calibration. The tests did not show a material ranking improvement from cosine over L2 when both metrics were compared with equivalent recall settings.

## What Changed

- `vector_store.distance_metric` selects the Chroma HNSW space used by the persisted collection.
- `vector_store.embedding_base_url` and `vector_store.embedding_api_key_env` explicitly configure the embedding endpoint and key env var.
- Index settings now include `distance_metric`, embedding base URL, and embedding key env var, forcing rebuilds when those settings change.
- Retrieval artifacts and input snapshots include the distance metric used for the run.
- Markdown extraction audit now includes accepted and rejected chunk distance distributions:
  - `markdown_accepted_distance_count/min/p50/p90/p95/max`
  - `markdown_rejected_distance_count/min/p50/p90/p95/max`

## Test Setup

All live validation used OpenRouter for both embeddings and LLM calls:

- Embeddings: `https://openrouter.ai/api/v1/embeddings`
- LLM responses: `https://openrouter.ai/api/v1/responses`
- API key env var: `OPENROUTER_API_KEY`

Validation used two corpora:

| Corpus | Files | Chunks | Table chunks |
| --- | ---: | ---: | ---: |
| Reduced Munich/Leipzig | 2 | 77 | 36 |
| Full Munich/Leipzig | 2 | 3,188 | 1,704 |

The main comparison question was:

> Compare Munich and Leipzig on EV charging, building retrofit, climate targets, budgets, and implementation timelines. Cite concrete evidence from the source chunks.

## Reduced Corpus Results

Initial smoke test with the existing L2 cutoff showed cosine returning more useful context:

| Config | Retrieved | Accepted chunks | Excerpts | Accepted rate |
| --- | ---: | ---: | ---: | ---: |
| L2 `1.0` | 18 | 8 | 11 | 44.4% |
| Cosine `1.0` | 27 | 13 | 17 | 48.1% |

After disabling or recalibrating the cutoff, the difference largely disappeared:

| Config | Retrieved | Accepted chunks | Excerpts | Accepted rate |
| --- | ---: | ---: | ---: | ---: |
| L2 `null` | 27 | 14 | 17 | 51.9% |
| Cosine `null` | 27 | 14 | 18 | 51.9% |
| Cosine `0.6` | 27 | 14 | 18 | 51.9% |

Retrieval-only tests across 10 reduced-corpus questions confirmed that L2 and cosine retrieve nearly identical chunks when the cutoff is disabled:

| Metric | Value |
| --- | ---: |
| Seed chunk overlap | 238/240, 99.2% |
| Final chunk overlap | 323/325, 99.4% |

## Full Munich/Leipzig Results

Retrieval-only full-corpus tests also showed high overlap between L2 and cosine rankings:

| Metric | Value |
| --- | ---: |
| Seed chunk overlap | 236/240, 98.3% |
| Final chunk overlap | 300/306, 98.0% |
| Questions with identical seed order | 3/10 |

Full-corpus distance distributions show the practical calibration difference between L2 and cosine:

| Metric | p50 | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| L2 | 0.9223 | 1.0528 | 1.0982 | 1.1465 |
| Cosine | 0.4612 | 0.5265 | 0.5480 | 0.5735 |

Pipeline full-corpus runs delivered the same chunk set for L2 and cosine after calibration:

| Config | Retrieved | Accepted chunks | Rejected chunks | Excerpts | Accepted rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| L2 `null` | 26 | 20 | 6 | 29 | 76.9% |
| Cosine `0.55` | 26 | 20 | 6 | 37 | 76.9% |

The final chunk overlap for these two pipeline runs was 26/26. The excerpt-count difference is therefore attributable to markdown researcher variability, not retrieval differences.

## Findings

### Cosine improves interpretability more than ranking

Cosine did not prove materially better than L2 at ranking chunks for the tested corpora. Once the L2 cutoff was disabled or calibrated, the retrieved chunk sets were almost identical.

Cosine is still preferable operationally because the distance range is easier to explain and calibrate. In the full-corpus test, useful top-12-per-city/query cosine distances clustered below roughly `0.55`, while L2 clustered around roughly double that scale.

### The old L2 cutoff was under-recalling

The default L2 cutoff of `1.0` cut too aggressively in the reduced corpus. It reduced Munich from 12 candidate chunks to 6 for the main smoke question and reduced accepted chunks from 14 to 8 compared with L2 without cutoff.

### Accepted and rejected distances overlap

Accepted/rejected distance distributions overlap substantially for both metrics. Distance alone is not enough to separate useful from irrelevant chunks. The markdown extractor is still performing a meaningful filtering step after retrieval.

### Some irrelevant context comes from chunk quality

Qualitative review found boilerplate and low-information chunks such as short action-plan page fragments and repeated table headers. These are not fixed by metric choice.

Examples of improvement opportunities:

- suppress very short boilerplate chunks during indexing
- improve table chunk context and parent-heading attachment
- de-duplicate repeated page headers/footers before embedding
- split broad comparison questions into subtopic-specific retrieval queries
- add reranking or maximal marginal relevance for high-recall retrieval sets

## ChromaDB Assessment

The tests do not justify moving away from ChromaDB for this card.

ChromaDB is adequate for the current workload because:

- it supports the needed HNSW distance metrics
- metric changes can be captured in collection configuration and manifest settings
- the observed relevance issue was primarily threshold calibration and chunk quality, not vector database capability
- the high L2/cosine overlap suggests the backend is returning stable nearest-neighbor sets for this corpus

Moving away from ChromaDB should only be revisited if future needs include hybrid lexical/vector search, metadata-filter performance issues at larger scale, server-side reranking, or stronger operational guarantees than the local persistent store provides.

## Recommendation

Use cosine as the default metric with a calibrated cutoff:

```yaml
vector_store:
  distance_metric: "cosine"
  retrieval_max_distance: 0.55
```

This should be paired with a full vector-index rebuild. Existing L2 collections are not compatible with cosine collections because Chroma's HNSW metric is collection-level configuration.

If production rollout prioritizes recall over strictness, temporarily set `retrieval_max_distance: null` and use the new accepted/rejected audit metrics to choose a production cutoff from real runs.

## Follow-Up Work

- Build a small gold set of known relevant chunks for 10-20 representative questions.
- Add a retrieval-only benchmark that computes recall@k and precision@k against that gold set.
- Add chunk-quality filters for boilerplate, very short chunks, repeated page headers, and table fragments without enough semantic context.
- Evaluate query decomposition and optional reranking before changing vector database technology.
