# ON-6053 Model Update Analysis

## Summary

ON-6053 asked to review the GPT-5.4 defaults in `llm_config.yaml`, compare GPT-5.6 tiers (terra/luna, with stronger options allowed where justified), and update models based on measured results.

After a fixed Munich/Leipzig baseline and two GPT-5.6 candidates, the recommended default is a **tiered GPT-5.6 map**:

| Role | Model |
| --- | --- |
| markdown_researcher, initiative_extractor, tef_mapper, enrichment | `openai/gpt-5.6-luna` |
| orchestrator, chat, assumptions_reviewer, benchmark judges | `openai/gpt-5.6-terra` |
| writer | `openai/gpt-5.6-sol` |

This is Candidate B. It beat the 5.4 baseline on acceptance rate, excerpt yield, unresolved chunks, and batch failures, and beat all-terra (Candidate A) on acceptance/excerpts/writer richness while remaining faster than baseline.

## Evaluation Setup

Fixed inputs (same for all runs):

- Question: `Compare Munich and Leipzig on EV charging, building retrofit, climate targets, budgets, and implementation timelines. Cite concrete evidence from the source chunks.`
- Cities: Munich, Leipzig
- Vector store: disabled in this environment (`VECTOR_STORE_ENABLED=false`), so all runs used `standard_chunking` (31 chunks)
- Enrichment: disabled

Runs:

| Strategy | Run ID | Config used during evaluation |
| --- | --- | --- |
| Baseline (5.4) | `on6053_baseline_munich_leipzig` | pre-change `llm_config.yaml` |
| Candidate A | `on6053_candidate_a_terra` | temporary all-terra GPT-5.6 copy (removed before merge) |
| Candidate B | `on6053_candidate_b_tiered` | temporary tiered GPT-5.6 copy (removed before merge; became the checked-in default) |

Ticket-specific candidate YAML snapshots were deleted before merge to avoid repo clutter; results and recommendation remain in this doc. Metric snapshots lived under `output/on6053_validation/` locally.

## Results

| Strategy | Duration (s) | Accepted | Rejected | Unresolved | Accepted rate | Excerpts | Batch failures | Total tokens | Output tokens | final.md lines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline 5.4 | 127.0 | 19 | 11 | 1 | 61.3% | 98 | 1 | 1,226,512 | 30,641 | 73 |
| A all-terra | 54.8 | 21 | 10 | 0 | 67.7% | 111 | 0 | 1,217,487 | 19,424 | 68 |
| B tiered | 103.8 | 22 | 9 | 0 | 71.0% | 151 | 0 | 1,231,255 | 26,715 | 80 |

### Qualitative notes

- Baseline writer (`gpt-5.4`, high reasoning) produced a solid cited comparison but had 1 unresolved chunk and 1 markdown batch failure.
- Candidate A (`gpt-5.6-terra` everywhere) was the fastest run, cleaned up unresolved/failures, and improved acceptance/excerpts vs baseline. The writer answer was more compact.
- Candidate B (`luna` extractors + `terra` mid-tier + `sol` writer) produced the richest evidence set (151 excerpts) and the strongest acceptance rate. The writer executive summary was more concrete on charger counts and finance framing.

## Recommendation

Adopt Candidate B as the checked-in default:

1. High-volume extraction benefits from Luna: better acceptance/excerpts than all-terra at the cheaper tier.
2. Writer benefits from Sol on this synthesis task: denser citations and clearer cross-city framing than terra-only.
3. Terra remains the balanced default for orchestrator/chat/review/judges.

Do **not** use the bare `openai/gpt-5.6` alias; it routes to Sol and would accidentally upgrade every agent.

Embeddings stay on `text-embedding-3-large` (out of scope / unchanged).

### Alternative if cost must dominate

Use Candidate A (all `openai/gpt-5.6-terra`). It is still clearly better than the 5.4 baseline and roughly 2.3× faster in this smoke comparison, with slightly lower acceptance/excerpt yield than B.

## Changes Applied

- Updated `llm_config.yaml` to the tiered GPT-5.6 map.
- Synced AppConfig fallback defaults in `backend/utils/config.py`, including writer → `openai/gpt-5.6-sol`.
- Updated `BENCHMARK_JUDGE_MODEL` fallback in `backend/benchmarks/judge.py`.
- Updated docs/tests that asserted the old 5.4 defaults.
- Removed ticket-specific candidate YAML snapshots before merge; recommendation and metrics stay in this doc.
