# Retrieval Experiment Postmortem (2026-03-31)

## Purpose

This note records the two retrieval experiments we tried against the current
gold benchmark failure set, what changed in code, and why neither experiment
was stable enough to keep. It is meant to be a reference point for designing a
deeper retrieval plan without re-litigating the same two quick trials.

## Baseline We Started From

The work focused on the benchmark cases that were still failing after the gold
fixture adjustments:

- `capex_cross_ccc`
- `quantified_charging_targets_munster`
- `charging_targets_germany_seven_cities`
- `dresden_charging_pilots_and_retrofits`
- `mannheim_transport_electrification_capex`
- `krakow_warszawa_transport_electrification`

The earlier loss analysis showed that the misses were split evenly across:

- retrieval misses
- markdown-stage losses
- writer-stage losses

That meant any retrieval-only change had to do more than just move a few chunks
around. It had to improve the final fact set, not only Stage A recall.

## Approach 1: Fixed Evidence-Family Query Expansion

### What we changed

We temporarily expanded retrieval from the standard canonical question plus
refined variants to a deterministic evidence-family set:

- canonical question
- measure/program family
- quantitative family
- scenario/projection family
- implementation family
- tabular/indicator family

The goal was to diversify query intent without adding city-specific project
names.

### Where we tested it first

We first isolated it on:

- `charging_targets_germany_seven_cities`
- `krakow_warszawa_transport_electrification`

### What happened

Results on that two-case subset:

| Variant | Final facts kept | Total gold facts |
|---|---:|---:|
| Baseline | 4 | 13 |
| Query-family expansion only | 4 | 13 |

Case-level effect:

- `charging_targets_germany_seven_cities`: retrieval and delivery improved, but
  final fact recall stayed at `1/7`
- `krakow_warszawa_transport_electrification`: no change, stayed at `3/6`

### Why it did not work

- It changed upstream recall shape, but the extra retrieved context did not
  survive markdown extraction or final writing.
- The new queries were still generic. They diversified wording, but they did
  not guarantee better access to the specific evidence forms that were being
  lost.
- In practice, this experiment added more retrieval surface area without a
  reliable mechanism to preserve the extra evidence downstream.

### Conclusion

This did not improve end-to-end fact recall in the trial subset, so it was
removed.

## Approach 2: Reciprocal-Rank Fusion (RRF) Re-ranking

### What we changed

We temporarily replaced the direct-hit ranking rule from pure
`best_distance` ordering to reciprocal-rank fusion across query variants.

The goal was to promote chunks that appeared across multiple query variants, so
that repeated multi-query support would outrank generic one-query hits.

### Where we tested it first

We first isolated it on:

- `charging_targets_germany_seven_cities`
- `krakow_warszawa_transport_electrification`

### What happened in the first isolated trial

Results on that two-case subset:

| Variant | Final facts kept | Total gold facts |
|---|---:|---:|
| Baseline | 4 | 13 |
| RRF only | 6 | 13 |

Case-level effect:

- `charging_targets_germany_seven_cities`: improved from `1/7` to `3/7`
- `krakow_warszawa_transport_electrification`: stayed at `3/6`

This looked promising enough to keep testing.

### What happened on the remaining failing cases

We then ran RRF against the broader remaining failure set. The result was not a
stable improvement:

| Case | Baseline final | RRF final | Net |
|---|---:|---:|---|
| `capex_cross_ccc` | `2/4` | `1/4` | Worse |
| `quantified_charging_targets_munster` | `3/4` | `2/4` | Worse |
| `charging_targets_germany_seven_cities` | `1/7` | `1/7` | Flat on final recall |
| `dresden_charging_pilots_and_retrofits` | `2/5` | `1/5` | Worse |
| `mannheim_transport_electrification_capex` | `4/7` | `3/7` | Worse |
| `krakow_warszawa_transport_electrification` | `3/6` | `3/6` | Flat |

Important detail:

- `charging_targets_germany_seven_cities` did improve upstream retrieval in the
  broader run, but the final answer still stayed at `1/7`

### Why it did not work

- It was not robust. The same mechanism that helped one case also destabilized
  fact retention in others.
- In the broader run, RRF often changed which fact survived rather than
  increasing the total number of supported facts.
- The improvement signal was mostly at retrieval ranking level, but the
  benchmark bottleneck in several cases was still markdown extraction or writer
  retention.
- Because of that, better seed ranking did not translate into better final
  answers often enough to justify keeping the extra retrieval complexity.

### Conclusion

RRF showed a local upside on one subset run, but it was not stable across the
remaining weak cases, so it was removed.

## Combined Trial: Query-Family Expansion + RRF

We also tested both approaches together on the two-case subset.

| Variant | Final facts kept | Total gold facts |
|---|---:|---:|
| Baseline | 4 | 13 |
| Query-family expansion + RRF | 3 | 13 |

Case-level effect:

- `charging_targets_germany_seven_cities`: dropped back to `1/7`
- `krakow_warszawa_transport_electrification`: regressed from `3/6` to `2/6`

This was the clearest sign that stacking both quick retrieval changes was not a
safe direction.

## What We Learned

- Retrieval diversity by itself is not enough when the downstream markdown and
  writer stages still compress or substitute the new evidence.
- Re-ranking alone can improve retrieval ordering without improving the final
  answer, which means Stage A gains are not sufficient success criteria.
- The current failure set is not explained by a single retrieval-ranking issue.
- Quick retrieval-only experiments can change which facts survive without
  increasing the final supported fact count.

## Implication For Next Plans

Future retrieval work should be designed as a deeper, testable plan rather than
another quick ranking tweak. The strongest remaining hypotheses are still:

- table/index representation improvements
- heading-aware embedding text
- evidence-shape coverage checks between retrieval and markdown
- tighter evaluation that separates Stage A gains from final-answer gains

This document should be read together with:

- [`backend/benchmarks/retrieval_evidence_diversity_analysis.md`](./retrieval_evidence_diversity_analysis.md)
- [`output/benchmarks/recall/full_live_20260330_codex/current_state_loss_analysis.md`](../../output/benchmarks/recall/full_live_20260330_codex/current_state_loss_analysis.md)
