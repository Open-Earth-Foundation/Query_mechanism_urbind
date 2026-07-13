# Enrichment Layer - Known Issues

Tracking open issues observed in prior real runs of the enrichment pipeline
(`backend/modules/web_researcher/`). These are not merge blockers for the
current branch but should be addressed before relying on enrichment outputs in
production.

## 1. Off-topic web-source false positives are only partly mitigated

**Symptom.** In a prior run over 8 German cities, Heidelberg's
`total_targeted_vehicles` was estimated at **19,000**, a figure pulled from an
article about **SAP's corporate EV fleet charging infrastructure**. SAP is
headquartered near Heidelberg but has no relationship to Heidelberg's municipal
climate action plan.

**Location.** `backend/modules/web_researcher/relevance.py` (pre-scrape
filter) and/or `backend/modules/web_researcher/extractor.py` (post-scrape
filter).

**Status.** Partly mitigated, not closed. The relevance checker now asks the
LLM to distinguish a municipality from a company, brand, product, or person with
the same name, and it forces `likely_false_match` results to be rejected. The
web extractor also includes semantic-alignment rules that tell it not to treat
private or regional/national figures as city-field values. However, the
relevance stage still intentionally fails open when uncertain or when the LLM
call fails, and there is not yet a deterministic regression fixture for the
SAP/Heidelberg-style case.

**Fix direction.** Add regression tests and deterministic guardrails around
known false-match patterns. Candidate heuristics: reject obvious corporate PR
sources when the target field is municipal inventory; require the city name to
appear in a governmental, plan, procurement, or operator context near the
extracted value; keep the LLM disambiguation as a useful layer but not the only
line of defense.

## 2. Assumptions estimator source-figure selection is still weak for roll-ups

**Symptom.** In a prior run, Aachen's `total_targeted_vehicles` estimate came
back as **2** (range 1-3) even though the same CCC excerpts explicitly contain
**4** municipal/fire-rescue replacements, **90** electric/hydrogen buses, and
**2** hydrogen waste-collection vehicles. The estimator picked the smallest
numeric mention (the 2 waste vehicles) as the source figure and built a
peer-sanity range around it, so the estimated "total" was an order of magnitude
below the numbers plainly visible in the same bundle.

**Location.** `backend/modules/web_researcher/assumptions_estimator.py`,
especially the estimator prompt and peer-reference construction.

**Status.** Still relevant, but the original "min-first behavior" wording is
now too specific. Current code sends the fixed gaps, enriched fields, benchmark
findings, and peer reference table to the estimator LLM; it does not expose a
structured `known_values[]` roll-up for all numeric mentions found in CCC prose.
That means roll-up fields such as `total_targeted_vehicles` still depend on the
LLM selecting and combining the right anchor values from unstructured context.

**Fix direction.** Surface structured CCC numeric facts before estimation. When
multiple numeric mentions are present for a city-field pair, the estimator
should either sum category-appropriate counts if the field is a roll-up, or
anchor on the largest comparable figure and explicitly note excluded
sub-category counts in the rationale.

## 3. Freshness "uncertain" dominates on prose-only CCC evidence

**Symptom.** In the same prior run, the per-city audit column showed freshness
flags of `uncertain` for 7/8 cities (Munich was the only
`consistent/uncertain mix`). Web findings were not classified as `consistent`
or `superseded` very often.

**Status.** Expected, not a bug. CCC values live only in markdown prose. When
the freshness LLM reads a qualitative sentence like "Dresden includes city
council vehicles, city logistics, bicycle couriers, buses, taxis, and delivery
vehicles in its electrification-oriented mobility measures", there is no
numeric anchor to compare against a specific web value, so it correctly returns
`uncertain` rather than a false `consistent`.

**What would improve the signal.** The gap-analyst agent could be extended to
emit `known_values[]` per (city, field), structured numeric facts it already
has to locate to classify blanks. Freshness could then skip the prose hunt and
compare web values against those structured facts directly. This is a feature,
not a fix: the pipeline works correctly today, but the proportion of
`consistent`/`superseded` classifications will stay low until structured CCC
values are surfaced as first-class artifacts.

## 4. Context merger still uses naive `.lower()` for city keys

**Location.** `backend/modules/web_researcher/context_merger.py`, especially
the web/freshness/external-resolution key construction paths that still use
`(city.lower(), field.lower())` style lookups.

**Problem.** Web findings carry display names like `"Frankfurt am Main"`; gap
manifest entries carry the same; excerpts carry the normalized key
`"frankfurt_am_main"`. A naive `.lower()` keeps the space and never matches the
normalized form.

**Scope note.** The freshness module was fixed to use `normalize_city_key` on
both sides of the lookup. The context merger still uses `.lower()`, which is
fine for single-word cities (Dresden, Munich, Leipzig, Aachen, Mannheim,
Muenster, Klagenfurt, Heidelberg) but will break on multi-word or hyphenated
cities the moment they appear. Applying `normalize_city_key` consistently there
is a small lookup-normalization change plus corresponding tests for web,
freshness, and external-resolution key construction.

## 5. Saturation warning is firing at 88% Method C

**Symptom.** Prior real runs showed the saturation warning triggering at "7/8
(88%) rely on expert_heuristic_scaling". The warning's threshold is >60%.

**Status.** The warning is doing its job: Method C (expert heuristic scaling)
carries the widest uncertainty bands (+/-40-50%), and the reader is being told
that the numbers are order-of-magnitude rather than precise. But the underlying
cause is that Methods A (national/regional average) and B (peer-city proxy)
rarely fire because there are not enough peer-resolved fields to seed them.

**Fix direction.** Method A can use national/regional benchmark findings when
the web-research benchmark batches run, but those batches are still
budget-dependent (see issue 7). Pre-populate a small registry of stable
national/regional averages for common fields (bus fleet size per 100k residents,
taxi licenses per 100k, municipal fleet size per 100k) so Method A is available
even before any peer city is resolved or benchmark search has spare capacity.

## 6. Writer prompts still reference enrichment sections that assume structure

**Location.** `backend/prompts/writer_system_aggregate.md` and
`backend/prompts/writer_system_city_by_city.md`.

**Status.** Fine today, but worth watching. The writer assumes
`enrichment.enriched_fields[]`, `enrichment.web_findings[]`,
`enrichment.freshness_results[]`, and
`context_bundle.assumptions.assumptions[]` /
`context_bundle.assumptions.non_estimable[]` are present when those stages run.
Several tables in the prompt (per-city audit, augmented-data-insights) are only
useful when web findings and assumption estimates agree or clearly disagree.
When most freshness flags are `uncertain` (see issue 3), those tables can
become thin and repetitive. Consider reworking the writer prompt to collapse
the audit table into a compact "n still_missing, m partially_resolved" summary
when there are no `resolved` web findings.

## 7. Benchmark web searches are silently budget-gated

**Location.** `backend/modules/web_researcher/search_planner.py`.

**Problem.** National and comparative benchmark query batches are only planned
after city-specific web search batches consume their share of
`max_total_queries_per_run`. If city-search planning uses the full budget, the
benchmark batches do not run. This makes broader benchmark coverage vary by run
shape instead of by an explicit product/config decision.

**Fix direction.** Make benchmark search behavior explicit. Either add a config
or env flag that clearly enables/disables national and comparative benchmark
search, or give benchmark batches their own budget so they can run alongside
city-specific search without depending on leftover query capacity.

## 8. Assumption peer anchors use exact LLM-generated field names

**Location.** `backend/modules/web_researcher/gap_analysis.py` and
`backend/modules/web_researcher/assumptions_estimator.py`.

**Problem.** Gap analysis asks the LLM to create short `snake_case` field names.
Those names are then reused through city-gap detection, web/external search, and
assumptions. The assumptions estimator builds peer anchors only from resolved
`enriched_fields` with the exact same field name. There is no canonical field
taxonomy or alias map today, so conceptually similar fields such as
`dc_chargers`, `public_dc_charger_count`, and `fast_charging_points` are not
automatically treated as comparable anchors.

**Fix direction.** Add a canonical field layer before downstream enrichment:
each decomposed field should carry a stable `canonical_field_id`, display label,
aliases/synonyms, unit family, and scope. Assumptions should match peer anchors
on the canonical id plus compatible scope/unit family, while still preserving
the LLM-generated field label for readability.
