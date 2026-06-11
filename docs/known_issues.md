# Enrichment Layer — Known Issues

Tracking open issues observed in real runs of the enrichment pipeline
(`backend/modules/web_researcher/`). These are not merge blockers for the
current branch but should be addressed before relying on enrichment outputs
in production.

## 1. Relevance filter misses off-topic web sources

**Symptom.** In a run over 8 German cities, Heidelberg's
`total_targeted_vehicles` was estimated at **19,000** — a figure pulled from
an article about **SAP's corporate EV fleet charging infrastructure**. SAP
is headquartered near Heidelberg but has no relationship to Heidelberg's
municipal climate action plan.

**Location.** `backend/modules/web_researcher/relevance.py` (pre-scrape
filter) and/or `backend/modules/web_researcher/extractor.py` (post-scrape
filter).

**What's happening.** The city-name disambiguation heuristic accepts any
search result that mentions the target city, even when the result is about
an unrelated private entity co-located with the city. The extractor then
happily pulls a "fleet" number from that page.

**Fix direction.** Tighten relevance scoring to require that the document's
main subject is the *municipality*, not a private entity. Candidate
heuristics: reject domains that read as corporate press releases when the
target field is a municipal fleet count; require the city name to appear in
a governmental/plan context near the extracted value; add domain blocklists
for obvious corporate PR sources when the query is about municipal
inventory.

## 2. Assumptions estimator anchors on the wrong CCC fragment

**Symptom.** Aachen's `total_targeted_vehicles` estimate came back as
**2** (range 1–3) even though the same CCC excerpts explicitly contain
**4** municipal/fire-rescue replacements, **90** electric/hydrogen buses,
and **2** hydrogen waste-collection vehicles. The estimator picked the
smallest numeric mention (the 2 waste vehicles) as the source figure and
built a peer-sanity range around it, so the estimated "total" is an order
of magnitude below the numbers plainly visible in the same bundle.

**Location.** `backend/modules/web_researcher/assumptions_estimator.py`,
specifically the step that selects the "source figure" for
`expert_heuristic_scaling`.

**Fix direction.** When multiple numeric mentions are present in the CCC
evidence for a city-field pair, the estimator should either (a) sum
category-appropriate counts if the field is a roll-up, or (b) anchor on the
largest comparable figure and explicitly note the sub-category counts in
the rationale. The current min-first behavior produces systematically
low-biased estimates.

## 3. Freshness "uncertain" dominates on prose-only CCC evidence

**Symptom.** In the same run, the per-city audit column shows freshness
flags of `uncertain` for 7/8 cities (Munich is the only
`consistent/uncertain mix`). Web findings are not being classified as
`consistent` or `superseded` very often.

**Status.** Expected, not a bug. CCC values live only in markdown prose.
When the freshness LLM reads a
qualitative sentence like *"Dresden includes city council vehicles, city
logistics, bicycle couriers, buses, taxis, and delivery vehicles in its
electrification-oriented mobility measures"*, there is no numeric anchor to
compare against a specific web value — so it correctly returns `uncertain`
rather than a false `consistent`.

**What would improve the signal.** The gap-analyst agent could be extended
to emit `known_values[]` per (city, field) — structured numeric facts it
already has to locate to classify blanks. Freshness could then skip the
prose hunt and compare web values against those structured facts directly.
This is a feature, not a fix: the pipeline works correctly today, but the
proportion of `consistent`/`superseded` classifications will stay low until
structured CCC values are surfaced as first-class artifacts.

## 4. Context merger still uses naive `.lower()` for city keys

**Location.** `backend/modules/web_researcher/context_merger.py` lines
57, 66, 75 — all `(city.lower(), field.lower())` lookups.

**Problem.** Web findings carry display names like `"Frankfurt am Main"`;
gap manifest entries carry the same; excerpts carry the normalized key
`"frankfurt_am_main"`. A naive `.lower()` keeps the space and never matches
the normalized form.

**Scope note.** The freshness module was fixed to use
`normalize_city_key` on both sides of the lookup. The context_merger still
uses `.lower()`, which is fine for single-word cities (Dresden, Munich,
Leipzig, Aachen, Mannheim, Münster, Klagenfurt, Heidelberg) but will break
on multi-word or hyphenated cities the moment they appear. Applying
`normalize_city_key` consistently there is a one-line change per lookup
plus a corresponding fix in the web-index and freshness-index key
construction.

## 5. Saturation warning is firing at 88% Method C

**Symptom.** Real runs show the saturation warning triggering at "7/8
(88%) rely on expert_heuristic_scaling". The warning's threshold is >60%.

**Status.** The warning is doing its job — Method C (expert heuristic
scaling) carries the widest uncertainty bands (±40–50%), and the reader is
being told loudly that the numbers are order-of-magnitude rather than
precise. But the underlying cause is that Methods A (national/regional
average) and B (peer-city proxy) rarely fire because there aren't enough
peer-resolved fields to seed them.

**Fix direction.** Pre-populate a small registry of national/regional
averages for common fields (bus fleet size per 100k residents, taxi
licenses per 100k, municipal fleet size per 100k) so Method A becomes
available even before any peer city is resolved.

## 6. Writer prompts still reference "enrichment" sections that assume some structure

**Location.** `backend/prompts/writer_system_aggregate.md` (sections 4–8),
`backend/prompts/writer_system_city_by_city.md` (equivalent sections).

**Status.** Fine today, but worth watching. The writer assumes
`enrichment.enriched_fields[]`, `enrichment.web_findings[]`,
`enrichment.freshness_results[]`, and top-level
`assumptions.assumptions[]` / `assumptions.non_estimable[]` are present
when those stages run. Several tables in the prompt
(per-city audit, augmented-data-insights) are only useful when web
findings and assumption estimates agree or clearly disagree. When most
freshness flags are `uncertain` (see issue 3), those tables end up thin
and repetitive. Consider reworking the writer prompt to collapse the
audit table into a compact "n still_missing, m partially_resolved"
summary when there are no `resolved` web findings.
