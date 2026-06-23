# Findings: why a "commitments" query returns 74/74 non-estimable

**TL;DR.** The pipeline isn't failing to *find* data — it's asking for the wrong
*shape* of data. A query about EV-charging **commitments** got decomposed into
**current-inventory** fields (chargers by connector type) that City Climate
Contracts (CCCs) don't report. The relevant numbers are right there in the
corpus as *commitments* ("400 new charging stations"), but they map to none of
the requested fields, so every field comes back non-estimable. This is a
field-decomposition problem, not a retrieval problem.

Grounded in run `20260622_2313` — question *"compare public EV charging
commitments"*, multi-city (Bratislava, Bucharest, Cluj-Napoca, Liberec, …).

## What happened

| Stage | Result |
|---|---|
| Gap analysis | 74 fields classified, **all** `estimable_numerical` |
| External + web search | 92 candidates found, **1** validated |
| Assumptions | **0 estimated, 74 non-estimable** |

The new Enrichment Process audit view breaks the 74 down by *why*:

| Reason | Count |
|---|---|
| **Source has related data, in a different shape** | **67** |
| Found, but not validated | 3 |
| No source data found | 3 |
| Too few comparable cities | 1 |

## Root cause: decomposition drift

The question asked about **commitments**. Decomposition produced **current-inventory
fields by connector type**: `public_ac_charger_count`, `public_dc_charger_count`,
`public_fast_charger_count`, `offstreet_public_charger_count`,
`onstreet_public_charger_count`, … per city.

CCCs don't report today's chargers split by AC/DC/fast/ultrafast. They state
*future commitments*. The accepted excerpts contain exactly that, and it is
directly comparable across cities:

- Bucharest — *"200 new public charging stations • 300 new private"*
- Cluj-Napoca — *"commits to 200 new public EV charging stations, alongside 300 new private"*
- Bratislava — *"deploy 400 new EV charging stations"*

So 67/74 fields are flagged **shape mismatch**: the corpus has the topic (with
numbers), just not in the requested field shape. The estimator then finds **0
peer anchors** for `public_ac_charger_count` and friends → non-estimable.

**Consequence:** the prose report narrates the commitment numbers, while the
structured enrichment layer reports "nothing estimable." Two layers, two field
schemas, opposite conclusions.

## Secondary findings

1. **Candidate generation is noisy.** 91 of 92 candidates were unused — mostly
   report *titles* and *URLs* (`acea-charging-ahead-2024`, iea.org links), not
   values. Candidate matching keys on the topic word, not on a number in context.
2. **The one validation is a cross-city error.** Liberec's `target_year` (2030)
   was borrowed from **Krakow's** contract at 0.7 confidence — no provenance
   guardrail against cross-city anchoring.
3. **Sparse / mistagged source library.** Several cities have no tagged docs
   (e.g. *"Approved source library does not include Bucharest-tagged documents"*).
4. **Classifier over-optimism.** Gap analysis labels everything `estimable_numerical`
   based on the field being numeric *in principle*, decoupled from whether anchors
   plausibly exist in the corpus.

## Recommendations (priority order)

1. **Ground decomposition in the question intent + the corpus.** Decompose
   "commitments" into commitment-shaped fields (e.g. `committed_new_public_charging_stations`,
   `target_year`) — ideally after a peek at retrieved excerpts. Highest leverage;
   directly relevant to the assumption-model PRD.
2. **Drop/merge unrealistic granularity** (AC/DC/fast/ultrafast splits) for the
   CCC corpus.
3. **Filter candidate generation** to require a numeric value in context — kills
   the title/URL noise before validation.
4. **Add a provenance guardrail** so a value can't be silently borrowed across
   cities.
5. **Make the classifier corpus-aware**, or have it distinguish "estimable in
   principle" from "estimable from these sources."

## How this surfaces in the UI

The Enrichment Process view (per-field reason + the "Why fields broke" rollup)
turns this from "everything is red" into a precise, per-field diagnosis. The
67-field shape-mismatch signal is the headline; it's the evidence base for the
decomposition redesign above.
