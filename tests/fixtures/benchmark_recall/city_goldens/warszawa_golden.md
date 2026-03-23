# Warszawa Golden

## Question
What concrete bus-electrification and charging actions does Warsaw describe for reducing transport emissions in the mission districts, including planned investment and expected impact?

## Run Used
- Run date: 2026-03-22
- Requested run id: `gold-warszawa-20260322`
- Actual run id: `gold-warszawa-20260322_01`
- Run dir: `D:\GitHub\Query_mechanism_urbind\output\gold-warszawa-20260322_01`

## Retrieval Artifact Check
`retrieval.json` captures pre-verification retrieval.

Evidence:
- `retrieval.json` contains the raw retrieved chunk set before excerpt acceptance.
- Researcher decisions are stored later in:
  - `accepted_excerpts.json`
  - `rejected_excerpts.json`
  - `excerpts.json`

Run counts:
- Seed retrieved chunks: `78`
- Neighbor-expanded chunks: `27`
- Total delivered chunks: `105`

## Must-Have Chunk IDs
These chunk ids should always be present in retrieval for this question.

### 1. `chunk_d4ec7d0e245e9ea3c2d964bf`
Why it matters:
- This is the single most important chunk for the benchmark.
- It carries the core investment and impact numbers.

Facts protected:
- 202 electric buses in the citywide fleet by 2025.
- Further fleet replacement needed especially for Praga-Południe and Ursynów.
- Hydrogen bus testing.
- Estimated GHG reduction: 144,555 tCO2e.
- 234 new electric buses cost PLN 739,440,000.
- Charging infrastructure on 48 loops costs PLN 28,800,000.
- Pilot purchase of 10 hydrogen buses costs PLN 49,100,000.
- Total estimated cost is PLN 979,840,000.

### 2. `chunk_352cd2992198e43f63c2046a`
Why it matters:
- This is the best narrative chunk for current fleet status and ongoing electrification.

Facts protected:
- More than 160 electric buses already in operation.
- 70 hybrid buses.
- 418 gas-powered buses.
- Contract signed in 2023 for 12 additional electric buses.
- Increased electric-bus service in the mission districts.

### 3. `chunk_0f84fce506f28ba06d7dd56d`
Why it matters:
- This is the best action-framing chunk for the transition logic.

Facts protected:
- Public-transport expansion with zero-emission buses, metro, and trams.
- Bus-fleet replacement with electric and hydrogen vehicles.
- This is part of the mission-district decarbonisation plan.

## Recommended Gold Retrieval Set
```json
[
  "chunk_d4ec7d0e245e9ea3c2d964bf",
  "chunk_352cd2992198e43f63c2046a",
  "chunk_0f84fce506f28ba06d7dd56d"
]
```

## Optional Extended Set
Add this if the benchmark should also enforce district-level charging-policy detail:

```json
[
  "chunk_d4ec7d0e245e9ea3c2d964bf",
  "chunk_352cd2992198e43f63c2046a",
  "chunk_0f84fce506f28ba06d7dd56d",
  "chunk_487c5eb9fbfc3554772d7458"
]
```

### Optional Supporting Chunk
- `chunk_487c5eb9fbfc3554772d7458`
  - Adds the “at least one charging station in each car park” district charging action.

## Retrieval Judgment
Retrieval for this question should be treated as incomplete if any of the three core chunks above is missing.

Missing `chunk_487c5eb9fbfc3554772d7458` should be treated as a meaningful downgrade in charging-action completeness, but not necessarily a total retrieval failure for this exact question wording.

## Manual Vector DB Sweep Additions
These chunks were identified by direct vector-DB inspection after the initial run review. They are candidates for extending the benchmark beyond the current headline bus-loop and fleet-procurement facts.

### Strong Additions
- `chunk_6b120f2a5edcb5c5c9eac7ff`
  - Why it matters: this is the strongest additional charging-action chunk beyond the current gold set.
  - Key facts added:
    - `94 charging points` are planned in total
    - estimated GHG reduction: `100,466.29 tCO2e`
    - estimated cost: `PLN 8,516,969.70 / EUR 1,892,659`
  - Retrieved in `gold-warszawa-20260322_01`: `yes`

- `chunk_795bf778f3cdc07ce3c1a64b`
  - Why it matters: adds broader electromobility strategy detail that supports the charging-action narrative.
  - Key facts added:
    - Warsaw plans to invest with private partners in charging infrastructure
    - Warsaw plans to expand the network of EV charging stations
    - the city ties this to car-sharing and other vehicle-sharing measures
  - Retrieved in `gold-warszawa-20260322_01`: `yes`

### Monitoring / KPI Support
- `chunk_5b1dd8ffc9014d201131631d`
  - Why it matters: weak as a primary answer chunk, but useful if the benchmark should also enforce monitoring coverage.
  - Key facts added:
    - `Electric vehicle charging stations` is a named monitoring indicator under transport action `T-3`
    - target values are intentionally deferred to later iterations
  - Retrieved in `gold-warszawa-20260322_01`: `no`
