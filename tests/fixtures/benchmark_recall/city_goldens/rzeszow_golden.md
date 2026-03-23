# Rzeszow Golden

## Question
What concrete public-transport electrification and EV-charging measures does Rzeszow describe in its climate-neutrality plan?

## Run Used
- Run date: 2026-03-22
- Requested run id: `rzeszow-golden-20260322`
- Actual run id: `rzeszow-golden-20260322_01`
- Run dir: `D:\GitHub\Query_mechanism_urbind\output\rzeszow-golden-20260322_01`

## Retrieval Artifact Check
`retrieval.json` captures pre-verification retrieval.

Evidence:
- `retrieval.json` contains both `seed_chunks` and expanded `chunks`.
- Researcher filtering happens later in `accepted_excerpts.json`, `rejected_excerpts.json`, and `excerpts.json`.

## Must-Have Chunk IDs
These chunk ids should always be present in retrieval for this question.

### 1. `chunk_c68062ec0f72131f230a1360`
Why it matters:
- This is the strongest single narrative chunk for the answer.

Facts protected:
- 20 zero-emission 12-meter hydrogen fuel-cell buses.
- Necessary charging infrastructure.
- EV charging-station expansion.
- EV parking spaces.
- Bus-lane access.
- Shared electric bicycles, scooters, and mopeds.

### 2. `chunk_fcef47b9e985a1523e456fcd`
Why it matters:
- This is the strongest explicit EV-charging implementation chunk.

Facts protected:
- At least 100 publicly accessible EV charging stations.
- Replacement of municipal service vehicles with electric ones.

### 3. `chunk_05ca67112c28dc4b032c3c13`
Why it matters:
- This is the strongest explicit bus-electrification action chunk.

Facts protected:
- Purchase of 20 hydrogen fuel-cell buses.
- Charging infrastructure bundled with the purchase.
- 282 bus shelters and 48 timetable pylons.
- At least 25% low-/zero-emission fleet by 2027.
- 100% low-/zero-emission fleet by 2030.

### 4. `chunk_bda8c597edd57dae46195ef6`
Why it matters:
- This chunk protects the charging-network coverage detail.

Facts protected:
- Access to charging stations at every parking lot.
- Charger availability for 30% of parking spaces.

### 5. `chunk_c31b78317c9fffff38d0ebff`
Why it matters:
- This is a supporting implementation/timeline chunk for the charging programme.

Facts protected:
- Charging-network expansion outcomes.
- Gradual municipal-fleet replacement through 2027.

## Recommended Gold Retrieval Set
```json
[
  "chunk_c68062ec0f72131f230a1360",
  "chunk_fcef47b9e985a1523e456fcd",
  "chunk_05ca67112c28dc4b032c3c13",
  "chunk_bda8c597edd57dae46195ef6",
  "chunk_c31b78317c9fffff38d0ebff"
]
```

## Strict Minimal Set
Use this smaller set if you want the benchmark to fail only on the most answer-critical omissions:

```json
[
  "chunk_c68062ec0f72131f230a1360",
  "chunk_fcef47b9e985a1523e456fcd",
  "chunk_05ca67112c28dc4b032c3c13"
]
```

## Useful But Not Required
- `chunk_598b249e84f02c0671634a92`
- `chunk_e487f444cb9b0272b8948f3c`
- `chunk_6b105b3aefa104aefe704785`

## Retrieval Judgment
Retrieval for this question should be considered incomplete if it misses any of:
- `chunk_c68062ec0f72131f230a1360`
- `chunk_fcef47b9e985a1523e456fcd`
- `chunk_05ca67112c28dc4b032c3c13`

Missing `chunk_bda8c597edd57dae46195ef6` or `chunk_c31b78317c9fffff38d0ebff` should be treated as a meaningful degradation in EV-charging completeness, even if the answer remains partially correct.

## Manual Vector DB Sweep Additions
These chunks were identified by direct vector-DB inspection after the initial run review. They extend the benchmark toward policy grounding, service-design detail, and KPI coverage.

### Strong Good-to-Have
- `chunk_ba4ac2be81a4c974306e3ae8`
  - Why it matters: adds transport-service detail that makes the electrification pathway more operational.
  - Key facts added:
    - the city aims to systematically increase the share of zero-emission buses until reaching `100%`
    - public-transport attractiveness measures include pricing, safety, timetable optimization, and accessibility
    - the city is expanding integrated ticketing for buses and trains
  - Retrieved in `rzeszow-golden-20260322_01`: `yes`

- `chunk_f2737189d4256c5a47c38f22`
  - Why it matters: gives the strongest policy-context chunk for bus-fleet renewal and modal shift.
  - Key facts added:
    - public transport should function as a real alternative to private cars
    - collective transport vehicles should be prioritized in road traffic
    - the plan calls for expanding the zero-emission fleet and gradually replacing the worn-out bus fleet
  - Retrieved in `rzeszow-golden-20260322_01`: `yes`

- `chunk_9a9b212e45bf8c2ad39e3583`
  - Why it matters: this is the cleanest provenance chunk for the underlying transport and electromobility plan.
  - Key facts added:
    - in `2021`, Rzeszow adopted the Sustainable Development Plan for Public Mass Transport for `2021-2030`
    - that plan explicitly includes elements of an electromobility development strategy
    - the plan covers Rzeszow and neighboring municipalities
  - Retrieved in `rzeszow-golden-20260322_01`: `yes`

### Monitoring / KPI Support
- `chunk_6cc5030be2077a2ec7d88d20`
  - Why it matters: strongest unretrieved KPI chunk found in the vector DB.
  - Key facts added:
    - indicator `RZE-AP-4` tracks the number of zero-emission buses in the urban public-transport fleet
    - target values are `20` in 2025, `60` in 2027, and `95` in 2030
    - it also links passenger-growth monitoring to the `Increasing accessibility for public mass transportation` task
  - Retrieved in `rzeszow-golden-20260322_01`: `no`
