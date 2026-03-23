# Krakow Golden

## Question
For Krakow's Climate City Contract, what concrete electrification and charging initiatives are stated for public buses, passenger cars, freight vehicles, and charging infrastructure?

## Run Used
- Run date: 2026-03-22
- Requested run id: `gold-krakow-20260322`
- Actual run id: `gold-krakow-20260322_01`
- Run dir: `D:\GitHub\Query_mechanism_urbind\output\gold-krakow-20260322_01`

## Retrieval Artifact Check
`retrieval.json` captures pre-verification retrieval.

Evidence:
- `retrieval.json` contains `seed_chunks` and final delivered `chunks`.
- Researcher acceptance happens later in:
  - `accepted_excerpts.json`
  - `rejected_excerpts.json`
  - `excerpts.json`
  - `decision_audit.json`

## Must-Have Chunk IDs
These chunk ids should always be present in retrieval for this question.

### 1. `chunk_af0660a148889a86e187d5dc`
Why it matters:
- This is the main fact-bearing chunk for the vehicle targets.

Facts protected:
- Electric buses rise from 4% to 74% by 2040.
- About 452 new electric buses.
- Electric passenger cars rise from 0% to 34% by 2040.
- About 124,388 electric passenger vehicles.
- Electric light trucks rise from 0% to 60% by 2040.
- Electric heavy trucks rise from 0% to 30% by 2040.
- About 35,565 light trucks and 4,028 heavy trucks.

### 2. `chunk_aab38f975a587c0f00e2477d`
Why it matters:
- This is the key bus-charging infrastructure chunk.

Facts protected:
- 2 pantograph charging stations on Łużycka Street.
- 2 pantograph charging stations on Stojałowskiego Street.
- 1 pantograph charging station on Rydgiera Street.

### 3. `chunk_21589c0fee371c9e14058a88`
Why it matters:
- This is the clearest general EV-charging rollout chunk.

Facts protected:
- At least 150 new public charging stations in the first five years.
- Rollout focus on high-traffic locations.

### 4. `chunk_382b5bbeb3e68cb54c1e2ef3`
Why it matters:
- This is the best programme-structure chunk tying the answer to named CCC actions.

Facts protected:
- `TR-8` is creation of new charging stations for electric buses.
- `TR-16` is the long-term electric mobility infrastructure programme.
- The contract links fleet expansion with supporting charging infrastructure.

## Recommended Gold Retrieval Set
```json
[
  "chunk_af0660a148889a86e187d5dc",
  "chunk_aab38f975a587c0f00e2477d",
  "chunk_21589c0fee371c9e14058a88",
  "chunk_382b5bbeb3e68cb54c1e2ef3"
]
```

## Useful But Not Required
- `chunk_6db5a49dd35c0277528089b9`
- `chunk_ad4ab546c4cf8993b01344c3`
- `chunk_6deca04a44983720aa3b3e59`

## Retrieval Judgment
Retrieval for this question should be treated as incomplete if any of the four gold chunks above is missing, because the answer will likely lose one of:
- quantified fleet targets
- concrete bus-charging rollout
- public charging rollout volume
- explicit CCC action linkage

## Manual Vector DB Sweep Additions
These chunks were identified by direct vector-DB inspection after the initial run review. They are not all required for the minimal benchmark, but they are strong candidates when the goal is to protect more implementation detail.

### Borderline Necessary / Strong Good-to-Have
- `chunk_d04872ca4abacbc84ea0ef43`
  - Why it matters: adds implementation framing beyond the headline bus-electrification target.
  - Key facts added:
    - expansion of the public transport fleet towards zero-emission
    - planned timeframe: `2024-2030`
    - estimated transport-sector reduction: `56,180 tCO2e`
    - estimated cost: `PLN 180,000,000`
  - Retrieved in `gold-krakow-20260322_01`: `yes`

- `chunk_2cac4e91b354af400b3257f3`
  - Why it matters: strengthens the freight-electrification section and guards against table-row splitting around truck targets.
  - Key facts added:
    - anchors the `Electrification of trucks` row as its own target block
    - supports the freight-vehicle electrification section beyond adjacency inference
  - Retrieved in `gold-krakow-20260322_01`: `yes`

- `chunk_39eb2d8903c3016b94622bab`
  - Why it matters: adds a concrete charging-related initiative that did not appear in the run retrieval.
  - Key facts added:
    - `TR-1` is participation in the SmartEPC project
    - Krakow is testing Smart City elements through street-lighting infrastructure
    - the project explicitly mentions testing LED lanterns for vehicle charging
  - Retrieved in `gold-krakow-20260322_01`: `no`

### Monitoring / KPI Support
- `chunk_3397ca7c5165c6112164ec29`
  - Why it matters: useful if the benchmark should protect monitoring structure as well as action text.
  - Key facts added:
    - indicator `W43_O` tracks the number of EV charging stations installed under Municipality of Krakow grants
    - the indicator is tied to `TR-1`, `TR-8`, and `TR-16`
    - the target trend is increasing over time
  - Retrieved in `gold-krakow-20260322_01`: `no`
