# Initiative Extraction And TEF Mapping Plan

## Purpose

This document captures the current agreement for integrating TEF-oriented initiative extraction into this repository.

The main design decision is:

- first extract rich initiative records from Climate City Contracts
- then classify extracted initiatives into one or more TEF categories in a separate pass

This avoids mixing extraction with taxonomy assignment and should improve recall, auditability, and future maintainability.

## Core Decision

We will **not** do TEF classification in the first extraction step.

We will use:

1. `Pass 1`: initiative extraction
2. `Pass 2`: TEF classification

This means the first pass should only capture what the document says about an initiative:

- what it is
- what it is trying to change
- how it will be done
- what outputs are planned
- who will deliver it
- what funding or costs are described
- what timing is described
- what quantities are stated

It should **not** assign TEF labels, TEF hints, or any pre-classified transition tags.

## Agreed V1 Extraction Shape

For `v1`, the canonical extraction object is a single initiative record with descriptive narrative fields and structured numbers.

```json
{
  "city": "Krakow",
  "initiative_name": "Local energy programme based on heat pumps",
  "general_description": "Krakow plans a local energy programme based on heat pumps to improve and decarbonise the city's district heating system. The initiative is presented as a technical and infrastructure measure for the city energy system, intended to help close the emissions gap in the buildings and heating sector.",
  "objective_text": "Improve and decarbonise the city's district heating system and support a change in the structure of energy generation.",
  "implementation_text": "Krakow plans to implement a local energy programme based on heat pumps as a technical and infrastructure measure for the city energy system.",
  "planned_outputs_text": "The initiative plans heat-pump-based energy capacity of approximately 1 MW for the city's energy system.",
  "delivery_text": "The initiative is to be implemented by the Municipal Heating Company (MPEC). It is described as an action to supply the energy system of the City of Krakow, with businesses and residents listed as involved stakeholders.",
  "funding_text": "Estimated investment outlay is PLN 7,000,000 (approximately EUR 1,555,000). The cited section does not specify the funding source or financing instrument.",
  "timeline_text": "The planned timeframe for implementation runs from 2024 to 2028. The document notes that a broader description of the measure will be provided in future iterations as the project develops.",
  "numbers": {
    "current": {},
    "planned": {
      "capacity_mw": 1,
      "start_year": 2024,
      "end_year": 2028,
      "emissions_reduction_tco2e": 3107,
      "investment_cost_pln": 7000000
    }
  }
}
```

## Why This Shape

This shape is intended to balance three needs:

- it is descriptive enough for later TEF mapping
- it is still extraction-only, not classification
- it keeps quantitative data machine-readable

Compared with earlier ideas, this avoids:

- storing TEF category directly on the initiative
- forcing one initiative to one TE node
- making the first pass produce overly normalized TE-shaped outputs
- relying on one giant free-text description blob

## What Is Clear

The following points are already agreed:

- The first pass is extraction only.
- TEF mapping must happen in a separate second pass.
- `initiative_code` is not part of the universal schema because it is document-specific and not guaranteed across cities.
- `general_description` should stay for readability.
- The more focused text fields should stay because they give later TEF mapping better raw material than a single description field.
- `numbers` should be split into `current` and `planned`.
- Evidence is deferred from the canonical `v1` object for now.
- Generated ids, document-local codes, source refs, extraction notes, quality
  flags, and review metadata are pipeline/audit artifacts, not fields in the
  canonical `v1` initiative object.
- Query-driven RAG should not be the primary extraction method for initiative harvesting.
- Full-document or full-section LLM extraction is preferred for ingestion.
- Agentic orchestration should act as a coverage and audit layer, not the main extraction path.

## What Is Still Open

The following design questions still need decisions:

### Field Rules

- Which fields are required vs optional?
- When should a missing field be `null`, empty string, or omitted?
- Should `delivery_text`, `funding_text`, and `timeline_text` always exist even when the document says nothing explicit?

### Numbers Schema

- What is the allowed universal key set under `numbers.current` and `numbers.planned`?
- How should units be normalized across sectors?
- How should currencies be represented when documents use different currencies?
- Should `start_year` and `end_year` stay inside `numbers.planned`, or move to top-level structured fields later?

### Storage Model

- Which fields should map directly into `Initiative`?
- Which fields should live in `misc` initially?
- When should `numbers` be split into a dedicated signal table or indicator structure?

### Evidence Layer

- How should provenance be added later?
- Should evidence be stored per initiative, per field, or per extracted quantity?
- Should evidence references point to section, chunk, quote, or all three?

### TEF Pass Output

- What should the TEF classification record contain?
- Recommended fields:
  - `initiative_id`
  - `tef_id`
  - `confidence`
  - `is_primary`
  - optional `rationale`
- How should multiple TE mappings be ranked?

### Coverage And Deduplication

- Exact dedupe by document-local code or normalized title is only a first pass.
- A second semantic dedupe pass is needed because the same initiative may be extracted under different names.
- Semantic dedupe should compare the agreed canonical fields only, not source refs, extraction traces, review notes, or TEF fields.
- Semantic dedupe should group records when they describe the same real-world initiative, project, programme, policy, investment, or action.
- Ambiguous near-duplicates should remain separate and be surfaced in review notes rather than force-merged.

## Recommended Pass Design

### Pass 1: Initiative Extraction

Recommended process:

1. Parse each Climate City Contract into ordered sections and tables.
2. Split each document into ordered, line-aware segments with a target cap of `50k` input tokens.
3. Process segments in source order, not in parallel, when duplicate-suppression context is enabled.
4. Send the model a rolling `already_extracted_initiatives` list capped at `20k` tokens.
5. Keep that rolling list to the agreed canonical v1 schema only: no source refs, extraction traces, review notes, or TEF fields.
6. Instruct the model to extract only materially new initiatives from the current segment.
7. Preserve table-based actions as first-class extraction inputs, not only prose sections.
8. Do not recursively split a segment because the model returned many initiatives.
9. If the first model call for a segment returns more than `3` initiatives, mark the segment as action-heavy.
10. For action-heavy segments, run follow-up calls over the same source segment and provide only initiatives already extracted from that segment as `already_extracted_initiatives`.
11. Continue action-heavy follow-up until the model either returns additional distinct initiatives or calls the stop tool to say no more initiatives remain.
12. This keeps dense segments as one auditable source artifact while making the model focus on missing initiatives instead of re-reading a long list of previous run-wide initiatives.

Canonical extraction artifacts should keep this same separation:

- `03_deduped/initiatives.jsonl` contains only canonical `v1` initiative
  objects.
- `03_deduped/initiative_records.jsonl` contains pipeline-generated ids and
  audit metadata needed by downstream TEF mapping.

### Coverage Pass

After extraction:

1. Check sections likely to contain initiatives.
2. Revisit sections such as:
   - Actions
   - Measures
   - Investments
   - Implementation
   - Governance
   - Financing
3. Flag segments that triggered action-heavy follow-up extraction.
4. Flag source quality issues such as OCR/encoding damage, page artifacts inside sections, or deferred values.

### Pass 2: TEF Classification

Only after initiative extraction is complete:

1. Take each extracted initiative object.
2. Map it to `1..n` TEF categories.
3. Mark one TE as primary when appropriate.
4. Store classification confidence.

## TEF Mapping Deep Dive

### Mapping Target

Initiatives should be mapped primarily to **Transition Elements**, not activities.

Transition Elements are the better aggregation unit because they represent comparable cross-city system shifts or emissions-relevant outcomes. Activities are useful as model internals and optional supporting links, but they are too granular and technical to be the main city-initiative bucket.

Recommended rule:

- map every initiative to `1..n` Transition Elements
- mark one Transition Element as primary when confidence is high enough
- optionally add activity links later only when the initiative explicitly mentions a concrete activity or technology

The product aggregation path should be:

```text
City initiatives -> Transition Elements -> TE hierarchy rollups
```

It should not be:

```text
City initiatives -> Activities
```

### TEF Root Categories

The current TEF repository has **six** top-level sectors, not five:

- `1-transport`
- `2-industry`
- `3-afolu`
- `4-buildings`
- `5-energy`
- `6-waste`

If URBIND wants a five-sector router, that should be an explicit product decision. The default technical design should preserve all six TEF sectors.

### Top-Level Documentation Quality

The TEF repository is strongest at the individual Transition Element level.

The generated top-level pages are uneven:

- `Transport` has a useful prose definition.
- `Industry`, `AFOLU`, `Buildings`, `Energy`, and `Waste` mostly expose title/frontmatter and generated child indexes.
- The detailed matching information lives mostly in the YAML transition files: `description`, `type`, `unitOfMeasure`, `shiftFrom`, `shiftTo`, and `carbonCausalChains`.

Implication:

- do not rely only on top-level TEF prose for classification
- build a compact local TEF catalog from YAML
- create short curated router definitions for the six sectors and major branches

### TEF Hierarchy For Routing

The staged classifier should use the TEF directory hierarchy as a narrowing structure.

```text
1 Transport
  1a Mobility
    1a1 Road
      Light-duty vehicles
      2/3 wheel vehicles
      Buses
    1a2 Rail
    1a3 Aviation
    1a4 Waterborne
  1b Freight
    Road
    Rail
    Aviation
    Shipping
  1c Other

2 Industry
  2a Minerals
  2b Chemicals
  2c Metals
  2d Other

3 AFOLU
  3a Livestock
  3b Land
  3c Aggregate sources / other

4 Buildings
  4a Residential
  4b Non-residential
  4c Building stocks

5 Energy
  5a Energy supply
    Electricity
    Heat
    Combined heat and power
    Fugitive emissions
  5b Energy transformation
  5c Energy transmission/storage
  5d Other

6 Waste
  6a Solids
  6b Non-solids
```

### TEF Catalog Fields For Routing And Ranking

The local TEF catalog should preserve these fields because they are useful for routing or final ranking:

- `sector`: top-level TEF sector
- `path`: directory-derived hierarchy path
- `class`: `transition` or `activity`
- `type`: mechanism such as `shift`, `update`, `resourceShift`, `supplyAlteration`, `supplyUpdate`, or `upshift`
- `unitOfMeasure`: useful for checking fit with extracted numbers
- `sustainability`: `green`, `amber`, or `red`
- `description`: primary natural-language definition
- `longName` and `shortName`: compact labels for matching
- `shiftFrom` and `shiftTo`: useful for final TE ranking
- `carbonCausalChains`: useful for supply/update-style transitions
- `ipccMitigationMethod`: useful when present, but sparse

### Recommended Staged Classifier

Do not put the full TEF catalog into one model call. Route to the relevant
category or subcategory first, then load all Transition Elements for that
category as the mapping universe. V1 should not add a separate narrowing,
filtering, or retrieval step between routing and mapping.

Use a staged design:

```text
Initiative extraction object
  -> sector router
  -> branch/category router
  -> category Transition Element loader
  -> final TE mapper
```

There is no LLM ambiguity verifier in the current implementation. Ambiguous
routes and low-confidence mappings are flagged in JSON for manual review.

#### Stage 1: Sector Router

Input:

- `initiative_name`
- `general_description`
- `objective_text`
- `implementation_text`
- `planned_outputs_text`
- `delivery_text`
- `funding_text`
- `timeline_text`
- `numbers`

Context:

- only six TEF sector definitions
- compact child labels for each sector

Output:

```json
{
  "sector": "energy",
  "confidence": 0.82,
  "alternatives": ["buildings"],
  "reason": "The initiative changes district heating supply using heat pumps."
}
```

The sector router must support alternatives because many city initiatives touch multiple systems.

#### Stage 2: Branch / Category Router

Input:

- initiative object
- selected sector
- direct children of selected sector

Example for `energy`:

```text
5a Energy supply
5b Energy transformation
5c Energy transmission/storage
5d Other
```

Then, if `5a Energy supply` is selected:

```text
5a1 Electricity
5a2 Heat
5a3 Combined heat and power
5a7 Fugitive emissions
```

After selecting the most specific applicable category or subcategory, filter
`tef_mapping/catalog/transition_elements.json` to records whose `path` exactly
matches the selected TEF path. If the selected category has no direct Transition
Elements but has child records in `subsubcategories.json`, continue routing into
those direct children. If the selected leaf has no direct Transition Elements and
no child categories, map the initiative directly to that selected category. This
is valid for no-transition leaves such as `2-industry/2a-minerals/2a5-soda-ash`.

#### Stage 3: Final TE Mapper

The final mapper sees the initiative and every Transition Element in the
selected category or subcategory. It may return only positive matches, but the
input set should be the full category list rather than a pre-filtered subset.

For each Transition Element, include:

- `tef_id`
- `title`
- `description`
- `sector`
- `path`
- `type`
- `unitOfMeasure`
- `shiftFrom`
- `shiftTo`
- `carbonCausalChains`

Recommended output:

```json
{
  "initiative_id": "generated_or_database_id",
  "matches": [
    {
      "tef_id": "district_heating_residual_heat",
      "confidence": 0.76,
      "is_primary": true,
      "rationale": "The initiative adds heat-pump-based capacity to the district heating system."
    }
  ],
  "needs_review": true
}
```

#### Manual Review Flags

Do not run an additional LLM verifier in the current implementation. Instead,
write review flags into JSON when:

- sector confidence is low
- top two sectors are close
- the initiative mentions multiple systems
- the CCC sector label disagrees with TEF sector routing
- the final TE confidence is below threshold
- a selected subcategory branch has no Transition Elements and is used as the
  final mapping target
- a category router returns a descendant path instead of one of the direct
  child candidates loaded for the current pass

Example:

- Krakow's heat-pump intervention is described in the CCC as `Buildings and heating`.
- TEF may classify district heat supply under `Energy -> Energy supply -> Heat`.
- This should not be treated as an error; it should be flagged for manual review.

### Local TEF Catalog Implementation

Build a local catalog from TEF YAML rather than reading the GitHub repository at runtime.

Implemented catalog layout:

```text
tef_mapping/
  README.md
  SOURCE.md
  prompts/
  catalog/
    sectors.json
    subcategories.json
    subsubcategories.json
    transition_elements.json
```

Implementation notes:

- `tef_mapping/prompts/` contains all shared prompt templates.
- Prompt templates use TOON for model output contracts.
- `tef_mapping/catalog/` contains exactly four runtime JSON indexes.
- `sectors.json` contains top-level TEF sectors.
- `subcategories.json` contains first-level subcategories under sectors.
- `subsubcategories.json` contains all deeper category levels.
- `transition_elements.json` contains compact Transition Element records filterable by `path`.
- Compact Transition Element records keep the original TEF `tef_source_path`; raw YAML files are not mirrored into the catalog.
- Do not create `tef_mapping/categories/`, `raw_transition_elements/`, or any other mirrored TEF folder tree.
- Only TEF records with `class: transition` are represented.
- TEF activities and parameters are intentionally excluded for now.

#### How The Catalog Was Generated

The local catalog was generated from a temporary sparse checkout of the TEF repository:

- Source repo: `https://github.com/ClimateView/transition-element-framework`
- Source ref: `main`
- Source commit: `befa1d1517bf1153158d6586679cc2f6cdb39461`
- Extracted on: `2026-04-20`
- Sparse checkout path: `models/`

Generation steps:

1. Read every YAML file under `models/`.
2. Exclude `models/parameters/`.
3. Keep only YAML records where `class: transition`.
4. Generate `sectors.json`, `subcategories.json`, `subsubcategories.json`, and `transition_elements.json` in `tef_mapping/catalog/`.
5. Store prompt-ready `card_text` directly on sector and category records.
6. Do not copy raw YAML into `tef_mapping/`; keep only compact JSON records with `tef_source_path` provenance.
7. Add central TOON-output prompt templates in `tef_mapping/prompts/`.
8. Store source attribution, extraction date, source commit, and license note in `tef_mapping/SOURCE.md`.

Validation results:

- `187` compact Transition Element records generated.
- `6` top-level sectors represented.
- `19` first-level subcategories represented.
- `95` deeper subsubcategories represented.
- `55` zero-transition subcategories preserved without invented transition entries.
- `4` runtime catalog JSON files parsed successfully.
- `6` prompt files created.
- Every compact transition record comes from a TEF YAML file with `class: transition`.
- No compact transition record comes from `models/parameters/`.
- Every compact transition record keeps the original TEF `tef_source_path`.
- Prompt files use TOON output examples and no JSON output examples.

Recommended catalog fields:

```json
{
  "tef_id": "shift_to_electric_vehicles",
  "title": "T-1A1a-TE-1 - Shift to electric cars",
  "sector": "transport",
  "path_code": "1a1a",
  "path_labels": ["Transport", "Mobility", "Road", "Light-duty vehicles"],
  "type": "shift",
  "unit_of_measure": "vehicles",
  "sustainability": "green",
  "description": "Shift vehicle kilometer from petrol, diesel, LPG and gas vehicles to battery electric vehicles in vehicle kilometer to fulfill the need of mobility",
  "shift_from": ["petrol_vehicles", "diesel_vehicles", "lpg_vehicles", "gas_vehicles"],
  "shift_to": ["battery_electric_vehicles"],
  "carbon_causal_chains": []
}
```

The catalog should be versioned or stamped with the TEF source commit so classification outputs can be audited later.

### TEF Mapping Output

The first durable mapping artifact should target Transition Elements when
available, and may target a subcategory when the selected TEF branch has no
Transition Elements anywhere.

Recommended fields:

- `initiative_id`
- `target_type`: `transition_element` or `subcategory`
- `target_id`: Transition Element `tef_id` or subcategory path
- `target_path`
- `confidence`
- `is_primary`
- `rationale`
- `mapper_version`
- `tef_source_version`
- `needs_review`

Optional later fields:

- `activity_id`
- `activity_confidence`
- `activity_rationale`

Activities should remain optional supporting links, not the primary aggregation layer.

### TEF Numeric Rollups

Numeric rollups must not ask the LLM to invent pipeline ids. The rollup layer
should:

- read generated ids and source metadata from `03_deduped/initiative_records.jsonl`
- read numeric values only from the clean canonical `initiative.numbers.current`
  and `initiative.numbers.planned` object inside each record
- join those numbers to `05_final_mappings/final_mappings.jsonl`
- emit one fact row per initiative number and TEF mapping
- include only primary mappings in default rollup totals to avoid double counting
- keep non-primary, nonnumeric, time/rate, and uncertain facts as reviewable
  unaggregated facts

Recommended JSON artifacts:

```text
output/tef_mapping/<run_id>/
  07_numeric_facts/initiative_numeric_facts.jsonl
  08_tef_groups/tef_grouped_initiatives.jsonl
  08_tef_groups/tef_metric_rollups.json
```

### Confidence Policy

Recommended initial thresholds:

- `>= 0.80`: accept primary TE mapping automatically
- `0.60-0.79`: accept but flag for review
- `< 0.60`: do not create a primary mapping automatically; keep the mapping record for review

For multi-system initiatives:

- allow multiple TE mappings
- require exactly one primary mapping only when confidence is high
- otherwise mark `needs_review=true`

### Working Example: Krakow Heat Pumps

Extracted initiative:

- local energy programme based on heat pumps
- improve and decarbonise the district heating system
- planned capacity around `1 MW`
- implementation by Municipal Heating Company
- planned timeframe `2024-2028`

Likely staged route:

```text
Root sector options:
  1. Energy
  2. Buildings

Likely branch:
  Energy -> Energy supply -> Heat

Review reason:
  The source document frames the action as Buildings and heating, but the TEF hierarchy may place district heat supply changes under Energy.
```

This example should be used as a test case for manual review flagging.

## Why Not Query-RAG For Canonical Extraction

Query-RAG is suitable for answering user questions, but not for exhaustive initiative harvesting.

In this repository, vector retrieval is query-driven, which makes it helpful for question answering but risky for recall if used as the canonical ingestion path.

For initiative harvesting, the safer approach is:

- full document parsing
- section-by-section LLM reading
- coverage and audit checks afterward

## Repository Fit

The current repository already contains structures that can support this direction:

- `backend/db_models/initiatives.py`
- `backend/db_models/tef.py`
- `backend/db_models/indicators.py`

Current likely implementation strategy:

- store the initiative narrative fields in the initiative record, likely using `misc` first for the richer fields
- store TEF mappings separately in the initiative-to-TE relation
- keep quantitative normalization flexible until the signal model stabilizes

## Recommended Near-Term Implementation Order

1. Define a concrete `InitiativeExtraction` schema in code.
2. Define validation rules for required and optional fields.
3. Define the allowed numeric key set for `numbers.current` and `numbers.planned`.
4. Build a section-by-section extraction flow for CCCs.
5. Add a coverage pass for missed initiatives.
6. Add duplicate merge rules within one city contract.
7. Add a second-pass TEF classifier.
8. Add evidence/provenance after the extraction model is stable.

## Bottom Line

The agreed direction is:

- extract `initiative + descriptive fields + numbers` first
- keep extraction free of TEF labels
- classify to TEF second
- defer evidence from the canonical schema for now
- use full-document LLM extraction, not query-RAG, for ingestion
- use agentic review only as a recall and coverage auditor

## Reference Example

The Krakow heat-pump initiative remains the working reference example for this model.

Primary source section:

- `documents/Krakow.md`, BIC-7 action sheet

Relevant external references for TEF understanding:

- https://github.com/ClimateView/transition-element-framework
- https://knowledgebase.climateview.global/en/knowledge/transition-target
