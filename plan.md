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

- How should section-level duplicates be merged?
- How should repeated mentions of one initiative across tables and prose be consolidated?
- What rules define when two extractions are the same initiative?

## Recommended Pass Design

### Pass 1: Initiative Extraction

Recommended process:

1. Parse each Climate City Contract into ordered sections and tables.
2. Run the model over every relevant section or bounded chunk.
3. Extract initiative objects using the agreed schema.
4. Preserve table-based actions as first-class extraction inputs, not only prose sections.

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
3. Flag suspicious cases where action-heavy sections produced zero initiatives.

### Pass 2: TEF Classification

Only after initiative extraction is complete:

1. Take each extracted initiative object.
2. Map it to `1..n` TEF categories.
3. Mark one TE as primary when appropriate.
4. Store classification confidence.

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
