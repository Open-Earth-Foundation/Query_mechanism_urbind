# Canonical Pipeline V2 Transition Plan

<<<CHANGE: Reframed the plan from "assumption model v2" to a full PRD-aligned `pipeline_v2`. The PRD's diagram shows a broader Foundation pipeline: L1 CCC extraction, L2 third-party enrichment, estimator flow, review loop, archetype classification, Demand Engine, and Demand Atlas. The current repo's gap analysis, web research, and automatic assumptions estimator map mainly into the PRD's estimator flow.>>>

## Purpose

This document proposes a v2 pipeline for `Query_mechanism_urbind` that follows the PRD's Foundation diagram while preserving as much existing working logic as possible.

The core goal is not to bolt "assumptions v2" onto the current pipeline. The goal is to build a clearer, reusable, PRD-aligned pipeline with:

- one canonical Pydantic handoff model, or one tightly coordinated model family under a single root model
- explicit typed handoffs between every stage
- clear module boundaries
- reusable current logic placed under the right PRD stage
- a clean comparison path against the current v1 implementation
- Demand Engine and Demand Atlas outputs as first-class pipeline products

The current pipeline should remain usable while v2 is built. V2 should run in parallel, consume the same inputs where possible, and produce comparison artifacts before any cutover.

Primary references:

- Current pipeline docs: `docs/README.md` and `docs/pipeline/`
- Diagnostic: [01_diagnostic.md](https://github.com/Open-Earth-Foundation/urbind-architecture-review/blob/main/analysis/assumption-model/01_diagnostic.md)
- PRD: [02_prd.md](https://github.com/Open-Earth-Foundation/urbind-architecture-review/blob/main/analysis/assumption-model/02_prd.md)
- Prior execution plan: [03_execution_plan.md](https://github.com/Open-Earth-Foundation/urbind-architecture-review/blob/main/analysis/assumption-model/03_execution_plan.md)

## Key Interpretation

<<<CHANGE: Added explicit validation of the user's interpretation against the current build and PRD.>>>

The user's interpretation is directionally correct:

- The PRD's "assumption model" is broader than the current `010_assumptions` stage.
- Current `010_assumptions` is mostly a conservative gap-filling estimator for unresolved enriched fields.
- In the PRD logic, that current estimator belongs inside the **Estimator Flow**, where L1 and L2 evidence are turned into initial `AssumptionRecord[]`.
- The larger, newer PRD piece is the **Demand Engine**, which consumes reviewed assumptions plus archetype classifications and domain methodology to create demand assumptions and the Demand Atlas.
- Therefore, a v2 implementation should not be named or scoped as only `assumptions_v2`.

The better target is a **Canonical Pipeline V2**:

```text
L1 CCC extraction
  + L2 third-party enrichment
      -> Estimator Flow
          -> Initial AssumptionRecord[]
              -> Review Loop
                  -> Reviewed AssumptionRecord[]

L1 CCC extraction
  -> Six archetype signals
      -> Archetype classifier

Reviewed AssumptionRecord[] + Archetype assignments + Domain methodology
  -> Demand Engine
      -> Demand Atlas
```

## PRD-Aligned V2 Pipeline

<<<CHANGE: Added the v2 pipeline shape that follows the PRD diagram directly.>>>

### 1. L1 CCC Extraction

Purpose: produce structured CCC-derived evidence from the 112 Climate City Contracts.

Current logic to reuse:

- `backend/modules/vector_store/`
- `backend/modules/markdown_researcher/`
- current retrieval, batching, source chunk index, accepted excerpts, rejected chunks, and decision audit behavior

V2 responsibility:

- Wrap current CCC extraction behavior behind a typed stage facade.
- Output `L1CccStageResult` as a Pydantic model.
- Preserve source traceability from every accepted evidence item back to city, source file, heading path, chunk id, quote, and extraction rationale.
- Expose enough typed CCC evidence for a future or external archetype-signal process to consume, but do not make six-signal extraction part of the first L1 implementation.

### 2. L2 Third-Party Enrichment

Purpose: add non-CCC evidence that can confirm, fill, challenge, or contextualize CCC evidence.

Current logic to reuse:

- governed external Markdown source registry and search
- external source agent and resolver
- tier-1/open web search planning
- web search worker
- scraping and extraction
- national/comparative benchmark findings

V2 responsibility:

- Split "evidence acquisition" from "estimation."
- Treat governed external Markdown and web/benchmark search as L2 evidence acquisition.
- Output `L2EnrichmentStageResult` as a Pydantic model containing candidate evidence, resolved evidence, no-evidence records, freshness inputs, benchmark findings, and source provenance.
- Keep current cost and billing diagnostics where useful, but expose them as metadata rather than mixing them into the semantic evidence model.

### 3. Estimator Flow

Purpose: combine L1 and L2 evidence into initial assumption records for missing, stale, bundled, or uncertain city-field needs.

Current logic to reuse:

- `gap_analysis.py`
- field decomposition
- gap manifest generation
- freshness comparison
- `compute_field_statuses`
- current `assumptions_estimator.py` logic for conservative per-city gap filling
- current non-estimable routing and saturation warning logic

V2 responsibility:

- Treat current gap analysis, freshness, field-status resolution, and automatic assumptions as one PRD Estimator Flow.
- Output `EstimatorStageResult` containing `InitialAssumptionSet`, `AssumptionRecord[]`, and `NonEstimableRecord[]`.
- Use the PRD method taxonomy and confidence rubric.
- Keep the current conservative pre-checks, but make their inputs and outputs typed.
- Make the estimator's input explicit: `EstimatorStageInput = L1CccStageResult + L2EnrichmentStageResult + requested fields + domain config hints`.

Important interpretation:

- The old/current assumptions stage does become part of the broader enrichment/estimator area.
- It should not be treated as the final PRD "assumption model."
- It creates initial gap-filling assumptions that still need review before the Demand Engine uses them.

### 4. Review Loop

Purpose: turn initial assumption records into reviewed records that analysts and domain partners can trust enough to feed into demand calculations.

Current logic to reuse:

- `backend/api/services/assumptions_review.py`
- current discover/apply/regenerate ideas
- frontend assumptions workspace concepts

V2 responsibility:

- Make review a first-class pipeline stage, not a separate shape.
- Output `ReviewStageResult` containing `ReviewedAssumptionSet` and the review actions that produced it.
- Preserve review decisions, corrections, rationale changes, reviewer identity, status, and partner/domain sign-off metadata.
- Refactor current `MissingDataItem` flow into canonical `AssumptionRecord` review operations.
- Keep post-run operation possible, but persist review state through the same canonical model.

### 5. Archetype Inputs And Classifier

This is the left-side support path in the diagram, separate from the right-side Estimator Flow -> Review Loop path.
**It does not need to be implemented in the first v2 pipeline.**

Purpose: provide governance/demand archetype assignments to the Demand Engine.

PRD reading:

- The diagram shows six archetype signals as an output of L1 CCC processing.
- Those signals feed a separate archetype classifier.
- The classifier output feeds the Demand Engine.
- This is the left-side support path in the diagram, separate from the right-side Estimator Flow -> Review Loop path.

Implementation stance:

- Treat archetype assignment as an input contract to v2, not as a blocker for the first v2 pipeline.
- Support externally provided archetype assignments first, with provenance and confidence.
- Define the future six-signal extraction contract so it can be added later or supplied by another service.
- Do not mix archetype classification into the Estimator Flow.

V2 contract:

- `ArchetypeSignals` may include:
  - governance backbone
  - stakeholder composition
  - policy levers
  - finance model
  - capacity gaps
  - monitoring infrastructure
- `ArchetypeAssignment[]` should include city, archetype, confidence, source, provenance, version, and whether it is `external`, `l1_extracted`, `manual_override`, or `partner_validated`.

Placement:

- The first v2 implementation should accept `ArchetypeAssignment[]` as externally provided or manually curated input.
- Later, if needed, v2 can add an L1-derived six-signal extractor and classifier that produce the same typed assignment contract.
- Demand Engine should depend on `ArchetypeAssignment[]`, not on the raw six signals.

<<<CHANGE: Updated archetype handling: six signals are PRD L1-derived support data, but v2 can initially consume externally provided archetype assignments. The right-side Estimator Flow -> Review Loop remains the main path to build first.>>>

### 6. Domain Methodology

Purpose: hold domain-specific assumptions and parameters that the Demand Engine needs.

Current logic to reuse:

- existing config patterns
- YAML configuration conventions
- TEF catalog and mapping concepts

New v2 logic:

- Add versioned domain methodology configs for Transport, Energy, Built Environment, and later other domains.
- Store benchmark sources, stock units, domain-specific levers, TEF mappings, feasible transition rates, readiness multipliers, and parameter provenance.
- Keep `default`, `analyst_override`, and `partner_validated` provenance distinct.

### 7. Demand Engine

Purpose: create forward-looking demand assumptions and cohort-level demand outputs.

Current logic to reuse:

- little or none directly; this is largely new
- reuse config loading, artifact writing, and test harness patterns

New v2 logic:

- Consume reviewed assumptions, archetype assignments, and domain methodology.
- Implement the PRD's demand logic, including `Cities x Stock x Rate x Readiness`.
- Produce per-domain, per-intervention, per-archetype, per-wave demand outputs.
- Keep range, confidence, caveat, and provenance discipline from `AssumptionRecord[]`.

Important distinction:

- This is the PRD's bigger and newer "assumptions model" layer.
- It is different from current gap-filling assumptions.
- It creates demand assumptions, not merely missing-data estimates.

### 8. Demand Atlas

Purpose: compose Demand Engine outputs into a product-ready view.

New v2 logic:

- Output `DemandAtlasStageResult` containing the product-facing `DemandAtlasView`.
- Include rows, domains, intervention typologies, demand waves, cohort totals, confidence labels, caveats, drill-down links to reviewed assumptions, and parameter provenance.
- Keep the Atlas as structured data first; writer, slide studio, or frontend rendering should be downstream projections.

## Canonical Pydantic Handoff Model

<<<CHANGE: Replaced the previous stage-specific schema approach with one root canonical model family.>>>

The v2 pipeline should define one canonical Pydantic root model, for example:

```python
class PipelineContextV2(BaseModel):
    schema_version: str
    run: RunContextV2
    l1_ccc: L1CccStageResult | None = None
    l2_enrichment: L2EnrichmentStageResult | None = None
    estimator: EstimatorStageResult | None = None
    review: ReviewStageResult | None = None
    archetypes: ArchetypeStageResult | None = None
    demand: DemandEngineStageResult | None = None
    atlas: DemandAtlasStageResult | None = None
    artifacts: ArtifactIndexV2
```

This is still modular, but it has one root contract. Each stage reads a typed subset and writes a typed subset:

| Stage | Reads | Writes |
| --- | --- | --- |
| L1 CCC extraction | `RunContextV2` | `L1CccStageResult` |
| L2 enrichment | `RunContextV2`, `L1CccStageResult` | `L2EnrichmentStageResult` |
| Estimator flow | `L1CccStageResult`, `L2EnrichmentStageResult`, domain hints | `EstimatorStageResult` |
| Review loop | `EstimatorStageResult.initial_assumptions` | `ReviewStageResult` |
| Archetype input/classifier | `RunContextV2`, optional `L1CccStageResult`, external assignment source | `ArchetypeStageResult` |
| Demand Engine | `ReviewStageResult.reviewed_assumptions`, `ArchetypeStageResult.assignments`, domain methodology | `DemandEngineStageResult` |
| Demand Atlas | `DemandEngineStageResult` | `DemandAtlasStageResult` |

Rules:

- No v2 stage should pass loose dicts as its primary handoff.
- Existing v1 dict/json shapes can be parsed at the edge, but v2 internals should use `PipelineContextV2`.
- If one root model becomes too large, split internals into submodels but keep the single root object.
- Stage artifacts should be serialized views of the canonical model, not independent incompatible schemas.

### Model Naming Convention

<<<CHANGE: Added a model naming convention so objects like `ReviewStageResult` are clear and related pieces are named together.>>>

Use these suffixes consistently:

- `*StageInput`: the complete typed input a module-level `pipeline.py` accepts.
- `*StageResult`: the complete typed output a module writes into `PipelineContextV2`.
- `*Set`: a typed collection of domain records inside a stage result.
- `*Record`: one domain item, such as one evidence claim, assumption, review action, or archetype assignment.
- `*View`: a product/export projection derived from canonical data, not the source of truth.
- `*Diagnostics`: operational metrics, costs, skipped reasons, retries, and warnings; diagnostics should not be mixed into semantic records.

The important distinction:

- There should be no separate `ReviewFlowResult` object. Use `ReviewStageResult` so the review loop follows the same naming as every other stage.
- `ReviewStageResult` is the whole review-loop stage output. It contains review metadata, review actions, status, optional regeneration projection metadata, diagnostics, and `reviewed_assumptions`.
- `ReviewedAssumptionSet` is the reviewed assumptions collection inside `ReviewStageResult`. This is what the Demand Engine consumes.

### Canonical Model Catalog

| Module folder | Stage input | Stage result | Main records/sets kept nearby |
| --- | --- | --- | --- |
| `stages/l1_ccc/` | `L1CccStageInput` | `L1CccStageResult` | `CccEvidenceRecord`, `CccEvidenceSet`, `RetrievalDiagnostics`, `MarkdownExtractionDiagnostics` |
| `stages/l2_enrichment/` | `L2EnrichmentStageInput` | `L2EnrichmentStageResult` | `ExternalEvidenceRecord`, `WebFindingRecord`, `BenchmarkFindingRecord`, `NoEvidenceRecord`, `L2Diagnostics` |
| `stages/estimator_flow/` | `EstimatorStageInput` | `EstimatorStageResult` | `FieldNeedRecord`, `GapRecord`, `FieldStatusRecord`, `InitialAssumptionSet`, `NonEstimableRecord`, `EstimatorDiagnostics` |
| `stages/review_loop/` | `ReviewStageInput` | `ReviewStageResult` | `ReviewActionRecord`, `CorrectionRecord`, `ReviewedAssumptionSet`, `ReviewDiagnostics` |
| `stages/archetype_inputs/` | `ArchetypeStageInput` | `ArchetypeStageResult` | `ArchetypeAssignmentRecord`, `ArchetypeAssignmentSet`, optional `ArchetypeSignalRecord` |
| `stages/demand_engine/` | `DemandEngineStageInput` | `DemandEngineStageResult` | `DemandCellRecord`, `DemandScenarioRecord`, `DemandWaveRecord`, `DemandEngineDiagnostics` |
| `stages/demand_atlas/` | `DemandAtlasStageInput` | `DemandAtlasStageResult` | `DemandAtlasView`, `DemandAtlasCellView`, `DemandAtlasDrilldownRef` |
| `stages/writer_projection/` | `WriterProjectionStageInput` | `WriterProjectionStageResult` | `WriterContextView`, `WriterProjectionDiagnostics` |

Keep each stage's public models in that stage folder's `models.py`. Cross-stage shared primitives, such as `CityRef`, `SourceRef`, `EvidenceRef`, `EstimateRange`, `ConfidenceLevel`, and `ArtifactRef`, belong in `backend/modules/pipeline_v2/models.py`.

## Current Module Reuse Map

<<<CHANGE: Added where existing modules should move conceptually in the PRD-aligned v2 pipeline.>>>

| Current code / behavior                                   | V2 placement              | Reuse approach                                                      |
| --------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------- |
| `vector_store` retrieval                                  | L1 CCC extraction         | Reuse through typed facade                                          |
| `markdown_researcher` extraction                          | L1 CCC extraction         | Reuse through typed facade                                          |
| retrieval artifacts and chunk index                       | L1 CCC extraction         | Preserve, but map to typed evidence records                         |
| `external_sources`, `external_agent`, `external_resolver` | L2 third-party enrichment | Reuse as governed external evidence acquisition                     |
| `search_planner`, `search_worker`, `scraper`, `extractor` | L2 third-party enrichment | Reuse as web/benchmark evidence acquisition                         |
| `freshness.py`                                            | Estimator flow            | Reuse for resolving field status against L1/L2 evidence             |
| `gap_analysis.py`                                         | Estimator flow            | Reuse for requested-field decomposition and gap manifest generation |
| `compute_field_statuses`                                  | Estimator flow            | Reuse, but return typed estimator inputs/outputs                    |
| `assumptions_estimator.py`                                | Estimator flow            | Reuse conservative gap-filling logic; output initial assumptions    |
| `assumptions_review.py`                                   | Review loop               | Reuse behavior; replace `MissingDataItem` with canonical records    |
| current writer context projection                         | Writer/export projection  | Reuse as temporary projection from `PipelineContextV2`              |
| current run logger/artifact writer                        | Shared runtime            | Reuse if it does not force v1 schemas into v2                       |

## Proposed Package Structure

<<<CHANGE: Renamed target package from `assumption_model` to `pipeline_v2` because the PRD change is broader than assumptions.>>>

```text
backend/modules/pipeline_v2/
  __init__.py
  models.py
  handoffs.py
  artifacts.py
  compare.py

  stages/
    __init__.py
    l1_ccc/
      __init__.py
      models.py
      pipeline.py
      vector_retrieval.py
      markdown_extraction.py
      artifacts.py
    l2_enrichment/
      __init__.py
      models.py
      pipeline.py
      external_sources.py
      web_search.py
      benchmarks.py
      artifacts.py
    estimator_flow/
      __init__.py
      models.py
      pipeline.py
      gap_analysis.py
      freshness_resolution.py
      field_status.py
      gap_filling.py
      artifacts.py
    review_loop/
      __init__.py
      models.py
      pipeline.py
      discovery.py
      review_state.py
      regeneration_projection.py
      artifacts.py
    archetype_inputs/
      __init__.py
      models.py
      pipeline.py
      assignment_loader.py
      classifier_interface.py
      artifacts.py
    demand_engine/
      __init__.py
      models.py
      pipeline.py
      methodology_loader.py
      engine.py
      artifacts.py
    demand_atlas/
      __init__.py
      models.py
      pipeline.py
      view_builder.py
      artifacts.py
    writer_projection/
      __init__.py
      models.py
      pipeline.py
      projection.py
      artifacts.py

  domain_methodology/
    __init__.py
    config.py
    transport.yaml
    energy.yaml
    built_environment.yaml

  adapters/
    __init__.py
    from_v1_context.py
    from_v1_enrichment.py
    from_v1_assumptions.py

tests/
  pipeline_v2/
    test_models.py
    l1_ccc/
      test_pipeline.py
      test_vector_retrieval.py
      test_markdown_extraction.py
    l2_enrichment/
      test_pipeline.py
      test_external_sources.py
      test_web_search.py
      test_benchmarks.py
    estimator_flow/
      test_pipeline.py
      test_gap_analysis.py
      test_freshness_resolution.py
      test_field_status.py
      test_gap_filling.py
    review_loop/
      test_pipeline.py
      test_review_state.py
    archetype_inputs/
      test_pipeline.py
      test_assignment_loader.py
    demand_engine/
      test_pipeline.py
      test_engine.py
    demand_atlas/
      test_pipeline.py
      test_view_builder.py
    writer_projection/
      test_pipeline.py
```

Adapter rule:

- Adapters exist only at v1/v2 boundaries.
- New v2 stage code should not depend on v1 model shapes.
- Each adapter must have a deletion condition or a declared long-term purpose, such as old-run import.

<<<CHANGE: Made the filesystem hierarchy mirror the conceptual hierarchy: v2 has modules, modules have substages, and each substage has scoped models, pipeline orchestration, artifact serialization, and matching tests under `tests/pipeline_v2/`.>>>

## Module And Substage Boundaries

<<<CHANGE: Added explicit guidance that every module and substage needs a clear scope and interface contract.>>>

Each v2 module should be a contained pipeline area with its own substage contracts. A module can contain multiple substages, but those substages must not communicate through loose dicts or hidden shared state.

Module rules:

- Every module has a `models.py` defining its public input/output Pydantic contracts.
- Every module has a `pipeline.py` that orchestrates only that module's substages.
- Every module has an `artifacts.py` that serializes module outputs from typed models.
- Every module exposes one public function, for example `run_l2_enrichment(...)`, that accepts and returns typed models.
- Cross-module handoff happens only through `PipelineContextV2` stage fields or explicit typed handoff models.
- Module internals may reuse current v1 code, but the module boundary must convert into v2 Pydantic models.

Substage rules:

- Every substage has a narrow responsibility and a named input/output type.
- Substages should be individually testable without running the whole pipeline.
- Substage diagnostics should be separate from semantic outputs.
- Substage artifacts should be derived from typed outputs, not assembled ad hoc from local variables.
- A substage may be disabled if its module contract defines what "skipped" means.

Examples:

| Module | Substages | Public module output |
| --- | --- | --- |
| `l1_ccc` | vector retrieval, markdown extraction | `L1CccStageResult` |
| `l2_enrichment` | external sources, web search, benchmark acquisition | `L2EnrichmentStageResult` |
| `estimator_flow` | gap analysis, freshness resolution, field status, gap filling | `EstimatorStageResult` |
| `review_loop` | discovery, analyst/agent review, correction state, regeneration projection | `ReviewStageResult` |
| `archetype_inputs` | assignment loading, optional classifier interface | `ArchetypeStageResult` |
| `demand_engine` | methodology loading, demand calculation | `DemandEngineStageResult` |
| `demand_atlas` | atlas composition, view metadata | `DemandAtlasStageResult` |

Concrete meaning for L2:

- `external_sources.py` owns governed Markdown evidence acquisition and resolver outputs.
- `web_search.py` owns search planning, Serper/tier-1/open-web calls, scraping, and web finding extraction.
- `benchmarks.py` owns national/comparative benchmark acquisition if it is not simply a web-search mode.
- `pipeline.py` combines those substages into one `L2EnrichmentStageResult`.
- Estimator Flow consumes `L2EnrichmentStageResult`; it does not run a second web search.

Concrete meaning for L1:

- `vector_retrieval.py` owns chunk retrieval and retrieval diagnostics.
- `markdown_extraction.py` owns LLM extraction of cited CCC evidence.
- `pipeline.py` combines retrieval and extraction into one `L1CccStageResult`.
- Future six-signal extraction can consume `L1CccStageResult`, but it should not be hidden inside vector retrieval or markdown extraction.

## V2 Artifact Layout

<<<CHANGE: Moved away from writing v2 sidecars into v1 `010_assumptions`; the new pipeline should have its own artifact root so v1 remains untouched.>>>

Use a separate v2 artifact root, for example:

```text
output/<run_id>/pipeline_v2/
  pipeline_context_v2.json
  stage_files/
    001_input_snapshot/
    010_l1_ccc_extraction/
    020_l2_third_party_enrichment/
    030_estimator_flow/
    040_review_loop/
    050_archetype_classification/
    060_demand_engine/
    070_demand_atlas/
    080_writer_projection/
  comparison/
    v1_v2_comparison.md
```

This avoids mixing canonical v2 artifacts with v1 stage files. The comparison harness can still link v1 and v2 artifacts side by side.

## Implementation Work Packages

### WP-0: Freeze V1 Baseline And Build Comparison Fixtures

Goal: keep the current pipeline untouched and measurable.

Work:

- Pick representative v1 runs for comparison.
- Preserve current v1 artifact expectations.
- Add a comparison fixture format that captures inputs, selected cities, question, config, and expected v1 artifact paths.
- Do not refactor v1 while building v2 unless a bug blocks comparison.

Acceptance criteria:

- V1 runs remain reproducible.
- V2 work can compare against v1 without modifying v1 internals.

### WP-1: Define `PipelineContextV2`

Goal: establish the canonical data model before moving logic.

Work:

- Create `backend/modules/pipeline_v2/models.py`.
- Define the root model and stage submodels.
- Define typed identifiers for city, source, evidence, field, assumption, archetype, demand cell, and artifact references.
- Add model-level serialization tests.

Acceptance criteria:

- `PipelineContextV2` round-trips through JSON.
- Each stage has a typed read/write contract.
- No stage implementation begins until the handoff model for that stage exists.

### WP-2: Implement L1 CCC Extraction Facade

Goal: reuse current CCC extraction while producing typed L1 output.

Work:

- Wrap current retrieval/vector/markdown researcher behavior.
- Convert accepted excerpts, rejected chunks, and source chunk index into `L1CccStageResult`.
- Keep current source traceability.
- Add optional archetype-signal extraction inputs, but do not require classifier output yet.

Acceptance criteria:

- Same inputs produce equivalent L1 evidence counts to v1.
- Typed L1 output links back to source chunks and quotes.

### WP-3: Implement L2 Third-Party Enrichment Facade

Goal: reuse current external source and web evidence acquisition under the PRD L2 stage.

Work:

- Wrap governed external Markdown search and resolver.
- Wrap web search/scrape/extract and benchmark finding collection.
- Split acquisition outputs from estimator decisions.
- Persist billing/search diagnostics separately from semantic evidence.

Acceptance criteria:

- L2 output clearly separates candidates, resolved evidence, no-evidence records, benchmark findings, and diagnostics.
- V2 can disable L2 without breaking L1 or later stages.

### WP-4: Implement Estimator Flow

Goal: place current gap analysis, freshness, field-status logic, and current assumptions estimator into the PRD estimator flow.

Work:

- Use L1 and L2 typed evidence as input.
- Run field decomposition and gap detection.
- Run freshness/field-status resolution.
- Run conservative gap-filling estimator.
- Output `EstimatorStageResult` containing `initial_assumptions: InitialAssumptionSet`.
- Preserve non-estimable decisions and saturation warnings.

Acceptance criteria:

- Current v1 gap-filling outcomes can be reproduced or intentionally improved.
- Initial assumptions are typed, traceable, and ready for review.
- The estimator flow does not call the Demand Engine.

### WP-5: Implement Review Loop

Goal: make reviewed assumptions a first-class handoff before demand calculation.

Work:

- Port current post-run review behavior to canonical records.
- Support analyst edits, critique, corrections, and status changes.
- Preserve regenerated document behavior only as an output projection, not as the core data model.

Acceptance criteria:

- Initial assumptions can become reviewed assumptions.
- Review history is append-only.
- Demand Engine only consumes reviewed assumptions, unless explicitly configured for draft mode.

### WP-6: Implement Archetype Input Contract

Goal: produce the second key input to the Demand Engine.

Work:

- Define `ArchetypeAssignment[]`.
- Add a loader for externally provided or manually curated archetype assignments.
- Validate city keys, archetype labels, confidence, source, version, and provenance.
- Keep a placeholder interface for future L1 six-signal extraction and classifier output.
- Preserve confidence and provenance.

Acceptance criteria:

- Each city has either an archetype assignment or an explicit missing-classification reason.
- Externally provided assignments cannot be mistaken for L1-extracted or partner-validated classifications.
- Demand Engine can consume assignments without knowing whether they came from an external source, manual override, or future L1 classifier.

### WP-7: Implement Domain Methodology Configs

Goal: make domain assumptions adjustable without code changes.

Work:

- Add Transport config first.
- Add Energy and Built Environment when defaults are available.
- Include parameter provenance and sign-off state.
- Include stock units, benchmark sources, levers, TEF mapping, rates, readiness multipliers, and caveats.

Acceptance criteria:

- Demand Engine parameters come from config.
- Defaults and partner-validated parameters are distinguishable.

### WP-8: Implement Demand Engine

Goal: implement the PRD's forward-looking demand assumption layer.

Work:

- Consume `ReviewedAssumptionSet`, `ArchetypeStageResult.assignments`, and domain methodology.
- Implement cohort-level calculations.
- Support demand waves.
- Preserve ranges, confidence, caveats, and provenance.

Acceptance criteria:

- Deterministic tests reproduce the PRD electric-bus worked example.
- Demand outputs are traceable to reviewed assumptions, archetype assignments, and domain parameters.

### WP-9: Implement Demand Atlas

Goal: produce the PRD's first product-facing output.

Work:

- Create `DemandAtlasStageResult` schema with a product-facing `DemandAtlasView`.
- Compose Demand Engine outputs by domain, intervention typology, archetype, wave, and time horizon where supported.
- Add drill-down references to assumptions and parameters.

Acceptance criteria:

- Demand Atlas can be rendered from structured data without reading free-form reports.
- Every cell has provenance and confidence metadata.

### WP-10: Implement V1/V2 Comparison Harness

Goal: prove v2 before replacing v1.

Work:

- Add `backend/scripts/compare_pipeline_v1_v2.py`.
- Run v1 and v2 on the same question/city/config inputs.
- Compare:
  - L1 evidence
  - L2 evidence
  - gap fields
  - initial assumptions
  - reviewed assumptions when present
  - writer projection
  - new demand outputs
- Write `output/<run_id>/pipeline_v2/comparison/v1_v2_comparison.md`.

Acceptance criteria:

- Differences are explicit, reviewable, and categorized as expected, improvement, regression, or new v2-only output.
- Promotion is blocked until comparison output is reviewed.

## What To Keep From The Current Build

<<<CHANGE: Added explicit "keep" guidance so implementation does not turn into an unnecessary rewrite.>>>

Keep where it makes sense:

- vector store retrieval mechanics
- markdown researcher extraction and evidence validation
- governed external source tools and resolver
- web search planning, scraping, and extraction machinery
- freshness comparison
- gap analysis logic
- conservative current assumptions estimator behavior
- run logging and artifact-writing utilities if they remain schema-neutral
- city normalization utilities
- tests and benchmarks that verify existing behavior

Clean up or wrap:

- loose dict handoffs
- tuple return values across stage boundaries
- mixed diagnostic/semantic payloads
- multiple assumption shapes
- writer-specific filtering embedded too deeply in upstream stages

Avoid:

- v2 internals depending on v1 `context_bundle` dict shapes
- v2 artifacts written into v1 stage folders as if they are the same pipeline
- treating current gap-filling assumptions as the final PRD Demand Engine
- keeping multiple "almost canonical" schemas alive

## Testing Matrix

| Area               | Required coverage                                                                              |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| Canonical model    | `PipelineContextV2` round-trip, schema validation, stage-slice validation                      |
| L1 CCC extraction  | v1/v2 evidence-count comparison, source traceability                                           |
| L2 enrichment      | external/web evidence acquisition, no-evidence records, benchmark findings                     |
| Estimator flow     | gap analysis, freshness, field status, initial assumptions, non-estimable routing              |
| Review loop        | correction history, status transitions, reviewed assumption output                             |
| Archetype input    | externally provided assignment loading, confidence, provenance, missing classification reasons |
| Domain methodology | config validation, parameter provenance                                                        |
| Demand Engine      | deterministic PRD worked examples, range/confidence propagation                                |
| Demand Atlas       | structured cell outputs, drill-down provenance                                                 |
| Comparison harness | same-input v1/v2 report with categorized differences                                           |

## Open Decisions

<<<CHANGE: Added explicit decisions that should be resolved before implementation starts.>>>

- Should v2 use `output/<run_id>/pipeline_v2/` inside the same run folder, or create separate run ids such as `<run_id>_v2`?
- Should the first v2 implementation use the current writer as a projection, or skip writer output until Demand Atlas is available?
- Should web search be entirely L2 evidence acquisition, or should some benchmark search remain inside Estimator Flow? Recommended default: acquisition in L2, resolution in Estimator Flow.
- Should review be mandatory before Demand Engine runs? Recommended default: mandatory for product-facing Atlas, optional draft mode for internal experiments.
- Which domain gets implemented first? Recommended default: Transport, because the PRD has the electric-bus worked example.

## Final Recommendation

Build a parallel **Canonical Pipeline V2**, not an assumptions-only v2.

Use the current codebase as a library of proven logic:

- L1 reuses vector retrieval and markdown extraction.
- L2 reuses governed external sources and web evidence acquisition.
- Estimator Flow reuses gap analysis, freshness, field-status resolution, and current gap-filling assumptions.
- Review Loop reuses current post-run assumptions-review behavior after converting it to canonical records.

Then add the genuinely new PRD pieces:

- archetype assignment input contract, with future support for six-signal extraction/classification
- domain methodology configs
- Demand Engine
- Demand Atlas

This keeps working functionality where it makes sense, but places it under the PRD's logic and gives every stage clear Pydantic handoffs. V1 remains untouched until v2 can be compared and reviewed on the same inputs.
