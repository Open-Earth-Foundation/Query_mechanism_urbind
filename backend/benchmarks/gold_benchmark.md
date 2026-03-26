# Gold Benchmark Design

## Purpose

This document now does two things:

1. describes the benchmark setup that currently exists in the repository
2. states the target design we want to move toward later

It is intentionally aligned with the current fixture and runner behavior as of
March 26, 2026. It does not assume the migration has already happened.

## Current State

The current gold fixture lives in:

- `tests/fixtures/benchmark_gold.json`

The current fixture is still the old path-based format:

- schema version is `1`
- top-level keys are `version` and `cases`
- there is no top-level `chunks` catalog yet
- there is no `split_spec` yet
- each case still carries a `cached_run_dir`
- `gold_chunk_ids` are still runtime-style chunk ids such as `chunk_...`

Each current case contains:

- `case_id`
- `question`
- `gold_chunk_ids`
- optional `gold_chunk_texts` for canonical chunk-text or excerpt fallback matching
- `gold_facts`
- `gold_city`
- `selected_cities`
- `cached_run_dir`

This means the benchmark is only partially self-contained today:

- the scoring expectations live in the fixture
- some cases may now include fixture-owned canonical chunk text or excerpts
- cached-mode execution still depends on historical run folders
- chunk identity still follows current runtime ids rather than canonical
  benchmark-owned ids

## Current Benchmark Flow

Today the benchmark can run in two practical modes.

### 1. Cached-mode verification

The runner reads `cached_run_dir` from the fixture and scores a previously
generated run.

This is useful for:

- fast fixture verification
- comparing scoring logic without rerunning the pipeline

This is limited because:

- it depends on external run artifacts
- the fixture is not portable by itself

### 2. Live benchmark execution

The runner executes the live pipeline for the case question and selected cities,
then scores the run against the fixture's `gold_chunk_ids` and `gold_facts`.

When `gold_chunk_texts` is present for a case, scoring can also fall back to
canonical chunk-text matching or canonical excerpt containment when runtime chunk
ids drift.

This is useful for:

- checking current retrieval behavior
- measuring stage-by-stage information loss

This still depends on current runtime chunk ids matching what the fixture
expects.

## Current Scoring Stages

The benchmark currently measures loss across three stages:

1. retrieval
2. markdown extraction
3. final answer generation

Current primary metrics:

- retrieval recall
- retrieval precision
- MRR
- delivery recall
- delivery precision
- extraction recall
- fact extraction rate
- end-to-end fact recall
- citation coverage

## Current Case Inventory

The current real gold fixture contains these 9 cases:

- `vehicle_targets_cross_ccc`
- `capex_cross_ccc`
- `quantified_charging_targets_munster`
- `charging_targets_germany_seven_cities`
- `dresden_charging_pilots_and_retrofits`
- `heidelberg_transport_electrification_capex`
- `mannheim_transport_electrification_capex`
- `krakow_vehicle_targets`
- `krakow_warszawa_transport_electrification`

## What Is Good About The Current Setup

The current setup is already useful because:

- benchmark questions, gold facts, and gold chunk ids live in one fixture
- live runs can be scored stage by stage
- benchmark reports explain where evidence was lost
- the fixture is easy to edit when refining case quality

## Current Limitations

The current setup still has important structural weaknesses:

- `cached_run_dir` keeps cached-mode verification tied to `output/` artifacts
- runtime chunk ids are used as benchmark ids
- there is no canonical chunk catalog inside the fixture
- the benchmark cannot yet validate chunk text directly from fixture-owned data
- portability across machines is weaker than it should be

## Direction We Want To Go

We still want the benchmark to become self-contained.

The target design is:

- one versioned fixture schema
- one canonical benchmark-owned chunk catalog
- chunk text embedded directly in the fixture
- no dependency on `cached_run_dir`
- no dependency on external `output/` folders for scoring
- stable benchmark chunk ids that are independent from production vector-store
  ids

The benchmark should still be able to run the live pipeline, but scoring should
be anchored to canonical benchmark data rather than historical runs.

## Target Fixture Shape

Recommended future structure:

```json
{
  "version": 2,
  "split_spec": {
    "name": "benchmark_split_v1",
    "description": "Canonical markdown split used by the gold benchmark"
  },
  "chunks": [
    {
      "chunk_id": "aachen_0001",
      "city": "Aachen",
      "document_title": "Aachen CCC",
      "section_path": ["Charging Infrastructure"],
      "text": "Canonical benchmark chunk text...",
      "text_sha256": "..."
    }
  ],
  "cases": [
    {
      "case_id": "charging_targets_cross_ccc",
      "question": "What charging infrastructure volume targets by 2030 or beyond are explicitly referenced or inferable in the CCCs...?",
      "selected_cities": ["Aachen"],
      "gold_chunk_ids": ["aachen_0001", "aachen_0002"],
      "gold_facts": [
        "Aachen targets approximately 2,400 charging points by 2030."
      ]
    }
  ]
}
```

## What Changes Later, Not Now

This document is not saying the migration is complete.

What remains deferred:

- replacing `cached_run_dir`
- materializing a canonical chunk catalog
- changing the fixture to `version: 2`
- mapping runtime artifacts to benchmark-owned chunk ids

## Migration Direction

When we choose to do the migration, the intended order is:

### Phase 1: Add the new fixture schema

- introduce `version: 2`
- add `split_spec`
- add a top-level `chunks` catalog

### Phase 2: Materialize benchmark-owned chunks

- extract the exact gold chunk text for each current case
- assign stable benchmark chunk ids
- store chunk text directly in the fixture

### Phase 3: Update scoring

- score against canonical chunk ids instead of runtime vector ids
- stop requiring `cached_run_dir` for benchmark validity

### Phase 4: Remove the old path-based dependency

- retire the old external-path pattern once the new fixture is stable

## Recommended Direction

For now, the benchmark should be maintained as:

- a high-quality `version: 1` gold fixture
- live-run capable
- explicit about its dependency on current runtime chunk ids and cached runs

Later, it should be upgraded into:

- a self-contained canonical benchmark dataset
- portable across machines
- reviewable directly from fixture contents
- robust to production vector-store changes
