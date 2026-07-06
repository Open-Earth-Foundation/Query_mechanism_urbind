# Project Documentation

This folder is organized for onboarding first, then deeper design history.

## Start Here

Read the pipeline docs in order under [docs/pipeline/](pipeline/00_overview.md):

- [00 Overview](pipeline/00_overview.md): end-to-end run flow and how `context_bundle.json` evolves.
- [01 Input Snapshot](pipeline/01_input-snapshot.md): reproducibility snapshots and planned stages.
- [02 Query Preparation](pipeline/02_query-preparation.md): research question and retrieval queries.
- [03 Retrieval](pipeline/03_retrieval.md): how markdown chunks are selected before extraction.
- [04 Markdown Researcher](pipeline/04_markdown-researcher.md): how retrieved CCC context becomes cited excerpts.
- [05 Enrichment Overview](pipeline/05_enrichment-overview.md): how gap analysis, governed external sources, and web research fit together.
- [06 Gap Analysis](pipeline/06_gap-analysis.md): how missing fields are decomposed and scoped.
- [07 External Sources](pipeline/07_external-sources.md): how governed Markdown sources are searched and resolved.
- [08 Web Research](pipeline/08_web-research.md): how open/tier-1 web search and freshness checks work.
- [09 Enrichment Context Handoff](pipeline/09_enrichment-context-handoff.md): freezing post-enrichment context.
- [10 Assumptions](pipeline/10_assumptions.md): how unresolved gaps are estimated or marked non-estimable.
- [11 Assumptions Context Handoff](pipeline/11_assumptions-context-handoff.md): freezing post-assumptions context.
- [12 Writer](pipeline/12_writer.md): final report generation and citation coverage.

## Operational Notes

- [Known Issues](known_issues.md): current enrichment risks and open follow-ups.
- [Vector Store Kubernetes Maintenance](vector_store_kubernetes_maintenance.md): deployed vector-store maintenance flow.
- [Data Pipeline Modularization Report](data_pipeline_modularization_report.md): architecture assessment and extraction-boundary analysis.

## Archive

Historical plans and deep implementation notes live under [archive](archive/README.md). They are useful for context, but they should not be treated as the current onboarding path.
