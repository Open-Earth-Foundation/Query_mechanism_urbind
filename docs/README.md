# Project Documentation

This folder is organized for onboarding first, then deeper design history.

## Start Here

- [Pipeline Overview](pipeline/overview.md): end-to-end run flow and how `context_bundle.json` evolves.
- [Retrieval](pipeline/retrieval.md): how markdown chunks are selected before extraction.
- [Markdown Researcher](pipeline/markdown-researcher.md): how retrieved CCC context becomes cited excerpts.
- [Enrichment Overview](pipeline/enrichment-overview.md): how gap analysis, governed external sources, web research, freshness, and assumptions fit together.
- [Gap Analysis](pipeline/gap-analysis.md): how missing fields are decomposed and scoped.
- [External Sources](pipeline/external-sources.md): how governed Markdown sources are searched and resolved.
- [Web Research](pipeline/web-research.md): how open/tier-1 web search and freshness checks work.
- [Assumptions](pipeline/assumptions.md): how unresolved gaps are estimated or marked non-estimable.

## Operational Notes

- [Known Issues](known_issues.md): current enrichment risks and open follow-ups.
- [Vector Store Kubernetes Maintenance](vector_store_kubernetes_maintenance.md): deployed vector-store maintenance flow.
- [Data Pipeline Modularization Report](data_pipeline_modularization_report.md): architecture assessment and extraction-boundary analysis.

## Archive

Historical plans and deep implementation notes live under [archive](archive/README.md). They are useful for context, but they should not be treated as the current onboarding path.
