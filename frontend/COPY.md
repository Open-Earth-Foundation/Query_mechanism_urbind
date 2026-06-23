# UI Copy & Terminology Guide

Source of truth for user-facing language in the Query Mechanism frontend. The
primary user is a **climate data analyst** who is *not* assumed to be technical
or familiar with the pipeline's internals. When in doubt, name things by the
analyst's goal, not by the engineering component that produces them.

## Voice

- **Plain over precise.** Prefer everyday words to internal/engineering terms.
  "Run query", not "submit a build run".
- **Goal-oriented.** Describe what the user gets, not how the system works.
  "Query demand signals from city plans", not "document-first retrieval flow".
- **Action labels are verbs.** Buttons: "Run query", "Load previous answer".
- **Define a domain term once, then reuse it.** Acronyms get a tooltip/expansion
  on first use (e.g. `CCC` → "City Climate Contracts").
- **No internal nouns in the UI.** Avoid "writer", "build", "pipeline artifacts",
  "retrieval query" (in standard mode), "context bundle" — these are implementation
  details. They may appear in **dev mode** only.
- **Sentence case** for labels and buttons (not Title Case), except product nouns
  used as proper names ("Enrichment Process", "Query Engine").

## Canonical terms

| Concept | Use this | Avoid | Notes |
|---|---|---|---|
| The product surface | **Query Engine** | Document Builder | Header eyebrow |
| The thing the user does | **Run a query** / **query** | build, run a build, generate a report | The verb is "query" |
| Primary input | **What demand signal or target are you looking for?** | Question 1 (required) | Demand-signal framing |
| Optional extra inputs | **Follow-up queries** | Additional retrieval questions, Question 2/3 | Lives under Advanced |
| The answer | **Report** | Generated Document, Writer Document, Writer Doc | One word everywhere |
| Left input panel | **Query Controls** | Build Controls | |
| Run trigger button | **Run query** | Generate Report | |
| Source documents | **City Climate Contracts** (`CCC`) | raw CCC, CCC Source | Expand acronym on first use |
| The data-gap audit view | **Enrichment Process** | Pipeline Artifacts | Audits classify → search → estimate |
| A field that got a value | **Estimated** (🔵 blue) | — | |
| A field with no value | **Non-estimable** (🔴 red) | — | |
| An in-between gap | **Unresolved** (🟡 amber) | — | |
| Where the chain failed | **Break** | — | Badge on the step that stopped the flow |
| Advanced settings | **Advanced options** | dev/expert toggles | Collapsed in standard, open in dev |

## Enrichment Process step names (analyst-facing)

| Step | Label | One-line summary pattern |
|---|---|---|
| Gap analysis | **Gap analysis** | "N fields classified" |
| External + web search | **External + web search** | "N found · M validated" (⚠ Break if found > 0, validated = 0) |
| Assumptions | **Assumptions** | "N estimated · M non-estimable" |

## Dev mode

Dev mode may surface internal terms (run id, context bundle, pipeline diagnostics,
writer context export). Keep that jargon **out of standard mode**. Nothing in
standard mode should require knowing how the pipeline is built.

## Applied vs. pending

Applied in this pass: Query Engine, Query demand signals…, Run query, Query
Controls, Report (rail + card + toggle), City Climate Contracts, Enrichment
Process, Advanced options, Follow-up queries, empty-state copy.

Pending / open questions for sign-off:
- Should `CCC` be fully spelled out as a tab label, or kept short with a tooltip?
  (Currently: short label "CCC" + tooltip "City Climate Contracts".)
- "Demand signal" vs "target" in the primary input label — confirm the term that
  resonates with analysts.
