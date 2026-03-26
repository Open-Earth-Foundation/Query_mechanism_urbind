---
name: markdown-artifact-qa
description: Review generated, imported, or cleaned-up Markdown artifacts for conversion damage before changes are applied. Use when Codex needs to inspect Markdown files for encoding corruption, broken tables, whitespace merges, repeated extraction fragments, malformed blockquotes, or similar document-conversion artifacts and propose exact fixes before editing.
---

# markdown-artifact-qa

Review Markdown artifacts for conversion and cleanup defects, then surface exact proposed fixes before making edits.

## Workflow

1. Identify the artifact source and likely failure mode.

- Determine whether the Markdown came from PDF conversion, OCR, HTML export, copy/paste cleanup, or manual edits after conversion.
- Note the highest-risk structures first: tables, lists, blockquotes, headings, footnotes, and repeated fragments.

2. Inspect for the common artifact classes.

- Encoding corruption or replacement-character issues.
- Missing whitespace or merged words.
- Broken or duplicated table rows and cells.
- Repeated extraction fragments, single-character noise, or orphaned lines.
- Escaped or malformed blockquote markers such as HTML entities where Markdown syntax is expected.
- Heading, list, or fence imbalance introduced by conversion.

3. Surface findings in review-first form.

- Point to the affected file and the specific artifact pattern.
- Explain the likely cause briefly.
- Propose the minimum safe fix.
- State whether the issue looks local or whether it suggests a broader batch cleanup pass.

4. Propose validation after the fix.

- Re-check Markdown rendering around the changed block.
- Re-check adjacent table rows, list items, or quote blocks for collateral corruption.
- If the artifact source is a conversion pipeline, propose whether a generator-side fix is better than repeated manual cleanup.

## Output requirements

- Keep the result review-only until the user confirms edits.
- Prefer grouped findings by artifact class when the same defect repeats across files.
- When no artifact issues are found, say that explicitly and mention the highest residual risk briefly.
