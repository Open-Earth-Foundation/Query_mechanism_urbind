# TEF Mapping Catalog

This directory contains local reference material for staged mapping from extracted city initiatives to TEF Transition Elements or, when no Transition Element exists, the best matching TEF category.

The mapping flow is:

1. Route an extracted initiative to one of six TEF sectors.
2. Route within the selected sector through first-level subcategories and deeper subsubcategories.
3. Rank the current category's direct Transition Elements when they exist.
4. Map to the selected category itself when the selected leaf has no Transition Elements.
5. Send ambiguous mappings to review instead of forcing a match.

This directory is a catalog and prompt asset bundle only. It does not implement runtime orchestration, database writes, or initiative extraction.

## Layout

- `catalog/sectors.json`: top-level TEF sectors.
- `catalog/subcategories.json`: first-level subcategories under sectors.
- `catalog/subsubcategories.json`: all deeper category levels.
- `catalog/transition_elements.json`: compact Transition Element records filterable by `path`.
- `prompts/`: reference prompt templates for sector routing, category routing, and Transition Element ranking.

Sector, subcategory, and subsubcategory cards include prompt-ready routing guidance
directly in JSON. For category records, `description` and `card_text` include
sector-style `Routing Definition`, `Use This Category When`, and `Avoid This Category
When` sections for every record in `subcategories.json` and `subsubcategories.json`.

## Mapping Target

Initiatives are mapped to TEF `Transition Elements` when a selected category has direct Transition Elements. A no-transition leaf, such as `2-industry/2a-minerals/2a5-soda-ash`, is a valid final `subcategory` target. TEF `activities` and `parameters` are intentionally excluded from this catalog.

## Source

See `SOURCE.md` for TEF source, commit, extraction date, and license attribution.
