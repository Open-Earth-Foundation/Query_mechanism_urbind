# Example: External Tagged Markdown Search for Krakow

## Scenario

The current CCC search remains unchanged. It runs first and tries to answer the
question using the existing CCC-derived Markdown corpus.

Example user question:

```text
For Krakow, what 2030 public EV charging infrastructure target is stated, and
how does it affect the city's mobility transition planning?
```

Assume this run has tagged Krakow external sources available and the CCC
extraction finds no useful Krakow evidence:

```json
{
  "city": "Krakow",
  "field": "public_ev_chargers_2030_target",
  "ccc_status": "missing",
  "ccc_evidence": []
}
```

Because tagged Krakow external sources exist, the pipeline runs the external
tagged Markdown search stage by default before the assumptions estimator. It
does not wait for an explicit user request to search external sources.

## Example Tagged File

We do not create this file here. This is an example of how a converted external
Markdown document would be represented by folder-level metadata.

File:

```text
external_docs/krakow/krakow_electromobility_strategy_2030.md
```

Folder-level metadata entry in `external_docs/sources.yaml`:

```yaml
sources:
  - source_id: krakow_electromobility_strategy_2030
    title: Krakow Electromobility Strategy 2030
    city: Krakow
    country: Poland
    publication_year: 2024
    description: City electromobility strategy covering public charging infrastructure, zero-emission buses, municipal fleet electrification, and low-emission transport zones.
    source_type: mobility_plan
    publisher: City of Krakow
    verticals: [mobility]
    tef_sectors: [transport]
    source_url: https://example.krakow.pl/electromobility-strategy-2030
```

The loader resolves this metadata to
`external_docs/krakow/krakow_electromobility_strategy_2030.md` by matching the
`source_id` to the Markdown filename stem. No manual `path` tag is needed.

## Step-by-Step Flow

1. Existing CCC search runs first

   The current CCC search and extraction pipeline receives the question and
   selected city, then searches existing CCC Markdown sources.

   Result:

   ```json
   {
     "city": "Krakow",
     "field": "public_ev_chargers_2030_target",
     "status": "missing",
     "reason": "No CCC excerpt contains a concrete public charging target for Krakow."
   }
   ```

2. External search starts by default when tagged Krakow sources exist

   The enrichment stage sees that tagged Krakow sources exist, notes that the
   field is unresolved in CCC, and starts external search before assumptions:

   ```json
   {
     "city": "Krakow",
     "country": "Poland",
     "verticals": ["mobility"],
     "tef_sectors": ["transport"],
     "field": "public_ev_chargers_2030_target"
   }
   ```

3. The agent asks what tags are available

   Tool call:

   ```python
   get_tag_options()
   ```

   Example output:

   ```json
   {
     "cities": ["Aachen", "Dresden", "Klagenfurt", "Krakow", "Leipzig", "Munich"],
     "countries": ["Austria", "Germany", "Poland"],
     "source_types": ["city_cap", "mobility_plan", "national_dataset"],
     "verticals": ["mobility", "energy", "built_environment"],
     "tef_sectors": ["transport", "energy", "buildings"]
   }
   ```

   The agent learns that Krakow, Poland, mobility, and transport are valid
   filters.

4. The agent lists candidate sources

   Tool call:

   ```python
   list_candidate_sources(
       cities=["Krakow"],
       countries=["Poland"],
       verticals=["mobility"],
       tef_sectors=["transport"],
   )
   ```

   Example output:

   ```json
   [
     {
       "source_id": "krakow_electromobility_strategy_2030",
       "title": "Krakow Electromobility Strategy 2030",
       "city": "Krakow",
       "country": "Poland",
       "publication_year": 2024,
       "source_type": "mobility_plan",
       "verticals": ["mobility"],
       "tef_sectors": ["transport"]
     }
   ]
   ```

   The agent now has one relevant file to search.

5. The agent generates source-language synonyms and starts with bounded search

   Before searching, the agent drafts source-language variants with the LLM,
   such as `charging points`, `charging infrastructure`, `punkty ladowania`,
   `infrastruktura ladowania`, `2030`, `target`, and `cel`.

   Tool call:

   ```python
   regex_search(
       pattern="(?i)(charging points|charging infrastructure|punkty ladowania|infrastruktura ladowania).{0,120}(2030|target|goal|cel)",
       cities=["Krakow"],
       countries=["Poland"],
       verticals=["mobility"],
       tef_sectors=["transport"],
       context_words=90,
       context_lines=3,
       max_matches=20,
   )
   ```

   Example output:

   ```json
   [
     {
       "hit_id": "hit_001",
       "source_id": "krakow_electromobility_strategy_2030",
       "title": "Krakow Electromobility Strategy 2030",
       "city": "Krakow",
       "line_start": 142,
       "line_end": 149,
       "matched_text": "public charging points by 2030",
       "snippet": "By 2030, Krakow plans to expand public charging infrastructure to 1,200 public charging points, prioritising park-and-ride sites, municipal car parks, and transport interchanges. The programme supports the wider zero-emission mobility transition and municipal fleet electrification.",
       "heading_path": ["3. Charging Infrastructure", "3.2 Public network targets"]
     }
   ]
   ```

6. The agent expands the promising hit

   The snippet already looks useful, but the agent wants more context to verify
   whether the number is a city target and not a national reference.

   Tool call:

   ```python
   expand_hit(
       hit_id="hit_001",
       context_words=250,
       context_lines=10,
   )
   ```

   Example output:

   ```json
   {
     "hit_id": "hit_001",
     "source_id": "krakow_electromobility_strategy_2030",
     "line_start": 136,
     "line_end": 158,
     "snippet": "The municipal charging programme defines city-level infrastructure targets for 2030. By 2030, Krakow plans to expand public charging infrastructure to 1,200 public charging points, prioritising park-and-ride sites, municipal car parks, and transport interchanges. The city will coordinate deployment with distribution grid upgrades and the electrification of public transport depots.",
     "heading_path": ["3. Charging Infrastructure", "3.2 Public network targets"]
   }
   ```

7. The agent saves the hit as evidence

   Tool call:

   ```python
   add_evidence_candidate(
       hit_id="hit_001",
       city="Krakow",
       field="public_ev_chargers_2030_target",
       reason="Contains a concrete 2030 city-level target for public charging points.",
       confidence=0.9,
   )
   ```

   Example saved evidence:

   ```json
   {
     "candidate_id": "ev_001",
     "hit_id": "hit_001",
     "source_id": "krakow_electromobility_strategy_2030",
     "city": "Krakow",
     "field": "public_ev_chargers_2030_target",
     "line_start": 136,
     "line_end": 158,
     "matched_text": "1,200 public charging points",
     "quote": "By 2030, Krakow plans to expand public charging infrastructure to 1,200 public charging points, prioritising park-and-ride sites, municipal car parks, and transport interchanges.",
     "confidence": 0.9,
     "reason": "Contains a concrete 2030 city-level target for public charging points."
   }
   ```

8. Claim extraction converts evidence into a structured fact

   The claim extractor reads saved evidence candidates and returns structured
   claims:

   ```json
   {
     "city": "Krakow",
     "field": "public_ev_chargers_2030_target",
     "value": 1200,
     "unit": "public charging points",
     "target_year": 2030,
     "source_id": "krakow_electromobility_strategy_2030",
     "source_type": "mobility_plan",
     "publication_year": 2024,
     "line_start": 136,
     "line_end": 158,
     "quote": "By 2030, Krakow plans to expand public charging infrastructure to 1,200 public charging points...",
     "confidence": 0.9,
     "claim_role": "fills_missing"
   }
   ```

9. Resolver compares CCC and external evidence

   Resolver input:

   ```json
   {
     "ccc": {
       "city": "Krakow",
       "field": "public_ev_chargers_2030_target",
       "status": "missing",
       "claims": []
     },
     "external": [
       {
         "city": "Krakow",
         "field": "public_ev_chargers_2030_target",
         "value": 1200,
         "unit": "public charging points",
         "target_year": 2030,
         "source_id": "krakow_electromobility_strategy_2030",
         "confidence": 0.9
       }
     ]
   }
   ```

   Resolver output:

   ```json
   {
     "city": "Krakow",
     "field": "public_ev_chargers_2030_target",
     "resolution": "external_fills_ccc_gap",
     "selected_value": 1200,
     "unit": "public charging points",
     "source_id": "krakow_electromobility_strategy_2030",
     "reason": "CCC has no value. Tagged external mobility plan provides a concrete city-level 2030 target with high confidence.",
     "requires_review": false
   }
   ```

10. The assumptions estimator is not used for this field

   The example stops before assumptions because the missing value was found in
   tagged external evidence. No estimate is needed for this field.

11. Writer receives enriched context

   The writer can now answer with clear provenance:

   ```text
   Krakow's external electromobility strategy states a 2030 target of 1,200
   public charging points. This fills a gap in the CCC evidence and supports the
   city's mobility transition planning around charging infrastructure, park-and-
   ride sites, transport interchanges, and fleet electrification.
   ```

   The final answer can cite the external source and line-backed quote instead
   of inventing an assumption.

## What This Adds to the Current Search

Current behavior:

```text
CCC search finds no Krakow EV charging target
-> field remains missing
-> assumptions estimator may estimate or mark non-estimable
```

Enhanced behavior:

```text
CCC search finds no Krakow EV charging target
-> external tagged Markdown search runs only over relevant Krakow mobility docs
-> agent finds a line-backed 2030 target
-> evidence is saved and converted into a structured claim
-> resolver fills the CCC gap
-> assumptions estimator is skipped for this field
```

The key improvement is that additional documents become governed evidence, not
free-form web context and not assumptions.

## If No Evidence Is Found

If the agent searches the relevant tagged files and finds nothing, it records
that explicitly:

```python
mark_no_evidence_found(
    city="Krakow",
    field="public_ev_chargers_2030_target",
    searched_source_ids=["krakow_electromobility_strategy_2030"],
    search_summary="Searched charging, charger, charging points, infrastructure, 2030, target, goal, and related proximity patterns. No concrete numeric target found.",
)
```

Then the field can proceed to assumptions or non-estimable handling with a clear
audit trail showing that CCC and tagged external sources were checked first.
