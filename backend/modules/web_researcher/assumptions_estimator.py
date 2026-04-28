"""Assumptions Estimator (Agent 6): produce model-estimated values for data gaps."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from openai import OpenAI

from backend.modules.sources.manifest import load_manifest
from backend.modules.web_researcher.data_lookups import (
    find_matching_structured_lookups,
)
from backend.modules.web_researcher.models import (
    AssumptionRecord,
    EnrichedField,
    EstimateRange,
    FieldClassification,
    FieldDecomposition,
    GapManifest,
    NonEstimableRecord,
    StructuredLookupResult,
    WebFinding,
    _AssumptionsEnvelope,
)
from backend.modules.web_researcher.utils.json_helpers import (
    extract_json_candidate,
    extract_message_text,
)
from backend.services.progress_tracker import ProgressTracker
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)


def _structured_lookup_to_assumption(
    lookup: StructuredLookupResult,
) -> AssumptionRecord | None:
    """Lift a deterministic lookup result into an AssumptionRecord.

    Returns None when the lookup row is empty or non-numeric (the estimator
    keeps the field for the LLM pass to handle).
    """
    if lookup.value is None:
        return None
    try:
        numeric = float(lookup.value)
    except (TypeError, ValueError):
        return None

    operators = lookup.extra.get("source_name") if isinstance(lookup.extra, dict) else None
    source_name = str(operators) if operators else lookup.ingestion_id
    unit = lookup.unit or ""
    asof = lookup.asof or "current registry snapshot"

    estimate = EstimateRange(low=numeric, mid=numeric, high=numeric)
    return AssumptionRecord(
        city=lookup.city,
        field_name=lookup.field,
        gap_description=f"Resolved deterministically from {source_name}.",
        method_used="structured_lookup",
        estimate=estimate,
        confidence="HIGH",
        reference_data=(
            f"{source_name}: {numeric:g} {unit} ({asof})".strip()
        ),
        rationale=(
            f"Direct lookup from {source_name} for {lookup.city} / {lookup.field}; "
            "no estimation required."
        ),
        basis="structured_lookup",
        is_replaceable=False,
    )


def _resolve_via_structured_lookups(
    estimable_fields: list[EnrichedField],
    gap_manifest: GapManifest,
) -> tuple[list[AssumptionRecord], list[EnrichedField]]:
    """Try to resolve gaps deterministically before invoking the LLM.

    Returns ``(resolved_assumptions, remaining_fields)``.  Resolution is
    keyed by ``(city, field)`` — fields not covered by any structured
    lookup pass through unchanged.
    """
    if not estimable_fields:
        return [], estimable_fields

    # Find lookups for the (cities, fields) the estimator would otherwise estimate.
    cities = sorted({field.city for field in estimable_fields})
    decomposition = FieldDecomposition(
        query_fields=[
            FieldClassification(
                field=fc.field,
                classification=fc.classification,
                searchable=fc.searchable,
                rationale=fc.rationale,
            )
            for fc in gap_manifest.query_fields
            if any(field.field == fc.field for field in estimable_fields)
        ],
        non_estimable_fields=list(gap_manifest.non_estimable_fields),
    )
    if not decomposition.query_fields:
        return [], estimable_fields

    try:
        manifest = load_manifest()
    except FileNotFoundError:
        logger.info(
            "Assumptions estimator: no sources manifest; skipping structured-lookup grounding."
        )
        return [], estimable_fields
    except Exception:  # noqa: BLE001
        logger.warning(
            "Assumptions estimator: failed to load sources manifest", exc_info=True
        )
        return [], estimable_fields

    try:
        lookups = find_matching_structured_lookups(decomposition, cities, manifest)
    except Exception:  # noqa: BLE001
        logger.warning("Assumptions estimator: structured lookups raised", exc_info=True)
        return [], estimable_fields

    if not lookups:
        return [], estimable_fields

    resolved_index: dict[tuple[str, str], AssumptionRecord] = {}
    for lookup in lookups:
        record = _structured_lookup_to_assumption(lookup)
        if record is None:
            continue
        resolved_index[(record.city, record.field_name)] = record

    if not resolved_index:
        return [], estimable_fields

    resolved_assumptions: list[AssumptionRecord] = []
    remaining: list[EnrichedField] = []
    for field in estimable_fields:
        record = resolved_index.get((field.city, field.field))
        if record is not None:
            resolved_assumptions.append(record)
        else:
            remaining.append(field)

    logger.info(
        "Assumptions estimator: structured lookups resolved %d/%d gap(s).",
        len(resolved_assumptions),
        len(estimable_fields),
    )
    return resolved_assumptions, remaining

_METHOD_C_SATURATION_THRESHOLD = 0.60
_MIN_PEER_ANCHORS_METHOD_B = 2
_MIN_PEER_ANCHORS_METHOD_C = 3
_MIN_PEER_CONFIDENCE = 0.7
_MIN_AUTHORITATIVE_CONFIDENCE = 0.8


def run_assumptions_estimator(
    question: str,
    context_bundle: dict[str, Any],
    gap_manifest: GapManifest,
    enriched_fields: list[EnrichedField],
    config: AppConfig,
    api_key: str,
    progress: ProgressTracker | None = None,
    national_benchmarks: list[WebFinding] | None = None,
    comparative_data: list[WebFinding] | None = None,
) -> tuple[list[AssumptionRecord], list[NonEstimableRecord], str | None]:
    """Estimate values for remaining data gaps using a priority ladder.

    Returns (assumptions, non_estimable_records, saturation_warning).
    On failure returns empty lists.
    """
    # Determine which fields still need estimation
    still_missing = [f for f in enriched_fields if f.status == "still_missing"]
    partially_resolved = [f for f in enriched_fields if f.status == "partially_resolved"]
    fields_to_estimate = still_missing + partially_resolved

    if not fields_to_estimate:
        logger.info("Assumptions estimator: no gaps to estimate.")
        return [], [], None

    if progress:
        for f in fields_to_estimate:
            progress.add_item(
                "assumptions",
                f"Gap: {f.city} / {f.field} ({f.status})",
                item_type="gap",
                title=f"{f.city} / {f.field}",
                metadata={"city": f.city, "field": f.field, "status": f.status},
            )
        progress.add_item(
            "assumptions",
            f"{len(fields_to_estimate)} fields to estimate",
        )

    # Separate non-estimable fields immediately (no LLM needed)
    non_estimable_field_names = set(gap_manifest.non_estimable_fields)
    non_estimable_records: list[NonEstimableRecord] = []
    estimable_fields: list[EnrichedField] = []

    for field in fields_to_estimate:
        if field.field in non_estimable_field_names:
            non_estimable_records.append(
                NonEstimableRecord(
                    city=field.city,
                    field_name=field.field,
                    gap_description=f"Missing {field.field} for {field.city}",
                    explanation=(
                        "This field is classified as non-estimable: qualitative, "
                        "unique-to-city, or legally specific data that cannot be "
                        "reliably estimated from peer data."
                    ),
                    recommendation=(
                        f"Contact {field.city} directly or consult local policy "
                        "documents for authoritative data on this field."
                    ),
                )
            )
        else:
            estimable_fields.append(field)

    if progress and non_estimable_records:
        for nr in non_estimable_records:
            progress.add_item(
                "assumptions",
                f"{nr.city}: {nr.field_name} — non-estimable",
                item_type="field",
                title=f"{nr.city}: {nr.field_name}",
                metadata={"status": "non_estimable"},
            )

    if not estimable_fields:
        logger.info(
            "Assumptions estimator: all %d gaps are non-estimable.",
            len(non_estimable_records),
        )
        return [], non_estimable_records, None

    # Step 9: try deterministic structured lookups before any LLM call.
    # Fields directly answered by a structured source (e.g. Bnetza) get a
    # high-confidence ``method_used="structured_lookup"`` AssumptionRecord
    # and are removed from the LLM-driven pass.
    structured_assumptions, estimable_fields = _resolve_via_structured_lookups(
        estimable_fields, gap_manifest
    )
    if progress and structured_assumptions:
        for a in structured_assumptions:
            progress.add_item(
                "assumptions",
                f"{a.city} / {a.field_name}: {a.estimate.mid} [structured_lookup, HIGH]",
                item_type="estimate",
                title=f"{a.city} / {a.field_name}",
                metadata={
                    "method": a.method_used,
                    "confidence": a.confidence,
                    "mid": str(a.estimate.mid),
                },
            )

    if not estimable_fields:
        logger.info(
            "Assumptions estimator: all gaps resolved via structured lookups; "
            "skipping LLM passes."
        )
        return structured_assumptions, non_estimable_records, None

    # Build peer reference once — used for anchor check and both LLM passes
    peer_reference = _build_peer_reference_table(enriched_fields, estimable_fields)

    # Anchor sufficiency: fields with too few peer values can't use Method B
    # and are unlikely to produce reliable estimates.
    estimable_fields, insufficient_records = _check_anchor_sufficiency(
        peer_reference, estimable_fields,
    )
    non_estimable_records.extend(insufficient_records)

    if progress and insufficient_records:
        # Group by field so progress doesn't list every city×field pair
        _insuf_by_field: dict[str, list[str]] = defaultdict(list)
        for nr in insufficient_records:
            _insuf_by_field[nr.field_name].append(nr.city)
        for field_name, cities in _insuf_by_field.items():
            city_list = ", ".join(cities)
            progress.add_item(
                "assumptions",
                f"{field_name} — insufficient anchors ({len(cities)} cities: {city_list})",
                item_type="field",
                title=field_name,
                metadata={"status": "insufficient_anchors", "cities": cities},
            )

    if not estimable_fields:
        logger.info(
            "Assumptions estimator: all remaining gaps have insufficient anchors.",
        )
        return [], non_estimable_records, None

    # Pass 1: generate estimates
    if progress:
        progress.add_item(
            "assumptions",
            f"Estimating {len(estimable_fields)} fields via priority ladder...",
        )
    assumptions = _call_estimator(
        question=question,
        context_bundle=context_bundle,
        gap_manifest=gap_manifest,
        estimable_fields=estimable_fields,
        all_enriched_fields=enriched_fields,
        config=config,
        api_key=api_key,
        pass_name="generate",
        peer_reference=peer_reference,
        national_benchmarks=national_benchmarks,
        comparative_data=comparative_data,
    )

    if progress and assumptions:
        for a in assumptions:
            mid = a.estimate.mid if hasattr(a.estimate, "mid") else (
                a.estimate.get("mid", "?") if isinstance(a.estimate, dict) else "?"
            )
            low = a.estimate.low if hasattr(a.estimate, "low") else (
                a.estimate.get("low", "?") if isinstance(a.estimate, dict) else "?"
            )
            high = a.estimate.high if hasattr(a.estimate, "high") else (
                a.estimate.get("high", "?") if isinstance(a.estimate, dict) else "?"
            )
            progress.add_item(
                "assumptions",
                f"{a.city} / {a.field_name}: {mid} ({low}–{high}) "
                f"[{a.method_used}, {a.confidence}]",
                item_type="estimate",
                title=f"{a.city} / {a.field_name}",
                metadata={
                    "method": a.method_used,
                    "confidence": a.confidence,
                    "mid": str(mid),
                    "low": str(low),
                    "high": str(high),
                },
            )
        progress.add_item(
            "assumptions",
            f"Pass 1 done: {len(assumptions)} estimates",
        )

    # Pass 2: critique and revise (capped at 1 cycle)
    if assumptions:
        if progress:
            progress.add_item("assumptions", "Reviewing estimates (critique pass)...")
        revised = _call_estimator(
            question=question,
            context_bundle=context_bundle,
            gap_manifest=gap_manifest,
            estimable_fields=estimable_fields,
            all_enriched_fields=enriched_fields,
            config=config,
            api_key=api_key,
            pass_name="critique",
            prior_estimates=assumptions,
            peer_reference=peer_reference,
            national_benchmarks=national_benchmarks,
            comparative_data=comparative_data,
        )
        if revised:
            assumptions = revised
            if progress:
                for a in assumptions:
                    mid = a.estimate.mid if hasattr(a.estimate, "mid") else (
                        a.estimate.get("mid", "?") if isinstance(a.estimate, dict) else "?"
                    )
                    progress.add_item(
                        "assumptions",
                        f"Revised: {a.city} / {a.field_name}: {mid} [{a.method_used}, {a.confidence}]",
                        item_type="estimate",
                        title=f"{a.city} / {a.field_name}",
                        metadata={
                            "method": a.method_used,
                            "confidence": a.confidence,
                            "mid": str(mid),
                            "revised": True,
                        },
                    )
                progress.add_item(
                    "assumptions",
                    f"Pass 2 done: {len(assumptions)} revised estimates",
                )

    # Combine deterministic lookup results with the LLM-produced ones.
    final_assumptions = list(structured_assumptions) + list(assumptions)

    # Saturation is only meaningful for LLM estimates — lookups have HIGH
    # confidence by construction, so they shouldn't tilt the diagnostic.
    saturation_warning = _check_saturation(assumptions)

    logger.info(
        "Assumptions estimator completed: assumptions=%d (structured_lookup=%d) non_estimable=%d saturation=%s",
        len(final_assumptions),
        len(structured_assumptions),
        len(non_estimable_records),
        saturation_warning is not None,
    )
    return final_assumptions, non_estimable_records, saturation_warning


def _call_estimator(
    question: str,
    context_bundle: dict[str, Any],
    gap_manifest: GapManifest,
    estimable_fields: list[EnrichedField],
    all_enriched_fields: list[EnrichedField],
    config: AppConfig,
    api_key: str,
    pass_name: str,
    prior_estimates: list[AssumptionRecord] | None = None,
    peer_reference: dict[str, list[dict[str, object]]] | None = None,
    national_benchmarks: list[WebFinding] | None = None,
    comparative_data: list[WebFinding] | None = None,
) -> list[AssumptionRecord]:
    """Make a single LLM call for estimation or critique."""
    model = config.enrichment.assumptions_estimator_model or config.enrichment.model
    temperature = config.enrichment.assumptions_estimator_temperature

    if peer_reference is None:
        peer_reference = _build_peer_reference_table(all_enriched_fields, estimable_fields)

    client = OpenAI(api_key=api_key, base_url=config.openrouter_base_url)
    system_prompt = _build_system_prompt(pass_name)
    user_prompt = _build_user_prompt(
        question=question,
        context_bundle=context_bundle,
        gap_manifest=gap_manifest,
        estimable_fields=estimable_fields,
        pass_name=pass_name,
        prior_estimates=prior_estimates,
        peer_reference=peer_reference,
        national_benchmarks=national_benchmarks,
        comparative_data=comparative_data,
    )

    request_kwargs: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
    }
    if config.enrichment.reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = config.enrichment.reasoning_effort

    try:
        logger.info(
            "Assumptions estimator pass=%s model=%s fields=%d",
            pass_name,
            model,
            len(estimable_fields),
        )
        response = client.chat.completions.create(**request_kwargs)
        if not response.choices:
            raise ValueError("Assumptions estimator returned no choices.")

        content = extract_message_text(response.choices[0].message.content)
        candidate = extract_json_candidate(content)
        parsed = json.loads(candidate)
        envelope = _AssumptionsEnvelope.model_validate(parsed)
        return envelope.assumptions

    except Exception:
        logger.warning(
            "Assumptions estimator pass=%s failed; returning empty.",
            pass_name,
            exc_info=True,
        )
        return []


def _build_system_prompt(pass_name: str) -> str:
    base = (
        "You are an assumptions estimator for urban climate action data.\n"
        "You produce model-estimated values for missing data fields using a strict priority ladder.\n\n"
        "PRIORITY LADDER (apply in order — use the first method that applies):\n"
        "Method A (national_regional_average): City has the quantity but missing unit cost.\n"
        "  Use national/regional benchmarks from the 'National/regional benchmarks' section\n"
        "  when available. These are retrieved from real web searches, not training data.\n"
        "  Confidence: HIGH, range: +/-15-20%.\n"
        "Method B (peer_city_proxy): Peer cities have resolved values for the same field.\n"
        "  Ratio-scale from peers using city population, fleet size, or municipal employees.\n"
        "  PREFER this over Method C when the \"Peer reference data\" section contains\n"
        "  2+ resolved peer values for the target field.\n"
        "  Example: If Munich (pop 1.5M) has 280 municipal vehicles and target city\n"
        "  has pop 250K, estimate ≈ 280 × (250K/1.5M) = ~47, adjusted for local context.\n"
        "  Confidence: MEDIUM, range: +/-25-35%.\n"
        "Method C (expert_heuristic_scaling): Use comparable same-country/region peer cities.\n"
        "  Use cross-country comparative data from the 'Cross-country comparative data' section\n"
        "  when available. These are retrieved from real web searches.\n"
        "  Minimum 3 same-country peers; if fewer, widen to same-GDP-tier (+/-30%).\n"
        "  Confidence: LOW, range: +/-40-50%.\n"
        "If none apply → mark as non-estimable (skip, do not estimate).\n\n"
        "RULES:\n"
        "- When Peer reference data is provided, you MUST attempt Method B before falling\n"
        "  back to Method C. Cite specific peer city values and scaling ratios used.\n"
        "- Never average across different asset types (separate reference pools).\n"
        "- Zero-data cities get a single summary record tagged VERY_LOW.\n"
        "- Every estimate MUST be a range (low/mid/high), never a point estimate.\n"
        "- is_replaceable is always true.\n"
        "- Include reference_data (what peer data was used), rationale, and basis.\n\n"
    )
    if pass_name == "critique":
        base += (
            "CRITIQUE PASS:\n"
            "Review the prior estimates provided. For each:\n"
            "1. Check the method is correctly applied per the priority ladder.\n"
            "2. Verify the range is reasonable (not too narrow or wide).\n"
            "3. Ensure reference_data is specific (city names, values) not generic.\n"
            "4. If an estimate should be revised, include the revised version.\n"
            "5. If an estimate is correct, include it unchanged.\n"
            "Return the complete revised list.\n\n"
        )

    base += (
        "OUTPUT JSON SCHEMA:\n"
        "```json\n"
        "{\n"
        '  "assumptions": [\n'
        "    {\n"
        '      "city": "...",\n'
        '      "field_name": "...",\n'
        '      "gap_description": "...",\n'
        '      "method_used": "national_regional_average|peer_city_proxy|expert_heuristic_scaling",\n'
        '      "estimate": {"low": ..., "mid": ..., "high": ...},\n'
        '      "confidence": "HIGH|MEDIUM|LOW|VERY_LOW",\n'
        '      "reference_data": "...",\n'
        '      "rationale": "...",\n'
        '      "basis": "...",\n'
        '      "is_replaceable": true\n'
        "    }\n"
        "  ],\n"
        '  "non_estimable": [\n'
        "    {\n"
        '      "city": "...",\n'
        '      "field_name": "...",\n'
        '      "gap_description": "...",\n'
        '      "status": "NON_ESTIMABLE",\n'
        '      "explanation": "...",\n'
        '      "recommendation": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
    )
    return base


def _build_context_summary(context_bundle: dict[str, Any]) -> dict[str, Any]:
    """Extract a lightweight data summary from the context bundle for estimation.

    The full context bundle can be 100K+ tokens (all markdown excerpts, SQL
    results, retrieval metadata).  The estimator only needs a small fraction
    of that — mainly data values and city metadata.
    """
    summary: dict[str, Any] = {}

    for key in ("research_question", "original_question", "analysis_mode", "query_mode"):
        if key in context_bundle:
            summary[key] = context_bundle[key]

    # Markdown: city list + excerpt count — never the full excerpt text
    markdown = context_bundle.get("markdown")
    if isinstance(markdown, dict):
        summary["cities_inspected"] = markdown.get("inspected_city_names", [])
        summary["cities_selected"] = markdown.get("selected_city_names", [])
        summary["excerpt_count"] = markdown.get("excerpt_count", 0)

    return summary


def _format_national_benchmarks(
    national_benchmarks: list[WebFinding] | None,
) -> str:
    """Format national/regional benchmark findings into a prompt section.

    Returns an empty string if no benchmarks are available.
    """
    if not national_benchmarks:
        return ""

    lines = ["National/regional benchmarks (retrieved from web search):"]
    for finding in national_benchmarks:
        val = f"{finding.value} {finding.unit}" if finding.unit else str(finding.value)
        date_part = f", {finding.source_date}" if finding.source_date else ""
        conf = finding.extraction_confidence
        lines.append(
            f"- {finding.field}: {finding.city} avg = {val} "
            f"({finding.source_type}{date_part}) [conf: {conf}]"
        )

    lines.append("")
    lines.append(
        "USE for Method A when:\n"
        "- The city reports a quantity (e.g. fleet size) but is missing the unit cost\n"
        "- Example: Dresden has 120 electric buses (CCC) but no procurement cost\n"
        "  → Estimate: 120 × benchmark unit cost = total (±15-20%, HIGH confidence)"
    )
    lines.append("")
    lines.append(
        "DO NOT use when:\n"
        "- The field is qualitative or city-specific (operator names, policy terms)\n"
        "- The city has unusual local conditions that make national averages misleading\n"
        "  (e.g. island logistics, extreme climate adaptation costs)\n"
        "- Better peer city data exists in the 'Peer reference data' section\n"
        "  (prefer Method B with exact peer values over Method A with averages)"
    )

    return "\n".join(lines) + "\n"


def _format_comparative_data(
    comparative_data: list[WebFinding] | None,
) -> str:
    """Format cross-country comparative findings into a prompt section.

    Returns an empty string if no comparative data is available.
    """
    if not comparative_data:
        return ""

    lines = ["Cross-country comparative data (retrieved from web search):"]
    for finding in comparative_data:
        val = f"{finding.value} {finding.unit}" if finding.unit else str(finding.value)
        date_part = f", {finding.source_date}" if finding.source_date else ""
        conf = finding.extraction_confidence
        lines.append(
            f"- {finding.field}: {finding.city} = {val} "
            f"({finding.source_type}{date_part}) [conf: {conf}]"
        )

    lines.append("")
    lines.append(
        "USE for Method C when:\n"
        "- Fewer than 3 same-country peers available, need cross-border comparison\n"
        "- Scaling by per-capita or per-fleet-size ratios from international benchmarks\n"
        "- Cite the specific report and normalized figure used"
    )
    lines.append("")
    lines.append(
        "DO NOT use when:\n"
        "- Same-country peer data is available (prefer Method B)\n"
        "- National benchmarks apply (prefer Method A)\n"
        "- The field is not meaningfully comparable across countries\n"
        "  (e.g. regulatory-specific costs, local subsidy structures)"
    )

    return "\n".join(lines) + "\n"


def _build_user_prompt(
    question: str,
    context_bundle: dict[str, Any],
    gap_manifest: GapManifest,
    estimable_fields: list[EnrichedField],
    pass_name: str,
    prior_estimates: list[AssumptionRecord] | None = None,
    peer_reference: dict[str, list[dict[str, object]]] | None = None,
    national_benchmarks: list[WebFinding] | None = None,
    comparative_data: list[WebFinding] | None = None,
) -> str:
    summary = _build_context_summary(context_bundle)
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2, default=str)
    gap_json = json.dumps(gap_manifest.model_dump(mode="json"), indent=2, default=str)
    fields_json = json.dumps(
        [f.model_dump(mode="json") for f in estimable_fields],
        indent=2,
        default=str,
    )

    parts = [
        f"User question:\n{question.strip()}\n",
        f"Gap manifest:\n```json\n{gap_json}\n```\n",
        f"Fields needing estimation ({len(estimable_fields)}):\n```json\n{fields_json}\n```\n",
    ]

    # Insert peer reference data prominently before the context summary
    if peer_reference:
        lines = [
            "Peer reference data (resolved values from other cities for each target field):"
        ]
        for field_name, peers in peer_reference.items():
            peer_strs = [
                f"  {p['city']}: {p['value']} ({p['source']})" for p in peers
            ]
            lines.append(f"- {field_name}:")
            lines.extend(peer_strs)
        lines.append("")
        lines.append(
            "Use these as peer benchmarks for Method B (peer_city_proxy). "
            "Ratio-scale by city population or municipal fleet size rather than "
            "using flat benchmarks."
        )
        parts.append("\n".join(lines) + "\n")

    # Insert national/regional benchmarks (between peer reference and data summary)
    benchmarks_section = _format_national_benchmarks(national_benchmarks)
    if benchmarks_section:
        parts.append(benchmarks_section)

    # Insert cross-country comparative data (between national benchmarks and data summary)
    comparative_section = _format_comparative_data(comparative_data)
    if comparative_section:
        parts.append(comparative_section)

    parts.append(f"Data summary:\n```json\n{summary_json}\n```\n")

    if pass_name == "critique" and prior_estimates:
        prior_json = json.dumps(
            [e.model_dump(mode="json") for e in prior_estimates],
            indent=2,
            default=str,
        )
        parts.append(
            f"Prior estimates to critique and revise:\n```json\n{prior_json}\n```\n"
        )
        parts.append(
            "Review each estimate, correct any issues, and return the complete revised list.\n"
        )
    else:
        parts.append(
            "Generate estimates for each field using the priority ladder. "
            "Return only the JSON object.\n"
        )

    return "\n".join(parts)


def _check_anchor_sufficiency(
    peer_reference: dict[str, list[dict[str, object]]],
    estimable_fields: list[EnrichedField],
) -> tuple[list[EnrichedField], list[NonEstimableRecord]]:
    """Split fields into estimable vs insufficient-anchor.

    A field passes if it has ``_MIN_PEER_ANCHORS_METHOD_B`` (2) peers,
    or at least one authoritative CCC-sourced peer with confidence
    >= ``_MIN_AUTHORITATIVE_CONFIDENCE``.  Fields below that threshold
    are routed to non-estimable with a recommendation to find additional
    anchor data.
    """
    sufficient: list[EnrichedField] = []
    insufficient_records: list[NonEstimableRecord] = []

    for field in estimable_fields:
        peers = peer_reference.get(field.field, [])
        if len(peers) >= _MIN_PEER_ANCHORS_METHOD_B:
            sufficient.append(field)
            continue

        # Below Method B threshold — allow through only if at least one
        # authoritative CCC-sourced peer has high confidence.
        authoritative_peers = [
            p for p in peers
            if str(p.get("source", "")).lower() == "ccc"
            and float(p.get("confidence", 0)) >= _MIN_AUTHORITATIVE_CONFIDENCE
        ]
        if authoritative_peers:
            sufficient.append(field)
            continue

        insufficient_records.append(
            NonEstimableRecord(
                city=field.city,
                field_name=field.field,
                gap_description=f"Insufficient peer anchors for {field.field}",
                explanation=(
                    f"Only {len(peers)} peer value(s) available; "
                    f"minimum {_MIN_PEER_ANCHORS_METHOD_B} needed for Method B "
                    f"(peer proxy), {_MIN_PEER_ANCHORS_METHOD_C} for Method C "
                    "(heuristic scaling), or at least 1 authoritative "
                    f"CCC-sourced anchor with confidence >= "
                    f"{_MIN_AUTHORITATIVE_CONFIDENCE}."
                ),
                recommendation=(
                    "Search for operator fleet registries or municipal "
                    "procurement records to establish additional anchors."
                ),
            )
        )

    if insufficient_records:
        logger.info(
            "Anchor sufficiency check: %d sufficient, %d insufficient.",
            len(sufficient),
            len(insufficient_records),
        )

    return sufficient, insufficient_records


def _build_peer_reference_table(
    all_enriched: list[EnrichedField],
    estimable_fields: list[EnrichedField],
) -> dict[str, list[dict[str, object]]]:
    """Extract resolved peer values for each field needing estimation.

    Only includes peers whose extraction confidence meets
    ``_MIN_PEER_CONFIDENCE``.  CCC-sourced data defaults to 1.0
    confidence; web-sourced data defaults to 0.5 unless provenance
    carries an explicit ``extraction_confidence``.
    """
    fields_needing_estimation = {f.field for f in estimable_fields}

    table: dict[str, list[dict[str, object]]] = {}
    for ef in all_enriched:
        if (
            ef.field in fields_needing_estimation
            and ef.status == "resolved"
            and ef.value is not None
        ):
            default_conf = 1.0 if ef.source == "ccc" else 0.5
            confidence = float(ef.provenance.get("extraction_confidence", default_conf))
            if confidence < _MIN_PEER_CONFIDENCE:
                continue
            table.setdefault(ef.field, []).append({
                "city": ef.city,
                "value": ef.value,
                "source": ef.source,
                "confidence": confidence,
            })
    return table


def _check_saturation(assumptions: list[AssumptionRecord]) -> str | None:
    """Return a warning if >60% of estimates use Method C (expert heuristic)."""
    if not assumptions:
        return None
    method_c_count = sum(
        1 for a in assumptions if a.method_used == "expert_heuristic_scaling"
    )
    ratio = method_c_count / len(assumptions)
    if ratio > _METHOD_C_SATURATION_THRESHOLD:
        return (
            f"Methodological caveat: {method_c_count}/{len(assumptions)} "
            f"({ratio:.0%}) of estimates rely on expert heuristic scaling "
            "(Method C), which carries wider uncertainty ranges (+/-40-50%). "
            "Results should be treated as indicative order-of-magnitude figures."
        )
    return None


__all__ = ["run_assumptions_estimator"]
