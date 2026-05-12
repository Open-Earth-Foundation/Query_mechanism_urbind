"""Assumptions Estimator (Agent 6): produce model-estimated values for data gaps."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from openai import OpenAI

from backend.modules.web_researcher.models import (
    AssumptionRecord,
    EnrichedField,
    GapManifest,
    NonEstimableRecord,
    WebFinding,
    _AssumptionsEnvelope,
)
from backend.modules.web_researcher.utils.json_helpers import (
    extract_json_candidate,
    extract_message_text,
)
from backend.services.progress_tracker import ProgressTracker
from backend.utils.config import AppConfig
from backend.utils.llm_serialization import render_toon_block

logger = logging.getLogger(__name__)

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
    """Estimate remaining gaps with the LLM priority ladder."""
    still_missing = [f for f in enriched_fields if f.status == "still_missing"]
    partially_resolved = [f for f in enriched_fields if f.status == "partially_resolved"]
    bundled_only = [f for f in enriched_fields if f.status == "bundled_only"]
    fields_to_estimate = still_missing + partially_resolved + bundled_only

    if not fields_to_estimate:
        logger.info("Assumptions estimator: no gaps to estimate.")
        return [], [], None

    if progress:
        for field in fields_to_estimate:
            progress.add_item(
                "assumptions",
                f"Gap: {field.city} / {field.field} ({field.status})",
                item_type="gap",
                title=f"{field.city} / {field.field}",
                metadata={
                    "city": field.city,
                    "field": field.field,
                    "status": field.status,
                },
            )
        progress.add_item(
            "assumptions",
            f"{len(fields_to_estimate)} fields to estimate",
        )

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
        for record in non_estimable_records:
            progress.add_item(
                "assumptions",
                f"{record.city}: {record.field_name} - non-estimable",
                item_type="field",
                title=f"{record.city}: {record.field_name}",
                metadata={"status": "non_estimable"},
            )

    if not estimable_fields:
        logger.info(
            "Assumptions estimator: all %d gaps are non-estimable.",
            len(non_estimable_records),
        )
        return [], non_estimable_records, None

    peer_reference = _build_peer_reference_table(enriched_fields, estimable_fields)
    estimable_fields, insufficient_records = _check_anchor_sufficiency(
        peer_reference, estimable_fields
    )
    non_estimable_records.extend(insufficient_records)

    if progress and insufficient_records:
        records_by_field: dict[str, list[str]] = defaultdict(list)
        for record in insufficient_records:
            records_by_field[record.field_name].append(record.city)
        for field_name, cities in records_by_field.items():
            city_list = ", ".join(cities)
            progress.add_item(
                "assumptions",
                f"{field_name} - insufficient anchors ({len(cities)} cities: {city_list})",
                item_type="field",
                title=field_name,
                metadata={"status": "insufficient_anchors", "cities": cities},
            )

    if not estimable_fields:
        logger.info("Assumptions estimator: all remaining gaps lack anchors.")
        return [], non_estimable_records, None

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
        for record in assumptions:
            mid = _estimate_value(record, "mid")
            low = _estimate_value(record, "low")
            high = _estimate_value(record, "high")
            progress.add_item(
                "assumptions",
                f"{record.city} / {record.field_name}: {mid} ({low}-{high}) "
                f"[{record.method_used}, {record.confidence}]",
                item_type="estimate",
                title=f"{record.city} / {record.field_name}",
                metadata={
                    "method": record.method_used,
                    "confidence": record.confidence,
                    "mid": str(mid),
                    "low": str(low),
                    "high": str(high),
                },
            )
        progress.add_item("assumptions", f"Pass 1 done: {len(assumptions)} estimates")

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
                for record in assumptions:
                    mid = _estimate_value(record, "mid")
                    progress.add_item(
                        "assumptions",
                        f"Revised: {record.city} / {record.field_name}: {mid} "
                        f"[{record.method_used}, {record.confidence}]",
                        item_type="estimate",
                        title=f"{record.city} / {record.field_name}",
                        metadata={
                            "method": record.method_used,
                            "confidence": record.confidence,
                            "mid": str(mid),
                            "revised": True,
                        },
                    )
                progress.add_item(
                    "assumptions",
                    f"Pass 2 done: {len(assumptions)} revised estimates",
                )

    saturation_warning = _check_saturation(assumptions)
    logger.info(
        "Assumptions estimator completed: assumptions=%d non_estimable=%d saturation=%s",
        len(assumptions),
        len(non_estimable_records),
        saturation_warning is not None,
    )
    return assumptions, non_estimable_records, saturation_warning


def _estimate_value(record: AssumptionRecord, key: str) -> object:
    """Return an estimate value from either a model or dict-shaped estimate."""
    if hasattr(record.estimate, key):
        return getattr(record.estimate, key)
    if isinstance(record.estimate, dict):
        return record.estimate.get(key, "?")
    return "?"


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
    """Build the estimator system prompt for generation or critique."""
    base = (
        "You are an assumptions estimator for urban climate action data.\n"
        "You produce model-estimated values for missing data fields using a strict priority ladder.\n\n"
        "PRIORITY LADDER (apply in order; use the first method that applies):\n"
        "Method A (national_regional_average): City has the quantity but missing unit cost.\n"
        "  Use national/regional benchmarks from the 'National/regional benchmarks' section\n"
        "  when available. These are retrieved from real web searches, not training data.\n"
        "  Confidence: HIGH, range: +/-15-20%.\n"
        "Method B (peer_city_proxy): Peer cities have resolved values for the same field.\n"
        "  Ratio-scale from peers using city population, fleet size, or municipal employees.\n"
        "  PREFER this over Method C when the \"Peer reference data\" section contains\n"
        "  2+ resolved peer values for the target field.\n"
        "  Example: If Munich (pop 1.5M) has 280 municipal vehicles and target city\n"
        "  has pop 250K, estimate about 280 * (250K/1.5M) = 47, adjusted for local context.\n"
        "  Confidence: MEDIUM, range: +/-25-35%.\n"
        "Method C (expert_heuristic_scaling): Use comparable same-country/region peer cities.\n"
        "  Use cross-country comparative data from the 'Cross-country comparative data' section\n"
        "  when available. These are retrieved from real web searches.\n"
        "  Minimum 3 same-country peers; if fewer, widen to same-GDP-tier (+/-30%).\n"
        "  Confidence: LOW, range: +/-40-50%.\n"
        "If none apply, mark as non-estimable (skip, do not estimate).\n\n"
        "RULES:\n"
        "- When Peer reference data is provided, you MUST attempt Method B before falling\n"
        "  back to Method C. Cite specific peer city values and scaling ratios used.\n"
        "- Never average across different asset types or scopes.\n"
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
    """Extract a lightweight data summary from the context bundle for estimation."""
    summary: dict[str, Any] = {}

    for key in ("research_question", "original_question", "analysis_mode", "query_mode"):
        if key in context_bundle:
            summary[key] = context_bundle[key]

    markdown = context_bundle.get("markdown")
    if isinstance(markdown, dict):
        summary["cities_inspected"] = markdown.get("inspected_city_names", [])
        summary["cities_selected"] = markdown.get("selected_city_names", [])
        summary["excerpt_count"] = markdown.get("excerpt_count", 0)

    return summary


def _format_national_benchmarks(
    national_benchmarks: list[WebFinding] | None,
) -> str:
    """Format national/regional benchmark findings into a prompt section."""
    if not national_benchmarks:
        return ""

    lines = ["National/regional benchmarks (retrieved from web search):"]
    for finding in national_benchmarks:
        value = f"{finding.value} {finding.unit}" if finding.unit else str(finding.value)
        date_part = f", {finding.source_date}" if finding.source_date else ""
        lines.append(
            f"- {finding.field}: {finding.city} avg = {value} "
            f"({finding.source_type}{date_part}) [conf: {finding.extraction_confidence}]"
        )

    lines.extend(
        [
            "",
            "USE for Method A when:",
            "- The city reports a quantity but is missing the unit cost.",
            "- Prefer exact peer-city values over national averages when available.",
            "DO NOT use when:",
            "- The field is qualitative or city-specific.",
            "- Peer reference data has exact same-field values; use Method B first.",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_comparative_data(
    comparative_data: list[WebFinding] | None,
) -> str:
    """Format cross-country comparative findings into a prompt section."""
    if not comparative_data:
        return ""

    lines = ["Cross-country comparative data (retrieved from web search):"]
    for finding in comparative_data:
        value = f"{finding.value} {finding.unit}" if finding.unit else str(finding.value)
        date_part = f", {finding.source_date}" if finding.source_date else ""
        lines.append(
            f"- {finding.field}: {finding.city} = {value} "
            f"({finding.source_type}{date_part}) [conf: {finding.extraction_confidence}]"
        )

    lines.extend(
        [
            "",
            "USE for Method C when:",
            "- Fewer than 3 same-country peers are available.",
            "- Scaling by per-capita or per-fleet-size ratios is meaningful.",
            "DO NOT use when:",
            "- The field requires cross-border comparison without comparable context.",
            "- You can prefer Method B with direct peer city values.",
            "- You can prefer Method A with a national or regional benchmark.",
        ]
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
    """Build the estimator user prompt from the current gap state."""
    summary = _build_context_summary(context_bundle)

    parts = [
        f"User question:\n{question.strip()}\n",
        f"{render_toon_block('Gap manifest TOON', gap_manifest.model_dump(mode='json'))}\n",
        (
            f"{render_toon_block(
                f'Fields needing estimation TOON ({len(estimable_fields)})',
                [field.model_dump(mode='json') for field in estimable_fields],
            )}\n"
        ),
    ]

    if peer_reference:
        lines = [
            "Peer reference data (resolved values from other cities for each target field):"
        ]
        for field_name, peers in peer_reference.items():
            lines.append(f"- {field_name}:")
            lines.extend(
                f"  {peer['city']}: {peer['value']} ({peer['source']})"
                for peer in peers
            )
        lines.extend(
            [
                "",
                "Use these as peer benchmarks for Method B (peer_city_proxy).",
                "Ratio-scale by city population or municipal fleet size rather than using flat benchmarks.",
            ]
        )
        parts.append("\n".join(lines) + "\n")

    benchmarks_section = _format_national_benchmarks(national_benchmarks)
    if benchmarks_section:
        parts.append(benchmarks_section)

    comparative_section = _format_comparative_data(comparative_data)
    if comparative_section:
        parts.append(comparative_section)

    parts.append(f"{render_toon_block('Data summary TOON', summary)}\n")

    if pass_name == "critique" and prior_estimates:
        parts.append(
            f"{render_toon_block('Prior estimates to critique and revise TOON', [estimate.model_dump(mode='json') for estimate in prior_estimates])}\n"
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
    """Split fields into estimable and insufficient-anchor groups."""
    sufficient: list[EnrichedField] = []
    insufficient_records: list[NonEstimableRecord] = []

    for field in estimable_fields:
        peers = peer_reference.get(field.field, [])
        if len(peers) >= _MIN_PEER_ANCHORS_METHOD_B:
            sufficient.append(field)
            continue

        authoritative_peers = [
            peer
            for peer in peers
            if str(peer.get("source", "")).lower() == "ccc"
            and float(peer.get("confidence", 0)) >= _MIN_AUTHORITATIVE_CONFIDENCE
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
                    f"minimum {_MIN_PEER_ANCHORS_METHOD_B} needed for Method B, "
                    f"{_MIN_PEER_ANCHORS_METHOD_C} for Method C, or at least "
                    f"1 authoritative CCC-sourced anchor with confidence >= "
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
    """Extract resolved peer values for each field needing estimation."""
    fields_needing_estimation = {field.field for field in estimable_fields}

    table: dict[str, list[dict[str, object]]] = {}
    for enriched in all_enriched:
        if (
            enriched.field in fields_needing_estimation
            and enriched.status == "resolved"
            and enriched.value is not None
        ):
            default_confidence = 1.0 if enriched.source == "ccc" else 0.5
            confidence = float(
                enriched.provenance.get("extraction_confidence", default_confidence)
            )
            if confidence < _MIN_PEER_CONFIDENCE:
                continue
            table.setdefault(enriched.field, []).append(
                {
                    "city": enriched.city,
                    "value": enriched.value,
                    "source": enriched.source,
                    "confidence": confidence,
                }
            )
    return table


def _check_saturation(assumptions: list[AssumptionRecord]) -> str | None:
    """Return a warning if more than 60% of estimates use Method C."""
    if not assumptions:
        return None
    method_c_count = sum(
        1 for record in assumptions if record.method_used == "expert_heuristic_scaling"
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
