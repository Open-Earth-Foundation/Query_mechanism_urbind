"""Section-first aggregate writer helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from backend.modules.orchestrator.utils.references import is_valid_ref_id
from backend.modules.writer.models import WriterSectionPlan, WriterSectionSpec
from backend.modules.writer.utils.markdown_helpers import (
    city_display_name,
    city_key,
    dedupe_city_names,
    extract_markdown_bundle,
    extract_markdown_excerpts,
)
from backend.modules.writer.utils.multi_pass import build_writer_context_bundle
from backend.utils.tokenization import count_tokens

_GENERIC_SECTION_TITLES = {
    "analysis",
    "cross-city synthesis",
    "data gaps",
    "executive summary",
    "findings",
    "key findings",
    "methodology",
    "overview",
    "source registry",
    "summary",
    "synthesis",
}
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SECTION_ID_PATTERN = re.compile(r"[^a-z0-9_]+")
_NUMERIC_DATE_PATTERN = re.compile(
    r"(?:[€$£]\s*)?\d[\d,.\s]*(?:%|tCO2e|CO2e|MW|MWh|GWh|km|m|cars|buses|chargers)?"
    r"|(?:19|20)\d{2}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WriterSectionPlannerPayload:
    """Compact section-planner payload plus token diagnostics."""

    payload: dict[str, object]
    input_tokens: int
    catalog_truncated: bool


def build_section_planner_payload(
    *,
    question: str,
    context_bundle: dict[str, object],
    analysis_mode: str,
    selected_city_names: list[str],
    max_input_tokens: int,
) -> WriterSectionPlannerPayload:
    """Build the compact planner payload without sending the full context bundle."""
    preview_limits = [(360, 240), (180, 120), (90, 60), (0, 0)]
    for index, (partial_limit, quote_limit) in enumerate(preview_limits):
        payload = _build_planner_payload(
            question=question,
            context_bundle=context_bundle,
            analysis_mode=analysis_mode,
            selected_city_names=selected_city_names,
            partial_limit=partial_limit,
            quote_limit=quote_limit,
        )
        input_tokens = _count_json_tokens(payload)
        if input_tokens <= max_input_tokens:
            return WriterSectionPlannerPayload(
                payload=payload,
                input_tokens=input_tokens,
                catalog_truncated=index > 0,
            )

    fitted_payload = _fit_catalog_to_token_limit(payload, max_input_tokens)
    return WriterSectionPlannerPayload(
        payload=fitted_payload,
        input_tokens=_count_json_tokens(fitted_payload),
        catalog_truncated=True,
    )


def sanitize_writer_section_plan(
    *,
    plan: WriterSectionPlan,
    question: str,
    context_bundle: dict[str, object],
    selected_city_names: list[str],
) -> WriterSectionPlan:
    """Normalize an LLM-generated section plan against known refs and cities."""
    markdown_bundle = extract_markdown_bundle(context_bundle)
    excerpts = extract_markdown_excerpts(markdown_bundle)
    expected_refs = {
        str(excerpt.get("ref_id", "")).strip()
        for excerpt in excerpts
        if is_valid_ref_id(str(excerpt.get("ref_id", "")).strip())
    }
    ref_city_names = {
        str(excerpt.get("ref_id", "")).strip(): city_display_name(
            str(excerpt.get("city_name", ""))
        )
        for excerpt in excerpts
    }

    sections: list[WriterSectionSpec] = []
    used_ids: set[str] = set()
    for index, section in enumerate(plan.sections, start=1):
        ref_ids = _dedupe_valid_refs(section.required_ref_ids, expected_refs)
        city_names = dedupe_city_names(
            [
                *section.city_names,
                *[ref_city_names.get(ref_id, "") for ref_id in ref_ids],
            ]
        )
        if not ref_ids and not city_names:
            continue
        section_id = _normalize_section_id(section.section_id, index, used_ids)
        used_ids.add(section_id)
        sections.append(
            WriterSectionSpec(
                section_id=section_id,
                title=_normalize_section_title(section.title, question, section.section_type),
                section_type=section.section_type.strip() or "answer_section",
                purpose=section.purpose.strip() or f"Answer the question section for {question}",
                required_ref_ids=ref_ids,
                city_names=city_names,
                writing_instructions=section.writing_instructions.strip()
                or "Use the assigned evidence to answer this section of the question.",
            )
        )

    if not sections:
        sections = [_build_fallback_section(question, excerpts, selected_city_names)]

    sections = _backfill_unassigned_refs(
        sections=sections,
        question=question,
        excerpts=excerpts,
        selected_city_names=selected_city_names,
    )
    return WriterSectionPlan(sections=sections)


def build_section_context_bundle(
    *,
    context_bundle: dict[str, object],
    section: WriterSectionSpec,
) -> dict[str, object]:
    """Build a writer context bundle containing only one section's evidence."""
    markdown_bundle = extract_markdown_bundle(context_bundle)
    selected_refs = set(section.required_ref_ids)
    selected_excerpts = [
        excerpt
        for excerpt in extract_markdown_excerpts(markdown_bundle)
        if str(excerpt.get("ref_id", "")).strip() in selected_refs
    ]
    city_names = section.city_names or [
        str(excerpt.get("city_name", "")) for excerpt in selected_excerpts
    ]
    return build_writer_context_bundle(
        context_bundle=context_bundle,
        excerpts=selected_excerpts,
        city_names=city_names,
    )


def build_section_writer_payload(
    *,
    question: str,
    analysis_mode: str,
    selected_city_names: list[str],
    section: WriterSectionSpec,
    context_bundle: dict[str, object],
) -> dict[str, object]:
    """Build the payload sent to a single section writer."""
    return {
        "question": question,
        "analysis_mode": analysis_mode,
        "selected_cities": selected_city_names,
        "section": section.model_dump(),
        "context_bundle": context_bundle,
    }


def build_section_composer_payload(
    *,
    question: str,
    analysis_mode: str,
    selected_city_names: list[str],
    plan: WriterSectionPlan,
    section_drafts: list[dict[str, object]],
) -> dict[str, object]:
    """Build the final composer payload from section drafts."""
    return {
        "question": question,
        "analysis_mode": analysis_mode,
        "selected_cities": selected_city_names,
        "section_plan": plan.model_dump(),
        "section_drafts": section_drafts,
    }


def count_section_payload_tokens(payload: dict[str, object]) -> int:
    """Count tokens for a section-first runtime payload."""
    return _count_json_tokens(payload)


def _build_planner_payload(
    *,
    question: str,
    context_bundle: dict[str, object],
    analysis_mode: str,
    selected_city_names: list[str],
    partial_limit: int,
    quote_limit: int,
) -> dict[str, object]:
    """Build a planner payload for one preview-size setting."""
    markdown_bundle = extract_markdown_bundle(context_bundle)
    excerpts = extract_markdown_excerpts(markdown_bundle)
    evidence_catalog = [
        _build_catalog_entry(excerpt, partial_limit, quote_limit)
        for excerpt in excerpts
    ]
    return {
        "question": question,
        "analysis_mode": analysis_mode,
        "selected_cities": selected_city_names,
        "evidence_catalog": evidence_catalog,
        "saved_evidence_catalog": [
            entry for entry in evidence_catalog if entry.get("writer_saved_id")
        ],
        "fallback_evidence_catalog": [
            entry for entry in evidence_catalog if not entry.get("writer_saved_id")
        ],
        "enrichment_summary": _build_enrichment_summary(context_bundle),
    }


def _build_catalog_entry(
    excerpt: dict[str, object],
    partial_limit: int,
    quote_limit: int,
) -> dict[str, object]:
    """Return one compact evidence-catalog record."""
    partial_answer = str(excerpt.get("partial_answer", ""))
    quote = str(excerpt.get("quote", ""))
    return {
        "ref_id": str(excerpt.get("ref_id", "")).strip(),
        "city_name": city_display_name(str(excerpt.get("city_name", ""))),
        "city_key": city_key(str(excerpt.get("city_key", "") or excerpt.get("city_name", ""))),
        "source_chunk_ids": _coerce_string_list(excerpt.get("source_chunk_ids")),
        "source_kind": str(excerpt.get("source_kind", "")).strip() or "ccc_excerpt",
        "source_id": str(excerpt.get("source_id", "")).strip(),
        "field": str(excerpt.get("field", "")).strip(),
        "writer_saved_id": str(excerpt.get("writer_saved_id", "")).strip(),
        "partial_answer_preview": _preview(partial_answer, partial_limit),
        "quote_preview": _preview(quote, quote_limit),
        "numeric_date_snippets": _extract_numeric_date_snippets(
            f"{partial_answer}\n{quote}"
        ),
    }


def _build_enrichment_summary(context_bundle: dict[str, object]) -> dict[str, object]:
    """Summarize enrichment without sending full enrichment records to the planner."""
    enrichment = context_bundle.get("enrichment")
    if not isinstance(enrichment, dict):
        return {"present": False}

    record_counts = {
        key: len(value)
        for key, value in enrichment.items()
        if isinstance(value, list)
    }
    summary: dict[str, object] = {
        "present": True,
        "record_counts": record_counts,
    }
    meta = enrichment.get("meta")
    if isinstance(meta, dict):
        summary["meta"] = {
            key: meta[key]
            for key in (
                "total_gaps",
                "estimable_count",
                "non_estimable_count",
                "web_findings_count",
                "external_evidence_count",
            )
            if key in meta
        }
    saturation_warning = enrichment.get("saturation_warning")
    if isinstance(saturation_warning, str) and saturation_warning.strip():
        summary["saturation_warning"] = saturation_warning.strip()
    return summary


def _fit_catalog_to_token_limit(
    payload: dict[str, object],
    max_input_tokens: int,
) -> dict[str, object]:
    """Drop trailing catalog entries until the planner payload fits the token cap."""
    catalog = payload.get("evidence_catalog")
    if not isinstance(catalog, list):
        return payload

    fitted_catalog: list[object] = []
    fitted_payload = dict(payload)
    for entry in catalog:
        candidate_catalog = [*fitted_catalog, entry]
        fitted_payload["evidence_catalog"] = candidate_catalog
        if _count_json_tokens(fitted_payload) > max_input_tokens:
            break
        fitted_catalog = candidate_catalog

    fitted_payload["evidence_catalog"] = fitted_catalog
    fitted_payload["catalog_truncation_note"] = (
        "Planner evidence catalog was truncated to fit the configured token limit."
    )
    return fitted_payload


def _preview(value: str, limit: int) -> str:
    """Return a compact single-line preview."""
    if limit <= 0:
        return ""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 1, 0)].rstrip() + "..."


def _extract_numeric_date_snippets(value: str) -> list[str]:
    """Extract compact numeric and date-like hints for planner sectioning."""
    snippets: list[str] = []
    seen: set[str] = set()
    for match in _NUMERIC_DATE_PATTERN.finditer(value):
        snippet = " ".join(match.group(0).split())
        if not snippet or snippet in seen:
            continue
        seen.add(snippet)
        snippets.append(snippet)
        if len(snippets) >= 8:
            break
    return snippets


def _coerce_string_list(value: object) -> list[str]:
    """Normalize a list-like value to strings."""
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _dedupe_valid_refs(ref_ids: list[str], expected_refs: set[str]) -> list[str]:
    """Return valid known refs while preserving planner order."""
    refs: list[str] = []
    seen: set[str] = set()
    for value in ref_ids:
        ref_id = str(value).strip()
        if ref_id not in expected_refs or ref_id in seen:
            continue
        seen.add(ref_id)
        refs.append(ref_id)
    return refs


def _normalize_section_id(raw_section_id: str, index: int, used_ids: set[str]) -> str:
    """Return a stable section id suitable for diagnostics."""
    candidate = _SECTION_ID_PATTERN.sub("_", raw_section_id.strip().lower()).strip("_")
    if not candidate:
        candidate = f"section_{index}"
    while candidate in used_ids:
        candidate = f"{candidate}_{index}"
    return candidate


def _normalize_section_title(title: str, question: str, section_type: str) -> str:
    """Ensure a section title is specific to the user question."""
    normalized_title = " ".join(title.split()).strip()
    anchor = _question_anchor(question)
    if not normalized_title:
        return f"Evidence for {anchor}"

    title_key = normalized_title.lower().strip(":")
    if title_key in _GENERIC_SECTION_TITLES:
        label = section_type.replace("_", " ").strip().title() or "Evidence"
        return f"{label} for {anchor}"

    if not _has_question_overlap(normalized_title, question):
        return f"{normalized_title}: {anchor}"
    return normalized_title


def _has_question_overlap(title: str, question: str) -> bool:
    """Return True when a title shares meaningful tokens with the question."""
    title_tokens = _meaningful_tokens(title)
    question_tokens = _meaningful_tokens(question)
    return bool(title_tokens & question_tokens)


def _meaningful_tokens(value: str) -> set[str]:
    """Tokenize text for lightweight title relevance checks."""
    ignored = {
        "a",
        "about",
        "and",
        "are",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "the",
        "to",
        "what",
        "which",
        "with",
    }
    return {
        token.lower()
        for token in _TOKEN_PATTERN.findall(value)
        if len(token) > 2 and token.lower() not in ignored
    }


def _question_anchor(question: str) -> str:
    """Return a compact question-derived title suffix."""
    normalized = " ".join(question.strip().rstrip("?").split())
    if not normalized:
        return "the User Question"
    if len(normalized) <= 72:
        return normalized
    return normalized[:71].rstrip() + "..."


def _build_fallback_section(
    question: str,
    excerpts: list[dict[str, object]],
    selected_city_names: list[str],
) -> WriterSectionSpec:
    """Build a deterministic section plan when planner output is unusable."""
    ref_ids = [
        str(excerpt.get("ref_id", "")).strip()
        for excerpt in excerpts
        if is_valid_ref_id(str(excerpt.get("ref_id", "")).strip())
    ]
    city_names = selected_city_names or [
        city_display_name(str(excerpt.get("city_name", ""))) for excerpt in excerpts
    ]
    return WriterSectionSpec(
        section_id="question_evidence",
        title=f"Evidence for {_question_anchor(question)}",
        section_type="answer_section",
        purpose="Answer the user question from the available evidence.",
        required_ref_ids=ref_ids,
        city_names=dedupe_city_names(city_names),
        writing_instructions=(
            "Synthesize the assigned evidence into a focused answer section. "
            "Preserve citations on every factual claim."
        ),
    )


def _backfill_unassigned_refs(
    *,
    sections: list[WriterSectionSpec],
    question: str,
    excerpts: list[dict[str, object]],
    selected_city_names: list[str],
) -> list[WriterSectionSpec]:
    """Assign any missing excerpt refs to the closest existing section."""
    assigned_refs = {
        ref_id
        for section in sections
        for ref_id in section.required_ref_ids
    }
    missing_excerpts = [
        excerpt
        for excerpt in excerpts
        if str(excerpt.get("ref_id", "")).strip()
        and str(excerpt.get("ref_id", "")).strip() not in assigned_refs
    ]
    if not missing_excerpts:
        return sections
    if not sections:
        return [_build_fallback_section(question, excerpts, selected_city_names)]

    updated_sections = [section.model_copy(deep=True) for section in sections]
    for excerpt in missing_excerpts:
        target_index = _best_section_index(question, excerpt, updated_sections)
        target = updated_sections[target_index]
        ref_id = str(excerpt.get("ref_id", "")).strip()
        city_name = city_display_name(str(excerpt.get("city_name", "")))
        target.required_ref_ids.append(ref_id)
        target.city_names = dedupe_city_names([*target.city_names, city_name])
    return updated_sections


def _best_section_index(
    question: str,
    excerpt: dict[str, object],
    sections: list[WriterSectionSpec],
) -> int:
    """Choose the section with the highest lightweight token overlap."""
    excerpt_tokens = _meaningful_tokens(
        " ".join(
            [
                question,
                str(excerpt.get("partial_answer", "")),
                str(excerpt.get("quote", "")),
            ]
        )
    )
    best_index = 0
    best_score = -1
    for index, section in enumerate(sections):
        section_tokens = _meaningful_tokens(
            " ".join(
                [
                    section.title,
                    section.section_type,
                    section.purpose,
                    section.writing_instructions,
                ]
            )
        )
        score = len(excerpt_tokens & section_tokens)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _count_json_tokens(payload: dict[str, object]) -> int:
    """Count tokens for a JSON payload."""
    return count_tokens(json.dumps(payload, ensure_ascii=False))


__all__ = [
    "WriterSectionPlannerPayload",
    "build_section_composer_payload",
    "build_section_context_bundle",
    "build_section_planner_payload",
    "build_section_writer_payload",
    "count_section_payload_tokens",
    "sanitize_writer_section_plan",
]
