"""Schema-aware field extractor: extract specific URBIND fields from scraped content."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from backend.modules.web_researcher.models import WebFinding
from backend.modules.web_researcher.utils.json_helpers import (
    extract_json_candidate,
    extract_message_text,
)
from backend.services.llm_observability import (
    LlmCallContext,
    LlmCallRecorder,
    record_openai_chat_completion,
)
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)


def extract_fields_from_content(
    content: str,
    source_url: str,
    target_fields: list[str],
    cities: list[str],
    config: AppConfig,
    api_key: str,
    llm_recorder: LlmCallRecorder | None = None,
) -> list[WebFinding]:
    """Extract specific URBIND fields from scraped page content.

    Uses schema-first extraction: the LLM is told exactly which fields to
    look for, not asked generically "what data is on this page?".

    Returns a list of ``WebFinding`` objects. On failure returns empty list.
    """
    if not content.strip():
        return []

    client = OpenAI(api_key=api_key, base_url=config.openrouter_base_url)

    system_prompt = (
        "You are a structured data extractor for urban climate action plans.\n"
        "Extract ONLY the specified fields from the provided page content.\n\n"
        "RULES:\n"
        "- Extract only concrete, quantitative values (numbers with units).\n"
        "- Do not extract aspirational language ('plans to', 'targets').\n"
        "- Each finding must include the city, field, value, and unit.\n"
        "- source_type must be one of: government_report, operator_report,\n"
        "  press_release, news_article, academic_paper, industry_report, other.\n"
        "- extraction_confidence: 0.0-1.0 based on how clearly the value is stated.\n"
        "- source_date: extract the date of the data if mentioned (YYYY or YYYY-MM format).\n"
        "- If a field is not found in the content, do NOT fabricate a value.\n"
        "- SEMANTIC ALIGNMENT: only extract values that genuinely describe the requested field.\n"
        "  Do NOT extract:\n"
        "  - Private vehicle registration stats as fleet vehicle counts\n"
        "  - Policy targets or aspirations as current values\n"
        "  - Regional/national figures as city-specific values (unless the field is national-level)\n"
        "  - Passenger counts as vehicle counts\n"
        '  Example: "100,000 registered EVs in Baden-Württemberg" is NOT a valid value for "ev_bus_count" for Mannheim.\n'
        "- CANCELLATION SIGNALS: If a page states that a programme, project, or initiative\n"
        "  was cancelled, discontinued, halted, or reversed, extract it as:\n"
        '  value: 0, unit: "cancelled_programme", source_type: "cancellation_notice".\n'
        "  This captures the signal that a previously reported value is no longer valid.\n\n"
        "OUTPUT JSON SCHEMA:\n"
        "```json\n"
        '{"findings": [\n'
        '  {"city": "...", "field": "...", "value": ..., "unit": "...",\n'
        '   "source_type": "...", "source_date": "...",\n'
        '   "extraction_confidence": 0.85}\n'
        "]}\n"
        "```\n"
    )

    user_prompt = (
        f"Cities to look for: {', '.join(cities)}\n"
        f"Fields to extract: {', '.join(target_fields)}\n"
        f"Source URL: {source_url}\n\n"
        "Page content:\n"
        "```\n"
        f"{content}\n"
        "```\n\n"
        "Extract the specified fields if they appear in the content.\n"
    )

    try:
        request_kwargs: dict[str, object] = {
            "model": config.enrichment.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }

        response = record_openai_chat_completion(
            client,
            request_kwargs,
            context=LlmCallContext(
                stage_name="enrichment",
                stage_family="enrichment",
                agent="web_extractor",
                call_kind="field_extraction",
                model=config.enrichment.model,
                metadata={
                    "source_url": source_url,
                    "city_count": len(cities),
                    "target_field_count": len(target_fields),
                    "content_chars": len(content),
                },
            ),
            recorder=llm_recorder,
        )
        if not response.choices:
            return []

        text = extract_message_text(response.choices[0].message.content)
        candidate = extract_json_candidate(text)
        parsed = json.loads(candidate)

        findings: list[WebFinding] = []
        raw_findings: list[dict[str, Any]] = []

        if isinstance(parsed, dict):
            raw_findings = parsed.get("findings", [])
        elif isinstance(parsed, list):
            raw_findings = parsed

        for item in raw_findings:
            if not isinstance(item, dict):
                continue
            try:
                finding = WebFinding(
                    city=str(item.get("city", "")),
                    field=str(item.get("field", "")),
                    value=item.get("value"),
                    unit=item.get("unit"),
                    source_url=source_url,
                    source_type=str(item.get("source_type", "other")),
                    source_date=item.get("source_date"),
                    extraction_confidence=float(item.get("extraction_confidence", 0.5)),
                )
                # Validate city is one of the targets
                if finding.city.lower() in {c.lower() for c in cities}:
                    findings.append(finding)
            except (ValueError, TypeError):
                continue

        logger.info(
            "Extractor: %d findings from url=%r for %d cities.",
            len(findings),
            source_url,
            len(cities),
        )
        return findings

    except Exception:
        logger.warning("Field extraction failed for url=%r", source_url, exc_info=True)
        return []


__all__ = ["extract_fields_from_content"]
