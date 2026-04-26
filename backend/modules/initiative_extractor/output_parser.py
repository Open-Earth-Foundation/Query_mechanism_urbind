"""Output coercion helpers for initiative extractor tool calls."""

from __future__ import annotations

import json

from backend.modules.initiative_extractor.models import (
    InitiativeExtraction,
    InitiativeSegmentExtraction,
    InitiativeSegmentStop,
    JsonValue,
)

CANDIDATE_METADATA_FIELDS = {
    "document_local_code",
    "source_quote",
    "source_refs",
    "data_quality_flags",
    "number_context",
    "number_deferred",
    "number_uncertain",
    "extraction_notes",
}
CITY_OVERRIDDEN_FLAG = "city_overridden_from_segment"


def _get_field(value: object, key: str) -> object:
    """Read a field from a dict-like or object-like SDK payload."""
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _coerce_segment_output_payload(
    tool_name: str,
    payload: object,
    city_name: str | None = None,
) -> InitiativeSegmentExtraction | InitiativeSegmentStop:
    """Validate tool-call payloads against the matching segment output model."""
    if isinstance(payload, dict) and "result" in payload:
        payload = payload["result"]
    if tool_name == "stop_initiative_extraction":
        return InitiativeSegmentStop.model_validate(payload)
    return InitiativeSegmentExtraction.model_validate(
        _normalize_segment_extraction_payload(payload, city_name)
    )


def _normalize_json_object(value: object) -> dict[str, JsonValue]:
    """Return a dict for loose JSON metadata fields."""
    if isinstance(value, dict):
        return value
    if value in (None, []):
        return {}
    return {"items": value}


def _normalize_segment_extraction_payload(
    payload: object,
    city_name: str | None = None,
) -> object:
    """Repair common recoverable model shape errors before validation."""
    if not isinstance(payload, dict):
        return payload

    initiatives = payload.get("initiatives")
    if not isinstance(initiatives, list):
        return payload

    initiative_fields = set(InitiativeExtraction.model_fields)
    normalized_initiatives: list[object] = []
    for item in initiatives:
        if not isinstance(item, dict):
            normalized_initiatives.append(item)
            continue

        source_quote = _clean_source_quote(item.get("source_quote"))
        initiative = (
            item.get("initiative") if isinstance(item.get("initiative"), dict) else item
        )
        if source_quote is None:
            source_quote = _clean_source_quote(initiative.get("source_quote"))
        raw_flags = item.get("data_quality_flags")
        if not isinstance(raw_flags, list):
            raw_flags = initiative.get("data_quality_flags")
        data_quality_flags = list(raw_flags) if isinstance(raw_flags, list) else []
        original_city = initiative.get("city")

        for field_name in CANDIDATE_METADATA_FIELDS:
            initiative.pop(field_name, None)

        if city_name is not None:
            if (
                isinstance(original_city, str)
                and original_city.strip()
                and original_city.strip() != city_name
            ):
                data_quality_flags.append(CITY_OVERRIDDEN_FLAG)
            initiative["city"] = city_name
        initiative["numbers"] = _normalize_numbers_payload(initiative.get("numbers"))
        for field_name in list(initiative):
            if field_name not in initiative_fields:
                initiative.pop(field_name)
        normalized_item: dict[str, object] = {
            "initiative": initiative,
            "source_quote": source_quote,
        }
        if data_quality_flags:
            normalized_item["data_quality_flags"] = list(
                dict.fromkeys(data_quality_flags)
            )
        normalized_initiatives.append(normalized_item)

    payload["initiatives"] = normalized_initiatives
    return payload


def _clean_source_quote(value: object) -> str | None:
    """Return a trimmed source quote or None for blank/non-string values."""
    if not isinstance(value, str):
        return None
    quote = value.strip()
    return quote or None


def _normalize_numbers_payload(value: object) -> dict[str, dict[str, JsonValue]]:
    """Return canonical current/planned number buckets."""
    if not isinstance(value, dict):
        return {"current": {}, "planned": {}}
    current = value.get("current")
    planned = value.get("planned")
    return {
        "current": (
            current if isinstance(current, dict) else _normalize_json_object(current)
        ),
        "planned": (
            planned if isinstance(planned, dict) else _normalize_json_object(planned)
        ),
    }


def _extract_segment_tool_output(
    result: object,
    city_name: str | None = None,
) -> InitiativeSegmentExtraction | InitiativeSegmentStop | None:
    """Extract structured tool arguments from the Agents SDK raw response."""
    raw_responses = list(getattr(result, "raw_responses", []) or [])
    for response in reversed(raw_responses):
        output_items = _get_field(response, "output")
        if not isinstance(output_items, list):
            continue
        for item in reversed(output_items):
            if _get_field(item, "type") != "function_call":
                continue
            tool_name = str(_get_field(item, "name") or "")
            if tool_name not in {
                "submit_initiative_extractions",
                "stop_initiative_extraction",
            }:
                continue
            arguments = _get_field(item, "arguments")
            if not isinstance(arguments, str):
                continue
            return _coerce_segment_output_payload(
                tool_name, json.loads(arguments), city_name
            )
    return None


def _coerce_segment_output(
    output: object,
    city_name: str | None = None,
) -> InitiativeSegmentExtraction | InitiativeSegmentStop:
    """Coerce final output into one of the accepted segment output models."""
    if isinstance(output, (InitiativeSegmentExtraction, InitiativeSegmentStop)):
        return output
    if isinstance(output, dict):
        tool_name = (
            "stop_initiative_extraction"
            if "initiatives" not in output and "reason" in output
            else "submit_initiative_extractions"
        )
        return _coerce_segment_output_payload(tool_name, output, city_name)
    if isinstance(output, str) and output.strip().startswith("{"):
        payload = json.loads(output)
        return _coerce_segment_output(payload, city_name)
    raise TypeError(
        f"Unsupported initiative segment output type: {type(output).__name__}"
    )
