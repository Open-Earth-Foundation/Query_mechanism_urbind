from __future__ import annotations

import json
from typing import Any

from backend.modules.initiative_extractor.models import InitiativeExtractionRecord
from backend.modules.tef_mapper.models import TefTransitionElement


def initiative_payload(record: InitiativeExtractionRecord) -> dict[str, Any]:
    """Render one extracted initiative as a compact JSON payload."""
    return {
        "record_id": record.record_id,
        "source_document": record.source_document,
        "document_local_code": record.document_local_code,
        "initiative": record.initiative.model_dump(mode="json"),
        "data_quality_flags": record.data_quality_flags,
        "number_context": record.number_context,
        "number_deferred": record.number_deferred,
        "number_uncertain": record.number_uncertain,
        "extraction_notes": record.extraction_notes,
    }


def transition_candidate_payload(
    candidates: list[TefTransitionElement],
) -> list[dict[str, Any]]:
    """Render Transition Element candidates with fields needed for ranking."""
    return [
        {
            "tef_id": candidate.tef_id,
            "title": candidate.title,
            "description": candidate.description,
            "sector": candidate.sector,
            "path": candidate.path,
            "path_labels": candidate.path_labels,
            "type": candidate.type,
            "unit_of_measure": candidate.unit_of_measure,
            "sustainability": candidate.sustainability,
            "long_name": candidate.long_name,
            "short_name": candidate.short_name,
            "shift_from": candidate.shift_from,
            "shift_to": candidate.shift_to,
            "carbon_causal_chains": candidate.carbon_causal_chains,
        }
        for candidate in candidates
    ]


def json_input(payload: dict[str, Any]) -> str:
    """Serialize a stage payload as readable JSON for the LLM input."""
    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = ["initiative_payload", "json_input", "transition_candidate_payload"]
