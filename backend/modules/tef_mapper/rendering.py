from __future__ import annotations

from typing import Any

from backend.modules.initiative_extractor.models import InitiativeExtractionRecord
from backend.modules.tef_mapper.models import TefTransitionElement
from backend.utils.llm_serialization import serialize_for_llm


def initiative_payload(record: InitiativeExtractionRecord) -> dict[str, Any]:
    """Render one extracted initiative as a compact structured payload."""
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


def llm_input(payload: dict[str, Any]) -> str:
    """Serialize a TEF stage payload as TOON for the LLM input."""
    return serialize_for_llm(payload)


__all__ = ["initiative_payload", "llm_input", "transition_candidate_payload"]
