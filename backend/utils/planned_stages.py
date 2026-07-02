"""Build and merge the persisted run planned-stage contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.utils.artifact_writer import STAGE_NUMBERS
from backend.utils.config import AppConfig

PLANNED_STAGES_FILENAME = "planned_stages.json"
PLANNED_STAGES_ALIAS = "planned_stages"
PLANNED_STAGES_SCHEMA_VERSION = "1.0"

_PLANNED_STAGE_ORDER: tuple[tuple[str, str], ...] = (
    ("input_snapshot", "Capturing run inputs"),
    ("query_preparation", "Preparing retrieval queries"),
    ("retrieval", "Retrieving markdown context"),
    ("markdown_inputs", "Resolving markdown inputs"),
    ("markdown_batching", "Batching markdown context"),
    ("markdown_extraction", "Searching markdown documents"),
    ("markdown_context_handoff", "Freezing markdown context handoff"),
    ("enrichment", "Enrichment"),
    ("enrichment_context_handoff", "Freezing enrichment context handoff"),
    ("assumptions", "Assumptions"),
    ("assumptions_context_handoff", "Freezing assumptions context handoff"),
    ("writer_citation_coverage", "Recording writer citation coverage"),
    ("writer", "Generating final document"),
    ("finalize", "Finalizing run"),
)


def planned_stages_path(run_dir: Path) -> Path:
    """Return the canonical planned-stage artifact path for one run."""
    return run_dir / "stage_files" / "001_input_snapshot" / PLANNED_STAGES_FILENAME


def build_planned_stages_payload(config: AppConfig) -> dict[str, Any]:
    """Build the persisted planned-stage payload for a new run."""
    stages: list[dict[str, Any]] = []
    enrichment_enabled = bool(config.enrichment.enabled)
    disabled_when_no_enrichment = {
        "enrichment",
        "enrichment_context_handoff",
        "assumptions",
        "assumptions_context_handoff",
    }
    for index, (stage_name, label) in enumerate(_PLANNED_STAGE_ORDER, start=1):
        enabled = enrichment_enabled or stage_name not in disabled_when_no_enrichment
        stages.append(
            {
                "id": stage_name,
                "stage_name": stage_name,
                "stage_number": STAGE_NUMBERS.get(stage_name),
                "label": label,
                "planned_order": index,
                "enabled": enabled,
                "artifact_aliases": _artifact_aliases_for_stage(stage_name),
                "status": "pending" if enabled else "disabled",
                "started_at": None,
                "completed_at": None,
                "items": [],
            }
        )
    return {"schema_version": PLANNED_STAGES_SCHEMA_VERSION, "stages": stages}


def merge_progress_into_planned_stages(
    planned_payload: dict[str, Any],
    progress_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Overlay live progress entries onto the persisted planned-stage list."""
    raw_stages = planned_payload.get("stages")
    if not isinstance(raw_stages, list):
        return []
    stages = [_normalize_planned_stage(stage) for stage in raw_stages if isinstance(stage, dict)]
    by_stage_name = {stage["stage_name"]: stage for stage in stages}
    for progress in progress_steps:
        stage_name = progress.get("stage_name") or progress.get("id")
        if not isinstance(stage_name, str):
            continue
        target = by_stage_name.get(stage_name)
        if target is None:
            continue
        _merge_progress_entry(target, progress)
    for stage in stages:
        if not stage.get("enabled", True) and stage["status"] == "pending":
            stage["status"] = "disabled"
    return stages


def _artifact_aliases_for_stage(stage_name: str) -> list[str]:
    """Return known manifest aliases for one planned stage."""
    aliases_by_stage = {
        "input_snapshot": [
            PLANNED_STAGES_ALIAS,
            "execution_snapshot",
            "code_snapshot",
            "config_snapshot",
            "vector_store_snapshot",
            "documents_snapshot",
        ],
        "query_preparation": ["research_question"],
        "retrieval": ["retrieval"],
        "markdown_batching": ["markdown_batches", "source_chunk_index"],
        "markdown_extraction": [
            "markdown_excerpts",
            "markdown_decision_audit",
            "markdown_rejected_chunks",
            "markdown_city_summary",
        ],
        "markdown_context_handoff": ["markdown_context_handoff_context_snapshot"],
        "enrichment": [
            "enrichment_bundle",
            "enrichment_web_research_audit",
            "enrichment_external_source_search_audit",
        ],
        "enrichment_context_handoff": ["enrichment_context_handoff_context_snapshot"],
        "assumptions": [
            "assumptions_assumptions",
            "assumptions_non_estimable",
            "assumptions_assumptions_bundle",
            "assumptions_stage",
        ],
        "assumptions_context_handoff": ["assumptions_context_handoff_context_snapshot"],
        "writer": ["final_output"],
    }
    return aliases_by_stage.get(stage_name, [])


def _normalize_planned_stage(stage: dict[str, Any]) -> dict[str, Any]:
    """Return one planned stage with all fields expected by the API model."""
    stage_name = str(stage.get("stage_name") or stage.get("id") or "")
    enabled = bool(stage.get("enabled", True))
    status = str(stage.get("status") or ("pending" if enabled else "disabled"))
    return {
        "id": str(stage.get("id") or stage_name),
        "stage_name": stage_name,
        "stage_number": stage.get("stage_number"),
        "label": str(stage.get("label") or stage_name.replace("_", " ").title()),
        "status": status,
        "planned_order": stage.get("planned_order"),
        "enabled": enabled,
        "artifact_aliases": [
            alias for alias in (stage.get("artifact_aliases") or []) if isinstance(alias, str)
        ],
        "started_at": stage.get("started_at"),
        "completed_at": stage.get("completed_at"),
        "items": [item for item in (stage.get("items") or []) if isinstance(item, dict)],
    }


def _merge_progress_entry(stage: dict[str, Any], progress: dict[str, Any]) -> None:
    """Merge one progress entry into a planned stage in place."""
    progress_status = str(progress.get("status") or "running")
    stage["status"] = _merged_status(str(stage.get("status") or "pending"), progress_status)
    if stage.get("started_at") is None:
        stage["started_at"] = progress.get("started_at")
    if progress.get("completed_at") is not None:
        stage["completed_at"] = progress.get("completed_at")
    for item in progress.get("items") or []:
        if isinstance(item, dict):
            stage["items"].append(item)


def _merged_status(current: str, incoming: str) -> str:
    """Return the status that should represent a stage with multiple progress entries."""
    if current == "disabled":
        return current
    if incoming == "error" or current == "error":
        return "error"
    if incoming == "running" or current == "running":
        return "running"
    if incoming == "completed" or current == "completed":
        return "completed"
    if incoming == "skipped":
        return "skipped"
    return current


__all__ = [
    "PLANNED_STAGES_ALIAS",
    "PLANNED_STAGES_FILENAME",
    "build_planned_stages_payload",
    "merge_progress_into_planned_stages",
    "planned_stages_path",
]
