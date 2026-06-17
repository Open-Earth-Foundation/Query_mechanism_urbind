"""Helpers for assumptions context payloads and artifacts."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.modules.web_researcher.models import AssumptionRecord, NonEstimableRecord
from backend.services.run_logger import RunLogger
from backend.utils.artifact_writer import stage_file_dir_name

logger = logging.getLogger(__name__)


def city_field_pairs(
    records: list[dict[str, Any]],
    *,
    field_key: str = "field",
) -> list[dict[str, str]]:
    """Return unique city-field pairs in first-seen order."""
    pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        city = str(record.get("city", "")).strip()
        field = str(record.get(field_key, "")).strip()
        if not city or not field:
            continue
        key = (city.casefold(), field.casefold())
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"city": city, "field": field})
    return pairs


def build_assumptions_payload(
    *,
    assumptions_model: str,
    assumptions: list[AssumptionRecord],
    non_estimable: list[NonEstimableRecord],
    saturation_warning: str | None,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Build the top-level runtime assumptions context payload."""
    assumptions_payload = [record.model_dump(mode="json") for record in assumptions]
    non_estimable_payload = [
        record.model_dump(mode="json") for record in non_estimable
    ]
    return {
        "assumptions": assumptions_payload,
        "non_estimable": non_estimable_payload,
        "saturation_warning": saturation_warning,
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "assumptions_estimator_model": assumptions_model,
            "assumption_count": len(assumptions_payload),
            "non_estimable_output_count": len(non_estimable_payload),
            "elapsed_seconds": elapsed_seconds,
        },
    }


def build_assumptions_stage_artifact(
    assumptions_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the stage-010 assumptions artifact payload."""
    assumptions = assumptions_payload.get("assumptions")
    non_estimable = assumptions_payload.get("non_estimable")
    assumptions_list = assumptions if isinstance(assumptions, list) else []
    non_estimable_list = non_estimable if isinstance(non_estimable, list) else []
    added_city_fields = city_field_pairs(assumptions_list, field_key="field_name")
    non_estimable_city_fields = city_field_pairs(
        non_estimable_list,
        field_key="field_name",
    )
    meta = assumptions_payload.get("meta")
    return {
        "status": "completed",
        "flags": {
            "assumptions_enabled": True,
            "assumptions_executed": True,
            "assumptions_model": (
                meta.get("assumptions_estimator_model")
                if isinstance(meta, dict)
                else None
            ),
        },
        "outputs": {
            "assumptions": assumptions_list,
            "non_estimable": non_estimable_list,
            "saturation_warning": assumptions_payload.get("saturation_warning"),
            "added_city_fields": added_city_fields,
            "non_estimable_city_fields": non_estimable_city_fields,
        },
        "metrics": {
            "assumption_count": len(assumptions_list),
            "non_estimable_output_count": len(non_estimable_list),
            "added_city_field_count": len(added_city_fields),
            "non_estimable_city_field_count": len(non_estimable_city_fields),
        },
    }


def serialize_assumptions_artifacts(
    assumptions_payload: dict[str, Any],
    base_dir: Path,
    run_logger: RunLogger,
) -> None:
    """Write assumptions artifacts to their own stage folder."""
    if not assumptions_payload:
        return

    artifact_map = {
        "assumptions": ("assumptions.json", assumptions_payload.get("assumptions")),
        "non_estimable": (
            "non_estimable.json",
            assumptions_payload.get("non_estimable"),
        ),
        "assumptions_bundle": ("assumptions_bundle.json", assumptions_payload),
    }
    for name, artifact in artifact_map.items():
        filename, payload = artifact
        if payload is None:
            continue
        run_logger.write_stage_file(
            "assumptions",
            filename,
            payload,
            alias=f"assumptions_{name}",
        )

    stage_payload = build_assumptions_stage_artifact(assumptions_payload)
    stage_path = run_logger.write_stage_file(
        "assumptions",
        "assumptions_stage.json",
        stage_payload,
        alias="assumptions_stage",
    )
    run_logger.write_stage_detail(
        "assumptions",
        {
            "inputs": stage_payload["flags"],
            "outputs": {
                "assumptions_bundle": (
                    f"stage_files/{stage_file_dir_name('assumptions')}/"
                    "assumptions_bundle.json"
                ),
                "assumptions_stage": run_logger.artifact_label(stage_path),
            },
            "metrics": stage_payload["metrics"],
        },
    )
    logger.info(
        "Assumptions artifacts written to stage_files/%s",
        stage_file_dir_name("assumptions"),
    )


__all__ = [
    "build_assumptions_payload",
    "build_assumptions_stage_artifact",
    "city_field_pairs",
    "serialize_assumptions_artifacts",
]
