"""Run-scoped artifact writer for local-first pipeline observability."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.utils.artifact_manifest import read_artifact_manifest
from backend.utils.json_io import write_json

ARTIFACT_SCHEMA_VERSION = "1.0"
STAGE_NUMBERS: dict[str, int] = {
    "input_snapshot": 1,
    "query_preparation": 2,
    "retrieval": 3,
    "markdown_inputs": 4,
    "markdown_batching": 5,
    "markdown_extraction": 6,
    "markdown_context_handoff": 7,
    "enrichment": 8,
    "enrichment_context_handoff": 9,
    "assumptions": 10,
    "assumptions_context_handoff": 11,
    "writer_multi_pass": 12,
    "writer_citation_coverage": 13,
    "writer": 14,
    "finalize": 15,
    "assumptions_discovery": 101,
    "assumptions_apply": 102,
}
STAGE_NUMBER_ALIASES: dict[str, int] = {
    "context_bundle": 7,
    "enrichment_web_search_assumptions": 8,
}


def _utc_now_iso() -> str:
    """Return a UTC timestamp suitable for artifact metadata."""
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    """Convert a step or file label into a stable filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip().lower())
    return slug.strip("._") or "artifact"


def resolve_stage_number(step_name: str | None) -> int | None:
    """Return the canonical stage number for one known step name."""
    if not isinstance(step_name, str):
        return None
    normalized = step_name.strip()
    if not normalized:
        return None
    return STAGE_NUMBERS.get(normalized) or STAGE_NUMBER_ALIASES.get(normalized)


def stage_file_dir_name(stage_name: str) -> str:
    """Return the canonical numbered directory name for one stage's files."""
    stage_slug = _safe_name(stage_name)
    stage_number = resolve_stage_number(stage_name)
    if stage_number is None:
        return stage_slug
    return f"{stage_number:03d}_{stage_slug}"


class ArtifactWriter:
    """Write one run's summary events, stage details, files, and manifest."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.summary_path = run_dir / "summary.jsonl"
        self.manifest_path = run_dir / "manifest.json"
        self.stages_dir = run_dir / "stages"
        self.stage_files_dir = run_dir / "stage_files"
        self._event_index = 0
        self._aliases: dict[str, dict[str, Any]] = {}
        self._generated_files: list[str] = []

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        self.stage_files_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_state()

    def _load_existing_state(self) -> None:
        """Resume numbering and alias tracking when artifacts already exist."""
        if self.summary_path.exists():
            try:
                with self.summary_path.open("r", encoding="utf-8") as handle:
                    self._event_index = sum(1 for _ in handle)
            except OSError:
                self._event_index = 0

        manifest = read_artifact_manifest(self.run_dir)
        if manifest is None:
            return

        aliases = manifest.get("aliases")
        if isinstance(aliases, dict):
            self._aliases = {
                key: value
                for key, value in aliases.items()
                if isinstance(key, str) and isinstance(value, dict)
            }
        generated_files = manifest.get("generated_files")
        if isinstance(generated_files, list):
            self._generated_files = [
                item for item in generated_files if isinstance(item, str) and item.strip()
            ]

    def _relative_path(self, path: Path) -> str:
        """Return a run-local POSIX path when possible."""
        try:
            return path.resolve(strict=False).relative_to(
                self.run_dir.resolve(strict=False)
            ).as_posix()
        except ValueError:
            return str(path)

    def _remember_file(self, path: Path) -> str:
        relative_path = self._relative_path(path)
        if relative_path not in self._generated_files:
            self._generated_files.append(relative_path)
        return relative_path

    def register_file(
        self,
        alias: str,
        path: Path,
        *,
        artifact_type: str = "runtime_state",
    ) -> None:
        """Register an existing artifact path under a manifest alias."""
        relative_path = self._remember_file(path)
        self._aliases[alias] = {
            "path": relative_path,
            "type": artifact_type,
        }

    def resolve_alias_path(self, alias: str) -> Path | None:
        """Return the concrete path registered for one manifest alias."""
        alias_payload = self._aliases.get(alias)
        if not isinstance(alias_payload, dict):
            return None
        raw_path = alias_payload.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        return self.run_dir / candidate

    def stage_file_path(self, stage_name: str, filename: str) -> Path:
        """Return the canonical path for one stage-owned file."""
        return self.stage_files_dir / stage_file_dir_name(stage_name) / filename

    def write_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        stage_name: str | None = None,
        stage_number: int | None = None,
    ) -> int:
        """Append one summary event and return its event index."""
        self._event_index += 1
        resolved_stage_number = stage_number
        if resolved_stage_number is None:
            resolved_stage_number = resolve_stage_number(stage_name)
        if resolved_stage_number is None:
            resolved_stage_number = resolve_stage_number(str(payload.get("step", "")).strip())
        event = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "event_index": self._event_index,
            "event_type": event_type,
            "run_id": self.run_id,
            "created_at": _utc_now_iso(),
            "payload": payload,
        }
        if resolved_stage_number is not None:
            event["stage_number"] = resolved_stage_number
        with self.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._remember_file(self.summary_path)
        return self._event_index

    def write_step_detail(
        self,
        step_name: str,
        payload: dict[str, Any],
        *,
        event_index: int | None = None,
        event_type: str = "stage_completed",
        stage_number: int | None = None,
    ) -> Path:
        """Write a numbered stage detail JSON file."""
        resolved_event_index = event_index
        resolved_stage_number = stage_number
        if resolved_stage_number is None:
            resolved_stage_number = resolve_stage_number(step_name)
        if resolved_event_index is None:
            resolved_event_index = self.write_event(
                event_type,
                {"step": step_name},
                stage_name=step_name,
                stage_number=resolved_stage_number,
            )
        step_slug = _safe_name(step_name)
        detail_number = (
            resolved_stage_number
            if resolved_stage_number is not None
            else resolved_event_index
        )
        detail_path = self.stages_dir / f"{detail_number:03d}_{step_slug}.json"
        detail_payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "event_index": resolved_event_index,
            "event_type": event_type,
            "run_id": self.run_id,
            "step_name": step_name,
            "created_at": _utc_now_iso(),
            **payload,
        }
        if resolved_stage_number is not None:
            detail_payload["stage_number"] = resolved_stage_number
        write_json(detail_path, detail_payload, ensure_ascii=False, default=str)
        self.register_file(
            f"stage_{step_slug}",
            detail_path,
            artifact_type="stage_detail",
        )
        return detail_path

    def write_run_file(
        self,
        relative_path: str,
        payload: object,
        *,
        alias: str | None = None,
        artifact_type: str = "stage_file",
    ) -> Path:
        """Write a run-local JSON file and optionally register it in the manifest."""
        path = self.run_dir / relative_path
        write_json(path, payload, ensure_ascii=False, default=str)
        if alias:
            self.register_file(alias, path, artifact_type=artifact_type)
        else:
            self._remember_file(path)
        return path

    def write_stage_file(
        self,
        stage_name: str,
        filename: str,
        payload: object,
        *,
        alias: str | None = None,
        artifact_type: str = "stage_file",
    ) -> Path:
        """Write one JSON artifact under the canonical stage-files directory."""
        relative_path = self._relative_path(self.stage_file_path(stage_name, filename))
        return self.write_run_file(
            relative_path,
            payload,
            alias=alias,
            artifact_type=artifact_type,
        )

    def write_manifest(self, metadata: dict[str, Any] | None = None) -> Path:
        """Write the final manifest index for this run."""
        self._remember_file(self.summary_path)
        self._remember_file(self.manifest_path)
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at": _utc_now_iso(),
            "summary_events": self._relative_path(self.summary_path),
            "aliases": self._aliases,
            "generated_files": sorted(self._generated_files),
            "metadata": metadata or {},
        }
        write_json(self.manifest_path, manifest, ensure_ascii=False, default=str)
        return self.manifest_path


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactWriter",
    "STAGE_NUMBERS",
    "resolve_stage_number",
    "stage_file_dir_name",
]
