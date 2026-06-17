from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def build_run_id(now: datetime | None = None) -> str:
    timestamp = now or datetime.now(timezone.utc)
    return timestamp.strftime("%Y%m%d_%H%M")


def ensure_run_dir(runs_dir: Path, run_id: str) -> Path:
    base = runs_dir / run_id
    if not base.exists():
        return base

    counter = 1
    while True:
        candidate = runs_dir / f"{run_id}_{counter:02d}"
        if not candidate.exists():
            return candidate
        counter += 1


@dataclass(frozen=True)
class RunPaths:
    base_dir: Path
    api_state: Path
    manifest: Path
    summary_events: Path
    stages_dir: Path
    stage_files_dir: Path
    run_summary: Path
    error_log: Path
    context_bundle: Path
    final_output: Path


def create_run_paths(runs_dir: Path, run_id: str, context_bundle_name: str) -> RunPaths:
    base_dir = ensure_run_dir(runs_dir, run_id)

    return RunPaths(
        base_dir=base_dir,
        api_state=base_dir / "api_state.json",
        manifest=base_dir / "manifest.json",
        summary_events=base_dir / "summary.jsonl",
        stages_dir=base_dir / "stages",
        stage_files_dir=base_dir / "stage_files",
        run_summary=base_dir / "run_summary.txt",
        error_log=base_dir / "error_log.txt",
        context_bundle=base_dir / context_bundle_name,
        final_output=base_dir / "final.md",
    )


__all__ = ["build_run_id", "ensure_run_dir", "create_run_paths", "RunPaths"]
