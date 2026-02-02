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
    run_log: Path
    run_summary: Path
    context_bundle: Path
    schema_summary: Path
    city_list: Path
    sql_dir: Path
    sql_queries: Path
    sql_results: Path
    sql_results_full: Path
    markdown_dir: Path
    markdown_excerpts: Path
    drafts_dir: Path
    final_output: Path


def create_run_paths(runs_dir: Path, run_id: str, context_bundle_name: str) -> RunPaths:
    base_dir = ensure_run_dir(runs_dir, run_id)
    sql_dir = base_dir / "sql"
    markdown_dir = base_dir / "markdown"
    drafts_dir = base_dir / "drafts"

    return RunPaths(
        base_dir=base_dir,
        run_log=base_dir / "run.json",
        run_summary=base_dir / "run_summary.txt",
        context_bundle=base_dir / context_bundle_name,
        schema_summary=base_dir / "schema_summary.json",
        city_list=base_dir / "city_list.json",
        sql_dir=sql_dir,
        sql_queries=sql_dir / "queries.json",
        sql_results=sql_dir / "results.json",
        sql_results_full=sql_dir / "results_full.json",
        markdown_dir=markdown_dir,
        markdown_excerpts=markdown_dir / "excerpts.json",
        drafts_dir=drafts_dir,
        final_output=base_dir / "final.md",
    )


__all__ = ["build_run_id", "ensure_run_dir", "create_run_paths", "RunPaths"]
