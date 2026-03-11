"""Background executor for async backend runs."""

from __future__ import annotations

import json
import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from backend.api.models import RunError, RunStatus
from backend.api.services.city_catalog import build_city_subset
from backend.api.services.run_store import RunRecord, RunStore, TERMINAL_STATUSES
from backend.modules.orchestrator.module import run_pipeline
from backend.services.error_log_artifact import write_error_log_artifact
from backend.utils.city_normalization import normalize_city_keys
from backend.utils.config import load_config
from backend.utils.paths import RunPaths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StartRunCommand:
    """Parameters needed to submit a pipeline run."""

    question: str
    requested_run_id: str | None = None
    cities: list[str] | None = None
    config_path: str | None = None
    markdown_path: str | None = None
    log_llm_payload: bool = False
    api_key: str | None = None
    analysis_mode: Literal["aggregate", "city_by_city"] = "aggregate"


class RunExecutor:
    """Threaded executor that runs pipeline jobs without blocking HTTP requests."""

    def __init__(self, run_store: RunStore, max_workers: int = 2) -> None:
        self._run_store = run_store
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="run-worker"
        )

    def submit(self, command: StartRunCommand) -> RunRecord:
        """Create queued run state and dispatch worker thread."""
        normalized_cities = normalize_city_keys(command.cities)
        resolved_command = StartRunCommand(
            question=command.question,
            requested_run_id=command.requested_run_id,
            cities=normalized_cities or None,
            config_path=command.config_path,
            markdown_path=command.markdown_path,
            log_llm_payload=command.log_llm_payload,
            api_key=command.api_key,
            analysis_mode=command.analysis_mode,
        )
        record = self._run_store.create_queued_run(
            question=resolved_command.question, requested_run_id=resolved_command.requested_run_id
        )
        logger.info(
            "Run accepted run_id=%s cities=%s config_path=%s markdown_path=%s analysis_mode=%s log_llm_payload=%s api_key_override=%s",
            record.run_id,
            len(resolved_command.cities) if resolved_command.cities else "all",
            resolved_command.config_path,
            resolved_command.markdown_path,
            resolved_command.analysis_mode,
            resolved_command.log_llm_payload,
            resolved_command.api_key is not None,
        )
        self._executor.submit(
            self._execute, record.run_id, resolved_command.question, resolved_command
        )
        return record

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown worker threads."""
        self._executor.shutdown(wait=wait)

    def _execute(self, run_id: str, question: str, command: StartRunCommand) -> None:
        """Run pipeline and persist terminal state."""
        self._run_store.mark_running(run_id)
        try:
            logger.info("Run execution started run_id=%s", run_id)
            config = load_config(Path(command.config_path) if command.config_path else None)
            base_markdown_dir = (
                Path(command.markdown_path) if command.markdown_path else config.markdown_dir
            )
            logger.info(
                "Run config resolved run_id=%s runs_dir=%s markdown_dir=%s sql_enabled=%s",
                run_id,
                config.runs_dir,
                base_markdown_dir,
                config.enable_sql,
            )
            if command.cities:
                subset_dir = _prepare_selected_markdown_dir(config.runs_dir, run_id)
                copied_files = build_city_subset(
                    source_markdown_dir=base_markdown_dir,
                    target_markdown_dir=subset_dir,
                    selected_cities=command.cities,
                )
                logger.info(
                    "Run city filter run_id=%s requested_cities=%d copied_files=%d subset_dir=%s",
                    run_id,
                    len(command.cities),
                    len(copied_files),
                    subset_dir,
                )
                if not copied_files:
                    raise ValueError(
                        "No markdown files found for selected cities: "
                        + ", ".join(sorted(command.cities))
                    )
                config.markdown_dir = subset_dir
            else:
                config.markdown_dir = base_markdown_dir
                logger.info(
                    "Run city filter run_id=%s mode=all markdown_dir=%s",
                    run_id,
                    config.markdown_dir,
                )

            logger.info("Run pipeline invoking orchestrator run_id=%s", run_id)
            pipeline_kwargs: dict[str, object] = {
                "question": question,
                "config": config,
                "run_id": run_id,
                "log_llm_payload": command.log_llm_payload,
                "selected_cities": command.cities,
            }
            if command.api_key is not None:
                pipeline_kwargs["api_key_override"] = command.api_key
            pipeline_kwargs["analysis_mode"] = command.analysis_mode
            run_paths = run_pipeline(**pipeline_kwargs)
            logger.info(
                "Run pipeline finished run_id=%s run_log=%s",
                run_id,
                run_paths.run_log,
            )
            terminal = _build_terminal_update(run_id, run_paths)
            self._run_store.mark_terminal(
                run_id=run_id,
                status=terminal.status,
                finish_reason=terminal.finish_reason,
                error=terminal.error,
                final_output_path=terminal.final_output_path,
                context_bundle_path=terminal.context_bundle_path,
                run_log_path=terminal.run_log_path,
            )
            logger.info(
                "Run execution completed run_id=%s status=%s finish_reason=%s",
                run_id,
                terminal.status,
                terminal.finish_reason,
            )
        except Exception as exc:  # noqa: BLE001 - API must capture all worker errors
            logger.exception("Pipeline execution failed for run_id=%s", run_id)
            normalized_message = _normalize_error_message(str(exc))
            error_code = "RUN_EXECUTION_ERROR"
            finish_reason = "run_execution_error"
            if _looks_like_api_key_error(normalized_message):
                error_code = "API_KEY_ERROR"
                finish_reason = "api_key_error"
            run_log_path = _persist_executor_failure_artifacts(
                runs_dir=self._run_store.runs_dir,
                run_id=run_id,
                error_code=error_code,
                error_message=normalized_message,
                finish_reason=finish_reason,
            )
            self._run_store.mark_terminal(
                run_id=run_id,
                status="failed",
                finish_reason=finish_reason,
                error=RunError(code=error_code, message=normalized_message),
                run_log_path=run_log_path,
            )


_MASKABLE_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{20,}")


def _normalize_error_message(message: str) -> str:
    """Normalize error text and mask potential key-like fragments."""
    cleaned = message.strip() or "Unknown execution error."
    cleaned = _MASKABLE_KEY_PATTERN.sub("sk-***", cleaned)
    return cleaned


def _looks_like_api_key_error(message: str) -> bool:
    """Heuristic for authentication or API key errors."""
    lowered = message.lower()
    markers = (
        "api key",
        "invalid_api_key",
        "incorrect api key",
        "authentication",
        "unauthorized",
        "401",
        "403",
    )
    return any(marker in lowered for marker in markers)


@dataclass(frozen=True)
class TerminalUpdate:
    """Final state update extracted from run artifacts."""

    status: RunStatus
    finish_reason: str | None
    error: RunError | None
    final_output_path: Path | None
    context_bundle_path: Path | None
    run_log_path: Path | None


def _build_terminal_update(run_id: str, run_paths: RunPaths) -> TerminalUpdate:
    """Build terminal state from run artifacts produced by pipeline."""
    run_log_path = run_paths.run_log if run_paths.run_log.exists() else None
    final_output_path = run_paths.final_output if run_paths.final_output.exists() else None
    context_bundle_path = (
        run_paths.context_bundle if run_paths.context_bundle.exists() else None
    )
    status: RunStatus = "completed"
    finish_reason: str | None = None
    error_payload: RunError | None = None

    if run_log_path is not None:
        run_log_payload = _read_run_log_payload(run_log_path)
        if run_log_payload:
            parsed_status = run_log_payload.get("status")
            if isinstance(parsed_status, str) and parsed_status in TERMINAL_STATUSES:
                status = parsed_status
            elif isinstance(parsed_status, str):
                logger.warning(
                    "Unknown pipeline status `%s` for run_id=%s. Marking as failed.",
                    parsed_status,
                    run_id,
                )
                status = "failed"

            finish_value = run_log_payload.get("finish_reason")
            if isinstance(finish_value, str):
                finish_reason = finish_value

            artifacts = run_log_payload.get("artifacts")
            if isinstance(artifacts, dict):
                final_output_path = _resolve_artifact_path(
                    artifacts.get("final_output"), final_output_path
                )
                context_bundle_path = _resolve_artifact_path(
                    artifacts.get("context_bundle"), context_bundle_path
                )

    if status == "failed" and error_payload is None:
        error_payload = RunError(
            code="RUN_FAILED",
            message=f"Run {run_id} ended with failed status.",
        )

    return TerminalUpdate(
        status=status,
        finish_reason=finish_reason,
        error=error_payload,
        final_output_path=final_output_path,
        context_bundle_path=context_bundle_path,
        run_log_path=run_log_path,
    )


def _read_run_log_payload(path: Path) -> dict[str, object] | None:
    """Read run log JSON payload from file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to parse run log at %s", path)
        return None
    if isinstance(raw, dict):
        return raw
    return None


def _resolve_artifact_path(raw_value: object, fallback: Path | None) -> Path | None:
    """Resolve artifact path from run log or fallback path."""
    if isinstance(raw_value, str) and raw_value.strip():
        candidate = Path(raw_value)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate
    return fallback


def _prepare_selected_markdown_dir(runs_dir: Path, run_id: str) -> Path:
    """Create a clean temp directory for city-scoped markdown copies."""
    subset_dir = runs_dir / "_selected_markdown" / run_id
    if subset_dir.exists():
        shutil.rmtree(subset_dir, ignore_errors=True)
    subset_dir.mkdir(parents=True, exist_ok=True)
    return subset_dir


def _persist_executor_failure_artifacts(
    runs_dir: Path,
    run_id: str,
    *,
    error_code: str,
    error_message: str,
    finish_reason: str,
) -> Path | None:
    """Backfill failure metadata when pipeline exits before run finalization."""
    run_dir = runs_dir / run_id
    run_log_path = run_dir / "run.json"
    if not run_dir.exists():
        return None

    error_log_path = write_error_log_artifact(
        run_dir / "run.log", run_dir / "error_log.txt"
    )
    run_payload = _read_run_log_payload(run_log_path)
    if run_payload is None:
        run_payload = {
            "run_id": run_id,
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "decisions": [],
            "artifacts": {},
        }

    run_payload["status"] = "failed"
    run_payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    run_payload["finish_reason"] = finish_reason
    run_payload["error"] = {"code": error_code, "message": error_message}

    artifacts = run_payload.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    if error_log_path is not None:
        artifacts["error_log"] = str(error_log_path)
    run_payload["artifacts"] = artifacts

    try:
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        run_log_path.write_text(
            json.dumps(run_payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except OSError:
        logger.exception("Failed to persist fallback run.json for run_id=%s", run_id)
        return None
    return run_log_path


__all__ = ["RunExecutor", "StartRunCommand"]
