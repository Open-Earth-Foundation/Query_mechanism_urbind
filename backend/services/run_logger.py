from __future__ import annotations

import json
import logging
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from backend.services.error_log_artifact import write_error_log_artifact
from backend.utils.artifact_writer import ArtifactWriter, resolve_stage_number
from backend.utils.city_normalization import normalize_city_key
from backend.utils.json_io import write_json
from backend.utils.paths import RunPaths

logger = logging.getLogger(__name__)


def _collect_city_keys(values: list[str] | None) -> list[str]:
    """Normalize city labels into stable keys while preserving first-seen order."""
    if not values:
        return []
    keys: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        key = normalize_city_key(value.strip())
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


class RunLogger:
    def __init__(self, run_paths: RunPaths, question: str) -> None:
        """Initialize run-scoped structured and text artifacts."""
        self.run_paths = run_paths
        self.run_state: dict[str, Any] = {
            "run_id": run_paths.base_dir.name,
            "inputs": {
                "original_question": question,
                "canonical_research_query": question,
                "query_mode": "standard",
                "retrieval_queries": [question],
                "retrieval_query_count": 1,
                "retrieval_query_1": question,
                "retrieval_query_2": None,
                "retrieval_query_3": None,
                "city_scope_mode": "all_cities",
                "selected_cities_planned": [],
                "selected_cities_found": [],
                "markdown_dir": None,
                "markdown_file_count": 0,
                "markdown_chunk_count": 0,
                "markdown_excerpt_count": 0,
                "markdown_source_mode": "standard_chunking",
                "analysis_mode": "aggregate",
            },
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "decisions": [],
            "artifacts": {
                "manifest": str(run_paths.manifest),
                "summary_events": str(run_paths.summary_events),
            },
            "mlflow": None,
        }
        self.context_bundle: dict[str, Any] = {
            "markdown": None,
            "original_question": question,
            "research_question": question,
            "query_mode": "standard",
            "retrieval_queries": [question],
            "city_scope_mode": "all_cities",
            "selected_cities": [],
            "selected_city_names": [],
            "inspected_cities": [],
            "inspected_city_names": [],
            "final": None,
            "analysis_mode": "aggregate",
        }
        self.artifacts = ArtifactWriter(run_paths.base_dir, run_paths.base_dir.name)
        self._stage_event_indices: dict[str, int] = {}

        self._ensure_dirs()
        self.write_context_bundle()
        self.write_api_state()
        self.record_artifact("manifest", self.run_paths.manifest)
        self.record_artifact("summary_events", self.run_paths.summary_events)

    def _ensure_dirs(self) -> None:
        """Create the per-run artifact directories."""
        self.run_paths.base_dir.mkdir(parents=True, exist_ok=True)
        self.run_paths.stages_dir.mkdir(parents=True, exist_ok=True)
        self.run_paths.stage_files_dir.mkdir(parents=True, exist_ok=True)

    def write_input_snapshot_stage(
        self,
        *,
        snapshot_summary: Mapping[str, object] | None = None,
        snapshot_artifacts: Mapping[str, str] | None = None,
    ) -> None:
        """Persist the structured stage-001 overview after input snapshots are ready."""
        payload: dict[str, object] = {
            "inputs": self._build_input_snapshot_inputs(),
            "outputs": {
                "context_bundle": self._relative_path(self.run_paths.context_bundle),
                "api_state": self._relative_path(self.run_paths.api_state),
            },
            "metrics": {
                "retrieval_query_count": self.run_state["inputs"].get(
                    "retrieval_query_count", 0
                ),
                "selected_city_count": len(
                    self.run_state["inputs"].get("selected_cities_planned", []) or []
                ),
            },
        }
        if snapshot_summary:
            payload["snapshot_summary"] = dict(snapshot_summary)
        if snapshot_artifacts:
            payload["snapshots"] = dict(snapshot_artifacts)
        self.write_stage_detail(
            "input_snapshot",
            payload,
            event_type="stage_completed",
            reuse_existing_event=True,
        )

    def _build_input_snapshot_inputs(self) -> dict[str, object]:
        """Return the stage-001 input view without later discovery/extraction fields."""
        inputs = self.run_state.get("inputs", {})
        if not isinstance(inputs, dict):
            return {}
        keys = [
            "original_question",
            "canonical_research_query",
            "query_mode",
            "retrieval_queries",
            "retrieval_query_count",
            "retrieval_query_1",
            "retrieval_query_2",
            "retrieval_query_3",
            "city_scope_mode",
            "selected_cities_planned",
            "analysis_mode",
        ]
        return {key: inputs.get(key) for key in keys}

    def _relative_path(self, path: Path) -> str:
        """Return a run-local path label when possible."""
        try:
            return path.resolve(strict=False).relative_to(
                self.run_paths.base_dir.resolve(strict=False)
            ).as_posix()
        except ValueError:
            return str(path)

    def artifact_label(self, path: Path) -> str:
        """Return the stable run-local label for an artifact path."""
        return self._relative_path(path)

    def artifact_path(self, alias: str) -> Path | None:
        """Return the concrete path for one registered artifact alias."""
        return self.artifacts.resolve_alias_path(alias)

    def write_api_state(self) -> None:
        """Persist the structured API state JSON."""
        self.run_paths.api_state.write_text(
            json.dumps(
                self._build_serialized_api_state(),
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

    def _build_serialized_api_state(self) -> dict[str, Any]:
        """Return the persisted api_state.json payload for API and benchmark use."""
        payload: dict[str, Any] = {
            "run_id": self.run_state.get("run_id"),
            "question": self.run_state.get("inputs", {}).get("original_question")
            if isinstance(self.run_state.get("inputs"), dict)
            else None,
            "status": self.run_state.get("status"),
            "started_at": self.run_state.get("started_at"),
            "completed_at": self.run_state.get("completed_at"),
            "finish_reason": self.run_state.get("finish_reason"),
            "error": self.run_state.get("error"),
            "inputs": self.run_state.get("inputs"),
            "decisions": self.run_state.get("decisions"),
            "llm_usage": self.run_state.get("llm_usage"),
            "retry_summary": self.run_state.get("retry_summary"),
            "writer_citation_coverage": self.run_state.get("writer_citation_coverage"),
            "writer_multi_pass": self.run_state.get("writer_multi_pass"),
            "mlflow": self.run_state.get("mlflow"),
        }
        return payload

    def write_context_bundle(self) -> None:
        """Persist the current context bundle JSON."""
        write_json(
            self.run_paths.context_bundle,
            self.context_bundle,
            ensure_ascii=False,
            default=str,
        )
        if hasattr(self, "artifacts"):
            self.artifacts.register_file(
                "context_bundle",
                self.run_paths.context_bundle,
                artifact_type="runtime_state",
            )

    def _read_json_file(self, path: Path) -> object | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _format_json(self, payload: object | None) -> str:
        if payload is None:
            return "(missing)"
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str)

    def _summarize_markdown_failures(self, payload: object | None) -> dict[str, Any] | None:
        """Build an aggregate failure summary from markdown error details."""
        if not isinstance(payload, dict):
            return None
        error_payload = payload.get("error")
        if not isinstance(error_payload, dict):
            return None

        by_code: dict[str, int] = {}
        by_city: dict[str, int] = {}
        details = error_payload.get("details")
        if isinstance(details, list):
            for entry in details:
                if not isinstance(entry, str):
                    continue
                city_part, separator, reason_part = entry.partition(":")
                if not separator:
                    continue
                city_name, _, _batch_info = city_part.partition("#batch")
                reason = reason_part.strip()
                city = city_name.strip()
                if reason:
                    by_code[reason] = by_code.get(reason, 0) + 1
                if city:
                    by_city[city] = by_city.get(city, 0) + 1

        if not by_code:
            code = error_payload.get("code")
            if isinstance(code, str) and code:
                by_code[code] = 1

        if not by_code and not by_city:
            return None

        return {
            "total_failed_batches": sum(by_code.values()) if by_code else 0,
            "by_code": dict(sorted(by_code.items())),
            "by_city": dict(sorted(by_city.items())),
        }

    def write_query_preparation_stage(
        self,
        *,
        original_question: str,
        canonical_research_query: str,
        retrieval_queries: list[str],
        query_mode: str,
    ) -> None:
        """Persist the query-preparation stage detail."""
        self.write_stage_detail(
            "query_preparation",
            {
                "inputs": {
                    "original_question": original_question,
                    "query_mode": query_mode,
                },
                "outputs": {
                    "canonical_research_query": canonical_research_query,
                    "retrieval_queries": retrieval_queries,
                },
                "metrics": {
                    "retrieval_query_count": len(retrieval_queries),
                },
            },
        )

    def _read_text_file(self, path: Path, max_bytes: int = 200_000) -> str:
        if not path.exists():
            return "(missing)"
        size = path.stat().st_size
        if size > max_bytes:
            return f"(omitted {size} bytes; see {path})"
        return path.read_text(encoding="utf-8")

    def _extract_usage_value(self, usage: Mapping[str, Any], keys: list[str]) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return None

    def _parse_retry_payload_raw(self, payload_raw: str) -> dict[str, Any] | None:
        """Parse retry payload from either JSON or key=value format."""
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload

        try:
            tokens = shlex.split(payload_raw)
        except ValueError:
            return None
        parsed: dict[str, Any] = {}
        for token in tokens:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            key_clean = key.strip()
            if not key_clean:
                continue
            parsed[key_clean] = value.strip()
        if parsed:
            return parsed
        return None

    def _format_total_runtime(self) -> str:
        """Return elapsed runtime in seconds from run start/end timestamps."""
        started_raw = self.run_state.get("started_at")
        completed_raw = self.run_state.get("completed_at")
        if not isinstance(started_raw, str) or not isinstance(completed_raw, str):
            return "n/a"
        try:
            started_dt = datetime.fromisoformat(started_raw)
            completed_dt = datetime.fromisoformat(completed_raw)
        except ValueError:
            return "n/a"

        elapsed_seconds = (completed_dt - started_dt).total_seconds()
        if elapsed_seconds < 0:
            return "n/a"

        return f"{elapsed_seconds:.3f} seconds"

    def _summarize_llm_usage(self) -> dict[str, Any] | None:
        run_log_path = self.run_paths.base_dir / "run.log"
        if not run_log_path.exists():
            return None

        totals = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
        per_agent: dict[str, dict[str, int]] = {}
        calls = 0

        with run_log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "LLM_USAGE " not in line:
                    continue
                payload = line.split("LLM_USAGE ", 1)[1].strip()
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = data.get("usage")
                if not isinstance(usage, Mapping):
                    continue
                agent = str(data.get("agent") or "unknown")
                input_tokens = self._extract_usage_value(
                    usage, ["input_tokens", "prompt_tokens"]
                )
                output_tokens = self._extract_usage_value(
                    usage, ["output_tokens", "completion_tokens"]
                )
                total_tokens = self._extract_usage_value(
                    usage, ["total_tokens", "total"]
                )

                if total_tokens is None and input_tokens is not None and output_tokens is not None:
                    total_tokens = input_tokens + output_tokens

                if total_tokens is None:
                    continue

                calls += 1
                totals["total_tokens"] += total_tokens
                if input_tokens is not None:
                    totals["input_tokens"] += input_tokens
                if output_tokens is not None:
                    totals["output_tokens"] += output_tokens

                agent_totals = per_agent.setdefault(
                    agent,
                    {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
                )
                agent_totals["total_tokens"] += total_tokens
                if input_tokens is not None:
                    agent_totals["input_tokens"] += input_tokens
                if output_tokens is not None:
                    agent_totals["output_tokens"] += output_tokens

        if calls == 0:
            return None

        return {
            "calls": calls,
            "totals": totals,
            "per_agent": per_agent,
        }

    def _summarize_retry_events(self) -> dict[str, object] | None:
        """Build a compact retry summary from run.log RETRY_* lines."""
        run_log_path = self.run_paths.base_dir / "run.log"
        if not run_log_path.exists():
            return None

        total_events = 0
        exhausted_events = 0
        by_operation: dict[str, int] = {}

        with run_log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                marker = None
                if "RETRY_EVENT " in line:
                    marker = "RETRY_EVENT "
                elif "RETRY_EXHAUSTED " in line:
                    marker = "RETRY_EXHAUSTED "
                if marker is None:
                    continue
                payload_raw = line.split(marker, 1)[1].strip()
                payload = self._parse_retry_payload_raw(payload_raw)
                if not isinstance(payload, dict):
                    continue
                operation = str(payload.get("operation", "unknown")).strip() or "unknown"
                by_operation[operation] = by_operation.get(operation, 0) + 1
                total_events += 1
                if marker == "RETRY_EXHAUSTED ":
                    exhausted_events += 1

        if total_events == 0:
            return None
        return {
            "total_events": total_events,
            "exhausted_events": exhausted_events,
            "by_operation": dict(sorted(by_operation.items())),
        }

    def _write_error_log_artifact(self) -> Path | None:
        """Extract ERROR/CRITICAL and exhausted retry lines to error_log.txt."""
        run_log_path = self.run_paths.base_dir / "run.log"
        return write_error_log_artifact(run_log_path, self.run_paths.error_log)

    def write_text_log(self) -> None:
        """Write the human-readable run summary artifact."""
        lines: list[str] = []
        lines.append("RUN SUMMARY")
        lines.append(f"Run ID: {self.run_state.get('run_id')}")
        inputs = self.run_state.get("inputs", {})
        if isinstance(inputs, dict):
            original_question = (
                inputs.get("original_question")
                or inputs.get("initial_question")
                or "(missing)"
            )
            primary_retrieval_query = (
                inputs.get("canonical_research_query")
                or self.context_bundle.get("research_question")
                or "(missing)"
            )
            query_mode = inputs.get(
                "query_mode",
                self.context_bundle.get("query_mode", "standard"),
            )
            lines.append(f"Original question: {original_question}")
            lines.append(f"Query mode: {query_mode}")
            lines.append(f"Primary retrieval query: {primary_retrieval_query}")
            for index in range(1, 4):
                lines.append(
                    f"Retrieval query {index}: "
                    f"{inputs.get(f'retrieval_query_{index}') or '(none)'}"
                )
            lines.append(
                "Selected cities (planned): "
                f"{json.dumps(inputs.get('selected_cities_planned', []), ensure_ascii=False)}"
            )
            lines.append(
                "Selected cities (found): "
                f"{json.dumps(inputs.get('selected_cities_found', []), ensure_ascii=False)}"
            )
            lines.append(f"Markdown dir: {inputs.get('markdown_dir') or '(unknown)'}")
            lines.append(f"Markdown file count: {inputs.get('markdown_file_count', 0)}")
            lines.append(f"Markdown chunk count: {inputs.get('markdown_chunk_count', 0)}")
            lines.append(f"Markdown excerpt count: {inputs.get('markdown_excerpt_count', 0)}")
            lines.append(
                f"Markdown source mode: {inputs.get('markdown_source_mode', 'standard_chunking')}"
            )
        lines.append(f"Status: {self.run_state.get('status')}")
        lines.append(f"Finish reason: {self.run_state.get('finish_reason', 'n/a')}")
        lines.append(f"Started: {self.run_state.get('started_at')}")
        lines.append(f"Completed: {self.run_state.get('completed_at')}")
        lines.append(f"Total runtime: {self._format_total_runtime()}")
        llm_usage = self.run_state.get("llm_usage")
        if llm_usage:
            lines.append(f"LLM Usage: {json.dumps(llm_usage, ensure_ascii=False)}")
        retry_summary = self.run_state.get("retry_summary")
        if retry_summary:
            lines.append(f"Retry Summary: {json.dumps(retry_summary, ensure_ascii=False)}")
        writer_multi_pass = self.run_state.get("writer_multi_pass")
        if writer_multi_pass:
            lines.append(
                f"Writer multi-pass: {json.dumps(writer_multi_pass, ensure_ascii=False)}"
            )
        lines.append("")

        lines.append("ARTIFACTS")
        for key, value in self.run_state.get("artifacts", {}).items():
            lines.append(f"- {key}: {value}")
        lines.append("")

        lines.append("DECISIONS (LLM)")
        lines.append(self._format_json(self.run_state.get("decisions")))
        lines.append("")

        markdown_excerpts_path = self.artifact_path("markdown_excerpts")
        markdown_payload = (
            self._read_json_file(markdown_excerpts_path)
            if markdown_excerpts_path is not None
            else None
        )
        lines.append("MARKDOWN_FAILURE_SUMMARY")
        markdown_failure_summary = self._summarize_markdown_failures(markdown_payload)
        lines.append(
            self._format_json(markdown_failure_summary)
            if markdown_failure_summary is not None
            else "none"
        )
        lines.append("")
        lines.append("FULL PAYLOADS")
        lines.append("- context_bundle: context_bundle.json")
        lines.append("- markdown_excerpts: stage_files/006_markdown_extraction/accepted_excerpts.json")
        lines.append("- final_output: final.md")

        self.run_paths.run_summary.write_text("\n".join(lines), encoding="utf-8")

    def llm_token_count_for_agents(self, agent_names: set[str]) -> int | None:
        """Return total LLM tokens for selected agents from run.log."""
        usage = self._summarize_llm_usage()
        if not isinstance(usage, dict):
            return None
        per_agent = usage.get("per_agent")
        if not isinstance(per_agent, dict):
            return None
        total = 0
        matched = False
        for agent_name in agent_names:
            payload = per_agent.get(agent_name)
            if not isinstance(payload, dict):
                continue
            total_tokens = payload.get("total_tokens")
            if not isinstance(total_tokens, int):
                continue
            total += total_tokens
            matched = True
        return total if matched else None

    def record_decision(self, decision: dict[str, Any]) -> None:
        """Append one structured decision payload to run and stage state."""
        self.run_state["decisions"].append(decision)
        step_name = str(decision.get("step", "")).strip() or None
        self._append_decision_to_stage_detail(step_name, decision)
        self.write_api_state()

    def _append_decision_to_stage_detail(
        self,
        step_name: str | None,
        decision: dict[str, Any],
    ) -> None:
        """Attach a decision to its stage detail when that stage already exists."""
        stage_number = resolve_stage_number(step_name)
        if step_name is None or stage_number is None:
            return

        stage_path = self.run_paths.stages_dir / f"{stage_number:03d}_{step_name}.json"
        payload = self._read_json_file(stage_path)
        if not isinstance(payload, dict):
            return

        decisions = payload.get("decisions")
        if not isinstance(decisions, list):
            decisions = []
        decisions.append(decision)
        payload["decisions"] = decisions
        write_json(stage_path, payload, ensure_ascii=False, default=str)

    def record_artifact(self, name: str, path: Path) -> None:
        """Register one artifact path in the structured run state."""
        self.run_state["artifacts"][name] = str(path)
        self.artifacts.register_file(name, path)
        self.write_api_state()

    def _build_manifest_metadata(self) -> dict[str, Any]:
        """Return metadata persisted in the final artifact manifest."""
        metadata = {
            "status": self.run_state.get("status"),
            "finish_reason": self.run_state.get("finish_reason"),
            "llm_usage": self.run_state.get("llm_usage"),
            "retry_summary": self.run_state.get("retry_summary"),
        }
        mlflow_payload = self.run_state.get("mlflow")
        if mlflow_payload:
            metadata["mlflow"] = mlflow_payload
        return metadata

    def record_mlflow_metadata(self, metadata: dict[str, Any]) -> None:
        """Persist MLflow sync metadata into API state and manifest files."""
        self.run_state["mlflow"] = metadata
        self.write_api_state()
        manifest_path = self.artifacts.write_manifest(self._build_manifest_metadata())
        self.run_state["artifacts"]["manifest"] = str(manifest_path)
        self.write_api_state()

    def write_stage_detail(
        self,
        step_name: str,
        payload: dict[str, Any],
        *,
        event_type: str = "stage_completed",
        reuse_existing_event: bool = False,
    ) -> Path:
        """Write a standardized stage detail artifact."""
        event_payload = {
            "step": step_name,
            "metrics": payload.get("metrics", {}),
            "outputs": payload.get("outputs", {}),
        }
        stage_number = resolve_stage_number(step_name)
        event_index = self._stage_event_indices.get(step_name)
        if event_index is None or not reuse_existing_event:
            event_index = self.artifacts.write_event(
                event_type,
                event_payload,
                stage_name=step_name,
                stage_number=stage_number,
            )
            self._stage_event_indices[step_name] = event_index
        path = self.artifacts.write_step_detail(
            step_name,
            payload,
            event_index=event_index,
            event_type=event_type,
            stage_number=stage_number,
        )
        self.write_api_state()
        return path

    def write_stage_file(
        self,
        stage_name: str,
        filename: str,
        payload: object,
        *,
        alias: str | None = None,
    ) -> Path:
        """Write a run-local JSON stage file through the artifact writer."""
        path = self.artifacts.write_stage_file(
            stage_name,
            filename,
            payload,
            alias=alias,
        )
        if alias:
            self.run_state["artifacts"][alias] = str(path)
            self.write_api_state()
        return path

    def record_writer_citation_coverage(self, coverage: dict[str, Any]) -> None:
        """Persist final writer citation-coverage diagnostics for API consumers."""
        self.run_state["writer_citation_coverage"] = coverage
        self.write_stage_detail(
            "writer_citation_coverage",
            {
                "inputs": {},
                "outputs": {"writer_citation_coverage": coverage},
                "metrics": {
                    "citation_coverage_ratio": coverage.get("coverage_ratio"),
                    "confirmed_city_count": coverage.get("coverage_confirmed"),
                    "required_city_count": coverage.get("coverage_required"),
                },
            },
        )
        self.write_api_state()

    def record_writer_multi_pass(self, payload: dict[str, Any]) -> None:
        """Persist writer multi-pass diagnostics for API consumers."""
        self.run_state["writer_multi_pass"] = payload
        self.write_stage_detail(
            "writer_multi_pass",
            {
                "inputs": {},
                "outputs": {"writer_multi_pass": payload},
                "metrics": {
                    "batch_count": payload.get("batch_count"),
                    "input_tokens": payload.get("input_tokens"),
                },
            },
        )
        self.write_api_state()

    def update_enrichment_bundle(self, enrichment_payload: dict[str, Any]) -> None:
        """Persist the enrichment context bundle section."""
        self.context_bundle["enrichment"] = enrichment_payload
        self.write_context_bundle()

    def update_markdown_bundle(self, markdown_payload: dict[str, Any]) -> None:
        """Persist markdown payload and sync excerpt count in run inputs."""
        self.context_bundle["markdown"] = markdown_payload
        excerpt_count = markdown_payload.get("excerpt_count", 0)
        normalized_excerpt_count = excerpt_count if isinstance(excerpt_count, int) else 0
        inputs = self.run_state.get("inputs")
        if isinstance(inputs, dict):
            inputs["markdown_excerpt_count"] = normalized_excerpt_count
            self.run_state["inputs"] = inputs
            self.write_api_state()
        self.write_context_bundle()

    def update_query_inputs(
        self,
        *,
        original_question: str,
        canonical_research_query: str,
        retrieval_queries: list[str],
        query_mode: str,
        write_stage_detail: bool = True,
    ) -> None:
        """Persist query-mode metadata in both run-state inputs and context bundle."""
        normalized_original_question = original_question.strip() or original_question
        normalized_canonical_query = (
            canonical_research_query.strip() or normalized_original_question
        )
        normalized_retrieval_queries = [
            query.strip() for query in retrieval_queries if query.strip()
        ]
        if not normalized_retrieval_queries:
            normalized_retrieval_queries = [normalized_canonical_query]
        resolved_query_mode = query_mode.strip() if isinstance(query_mode, str) else ""
        if not resolved_query_mode:
            resolved_query_mode = "standard"

        self.context_bundle["original_question"] = normalized_original_question
        self.context_bundle["research_question"] = normalized_canonical_query
        self.context_bundle["query_mode"] = resolved_query_mode
        self.context_bundle["retrieval_queries"] = normalized_retrieval_queries

        inputs = self.run_state.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
        inputs["original_question"] = normalized_original_question
        inputs["canonical_research_query"] = normalized_canonical_query
        inputs["query_mode"] = resolved_query_mode
        inputs["retrieval_queries"] = normalized_retrieval_queries
        inputs["retrieval_query_count"] = len(normalized_retrieval_queries)
        inputs["retrieval_query_1"] = (
            normalized_retrieval_queries[0]
            if len(normalized_retrieval_queries) >= 1
            else None
        )
        inputs["retrieval_query_2"] = (
            normalized_retrieval_queries[1]
            if len(normalized_retrieval_queries) >= 2
            else None
        )
        inputs["retrieval_query_3"] = (
            normalized_retrieval_queries[2]
            if len(normalized_retrieval_queries) >= 3
            else None
        )
        self.run_state["inputs"] = inputs
        self.write_api_state()
        self.write_context_bundle()
        if write_stage_detail:
            self.write_query_preparation_stage(
                original_question=normalized_original_question,
                canonical_research_query=normalized_canonical_query,
                retrieval_queries=normalized_retrieval_queries,
                query_mode=resolved_query_mode,
            )

    def update_analysis_mode(self, analysis_mode: str) -> None:
        """Persist selected analysis mode in run-state inputs and context bundle."""
        normalized = analysis_mode.strip() if isinstance(analysis_mode, str) else ""
        resolved = normalized if normalized else "aggregate"
        inputs = self.run_state.get("inputs")
        if isinstance(inputs, dict):
            inputs["analysis_mode"] = resolved
            self.run_state["inputs"] = inputs
            self.write_api_state()
        self.context_bundle["analysis_mode"] = resolved
        self.write_context_bundle()

    def update_requested_city_scope(
        self,
        selected_cities: list[str] | None,
    ) -> None:
        """Persist the requested city scope before markdown discovery starts."""
        planned_keys = _collect_city_keys(selected_cities)
        scope_mode = "selected_cities" if planned_keys else "all_cities"
        inputs = self.run_state.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
        inputs["city_scope_mode"] = scope_mode
        inputs["selected_cities_planned"] = planned_keys
        self.run_state["inputs"] = inputs
        self.context_bundle["city_scope_mode"] = scope_mode
        self.context_bundle["selected_cities"] = planned_keys
        self.context_bundle["selected_city_names"] = planned_keys
        self.write_api_state()
        self.write_context_bundle()

    def record_markdown_inputs(
        self,
        markdown_dir: Path,
        selected_cities_planned: list[str] | None,
        markdown_chunks: list[dict[str, object]],
        markdown_source_mode: str = "standard_chunking",
        analysis_mode: str = "aggregate",
    ) -> None:
        """Capture markdown input snapshot for reproducible run summaries.

        ``markdown_chunks`` is expected to contain one entry per chunk.
        ``markdown_source_mode`` identifies whether chunks came from standard
        file chunking or vector store retrieval.
        """
        planned_keys = _collect_city_keys(selected_cities_planned)
        scope_mode = "selected_cities" if planned_keys else "all_cities"
        found_keys = _collect_city_keys(
            [
                str(doc.get("city_name", "")).strip() or str(doc.get("city_key", "")).strip()
                for doc in markdown_chunks
            ]
        )
        file_count = len(
            {
                str(doc.get("path", "")).strip()
                for doc in markdown_chunks
                if str(doc.get("path", "")).strip()
            }
        )
        inputs = self.run_state.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
        inputs["city_scope_mode"] = scope_mode
        inputs["selected_cities_planned"] = planned_keys
        inputs["selected_cities_found"] = found_keys
        inputs["markdown_dir"] = str(markdown_dir)
        inputs["markdown_file_count"] = file_count
        inputs["markdown_chunk_count"] = len(markdown_chunks)
        inputs["markdown_excerpt_count"] = 0
        inputs["markdown_source_mode"] = markdown_source_mode
        inputs["analysis_mode"] = analysis_mode
        self.run_state["inputs"] = inputs
        self.write_api_state()
        planned_set = set(planned_keys)
        found_set = set(found_keys)
        missing_keys = sorted(planned_set - found_set)
        self.write_stage_detail(
            "markdown_inputs",
            {
                "inputs": {
                    "markdown_dir": str(markdown_dir),
                    "city_scope_mode": scope_mode,
                    "selected_cities_planned": planned_keys,
                    "markdown_source_mode": markdown_source_mode,
                    "analysis_mode": analysis_mode,
                },
                "outputs": {
                    "selected_cities_found": found_keys,
                    "missing_selected_cities": missing_keys,
                },
                "metrics": {
                    "markdown_file_count": file_count,
                    "markdown_chunk_count": len(markdown_chunks),
                    "selected_city_count": len(planned_keys),
                    "found_city_count": len(found_keys),
                    "missing_selected_city_count": len(missing_keys),
                },
            },
        )

    def finalize(
        self,
        status: str,
        final_output_path: Path | None = None,
        finish_reason: str | None = None,
    ) -> None:
        self.run_state["status"] = status
        self.run_state["completed_at"] = datetime.now(timezone.utc).isoformat()
        if finish_reason:
            self.run_state["finish_reason"] = finish_reason
        usage_summary = self._summarize_llm_usage()
        if usage_summary:
            self.run_state["llm_usage"] = usage_summary
            logger.info("LLM_USAGE_SUMMARY %s", json.dumps(usage_summary, ensure_ascii=False))
        retry_summary = self._summarize_retry_events()
        if retry_summary:
            self.run_state["retry_summary"] = retry_summary
            logger.info("RETRY_SUMMARY %s", json.dumps(retry_summary, ensure_ascii=False))
        self.run_state["artifacts"]["run_summary"] = str(self.run_paths.run_summary)
        error_log_path = self._write_error_log_artifact()
        if error_log_path is not None:
            self.run_state["artifacts"]["error_log"] = str(error_log_path)
            self.artifacts.register_file("error_log", error_log_path)
        if final_output_path:
            self.run_state["artifacts"]["final_output"] = str(final_output_path)
            self.artifacts.register_file(
                "final_output",
                final_output_path,
                artifact_type="runtime_state",
            )
            self.context_bundle["final"] = str(final_output_path)
            self.write_context_bundle()
        self.write_api_state()
        self.write_text_log()
        self.record_artifact("run_summary", self.run_paths.run_summary)
        self.write_stage_detail(
            "finalize",
            {
                "inputs": {},
                "outputs": {
                    "status": self.run_state.get("status"),
                    "finish_reason": self.run_state.get("finish_reason"),
                    "final_output": (
                        self._relative_path(final_output_path)
                        if final_output_path is not None
                        else None
                    ),
                    "run_summary": self._relative_path(self.run_paths.run_summary),
                    "error_log": (
                        self._relative_path(error_log_path)
                        if error_log_path is not None
                        else None
                    ),
                },
                "metrics": {
                    "llm_calls": (
                        usage_summary.get("calls")
                        if isinstance(usage_summary, dict)
                        else None
                    ),
                    "retry_events": (
                        retry_summary.get("total_events")
                        if isinstance(retry_summary, dict)
                        else None
                    ),
                    "retry_exhausted_events": (
                        retry_summary.get("exhausted_events")
                        if isinstance(retry_summary, dict)
                        else None
                    ),
                },
            },
        )
        manifest_path = self.artifacts.write_manifest(self._build_manifest_metadata())
        self.run_state["artifacts"]["manifest"] = str(manifest_path)
        self.write_api_state()


__all__ = ["RunLogger"]
