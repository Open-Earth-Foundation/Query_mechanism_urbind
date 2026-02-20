from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from backend.utils.paths import RunPaths

logger = logging.getLogger(__name__)


class RunLogger:
    def __init__(self, run_paths: RunPaths, question: str) -> None:
        self.run_paths = run_paths
        self.run_log: dict[str, Any] = {
            "run_id": run_paths.base_dir.name,
            "inputs": {
                "initial_question": question,
                "refined_question": question,
                "selected_cities_planned": [],
                "selected_cities_found": [],
                "markdown_dir": None,
                "markdown_file_count": 0,
                "markdown_chunk_count": 0,
                "markdown_excerpt_count": 0,
                "markdown_source_mode": "standard_chunking",
            },
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "decisions": [],
            "artifacts": {},
        }
        self.context_bundle: dict[str, Any] = {
            "sql": None,
            "markdown": None,
            "research_question": question,
            "final": None,
        }

        self._ensure_dirs()
        self.write_context_bundle()
        self.write_run_log()

    def _ensure_dirs(self) -> None:
        self.run_paths.base_dir.mkdir(parents=True, exist_ok=True)
        self.run_paths.sql_dir.mkdir(parents=True, exist_ok=True)
        self.run_paths.markdown_dir.mkdir(parents=True, exist_ok=True)

    def write_run_log(self) -> None:
        self.run_paths.run_log.write_text(
            json.dumps(self.run_log, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def write_context_bundle(self) -> None:
        self.run_paths.context_bundle.write_text(
            json.dumps(self.context_bundle, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
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

    def _summarize_sql_results(self, payload: object | None) -> str:
        if not isinstance(payload, list):
            return "(missing)"
        lines: list[str] = []
        for result in payload:
            if not isinstance(result, dict):
                continue
            query_id = result.get("query_id")
            row_count = result.get("row_count")
            columns = result.get("columns", [])
            rows = result.get("rows", [])
            error = None
            if columns == ["error"] and rows:
                error = rows[0][0]
            lines.append(
                f"- {query_id}: rows={row_count}, columns={columns}"
                + (f" | error={error}" if error else "")
            )
        return "\n".join(lines) if lines else "(empty)"

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

    def _format_total_runtime(self) -> str:
        """Return elapsed runtime in seconds from run start/end timestamps."""
        started_raw = self.run_log.get("started_at")
        completed_raw = self.run_log.get("completed_at")
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

    def write_text_log(self) -> None:
        lines: list[str] = []
        lines.append("RUN SUMMARY")
        lines.append(f"Run ID: {self.run_log.get('run_id')}")
        inputs = self.run_log.get("inputs", {})
        if isinstance(inputs, dict):
            lines.append(
                f"Question: {inputs.get('initial_question', '(missing)')}"
            )
            lines.append(f"Refined question: {inputs.get('refined_question', self.context_bundle.get('research_question'))}")
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
        lines.append(f"Status: {self.run_log.get('status')}")
        lines.append(f"Finish reason: {self.run_log.get('finish_reason', 'n/a')}")
        lines.append(f"Started: {self.run_log.get('started_at')}")
        lines.append(f"Completed: {self.run_log.get('completed_at')}")
        lines.append(f"Total runtime: {self._format_total_runtime()}")
        llm_usage = self.run_log.get("llm_usage")
        if llm_usage:
            lines.append(f"LLM Usage: {json.dumps(llm_usage, ensure_ascii=False)}")
        lines.append("")

        lines.append("ARTIFACTS")
        for key, value in self.run_log.get("artifacts", {}).items():
            lines.append(f"- {key}: {value}")
        lines.append("")

        lines.append("DECISIONS (LLM)")
        lines.append(self._format_json(self.run_log.get("decisions")))
        lines.append("")

        lines.append("CONTEXT_BUNDLE (LLM)")
        lines.append(self._format_json(self.context_bundle))
        lines.append("")

        schema_payload = self._read_json_file(self.run_paths.schema_summary)
        lines.append("SCHEMA_SUMMARY")
        lines.append(self._format_json(schema_payload))
        lines.append("")

        city_payload = self._read_json_file(self.run_paths.city_list)
        lines.append("CITY_LIST")
        lines.append(self._format_json(city_payload))
        lines.append("")

        sql_queries = self._read_json_file(self.run_paths.sql_queries)
        lines.append("SQL_QUERIES (LLM)")
        lines.append(self._format_json(sql_queries))
        lines.append("")

        sql_rounds_path = self.run_paths.sql_dir / "rounds.json"
        sql_rounds = self._read_json_file(sql_rounds_path)
        lines.append("SQL_ROUNDS (LLM)")
        lines.append(self._format_json(sql_rounds))
        lines.append("")

        sql_results = self._read_json_file(self.run_paths.sql_results)
        lines.append("SQL_RESULTS_SUMMARY")
        lines.append(self._summarize_sql_results(sql_results))
        lines.append("")

        if self.run_paths.sql_results.exists():
            lines.append("SQL_RESULTS (CAPPED)")
            lines.append(self._read_text_file(self.run_paths.sql_results))
            lines.append("")

        sql_results_full_path = self.run_paths.sql_results_full
        if sql_results_full_path.exists():
            lines.append("SQL_RESULTS_FULL")
            lines.append(self._read_text_file(sql_results_full_path))
            lines.append("")

        markdown_payload = self._read_json_file(self.run_paths.markdown_excerpts)
        lines.append("MARKDOWN_EXCERPTS (LLM)")
        lines.append(self._format_json(markdown_payload))
        lines.append("")
        lines.append("MARKDOWN_FAILURE_SUMMARY")
        lines.append(self._format_json(self._summarize_markdown_failures(markdown_payload)))
        lines.append("")

        final_output = self.run_log.get("artifacts", {}).get("final_output")
        lines.append("FINAL_OUTPUT (LLM)")
        if final_output:
            lines.append(self._read_text_file(Path(final_output)))
        else:
            lines.append("(none)")
        lines.append("")

        self.run_paths.run_summary.write_text("\n".join(lines), encoding="utf-8")

    def record_decision(self, decision: dict[str, Any]) -> None:
        self.run_log["decisions"].append(decision)
        self.write_run_log()

    def record_artifact(self, name: str, path: Path) -> None:
        self.run_log["artifacts"][name] = str(path)
        self.write_run_log()

    def update_sql_bundle(self, sql_payload: dict[str, Any]) -> None:
        self.context_bundle["sql"] = sql_payload
        self.write_context_bundle()

    def update_markdown_bundle(self, markdown_payload: dict[str, Any]) -> None:
        """Persist markdown payload and sync excerpt count in run inputs."""
        self.context_bundle["markdown"] = markdown_payload
        excerpt_count = markdown_payload.get("excerpt_count", 0)
        normalized_excerpt_count = excerpt_count if isinstance(excerpt_count, int) else 0
        inputs = self.run_log.get("inputs")
        if isinstance(inputs, dict):
            inputs["markdown_excerpt_count"] = normalized_excerpt_count
            self.run_log["inputs"] = inputs
            self.write_run_log()
        self.write_context_bundle()

    def update_research_question(self, research_question: str) -> None:
        self.context_bundle["research_question"] = research_question
        inputs = self.run_log.get("inputs")
        if isinstance(inputs, dict):
            inputs["refined_question"] = research_question
            self.run_log["inputs"] = inputs
            self.write_run_log()
        self.write_context_bundle()

    def record_markdown_inputs(
        self,
        markdown_dir: Path,
        selected_cities_planned: list[str] | None,
        markdown_chunks: list[dict[str, str]],
        markdown_source_mode: str = "standard_chunking",
    ) -> None:
        """Capture markdown input snapshot for reproducible run summaries.

        ``markdown_chunks`` is expected to contain one entry per chunk.
        ``markdown_source_mode`` identifies whether chunks came from standard
        file chunking or vector store retrieval.
        """
        planned = sorted(
            {
                city.strip()
                for city in (selected_cities_planned or [])
                if isinstance(city, str) and city.strip()
            }
        )
        found = sorted(
            {
                str(doc.get("city_name", "")).strip()
                for doc in markdown_chunks
                if str(doc.get("city_name", "")).strip()
            }
        )
        file_count = len(
            {
                str(doc.get("path", "")).strip()
                for doc in markdown_chunks
                if str(doc.get("path", "")).strip()
            }
        )
        inputs = self.run_log.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
        inputs["selected_cities_planned"] = planned
        inputs["selected_cities_found"] = found
        inputs["markdown_dir"] = str(markdown_dir)
        inputs["markdown_file_count"] = file_count
        inputs["markdown_chunk_count"] = len(markdown_chunks)
        inputs["markdown_excerpt_count"] = 0
        inputs["markdown_source_mode"] = markdown_source_mode
        self.run_log["inputs"] = inputs
        self.write_run_log()

    def finalize(
        self,
        status: str,
        final_output_path: Path | None = None,
        finish_reason: str | None = None,
    ) -> None:
        self.run_log["status"] = status
        self.run_log["completed_at"] = datetime.now(timezone.utc).isoformat()
        if finish_reason:
            self.run_log["finish_reason"] = finish_reason
        usage_summary = self._summarize_llm_usage()
        if usage_summary:
            self.run_log["llm_usage"] = usage_summary
            logger.info("LLM_USAGE_SUMMARY %s", json.dumps(usage_summary, ensure_ascii=False))
        self.run_log["artifacts"]["run_summary"] = str(self.run_paths.run_summary)
        if final_output_path:
            self.run_log["artifacts"]["final_output"] = str(final_output_path)
            self.context_bundle["final"] = str(final_output_path)
            self.write_context_bundle()
        self.write_run_log()
        self.write_text_log()


__all__ = ["RunLogger"]
