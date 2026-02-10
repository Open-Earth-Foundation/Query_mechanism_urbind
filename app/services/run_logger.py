from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from app.utils.paths import RunPaths

logger = logging.getLogger(__name__)


class RunLogger:
    def __init__(self, run_paths: RunPaths, question: str) -> None:
        self.run_paths = run_paths
        self.run_log: dict[str, Any] = {
            "run_id": run_paths.base_dir.name,
            "question": question,
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "decisions": [],
            "artifacts": {},
            "drafts": [],
        }
        self.context_bundle: dict[str, Any] = {
            "sql": None,
            "markdown": None,
            "research_question": question,
            "drafts": [],
            "final": None,
        }

        self._ensure_dirs()
        self.write_context_bundle()
        self.write_run_log()

    def _ensure_dirs(self) -> None:
        self.run_paths.base_dir.mkdir(parents=True, exist_ok=True)
        self.run_paths.sql_dir.mkdir(parents=True, exist_ok=True)
        self.run_paths.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.run_paths.drafts_dir.mkdir(parents=True, exist_ok=True)

    def write_run_log(self) -> None:
        self.run_paths.run_log.write_text(
            json.dumps(self.run_log, indent=2, ensure_ascii=True, default=str),
            encoding="utf-8",
        )

    def write_context_bundle(self) -> None:
        self.run_paths.context_bundle.write_text(
            json.dumps(self.context_bundle, indent=2, ensure_ascii=True, default=str),
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
        return json.dumps(payload, indent=2, ensure_ascii=True, default=str)

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
        lines.append(f"Question: {self.run_log.get('question')}")
        lines.append(f"Status: {self.run_log.get('status')}")
        lines.append(f"Finish reason: {self.run_log.get('finish_reason', 'n/a')}")
        lines.append(f"Started: {self.run_log.get('started_at')}")
        lines.append(f"Completed: {self.run_log.get('completed_at')}")
        llm_usage = self.run_log.get("llm_usage")
        if llm_usage:
            lines.append(f"LLM Usage: {json.dumps(llm_usage, ensure_ascii=True)}")
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

        drafts = self.run_log.get("drafts", [])
        lines.append("DRAFTS (LLM)")
        if drafts:
            for draft in drafts:
                lines.append(f"--- Draft: {draft} ---")
                lines.append(self._read_text_file(Path(draft)))
        else:
            lines.append("(none)")
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

    def record_draft(self, path: Path) -> None:
        self.run_log["drafts"].append(str(path))
        self.context_bundle["drafts"].append(str(path))
        self.write_run_log()
        self.write_context_bundle()

    def update_sql_bundle(self, sql_payload: dict[str, Any]) -> None:
        self.context_bundle["sql"] = sql_payload
        self.write_context_bundle()

    def update_markdown_bundle(self, markdown_payload: dict[str, Any]) -> None:
        self.context_bundle["markdown"] = markdown_payload
        self.write_context_bundle()

    def update_research_question(self, research_question: str) -> None:
        self.context_bundle["research_question"] = research_question
        self.write_context_bundle()

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
            logger.info("LLM_USAGE_SUMMARY %s", json.dumps(usage_summary, ensure_ascii=True))
        self.run_log["artifacts"]["run_summary"] = str(self.run_paths.run_summary)
        if final_output_path:
            self.run_log["artifacts"]["final_output"] = str(final_output_path)
            self.context_bundle["final"] = str(final_output_path)
            self.write_context_bundle()
        self.write_run_log()
        self.write_text_log()


__all__ = ["RunLogger"]
