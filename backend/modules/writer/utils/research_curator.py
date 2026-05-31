"""Optional writer research curator orchestration."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agents import Agent, function_tool
from agents.exceptions import MaxTurnsExceeded
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel

from backend.modules.writer.models import WriterEvidenceSelection, WriterSavedEvidence
from backend.modules.writer.utils.markdown_helpers import extract_selected_city_names
from backend.modules.writer.utils.multi_pass import build_writer_payload
from backend.modules.writer.utils.research_context import (
    apply_saved_evidence_to_context,
    build_writer_context_index,
    build_writer_references_payload,
)
from backend.modules.writer.utils.research_session import (
    WriterResearchSession,
    WriterResearchToolError,
    build_writer_research_limits,
)
from backend.services.agents import (
    build_model_settings,
    build_openrouter_model,
    run_agent_sync,
)
from backend.services.run_logger import RunLogger
from backend.utils.config import AppConfig
from backend.utils.json_io import write_json
from backend.utils.paths import RunPaths
from backend.utils.prompts import load_prompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WriterResearchCuratorResult:
    """Curator result and the context bundle to send to the writer."""

    context_bundle: dict[str, object]
    saved_evidence: list[WriterSavedEvidence]
    status: str


def run_writer_research_curator(
    *,
    question: str,
    context_bundle: dict[str, object],
    analysis_mode: str,
    selected_city_names: list[str],
    config: AppConfig,
    api_key: str,
    run_id: str | None,
    paths: RunPaths | None,
    run_logger: RunLogger | None,
    log_llm_payload: bool,
) -> WriterResearchCuratorResult:
    """Run the optional evidence-prep curator and return writer-ready context."""
    if not config.writer.evidence_curator_enabled:
        return WriterResearchCuratorResult(
            context_bundle=context_bundle,
            saved_evidence=[],
            status="disabled",
        )
    if paths is None or run_id is None:
        return WriterResearchCuratorResult(
            context_bundle=context_bundle,
            saved_evidence=[],
            status="skipped_no_run_artifacts",
        )

    artifact_dir = paths.base_dir / "writer"
    session = _build_research_session(
        context_bundle=context_bundle,
        config=config,
        run_id=run_id,
        paths=paths,
        artifact_dir=artifact_dir,
    )
    session.write_state()
    if not session.items:
        _persist_curator_artifacts(
            session=session,
            selection=None,
            status="empty_context",
            paths=paths,
            run_logger=run_logger,
        )
        return WriterResearchCuratorResult(
            context_bundle=context_bundle,
            saved_evidence=[],
            status="empty_context",
        )

    try:
        agent = build_writer_research_curator_agent(
            config=config,
            api_key=api_key,
            session=session,
        )
        payload = _build_curator_payload(
            question=question,
            context_bundle=context_bundle,
            analysis_mode=analysis_mode,
            selected_city_names=selected_city_names,
            session=session,
        )
        result = run_agent_sync(
            agent,
            json.dumps(payload, ensure_ascii=False),
            max_turns=config.writer.evidence_curator_max_turns,
            log_llm_payload=log_llm_payload,
        )
        selection = _coerce_selection(result.final_output)
        saved_evidence = session.saved_evidence()
        status = selection.status if selection is not None else "completed"
    except (APIConnectionError, APIStatusError, APITimeoutError, MaxTurnsExceeded, RuntimeError, ValueError) as exc:
        logger.warning("Writer research curator failed; falling back to excerpts: %s", exc)
        _persist_curator_artifacts(
            session=session,
            selection=None,
            status="failed",
            paths=paths,
            run_logger=run_logger,
            error=str(exc),
        )
        return WriterResearchCuratorResult(
            context_bundle=context_bundle,
            saved_evidence=[],
            status="failed",
        )

    if not saved_evidence:
        _persist_curator_artifacts(
            session=session,
            selection=selection,
            status="empty_saved_evidence",
            paths=paths,
            run_logger=run_logger,
        )
        return WriterResearchCuratorResult(
            context_bundle=context_bundle,
            saved_evidence=[],
            status="empty_saved_evidence",
        )

    curated_context = apply_saved_evidence_to_context(
        context_bundle=context_bundle,
        saved_evidence=saved_evidence,
    )
    _persist_curator_artifacts(
        session=session,
        selection=selection,
        status=status,
        paths=paths,
        run_logger=run_logger,
    )
    write_json(
        artifact_dir / "references.json",
        build_writer_references_payload(run_id=run_id, saved_evidence=saved_evidence),
        ensure_ascii=False,
    )
    if run_logger is not None:
        run_logger.record_artifact("writer_references", artifact_dir / "references.json")
    return WriterResearchCuratorResult(
        context_bundle=curated_context,
        saved_evidence=saved_evidence,
        status=status,
    )


def build_writer_research_curator_agent(
    *,
    config: AppConfig,
    api_key: str,
    session: WriterResearchSession,
) -> Agent:
    """Build the tool-using writer research-curator agent."""
    instructions = load_prompt(_resolve_writer_research_curator_prompt_path())
    model = build_openrouter_model(
        config.writer.model,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )
    settings = build_model_settings(
        config.writer.temperature,
        config.writer.max_output_tokens,
        reasoning_effort=config.writer.reasoning_effort,
    )

    @function_tool
    def list_context_sources(
        cities: list[str] | None = None,
        source_kinds: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> dict[str, object]:
        """List writer-visible context sources matching optional filters."""
        return _safe_tool_result(
            "sources",
            lambda: session.list_context_sources(
                cities=cities,
                source_kinds=source_kinds,
                fields=fields,
            ),
        )

    @function_tool
    def regex_search_context(
        pattern: str,
        cities: list[str] | None = None,
        source_kinds: list[str] | None = None,
        fields: list[str] | None = None,
        case_sensitive: bool = False,
        context_words: int | None = None,
        max_matches: int | None = None,
    ) -> dict[str, object]:
        """Search writer-visible context with a bounded safe regex."""
        return _safe_tool_result(
            "hits",
            lambda: session.regex_search_context(
                pattern=pattern,
                cities=cities,
                source_kinds=source_kinds,
                fields=fields,
                case_sensitive=case_sensitive,
                context_words=context_words,
                max_matches=max_matches,
            ),
        )

    @function_tool
    def expand_context_hits(
        hit_ids: list[str],
        context_words: int | None = None,
    ) -> dict[str, object]:
        """Expand previously returned context hits."""
        return _safe_tool_result(
            "hits",
            lambda: session.expand_context_hits(
                hit_ids=hit_ids,
                context_words=context_words,
            ),
        )

    @function_tool
    def save_context_evidence(
        hit_ids: list[str],
        reason: str,
    ) -> dict[str, object]:
        """Save useful context hits as citation-compatible writer evidence."""
        return _safe_tool_result(
            "saved_evidence",
            lambda: session.save_context_evidence(hit_ids=hit_ids, reason=reason),
        )

    @function_tool
    def list_saved_context_evidence() -> dict[str, object]:
        """List saved writer context evidence."""
        return _safe_tool_result(
            "saved_evidence",
            session.list_saved_context_evidence,
        )

    @function_tool
    def mark_context_evidence_missing(
        reason: str,
        city_name: str = "",
        field: str = "",
        source_kind: str | None = None,
        searched_patterns: list[str] | None = None,
    ) -> dict[str, object]:
        """Record that useful evidence was searched for but not found."""
        return _safe_tool_result(
            "missing_record",
            lambda: session.mark_context_evidence_missing(
                reason=reason,
                city_name=city_name,
                field=field,
                source_kind=source_kind,
                searched_patterns=searched_patterns,
            ),
        )

    return Agent(
        name="WriterResearchCurator",
        instructions=instructions,
        model=model,
        model_settings=settings,
        tools=[
            list_context_sources,
            regex_search_context,
            expand_context_hits,
            save_context_evidence,
            list_saved_context_evidence,
            mark_context_evidence_missing,
        ],
        output_type=WriterEvidenceSelection,
    )


def _build_research_session(
    *,
    context_bundle: dict[str, object],
    config: AppConfig,
    run_id: str,
    paths: RunPaths,
    artifact_dir: Path,
) -> WriterResearchSession:
    """Build a search session from writer-safe context."""
    index = build_writer_context_index(
        context_bundle=context_bundle,
        run_dir=paths.base_dir,
        markdown_dir=config.markdown_dir,
        config=config,
        use_source_chunks=config.writer.evidence_curator_use_source_chunks,
    )
    return WriterResearchSession(
        run_id=run_id,
        items=index.items,
        limits=build_writer_research_limits(config),
        artifact_dir=artifact_dir,
        initial_missing_records=index.missing_records,
    )


def _build_curator_payload(
    *,
    question: str,
    context_bundle: dict[str, object],
    analysis_mode: str,
    selected_city_names: list[str],
    session: WriterResearchSession,
) -> dict[str, object]:
    """Build the runtime payload sent to the curator."""
    markdown_bundle = context_bundle.get("markdown")
    extracted_city_names = (
        extract_selected_city_names(context_bundle, markdown_bundle)
        if isinstance(markdown_bundle, dict)
        else selected_city_names
    )
    base_payload = build_writer_payload(
        question=question,
        context_bundle={
            "research_question": context_bundle.get("research_question"),
            "analysis_mode": context_bundle.get("analysis_mode"),
            "selected_cities": selected_city_names,
        },
        analysis_mode=analysis_mode,
        selected_city_names=selected_city_names,
    )
    return {
        "question": question,
        "analysis_mode": analysis_mode,
        "selected_cities": selected_city_names or extracted_city_names,
        "writer_payload_summary": {
            "question": base_payload["question"],
            "analysis_mode": base_payload["analysis_mode"],
            "selected_cities": base_payload["selected_cities"],
        },
        "context_source_summary": [
            summary.model_dump() for summary in session.source_summaries()
        ],
        "limits": {
            "max_saved_items": session.limits.max_saved_items,
            "max_regex_searches": session.limits.max_regex_searches,
            "max_matches_per_search": session.limits.max_matches_per_search,
        },
    }


def _persist_curator_artifacts(
    *,
    session: WriterResearchSession,
    selection: WriterEvidenceSelection | None,
    status: str,
    paths: RunPaths,
    run_logger: RunLogger | None,
    error: str | None = None,
) -> None:
    """Persist curator workspace, saved evidence, and run-log diagnostics."""
    artifact_dir = paths.base_dir / "writer"
    payload = session.saved_evidence_payload()
    payload["curator_status"] = status
    if selection is not None:
        payload["curator_selection"] = selection.model_dump()
    if error:
        payload["error"] = error
    write_json(artifact_dir / "saved_evidence.json", payload, ensure_ascii=False)
    write_json(artifact_dir / "evidence_workspace.json", session.workspace_payload(), ensure_ascii=False)
    if run_logger is not None:
        run_logger.record_writer_saved_evidence(payload)
        run_logger.record_artifact("writer_saved_evidence", artifact_dir / "saved_evidence.json")
        run_logger.record_artifact(
            "writer_evidence_workspace",
            artifact_dir / "evidence_workspace.json",
        )


def _safe_tool_result(
    key: str,
    callback: Callable[[], object],
) -> dict[str, object]:
    """Run a tool callback and convert structured errors into tool payloads."""
    try:
        result = callback()
    except WriterResearchToolError as exc:
        return exc.to_dict()
    if isinstance(result, list):
        return {
            key: [
                item.model_dump() if isinstance(item, BaseModel) else item
                for item in result
            ]
        }
    if isinstance(result, BaseModel):
        return {key: result.model_dump()}
    return {key: result}


def _coerce_selection(value: object) -> WriterEvidenceSelection | None:
    """Coerce an agent final output into the curator summary model."""
    if isinstance(value, WriterEvidenceSelection):
        return value
    if isinstance(value, dict):
        return WriterEvidenceSelection.model_validate(value)
    return None


def _resolve_writer_research_curator_prompt_path() -> Path:
    """Resolve the writer research-curator prompt path."""
    return Path(__file__).resolve().parents[3] / "prompts" / "writer_research_curator_system.md"


__all__ = [
    "WriterResearchCuratorResult",
    "build_writer_research_curator_agent",
    "run_writer_research_curator",
]
