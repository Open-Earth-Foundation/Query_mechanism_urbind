"""Services for discovering and applying missing-data assumptions per run."""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from backend.api.models import AssumptionsPayload, MissingDataItem, RegenerationResult
from backend.api.services.context_chat import load_context_bundle, load_final_document
from backend.api.services.run_store import RunRecord, RunStore
from backend.modules.writer.agent import write_markdown
from backend.services.llm_observability import (
    LlmCallContext,
    LlmCallRecorder,
    record_openai_chat_completion,
)
from backend.services.mlflow_observability import sync_run_to_mlflow
from backend.utils.artifact_manifest import resolve_manifest_alias
from backend.utils.artifact_writer import ArtifactWriter, stage_file_dir_name
from backend.utils.config import AppConfig, get_openrouter_api_key
from backend.utils.json_io import read_json, read_json_object, write_json
from backend.utils.paths import RunPaths

logger = logging.getLogger(__name__)

_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


class _MissingDataEnvelope(BaseModel):
    """LLM-facing envelope for structured missing-data extraction."""

    items: list[MissingDataItem] = Field(default_factory=list)


class _ExistingRunMlflowLogger:
    """Small adapter for syncing artifacts added after pipeline finalization."""

    def __init__(self, run_paths: RunPaths) -> None:
        self.run_paths = run_paths
        self._artifacts = ArtifactWriter(run_paths.base_dir, run_paths.base_dir.name)

    def register_llm_calls_index(self, index_path: Path) -> None:
        """Add the LLM call index to the manifest before a forced MLflow sync."""
        self._artifacts.register_file(
            "llm_calls_index",
            index_path,
            artifact_type="runtime_state",
        )
        self._artifacts.write_manifest(_build_existing_manifest_metadata(self.run_paths))

    def record_mlflow_metadata(self, metadata: dict[str, Any]) -> None:
        """Persist MLflow sync metadata into an existing run's API state."""
        api_state = read_json_object(self.run_paths.api_state) or {}
        api_state["mlflow"] = metadata
        write_json(self.run_paths.api_state, api_state, ensure_ascii=False, default=str)
        self._artifacts.write_manifest(
            _build_existing_manifest_metadata(self.run_paths, mlflow_metadata=metadata)
        )


def _build_existing_run_paths(
    *,
    run_store: RunStore,
    run_record: RunRecord,
    config: AppConfig,
) -> RunPaths:
    """Return canonical paths for an existing finalized run directory."""
    run_dir = run_store.runs_dir / run_record.run_id
    return RunPaths(
        base_dir=run_dir,
        api_state=run_dir / "api_state.json",
        manifest=run_dir / "manifest.json",
        summary_events=run_dir / "summary.jsonl",
        stages_dir=run_dir / "stages",
        stage_files_dir=run_dir / "stage_files",
        run_summary=run_dir / "run_summary.txt",
        error_log=run_dir / "error_log.txt",
        context_bundle=run_record.context_bundle_path
        or run_dir / config.orchestrator.context_bundle_name,
        final_output=run_record.final_output_path or run_dir / "final.md",
    )


def _build_existing_manifest_metadata(
    run_paths: RunPaths,
    *,
    mlflow_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve existing manifest metadata and refresh values from API state."""
    manifest = read_json_object(run_paths.manifest) or {}
    raw_metadata = manifest.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    api_state = read_json_object(run_paths.api_state) or {}
    for key in (
        "status",
        "finish_reason",
        "llm_usage",
        "retry_summary",
        "writer_citation_coverage",
        "writer_multi_pass",
    ):
        value = api_state.get(key)
        if value is not None:
            metadata[key] = value
    if mlflow_metadata is not None:
        metadata["mlflow"] = mlflow_metadata
    return metadata


def _sync_assumptions_run_to_mlflow(
    *,
    run_store: RunStore,
    run_record: RunRecord,
    config: AppConfig,
    llm_recorder: LlmCallRecorder | None,
) -> None:
    """Upload post-finalization assumptions artifacts into the existing MLflow run."""
    if not config.mlflow.enabled:
        return
    run_paths = _build_existing_run_paths(
        run_store=run_store,
        run_record=run_record,
        config=config,
    )
    mlflow_logger = _ExistingRunMlflowLogger(run_paths)
    if llm_recorder is not None and llm_recorder.index_path.exists():
        mlflow_logger.register_llm_calls_index(llm_recorder.index_path)
    sync_run_to_mlflow(
        run_logger=mlflow_logger,
        config=config.mlflow,
        recorder=llm_recorder,
        force=True,
    )


def discover_missing_data(
    question: str,
    final_document: str,
    context_bundle: dict[str, Any],
    config: AppConfig,
    api_key_override: str | None = None,
    llm_recorder: LlmCallRecorder | None = None,
) -> dict[str, object]:
    """Run two LLM passes to extract and verify missing-data assumptions."""
    pass_one_items = _run_discovery_pass_optional_recorder(
        pass_name="extract",
        question=question,
        final_document=final_document,
        context_bundle=context_bundle,
        existing_items=[],
        config=config,
        api_key_override=api_key_override,
        llm_recorder=llm_recorder,
    )
    pass_one_deduped = dedupe_missing_data_items(pass_one_items)
    pass_two_items = _run_discovery_pass_optional_recorder(
        pass_name="verify",
        question=question,
        final_document=final_document,
        context_bundle=context_bundle,
        existing_items=pass_one_deduped,
        config=config,
        api_key_override=api_key_override,
        llm_recorder=llm_recorder,
    )
    merged_items = dedupe_missing_data_items(pass_one_deduped + pass_two_items)
    verification_summary = {
        "first_pass_count": len(pass_one_deduped),
        "second_pass_count": len(pass_two_items),
        "merged_count": len(merged_items),
        "added_in_verification": max(0, len(merged_items) - len(pass_one_deduped)),
    }
    return {
        "pass_1_items": [item.model_dump() for item in pass_one_deduped],
        "pass_2_items": [item.model_dump() for item in pass_two_items],
        "items": [item.model_dump() for item in merged_items],
        "verification_summary": verification_summary,
    }


def _run_discovery_pass_optional_recorder(
    *,
    pass_name: str,
    question: str,
    final_document: str,
    context_bundle: dict[str, Any],
    existing_items: list[MissingDataItem],
    config: AppConfig,
    api_key_override: str | None,
    llm_recorder: LlmCallRecorder | None,
) -> list[MissingDataItem]:
    """Call the discovery pass while preserving compatibility with test doubles."""
    kwargs: dict[str, object] = {
        "pass_name": pass_name,
        "question": question,
        "final_document": final_document,
        "context_bundle": context_bundle,
        "existing_items": existing_items,
        "config": config,
        "api_key_override": api_key_override,
    }
    if llm_recorder is not None:
        kwargs["llm_recorder"] = llm_recorder
    return _run_discovery_pass(**kwargs)


def discover_missing_data_for_run(
    run_store: RunStore,
    run_record: RunRecord,
    config: AppConfig,
    persist_artifacts: bool = False,
    api_key_override: str | None = None,
) -> dict[str, object]:
    """Discover missing data for one completed run and persist discovery artifact."""
    final_output_path = _resolve_final_output_path(
        run_store=run_store,
        run_id=run_record.run_id,
        raw_path=run_record.final_output_path,
    )
    context_bundle_path = _resolve_context_bundle_path(
        run_store=run_store,
        run_id=run_record.run_id,
        raw_path=run_record.context_bundle_path,
    )
    final_document = load_final_document(final_output_path)
    context_bundle = load_context_bundle(context_bundle_path)
    llm_recorder = (
        LlmCallRecorder(run_store.runs_dir / run_record.run_id, run_record.run_id)
        if config.mlflow.enabled
        else None
    )

    try:
        discovery_payload = discover_missing_data(
            question=run_record.question,
            final_document=final_document,
            context_bundle=context_bundle,
            config=config,
            api_key_override=api_key_override,
            llm_recorder=llm_recorder,
        )
    except Exception:
        _sync_assumptions_run_to_mlflow(
            run_store=run_store,
            run_record=run_record,
            config=config,
            llm_recorder=llm_recorder,
        )
        raise
    final_items = [
        MissingDataItem.model_validate(item)
        for item in discovery_payload.get("items", [])
        if isinstance(item, dict)
    ]
    grouped = group_missing_data_by_city(final_items)

    if persist_artifacts:
        writer = ArtifactWriter(run_store.runs_dir / run_record.run_id, run_record.run_id)
        persisted = {
            "run_id": run_record.run_id,
            **discovery_payload,
            "grouped_by_city": {
                city: [item.model_dump() for item in city_items]
                for city, city_items in grouped.items()
            },
        }
        writer.write_stage_file(
            "assumptions",
            "discovered.json",
            persisted,
            alias="assumptions_discovered",
        )
        writer.write_step_detail(
            "assumptions_discovery",
            {
                "inputs": {"run_id": run_record.run_id},
                "outputs": {"grouped_city_count": len(grouped)},
                "metrics": {"item_count": len(final_items)},
            },
        )
        if llm_recorder is not None and llm_recorder.index_path.exists():
            writer.register_file(
                "llm_calls_index",
                llm_recorder.index_path,
                artifact_type="runtime_state",
            )
        writer.write_manifest()
    _sync_assumptions_run_to_mlflow(
        run_store=run_store,
        run_record=run_record,
        config=config,
        llm_recorder=llm_recorder,
    )

    return {
        "run_id": run_record.run_id,
        "items": [item.model_dump() for item in final_items],
        "grouped_by_city": {
            city: [item.model_dump() for item in city_items]
            for city, city_items in grouped.items()
        },
        "verification_summary": discovery_payload.get("verification_summary", {}),
    }


def group_missing_data_by_city(
    items: list[MissingDataItem],
) -> dict[str, list[MissingDataItem]]:
    """Group missing-data items by city using deterministic ordering."""
    grouped: dict[str, list[MissingDataItem]] = {}
    for item in items:
        city = item.city.strip()
        if city not in grouped:
            grouped[city] = []
        grouped[city].append(item)
    return dict(sorted(grouped.items(), key=lambda pair: pair[0].lower()))


def apply_assumptions_to_context(
    context_bundle: dict[str, Any],
    payload: AssumptionsPayload,
) -> dict[str, object]:
    """Attach user-edited assumptions as a dedicated context bundle section."""
    revised_context = deepcopy(context_bundle)
    assumptions_block: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items": [item.model_dump() for item in payload.items],
    }
    if payload.rewrite_instructions:
        assumptions_block["rewrite_instructions"] = payload.rewrite_instructions.strip()

    if not isinstance(revised_context, dict):
        return {
            "source_context_bundle": context_bundle,
            "assumptions": assumptions_block,
        }
    revised_context["assumptions"] = assumptions_block
    return revised_context


def apply_assumptions_and_regenerate(
    run_store: RunStore,
    run_record: RunRecord,
    payload: AssumptionsPayload,
    config: AppConfig,
    persist_artifacts: bool = False,
    api_key_override: str | None = None,
) -> RegenerationResult:
    """Persist edited assumptions, regenerate document, and return revised output metadata."""
    context_bundle_path = _resolve_context_bundle_path(
        run_store=run_store,
        run_id=run_record.run_id,
        raw_path=run_record.context_bundle_path,
    )
    context_bundle = load_context_bundle(context_bundle_path)
    revised_context_bundle = apply_assumptions_to_context(context_bundle, payload)
    llm_recorder = (
        LlmCallRecorder(run_store.runs_dir / run_record.run_id, run_record.run_id)
        if config.mlflow.enabled
        else None
    )

    try:
        revised_document = rewrite_document_with_assumptions(
            original_question=run_record.question,
            assumptions_payload=payload,
            revised_context_bundle=revised_context_bundle,
            config=config,
            api_key_override=api_key_override,
            run_id=run_record.run_id,
            llm_recorder=llm_recorder,
        )
    except Exception:
        _sync_assumptions_run_to_mlflow(
            run_store=run_store,
            run_record=run_record,
            config=config,
            llm_recorder=llm_recorder,
        )
        raise
    rendered = f"# Question\n{run_record.question.strip()}\n\n{revised_document.strip()}\n"

    revised_output_path: str | None = None
    assumptions_path: str | None = None
    if persist_artifacts:
        run_dir = run_store.runs_dir / run_record.run_id
        writer = ArtifactWriter(run_dir, run_record.run_id)

        edited_path = writer.write_stage_file(
            "assumptions",
            "edited.json",
            {
                "run_id": run_record.run_id,
                "edited_at": datetime.now(timezone.utc).isoformat(),
                **payload.model_dump(),
            },
            alias="assumptions_edited",
        )
        revised_context_path = writer.write_stage_file(
            "assumptions",
            "revised_context_bundle.json",
            revised_context_bundle,
            alias="assumptions_revised_context_bundle",
        )

        revised_output_file_path = writer.stage_file_path(
            "assumptions",
            "final_with_assumptions.md",
        )
        revised_output_file_path.parent.mkdir(parents=True, exist_ok=True)
        revised_output_file_path.write_text(rendered, encoding="utf-8")
        writer.register_file(
            "assumptions_final_output",
            revised_output_file_path,
            artifact_type="stage_file",
        )
        writer.write_step_detail(
            "assumptions_apply",
            {
                "inputs": {"item_count": len(payload.items)},
                "outputs": {
                    "edited": edited_path.relative_to(run_dir).as_posix(),
                    "revised_context_bundle": revised_context_path.relative_to(run_dir).as_posix(),
                    "final_output": revised_output_file_path.relative_to(run_dir).as_posix(),
                },
                "metrics": {"revised_output_chars": len(rendered)},
            },
        )
        if llm_recorder is not None and llm_recorder.index_path.exists():
            writer.register_file(
                "llm_calls_index",
                llm_recorder.index_path,
                artifact_type="runtime_state",
            )
        writer.write_manifest()
        revised_output_path = str(revised_output_file_path)
        assumptions_path = str(edited_path)
    _sync_assumptions_run_to_mlflow(
        run_store=run_store,
        run_record=run_record,
        config=config,
        llm_recorder=llm_recorder,
    )

    return RegenerationResult(
        run_id=run_record.run_id,
        revised_output_path=revised_output_path,
        revised_content=rendered,
        assumptions_path=assumptions_path,
    )


def rewrite_document_with_assumptions(
    original_question: str,
    assumptions_payload: AssumptionsPayload,
    revised_context_bundle: dict[str, object],
    config: AppConfig,
    api_key_override: str | None = None,
    run_id: str | None = None,
    llm_recorder: LlmCallRecorder | None = None,
) -> str:
    """Generate revised document content grounded in user-edited assumptions."""
    api_key = _resolve_api_key(api_key_override)
    rewritten_question = _build_rewrite_question(
        original_question=original_question,
        assumptions_payload=assumptions_payload,
    )
    writer_output = write_markdown(
        question=rewritten_question,
        context_bundle=revised_context_bundle,
        config=config,
        api_key=api_key,
        log_llm_payload=False,
        run_id=run_id,
        llm_recorder=llm_recorder,
        llm_stage_name="assumptions_apply",
        llm_stage_family="assumptions",
        llm_agent_name="assumptions_apply_writer",
        llm_call_kind="apply_assumptions",
    )
    return writer_output.content.strip()


def load_latest_assumptions_payload(run_store: RunStore, run_id: str) -> dict[str, object]:
    """Load most recent assumptions artifacts for a run when available."""
    run_dir = run_store.runs_dir / run_id
    assumptions_stage_dir = run_dir / "stage_files" / stage_file_dir_name("assumptions")
    discovered_path = resolve_manifest_alias(run_dir, "assumptions_discovered") or (
        assumptions_stage_dir / "discovered.json"
    )
    edited_path = resolve_manifest_alias(run_dir, "assumptions_edited") or (
        assumptions_stage_dir / "edited.json"
    )
    revised_output_path = resolve_manifest_alias(run_dir, "assumptions_final_output") or (
        assumptions_stage_dir / "final_with_assumptions.md"
    )
    revised_context_path = resolve_manifest_alias(
        run_dir, "assumptions_revised_context_bundle"
    ) or (assumptions_stage_dir / "revised_context_bundle.json")

    payload: dict[str, object] = {"run_id": run_id}
    if discovered_path.exists():
        payload["discovered"] = read_json(discovered_path)
        payload["discovered_path"] = str(discovered_path)
    if edited_path.exists():
        payload["edited"] = read_json(edited_path)
        payload["assumptions_path"] = str(edited_path)
    if revised_context_path.exists():
        payload["revised_context_bundle_path"] = str(revised_context_path)
    if revised_output_path.exists():
        payload["revised_output_path"] = str(revised_output_path)
        payload["revised_content"] = revised_output_path.read_text(encoding="utf-8")
    return payload


def dedupe_missing_data_items(items: list[MissingDataItem]) -> list[MissingDataItem]:
    """De-duplicate items by city + description while preserving first-seen order."""
    deduped: list[MissingDataItem] = []
    index_by_key: dict[tuple[str, str], int] = {}
    for item in items:
        key = (_normalize_key(item.city), _normalize_key(item.missing_description))
        existing_index = index_by_key.get(key)
        if existing_index is None:
            index_by_key[key] = len(deduped)
            deduped.append(item)
            continue
        existing = deduped[existing_index]
        if existing.proposed_number is None and item.proposed_number is not None:
            deduped[existing_index] = item
    return deduped


def _run_discovery_pass(
    pass_name: str,
    question: str,
    final_document: str,
    context_bundle: dict[str, Any],
    existing_items: list[MissingDataItem],
    config: AppConfig,
    api_key_override: str | None = None,
    llm_recorder: LlmCallRecorder | None = None,
) -> list[MissingDataItem]:
    """Run one missing-data extraction pass and validate structured output."""
    api_key = _resolve_api_key(api_key_override)
    client = OpenAI(api_key=api_key, base_url=config.openrouter_base_url)
    messages = _build_discovery_messages(
        pass_name=pass_name,
        question=question,
        final_document=final_document,
        context_bundle=context_bundle,
        existing_items=existing_items,
    )
    request_kwargs: dict[str, object] = {
        "model": config.assumptions_reviewer.model,
        "messages": messages,
    }
    request_kwargs["temperature"] = float(config.assumptions_reviewer.temperature)
    if config.assumptions_reviewer.reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = config.assumptions_reviewer.reasoning_effort
    if config.assumptions_reviewer.max_output_tokens is not None:
        request_kwargs["max_tokens"] = config.assumptions_reviewer.max_output_tokens

    logger.info(
        "Assumptions discovery pass=%s model=%s existing_items=%d",
        pass_name,
        config.assumptions_reviewer.model,
        len(existing_items),
    )
    response = record_openai_chat_completion(
        client,
        request_kwargs,
        context=LlmCallContext(
            stage_name="assumptions_discovery",
            stage_family="assumptions",
            agent="assumptions_reviewer",
            call_kind=f"{pass_name}_missing_data_review",
            model=config.assumptions_reviewer.model,
            metadata={
                "pass_name": pass_name,
                "existing_item_count": len(existing_items),
                "final_document_chars": len(final_document),
            },
        ),
        recorder=llm_recorder,
    )
    if not response.choices:
        raise ValueError("Assumptions reviewer returned no choices.")
    content = _extract_message_text(response.choices[0].message.content)
    envelope = _parse_missing_data_envelope(content)
    return dedupe_missing_data_items(envelope.items)


def _build_discovery_messages(
    pass_name: str,
    question: str,
    final_document: str,
    context_bundle: dict[str, Any],
    existing_items: list[MissingDataItem],
) -> list[dict[str, str]]:
    """Build prompt messages for extraction/verification passes."""
    if pass_name not in {"extract", "verify"}:
        raise ValueError(f"Unsupported pass_name `{pass_name}`.")
    context_bundle_json = json.dumps(context_bundle, ensure_ascii=True, indent=2, default=str)
    existing_items_json = json.dumps(
        [item.model_dump() for item in existing_items],
        ensure_ascii=True,
        indent=2,
        default=str,
    )
    system_prompt = (
        "You are a strict structured-data extractor.\n"
        "Return JSON only in this exact envelope shape:\n"
        '{"items":[{"city":"...","missing_description":"...","proposed_number":123}]}\n'
        "Rules:\n"
        "1. Output only fields: city, missing_description, proposed_number.\n"
        "2. Do not output run_id, grouped fields, paths, status, or explanations.\n"
        "3. proposed_number may be a number, short free-text assumption, or null.\n"
        "4. Keep one missing fact per item.\n"
        "5. Focus on city-level quantitative gaps needed for actionable recommendations.\n"
    )
    if pass_name == "extract":
        user_prompt = (
            "Pass 1: Extract missing quantitative data assumptions.\n\n"
            f"Question:\n{question.strip()}\n\n"
            "Final document:\n"
            "```markdown\n"
            f"{final_document.strip()}\n"
            "```\n\n"
            "Context bundle:\n"
            "```json\n"
            f"{context_bundle_json}\n"
            "```\n"
        )
    else:
        user_prompt = (
            "Pass 2: Verify pass-1 coverage.\n"
            "Return only additional missing items not already present.\n\n"
            f"Question:\n{question.strip()}\n\n"
            "Existing pass-1 items:\n"
            "```json\n"
            f"{existing_items_json}\n"
            "```\n\n"
            "Final document:\n"
            "```markdown\n"
            f"{final_document.strip()}\n"
            "```\n\n"
            "Context bundle:\n"
            "```json\n"
            f"{context_bundle_json}\n"
            "```\n"
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_rewrite_question(
    original_question: str,
    assumptions_payload: AssumptionsPayload,
) -> str:
    """Build writer question including assumptions and regeneration rules."""
    assumptions_json = json.dumps(
        [item.model_dump() for item in assumptions_payload.items],
        ensure_ascii=True,
        indent=2,
    )
    instructions = (
        assumptions_payload.rewrite_instructions.strip()
        if assumptions_payload.rewrite_instructions
        else "No additional rewrite instructions were provided."
    )
    return (
        f"{original_question.strip()}\n\n"
        "Regeneration instructions:\n"
        "1. Explicitly state which data points were missing.\n"
        "2. Explicitly list the assumptions used to fill gaps.\n"
        "3. Use provided assumptions consistently in recommendations.\n"
        "4. Keep uncertain assumptions clearly labeled as assumptions.\n\n"
        f"User rewrite instructions:\n{instructions}\n\n"
        "Approved assumptions:\n"
        "```json\n"
        f"{assumptions_json}\n"
        "```\n"
    )


def _extract_message_text(content: Any) -> str:
    """Extract plain text content from OpenAI chat message payload variants."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks).strip()
    return str(content).strip()


def _parse_missing_data_envelope(raw_text: str) -> _MissingDataEnvelope:
    """Parse and validate model output into the missing-data envelope."""
    candidate = _extract_json_candidate(raw_text)
    parsed = json.loads(candidate)
    if isinstance(parsed, list):
        return _MissingDataEnvelope.model_validate({"items": parsed})
    if isinstance(parsed, dict):
        if "items" in parsed:
            return _MissingDataEnvelope.model_validate(parsed)
        legacy_items = parsed.get("missing_data_items")
        if isinstance(legacy_items, list):
            return _MissingDataEnvelope.model_validate({"items": legacy_items})
    raise ValueError("Assumptions reviewer returned unsupported JSON structure.")


def _extract_json_candidate(raw_text: str) -> str:
    """Extract best JSON candidate from model response text."""
    stripped = raw_text.strip()
    if not stripped:
        return '{"items":[]}'

    fence_match = _JSON_FENCE_PATTERN.search(stripped)
    if fence_match:
        fenced = fence_match.group(1).strip()
        if fenced:
            return fenced

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        return stripped[first_brace : last_brace + 1]

    first_bracket = stripped.find("[")
    last_bracket = stripped.rfind("]")
    if first_bracket >= 0 and last_bracket > first_bracket:
        return stripped[first_bracket : last_bracket + 1]

    return stripped


def _normalize_key(value: str) -> str:
    """Normalize free-text values for deterministic de-duplication."""
    collapsed = _WHITESPACE_PATTERN.sub(" ", value.strip().lower())
    return collapsed


def _resolve_api_key(api_key_override: str | None) -> str:
    """Resolve OpenRouter API key from override or environment configuration."""
    if isinstance(api_key_override, str) and api_key_override.strip():
        return api_key_override.strip()
    return get_openrouter_api_key()


def _resolve_final_output_path(
    run_store: RunStore,
    run_id: str,
    raw_path: Path | None,
) -> Path:
    """Resolve final output path and ensure artifact exists."""
    run_dir = run_store.runs_dir / run_id
    candidates: list[Path] = []
    if raw_path is not None:
        candidates.append(raw_path)
        candidates.append(run_dir / raw_path.name)
    candidates.append(run_dir / "final.md")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ValueError(f"Final output is missing for run `{run_id}`.")


def _resolve_context_bundle_path(
    run_store: RunStore,
    run_id: str,
    raw_path: Path | None,
) -> Path:
    """Resolve context bundle path and ensure artifact exists."""
    run_dir = run_store.runs_dir / run_id
    candidates: list[Path] = []
    if raw_path is not None:
        candidates.append(raw_path)
        candidates.append(run_dir / raw_path.name)
    candidates.append(run_dir / "context_bundle.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ValueError(f"Context bundle is missing for run `{run_id}`.")
__all__ = [
    "apply_assumptions_and_regenerate",
    "apply_assumptions_to_context",
    "dedupe_missing_data_items",
    "discover_missing_data",
    "discover_missing_data_for_run",
    "group_missing_data_by_city",
    "load_latest_assumptions_payload",
    "rewrite_document_with_assumptions",
]
