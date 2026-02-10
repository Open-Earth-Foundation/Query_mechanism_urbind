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

from app.api.models import AssumptionsPayload, MissingDataItem, RegenerationResult
from app.api.services.context_chat import load_context_bundle, load_final_document
from app.api.services.run_store import RunRecord, RunStore
from app.modules.writer.agent import write_markdown
from app.utils.config import AppConfig, get_openrouter_api_key

logger = logging.getLogger(__name__)

_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


class _MissingDataEnvelope(BaseModel):
    """LLM-facing envelope for structured missing-data extraction."""

    items: list[MissingDataItem] = Field(default_factory=list)


def discover_missing_data(
    question: str,
    final_document: str,
    context_bundle: dict[str, Any],
    config: AppConfig,
    api_key_override: str | None = None,
) -> dict[str, object]:
    """Run two LLM passes to extract and verify missing-data assumptions."""
    pass_one_items = _run_discovery_pass(
        pass_name="extract",
        question=question,
        final_document=final_document,
        context_bundle=context_bundle,
        existing_items=[],
        config=config,
        api_key_override=api_key_override,
    )
    pass_one_deduped = dedupe_missing_data_items(pass_one_items)
    pass_two_items = _run_discovery_pass(
        pass_name="verify",
        question=question,
        final_document=final_document,
        context_bundle=context_bundle,
        existing_items=pass_one_deduped,
        config=config,
        api_key_override=api_key_override,
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

    discovery_payload = discover_missing_data(
        question=run_record.question,
        final_document=final_document,
        context_bundle=context_bundle,
        config=config,
        api_key_override=api_key_override,
    )
    final_items = [
        MissingDataItem.model_validate(item)
        for item in discovery_payload.get("items", [])
        if isinstance(item, dict)
    ]
    grouped = group_missing_data_by_city(final_items)

    if persist_artifacts:
        assumptions_dir = _assumptions_dir(run_store, run_record.run_id)
        assumptions_dir.mkdir(parents=True, exist_ok=True)
        discovered_path = assumptions_dir / "discovered.json"
        persisted = {
            "run_id": run_record.run_id,
            **discovery_payload,
            "grouped_by_city": {
                city: [item.model_dump() for item in city_items]
                for city, city_items in grouped.items()
            },
        }
        _write_json(discovered_path, persisted)
        _update_run_log_artifacts(
            run_store.runs_dir / run_record.run_id / "run.json",
            {"assumptions_discovered": str(discovered_path)},
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

    revised_document = rewrite_document_with_assumptions(
        original_question=run_record.question,
        assumptions_payload=payload,
        revised_context_bundle=revised_context_bundle,
        config=config,
        api_key_override=api_key_override,
    )
    rendered = f"# Question\n{run_record.question.strip()}\n\n{revised_document.strip()}\n"

    revised_output_path: str | None = None
    assumptions_path: str | None = None
    if persist_artifacts:
        assumptions_dir = _assumptions_dir(run_store, run_record.run_id)
        assumptions_dir.mkdir(parents=True, exist_ok=True)

        edited_path = assumptions_dir / "edited.json"
        _write_json(
            edited_path,
            {
                "run_id": run_record.run_id,
                "edited_at": datetime.now(timezone.utc).isoformat(),
                **payload.model_dump(),
            },
        )
        revised_context_path = assumptions_dir / "revised_context_bundle.json"
        _write_json(revised_context_path, revised_context_bundle)

        revised_output_file_path = assumptions_dir / "final_with_assumptions.md"
        revised_output_file_path.write_text(rendered, encoding="utf-8")
        revised_output_path = str(revised_output_file_path)
        assumptions_path = str(edited_path)

        _update_run_log_artifacts(
            run_store.runs_dir / run_record.run_id / "run.json",
            {
                "assumptions_edited": assumptions_path,
                "assumptions_revised_context_bundle": str(revised_context_path),
                "assumptions_final_output": revised_output_path,
            },
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
    )
    return writer_output.content.strip()


def load_latest_assumptions_payload(run_store: RunStore, run_id: str) -> dict[str, object]:
    """Load most recent assumptions artifacts for a run when available."""
    assumptions_dir = _assumptions_dir(run_store, run_id)
    discovered_path = assumptions_dir / "discovered.json"
    edited_path = assumptions_dir / "edited.json"
    revised_output_path = assumptions_dir / "final_with_assumptions.md"
    revised_context_path = assumptions_dir / "revised_context_bundle.json"

    payload: dict[str, object] = {"run_id": run_id}
    if discovered_path.exists():
        payload["discovered"] = _read_json(discovered_path)
        payload["discovered_path"] = str(discovered_path)
    if edited_path.exists():
        payload["edited"] = _read_json(edited_path)
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
    if config.assumptions_reviewer.temperature is not None:
        request_kwargs["temperature"] = config.assumptions_reviewer.temperature
    if config.assumptions_reviewer.max_output_tokens is not None:
        request_kwargs["max_tokens"] = config.assumptions_reviewer.max_output_tokens

    logger.info(
        "Assumptions discovery pass=%s model=%s existing_items=%d",
        pass_name,
        config.assumptions_reviewer.model,
        len(existing_items),
    )
    response = client.chat.completions.create(**request_kwargs)
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
    candidate = raw_path if raw_path is not None else run_store.runs_dir / run_id / "final.md"
    if not candidate.exists():
        raise ValueError(f"Final output is missing for run `{run_id}`.")
    return candidate


def _resolve_context_bundle_path(
    run_store: RunStore,
    run_id: str,
    raw_path: Path | None,
) -> Path:
    """Resolve context bundle path and ensure artifact exists."""
    candidate = (
        raw_path
        if raw_path is not None
        else run_store.runs_dir / run_id / "context_bundle.json"
    )
    if not candidate.exists():
        raise ValueError(f"Context bundle is missing for run `{run_id}`.")
    return candidate


def _assumptions_dir(run_store: RunStore, run_id: str) -> Path:
    """Return assumptions artifact directory for a run."""
    return run_store.runs_dir / run_id / "assumptions"


def _read_json(path: Path) -> object | None:
    """Read JSON file content, returning None on parse errors."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: object) -> None:
    """Write JSON payload to disk with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )


def _update_run_log_artifacts(run_log_path: Path, updates: dict[str, str]) -> None:
    """Update run log artifact mapping with additional assumptions artifacts."""
    if not run_log_path.exists():
        return
    payload = _read_json(run_log_path)
    if not isinstance(payload, dict):
        return
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        payload["artifacts"] = artifacts
    for key, value in updates.items():
        artifacts[key] = value
    _write_json(run_log_path, payload)


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
