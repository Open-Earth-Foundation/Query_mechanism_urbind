from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any

from backend.models import ErrorInfo
from backend.modules.initiative_extractor.models import (
    InitiativeDocumentSegment,
    InitiativeExtraction,
    InitiativeExtractionCandidate,
    InitiativeExtractionRecord,
    InitiativeExtractionRunResult,
    InitiativeRawSegmentResult,
    InitiativeReviewItem,
    InitiativeSemanticDedupeGroup,
    InitiativeSemanticDedupeResult,
    InitiativeSegmentExtraction,
    InitiativeSegmentStop,
    InitiativeSourceRef,
    JsonValue,
)
from backend.modules.initiative_extractor.segmentation import (
    build_document_segments,
    detect_source_quality_flags,
)
from backend.utils.city_normalization import normalize_city_key
from backend.utils.config import AppConfig
from backend.utils.json_io import write_json
from backend.utils.markdown_files import list_markdown_files
from backend.utils.prompts import load_prompt
from backend.utils.retry import RetrySettings, compute_retry_delay_seconds, log_retry_event
from backend.utils.tokenization import count_tokens

logger = logging.getLogger(__name__)
_thread_local = threading.local()
RETRYABLE_ERROR_NAMES = {
    "APIConnectionError",
    "MaxTurnsExceeded",
    "ModelBehaviorError",
}
CANDIDATE_METADATA_FIELDS = {
    "document_local_code",
    "source_quote",
    "source_refs",
    "data_quality_flags",
    "number_context",
    "number_deferred",
    "number_uncertain",
    "extraction_notes",
}
SOURCE_QUOTE_FLAGS = {"source_quote_missing", "source_quote_not_found"}
CITY_OVERRIDDEN_FLAG = "city_overridden_from_segment"
LOCAL_CODE_PATTERN = re.compile(
    r"\b([A-Z]{1,6}(?:[-.][A-Z0-9]{1,6})?[-.]\d+(?:[.-]\d+)*[A-Z]?)\b"
)


def run_agent_sync(*args: Any, **kwargs: Any) -> Any:
    """Lazy wrapper so schema/unit tests do not require the Agents SDK at import time."""
    from backend.services.agents import run_agent_sync as run_sync

    return run_sync(*args, **kwargs)


def _write_jsonl(path: Path, rows: list[object]) -> None:
    """Write JSONL rows with stable UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in rows:
        if hasattr(row, "model_dump"):
            payload = row.model_dump(mode="json")
        else:
            payload = row
        lines.append(json.dumps(payload, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _canonical_rows(records: list[InitiativeExtractionRecord]) -> list[InitiativeExtraction]:
    """Return v1 canonical initiative objects for public extraction artifacts."""
    return [record.initiative for record in records]


def _discover_markdown_files(
    markdown_path: Path,
    selected_cities: list[str] | None,
    config: AppConfig,
) -> list[Path]:
    """Discover selected markdown documents under a file or directory path."""
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown path not found: {markdown_path}")
    files = [markdown_path] if markdown_path.is_file() else list_markdown_files(markdown_path)
    if selected_cities:
        requested = {normalize_city_key(city) for city in selected_cities}
        files = [path for path in files if normalize_city_key(path.stem) in requested]
    selected_files: list[Path] = []
    for path in files:
        if path.stat().st_size <= config.initiative_extractor.max_file_bytes:
            selected_files.append(path)
        else:
            logger.warning("Skipping oversized markdown file during initiative extraction: %s", path)
    if len(selected_files) > config.initiative_extractor.max_files:
        logger.warning(
            "Limiting initiative extraction to the first %d markdown files.",
            config.initiative_extractor.max_files,
        )
    return selected_files[: config.initiative_extractor.max_files]


def _is_retryable_error(exc: Exception) -> bool:
    """Return whether an extractor failure should be retried."""
    return type(exc).__name__ in RETRYABLE_ERROR_NAMES or (
        isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc)
    )


def _get_field(value: object, key: str) -> object:
    """Read a field from a dict-like or object-like SDK payload."""
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _coerce_segment_output_payload(
    tool_name: str,
    payload: object,
    city_name: str | None = None,
) -> InitiativeSegmentExtraction | InitiativeSegmentStop:
    """Validate tool-call payloads against the matching segment output model."""
    if isinstance(payload, dict) and "result" in payload:
        payload = payload["result"]
    if tool_name == "stop_initiative_extraction":
        return InitiativeSegmentStop.model_validate(payload)
    return InitiativeSegmentExtraction.model_validate(
        _normalize_segment_extraction_payload(payload, city_name)
    )


def _normalize_json_object(value: object) -> dict[str, JsonValue]:
    """Return a dict for loose JSON metadata fields."""
    if isinstance(value, dict):
        return value
    if value in (None, []):
        return {}
    return {"items": value}


def _normalize_segment_extraction_payload(
    payload: object,
    city_name: str | None = None,
) -> object:
    """Repair common recoverable model shape errors before validation."""
    if not isinstance(payload, dict):
        return payload

    initiatives = payload.get("initiatives")
    if not isinstance(initiatives, list):
        return payload

    initiative_fields = set(InitiativeExtraction.model_fields)
    normalized_initiatives: list[object] = []
    for item in initiatives:
        if not isinstance(item, dict):
            normalized_initiatives.append(item)
            continue

        source_quote = _clean_source_quote(item.get("source_quote"))
        initiative = item.get("initiative") if isinstance(item.get("initiative"), dict) else item
        if source_quote is None:
            source_quote = _clean_source_quote(initiative.get("source_quote"))
        raw_flags = item.get("data_quality_flags")
        if not isinstance(raw_flags, list):
            raw_flags = initiative.get("data_quality_flags")
        data_quality_flags = list(raw_flags) if isinstance(raw_flags, list) else []
        original_city = initiative.get("city")

        for field_name in CANDIDATE_METADATA_FIELDS:
            initiative.pop(field_name, None)

        if city_name is not None:
            if (
                isinstance(original_city, str)
                and original_city.strip()
                and original_city.strip() != city_name
            ):
                data_quality_flags.append(CITY_OVERRIDDEN_FLAG)
            initiative["city"] = city_name
        initiative["numbers"] = _normalize_numbers_payload(initiative.get("numbers"))
        for field_name in list(initiative):
            if field_name not in initiative_fields:
                initiative.pop(field_name)
        normalized_item: dict[str, object] = {
            "initiative": initiative,
            "source_quote": source_quote,
        }
        if data_quality_flags:
            normalized_item["data_quality_flags"] = list(dict.fromkeys(data_quality_flags))
        normalized_initiatives.append(normalized_item)

    payload["initiatives"] = normalized_initiatives
    return payload


def _clean_source_quote(value: object) -> str | None:
    """Return a trimmed source quote or None for blank/non-string values."""
    if not isinstance(value, str):
        return None
    quote = value.strip()
    return quote or None


def _normalize_numbers_payload(value: object) -> dict[str, dict[str, JsonValue]]:
    """Return canonical current/planned number buckets."""
    if not isinstance(value, dict):
        return {"current": {}, "planned": {}}
    current = value.get("current")
    planned = value.get("planned")
    return {
        "current": current if isinstance(current, dict) else _normalize_json_object(current),
        "planned": planned if isinstance(planned, dict) else _normalize_json_object(planned),
    }


def _extract_segment_tool_output(
    result: object,
    city_name: str | None = None,
) -> InitiativeSegmentExtraction | InitiativeSegmentStop | None:
    """Extract structured tool arguments from the Agents SDK raw response."""
    raw_responses = list(getattr(result, "raw_responses", []) or [])
    for response in reversed(raw_responses):
        output_items = _get_field(response, "output")
        if not isinstance(output_items, list):
            continue
        for item in reversed(output_items):
            if _get_field(item, "type") != "function_call":
                continue
            tool_name = str(_get_field(item, "name") or "")
            if tool_name not in {"submit_initiative_extractions", "stop_initiative_extraction"}:
                continue
            arguments = _get_field(item, "arguments")
            if not isinstance(arguments, str):
                continue
            return _coerce_segment_output_payload(tool_name, json.loads(arguments), city_name)
    return None


def _coerce_segment_output(
    output: object,
    city_name: str | None = None,
) -> InitiativeSegmentExtraction | InitiativeSegmentStop:
    """Coerce final output into one of the accepted segment output models."""
    if isinstance(output, (InitiativeSegmentExtraction, InitiativeSegmentStop)):
        return output
    if isinstance(output, dict):
        tool_name = (
            "stop_initiative_extraction"
            if "initiatives" not in output and "reason" in output
            else "submit_initiative_extractions"
        )
        return _coerce_segment_output_payload(tool_name, output, city_name)
    if isinstance(output, str) and output.strip().startswith("{"):
        payload = json.loads(output)
        return _coerce_segment_output(payload, city_name)
    raise TypeError(f"Unsupported initiative segment output type: {type(output).__name__}")


def build_initiative_extractor_agent(config: AppConfig, api_key: str) -> object:
    """Build the initiative extractor agent with structured tool output."""
    from agents import Agent, function_tool
    from backend.services.agents import build_model_settings, build_openrouter_model

    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "initiative_extractor_system.md"
    )
    model = build_openrouter_model(
        config.initiative_extractor.model,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )
    settings = build_model_settings(
        config.initiative_extractor.temperature,
        config.initiative_extractor.max_output_tokens,
        reasoning_effort=config.initiative_extractor.reasoning_effort,
    )
    settings.tool_choice = "required"
    settings.parallel_tool_calls = False

    @function_tool(strict_mode=False)
    def submit_initiative_extractions(
        initiatives: list[dict[str, Any]] | None = None,
        segment_data_quality_flags: list[str] | None = None,
        segment_notes: list[str] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return raw tool args so segment-scoped parsing can assign city."""
        return {
            "initiatives": initiatives or [],
            "segment_data_quality_flags": segment_data_quality_flags or [],
            "segment_notes": segment_notes or [],
            "error": error,
        }

    @function_tool(strict_mode=False)
    def stop_initiative_extraction(
        reason: str | None = None,
        segment_data_quality_flags: list[str] | None = None,
        segment_notes: list[str] | None = None,
    ) -> InitiativeSegmentStop:
        return InitiativeSegmentStop.model_validate(
            {
                "reason": reason,
                "segment_data_quality_flags": segment_data_quality_flags or [],
                "segment_notes": segment_notes or [],
            }
        )

    return Agent(
        name="Initiative Extractor",
        instructions=load_prompt(prompt_path),
        model=model,
        model_settings=settings,
        tools=[submit_initiative_extractions, stop_initiative_extraction],
        tool_use_behavior="stop_on_first_tool",
    )


def build_initiative_semantic_dedupe_agent(config: AppConfig, api_key: str) -> object:
    """Build the semantic initiative dedupe agent."""
    from agents import Agent, AgentOutputSchema, function_tool
    from backend.services.agents import build_model_settings, build_openrouter_model

    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "initiative_semantic_dedupe_system.md"
    )
    model = build_openrouter_model(
        config.initiative_extractor.model,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )
    settings = build_model_settings(
        config.initiative_extractor.temperature,
        config.initiative_extractor.max_output_tokens,
        reasoning_effort=config.initiative_extractor.reasoning_effort,
    )
    settings.tool_choice = "submit_semantic_dedupe"
    settings.parallel_tool_calls = False

    @function_tool(strict_mode=False)
    def submit_semantic_dedupe(
        duplicate_groups: list[InitiativeSemanticDedupeGroup] | None = None,
        review_notes: list[str] | None = None,
    ) -> InitiativeSemanticDedupeResult:
        return InitiativeSemanticDedupeResult(
            duplicate_groups=duplicate_groups or [],
            review_notes=review_notes or [],
        )

    return Agent(
        name="Initiative Semantic Dedupe",
        instructions=load_prompt(prompt_path),
        model=model,
        model_settings=settings,
        tools=[submit_semantic_dedupe],
        output_type=AgentOutputSchema(
            InitiativeSemanticDedupeResult,
            strict_json_schema=False,
        ),
        tool_use_behavior="stop_on_first_tool",
    )


def _get_thread_agent(config: AppConfig, api_key: str) -> object:
    """Return a thread-local initiative extractor agent."""
    local_agent = getattr(_thread_local, "initiative_extractor_agent", None)
    if local_agent is None:
        local_agent = build_initiative_extractor_agent(config, api_key)
        _thread_local.initiative_extractor_agent = local_agent
    return local_agent


def _get_thread_semantic_dedupe_agent(config: AppConfig, api_key: str) -> object:
    """Return a thread-local semantic dedupe agent."""
    local_agent = getattr(_thread_local, "initiative_semantic_dedupe_agent", None)
    if local_agent is None:
        local_agent = build_initiative_semantic_dedupe_agent(config, api_key)
        _thread_local.initiative_semantic_dedupe_agent = local_agent
    return local_agent


def _build_prior_initiatives_context(
    prior_initiatives: list[InitiativeExtractionCandidate],
    max_tokens: int,
) -> list[dict[str, object]]:
    """Return recent canonical initiatives within a token budget."""
    if max_tokens <= 0 or not prior_initiatives:
        return []

    selected: list[dict[str, object]] = []
    selected_tokens = 0
    for candidate in reversed(prior_initiatives):
        item = candidate.initiative.model_dump(mode="json")
        item_tokens = count_tokens(json.dumps(item, ensure_ascii=False))
        if selected and selected_tokens + item_tokens > max_tokens:
            break
        selected.append(item)
        selected_tokens += item_tokens
    return list(reversed(selected))


def _segment_payload(
    segment: InitiativeDocumentSegment,
    prior_initiatives: list[InitiativeExtractionCandidate],
    config: AppConfig,
    *,
    extraction_mode: str = "initial",
    already_extracted_scope: str = "run",
) -> dict[str, object]:
    """Render one segment as the LLM input contract."""
    return {
        "city_name": segment.city_name,
        "source_document": segment.source_document,
        "source_path": segment.source_path,
        "segment_id": segment.segment_id,
        "start_line": segment.start_line,
        "end_line": segment.end_line,
        "heading_path": segment.heading_path,
        "content": segment.content,
        "extraction_mode": extraction_mode,
        "already_extracted_scope": already_extracted_scope,
        "already_extracted_initiatives": _build_prior_initiatives_context(
            prior_initiatives,
            config.initiative_extractor.prior_initiatives_max_tokens,
        ),
    }


def _default_source_ref(segment: InitiativeDocumentSegment) -> InitiativeSourceRef:
    """Build the canonical source ref from segment metadata."""
    return InitiativeSourceRef(
        source_document=segment.source_document,
        segment_id=segment.segment_id,
        start_line=segment.start_line,
        end_line=segment.end_line,
        section_heading=segment.heading_path,
    )


def _infer_document_local_code(
    candidate: InitiativeExtractionCandidate,
) -> str | None:
    """Infer a source-local action code from the initiative name or quote when possible."""
    if candidate.document_local_code:
        return candidate.document_local_code

    for value in (candidate.source_quote, candidate.initiative.initiative_name):
        if not value:
            continue
        match = LOCAL_CODE_PATTERN.search(value)
        if match:
            return match.group(1)
    return None


def _normalize_candidate(
    candidate: InitiativeExtractionCandidate,
    segment: InitiativeDocumentSegment,
) -> InitiativeExtractionCandidate:
    """Assign segment metadata and validate the quote-only citation."""
    source_refs = [_default_source_ref(segment)]
    source_quote = _clean_source_quote(candidate.source_quote)
    quote_flags: list[str] = []
    initiative = candidate.initiative
    if source_quote is None:
        quote_flags.append("source_quote_missing")
    elif source_quote not in segment.content:
        source_quote = None
        quote_flags.append("source_quote_not_found")
    if initiative.city != segment.city_name:
        initiative = initiative.model_copy(update={"city": segment.city_name})
        quote_flags.append(CITY_OVERRIDDEN_FLAG)
    flags = list(
        dict.fromkeys(
            [
                *candidate.data_quality_flags,
                *quote_flags,
                *detect_source_quality_flags(segment.content),
            ]
        )
    )
    return candidate.model_copy(
        update={
            "initiative": initiative,
            "document_local_code": _infer_document_local_code(candidate),
            "source_quote": source_quote,
            "source_refs": source_refs,
            "data_quality_flags": flags,
        },
        deep=True,
    )


def _run_segment_once(
    segment: InitiativeDocumentSegment,
    config: AppConfig,
    api_key: str,
    *,
    log_llm_payload: bool,
    prior_initiatives: list[InitiativeExtractionCandidate],
    extraction_mode: str,
    already_extracted_scope: str,
) -> InitiativeRawSegmentResult:
    """Run the LLM once for one document segment."""
    agent = _get_thread_agent(config, api_key)
    result = run_agent_sync(
        agent,
        json.dumps(
            _segment_payload(
                segment,
                prior_initiatives,
                config,
                extraction_mode=extraction_mode,
                already_extracted_scope=already_extracted_scope,
            ),
            ensure_ascii=False,
        ),
        max_turns=max(config.initiative_extractor.max_turns, 1),
        log_llm_payload=log_llm_payload,
    )
    output = _extract_segment_tool_output(
        result,
        segment.city_name,
    ) or _coerce_segment_output(result.final_output, segment.city_name)
    if isinstance(output, InitiativeSegmentStop):
        flags = list(
            dict.fromkeys([*output.segment_data_quality_flags, *detect_source_quality_flags(segment.content)])
        )
        return InitiativeRawSegmentResult(
            segment_id=segment.segment_id,
            source_document=segment.source_document,
            status="success",
            segment_data_quality_flags=flags,
            segment_notes=output.segment_notes,
            extraction_complete=True,
            stop_reason=output.reason,
        )
    initiatives = [_normalize_candidate(item, segment) for item in output.initiatives]
    flags = list(
        dict.fromkeys([*output.segment_data_quality_flags, *detect_source_quality_flags(segment.content)])
    )
    return InitiativeRawSegmentResult(
        segment_id=segment.segment_id,
        source_document=segment.source_document,
        status="success",
        initiatives=initiatives,
        segment_data_quality_flags=flags,
        segment_notes=output.segment_notes,
        error=output.error,
    )


def _run_segment_with_retries(
    segment: InitiativeDocumentSegment,
    config: AppConfig,
    api_key: str,
    *,
    log_llm_payload: bool,
    run_id: str,
    prior_initiatives: list[InitiativeExtractionCandidate],
    extraction_mode: str = "initial",
    already_extracted_scope: str = "run",
) -> InitiativeRawSegmentResult:
    """Run one segment with bounded retry handling."""
    retry_settings = RetrySettings.bounded(
        max_attempts=config.retry.max_attempts,
        backoff_base_seconds=config.retry.backoff_base_seconds,
        backoff_max_seconds=config.retry.backoff_max_seconds,
    )
    for attempt in range(1, retry_settings.max_attempts + 1):
        try:
            return _run_segment_once(
                segment,
                config,
                api_key,
                log_llm_payload=log_llm_payload,
                prior_initiatives=prior_initiatives,
                extraction_mode=extraction_mode,
                already_extracted_scope=already_extracted_scope,
            )
        except Exception as exc:  # noqa: BLE001
            if attempt >= retry_settings.max_attempts or not _is_retryable_error(exc):
                logger.exception("Initiative extraction failed for segment %s", segment.segment_id)
                return InitiativeRawSegmentResult(
                    segment_id=segment.segment_id,
                    source_document=segment.source_document,
                    status="error",
                    segment_data_quality_flags=detect_source_quality_flags(segment.content),
                    error=ErrorInfo(
                        code="INITIATIVE_SEGMENT_EXTRACTION_FAILED",
                        message="Initiative extraction failed for this segment.",
                        details=[str(exc)],
                    ),
                )
            delay_seconds = compute_retry_delay_seconds(attempt, retry_settings)
            log_retry_event(
                operation="initiative.segment_extraction",
                run_id=run_id,
                attempt=attempt,
                max_attempts=retry_settings.max_attempts,
                error_type=type(exc).__name__,
                error_message=str(exc),
                next_backoff_seconds=delay_seconds,
                context={"segment_id": segment.segment_id},
            )
            if delay_seconds > 0:
                time.sleep(delay_seconds)
    raise RuntimeError("Unreachable initiative extraction retry state.")


def _process_segment(
    segment: InitiativeDocumentSegment,
    config: AppConfig,
    api_key: str,
    *,
    log_llm_payload: bool,
    run_id: str,
    prior_initiatives: list[InitiativeExtractionCandidate],
) -> InitiativeRawSegmentResult:
    """Extract one segment, looping only when the first result shows density."""
    result = _run_segment_with_retries(
        segment,
        config,
        api_key,
        log_llm_payload=log_llm_payload,
        run_id=run_id,
        prior_initiatives=prior_initiatives,
        extraction_mode="initial",
        already_extracted_scope="run",
    )
    threshold = config.initiative_extractor.action_heavy_initiative_threshold
    if result.status != "success" or len(result.initiatives) <= threshold:
        return result

    result = result.model_copy(
        update={
            "action_heavy": True,
            "segment_data_quality_flags": list(
                dict.fromkeys([*result.segment_data_quality_flags, "action_heavy_segment"])
            ),
            "segment_notes": [
                *result.segment_notes,
                (
                    "Action-heavy extraction loop triggered because the first call returned "
                    f"more than {threshold} initiatives."
                ),
            ],
        },
        deep=True,
    )

    segment_initiatives = list(result.initiatives)
    max_followups = max(config.initiative_extractor.action_heavy_max_followup_calls, 0)
    for _ in range(max_followups):
        # Dense segments stay as one source segment artifact. Follow-up calls only
        # receive initiatives already extracted from this same chunk, which makes it
        # easier for the model to focus on what is still missing without creating
        # more split segments.
        followup = _run_segment_with_retries(
            segment,
            config,
            api_key,
            log_llm_payload=log_llm_payload,
            run_id=run_id,
            prior_initiatives=segment_initiatives,
            extraction_mode="dense_followup",
            already_extracted_scope="current_segment",
        )
        result.extraction_iterations += 1
        result.segment_data_quality_flags = list(
            dict.fromkeys([*result.segment_data_quality_flags, *followup.segment_data_quality_flags])
        )
        result.segment_notes = list(dict.fromkeys([*result.segment_notes, *followup.segment_notes]))

        if followup.status != "success":
            result.segment_data_quality_flags = list(
                dict.fromkeys([*result.segment_data_quality_flags, "action_heavy_followup_failed"])
            )
            result.error = followup.error
            result.stop_reason = "Action-heavy follow-up failed before model stop signal."
            break
        if followup.extraction_complete:
            result.extraction_complete = True
            result.stop_reason = followup.stop_reason
            break
        if not followup.initiatives:
            result.extraction_complete = True
            result.stop_reason = "Follow-up extraction returned no additional initiatives."
            break

        existing_keys = {
            _dedupe_key(candidate, segment.source_document)
            for candidate in segment_initiatives
        }
        new_initiatives = [
            candidate
            for candidate in followup.initiatives
            if _dedupe_key(candidate, segment.source_document) not in existing_keys
        ]
        if not new_initiatives:
            result.extraction_complete = True
            result.stop_reason = "Follow-up extraction returned only already extracted initiatives."
            result.segment_data_quality_flags = list(
                dict.fromkeys([*result.segment_data_quality_flags, "action_heavy_followup_duplicate_only"])
            )
            break
        result.initiatives.extend(new_initiatives)
        segment_initiatives.extend(new_initiatives)

    if (
        result.action_heavy
        and not result.extraction_complete
        and result.status == "success"
        and result.stop_reason is None
    ):
        result.stop_reason = "Action-heavy follow-up limit reached before model stop signal."
        result.segment_data_quality_flags = list(
            dict.fromkeys([*result.segment_data_quality_flags, "action_heavy_followup_limit_reached"])
        )

    return result


def _normalize_title(value: str) -> str:
    """Normalize initiative titles for deduplication."""
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _record_id_for(candidate: InitiativeExtractionCandidate, source_document: str) -> str:
    """Build a deterministic record id for one candidate."""
    city_key = normalize_city_key(candidate.initiative.city) or "unknown_city"
    doc_slug = normalize_city_key(Path(source_document).stem) or "document"
    local_code = candidate.document_local_code
    if local_code:
        local_part = normalize_city_key(local_code) or local_code.casefold()
        return f"{city_key}:{doc_slug}:{local_part}"
    title_hash = sha1(  # noqa: S324
        f"{city_key}|{doc_slug}|{_normalize_title(candidate.initiative.initiative_name)}".encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    return f"{city_key}:{doc_slug}:title_{title_hash}"


def _dedupe_key(candidate: InitiativeExtractionCandidate, source_document: str) -> tuple[str, str, str]:
    """Build a dedupe key using local code when available, otherwise title."""
    city_key = normalize_city_key(candidate.initiative.city)
    doc_key = normalize_city_key(Path(source_document).stem)
    if candidate.document_local_code:
        return (city_key, doc_key, candidate.document_local_code.casefold().strip())
    return (city_key, doc_key, _normalize_title(candidate.initiative.initiative_name))


def _candidate_source_document(
    candidate: InitiativeExtractionCandidate,
    fallback: str,
) -> str:
    """Return the best source document available for one candidate."""
    if candidate.source_refs:
        return candidate.source_refs[0].source_document
    return fallback


def _extend_prior_initiatives(
    prior_initiatives: list[InitiativeExtractionCandidate],
    raw_results: list[InitiativeRawSegmentResult],
) -> None:
    """Add newly extracted initiatives to the rolling canonical history."""
    seen_keys = {
        _dedupe_key(candidate, _candidate_source_document(candidate, ""))
        for candidate in prior_initiatives
    }
    for raw in raw_results:
        if raw.status != "success":
            continue
        for candidate in raw.initiatives:
            key = _dedupe_key(candidate, raw.source_document)
            if key in seen_keys:
                continue
            prior_initiatives.append(candidate)
            seen_keys.add(key)


def _merge_dicts(
    base: dict[str, JsonValue],
    extra: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Merge numeric dictionaries while preserving existing values."""
    merged = dict(base)
    for key, value in extra.items():
        merged.setdefault(key, value)
    return merged


def _source_quote_score(value: str | None) -> int:
    """Score a quote by useful word count, capped to avoid favoring huge excerpts."""
    if not value:
        return 0
    return min(len(value.split()), 80)


def _choose_source_quote(base: str | None, extra: str | None) -> str | None:
    """Choose the clearest available source quote during duplicate merges."""
    if _source_quote_score(extra) > _source_quote_score(base):
        return extra
    return base


def _merge_record(
    base: InitiativeExtractionRecord,
    extra: InitiativeExtractionCandidate,
) -> InitiativeExtractionRecord:
    """Merge duplicate candidate metadata into an existing record."""
    initiative = base.initiative.model_copy(deep=True)
    for field_name in (
        "general_description",
        "objective_text",
        "implementation_text",
        "planned_outputs_text",
        "delivery_text",
        "funding_text",
        "timeline_text",
    ):
        if not getattr(initiative, field_name) and getattr(extra.initiative, field_name):
            setattr(initiative, field_name, getattr(extra.initiative, field_name))
    initiative.numbers.current = _merge_dicts(
        initiative.numbers.current,
        extra.initiative.numbers.current,
    )
    initiative.numbers.planned = _merge_dicts(
        initiative.numbers.planned,
        extra.initiative.numbers.planned,
    )
    return base.model_copy(
        update={
            "initiative": initiative,
            "source_quote": _choose_source_quote(base.source_quote, extra.source_quote),
            "source_refs": [*base.source_refs, *extra.source_refs],
            "document_local_code": base.document_local_code or extra.document_local_code,
            "data_quality_flags": list(dict.fromkeys([*base.data_quality_flags, *extra.data_quality_flags])),
            "number_context": _merge_dicts(base.number_context, extra.number_context),
            "number_deferred": _merge_dicts(base.number_deferred, extra.number_deferred),
            "number_uncertain": _merge_dicts(base.number_uncertain, extra.number_uncertain),
            "extraction_notes": list(dict.fromkeys([*base.extraction_notes, *extra.extraction_notes])),
        },
        deep=True,
    )


def _semantic_dedupe_payload(records: list[InitiativeExtractionRecord]) -> dict[str, object]:
    """Render records for semantic dedupe without artifact traces."""
    return {
        "records": [
            {
                "record_id": record.record_id,
                "document_local_code": record.document_local_code,
                "source_quote": record.source_quote,
                **record.initiative.model_dump(mode="json"),
            }
            for record in records
        ]
    }


def _build_semantic_dedupe_batches(
    records: list[InitiativeExtractionRecord],
    config: AppConfig,
) -> list[list[InitiativeExtractionRecord]]:
    """Group semantic dedupe records into source-local token-bounded batches."""
    records_by_scope: dict[tuple[str, str], list[InitiativeExtractionRecord]] = {}
    for record in records:
        scope = (record.source_document, normalize_city_key(record.initiative.city))
        records_by_scope.setdefault(scope, []).append(record)

    max_records = max(config.initiative_extractor.semantic_dedupe_max_records_per_batch, 1)
    max_tokens = max(config.initiative_extractor.semantic_dedupe_max_input_tokens, 1)
    batches: list[list[InitiativeExtractionRecord]] = []
    for scoped_records in records_by_scope.values():
        current: list[InitiativeExtractionRecord] = []
        current_tokens = 0
        for record in sorted(scoped_records, key=lambda item: item.record_id):
            payload = {"record_id": record.record_id, **record.initiative.model_dump(mode="json")}
            item_tokens = count_tokens(json.dumps(payload, ensure_ascii=False))
            should_flush = current and (
                len(current) >= max_records or current_tokens + item_tokens > max_tokens
            )
            if should_flush:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(record)
            current_tokens += item_tokens
        if current:
            batches.append(current)
    return batches


def _run_semantic_dedupe_batch(
    records: list[InitiativeExtractionRecord],
    config: AppConfig,
    api_key: str,
    *,
    log_llm_payload: bool,
) -> InitiativeSemanticDedupeResult:
    """Run semantic dedupe once for one record batch."""
    agent = _get_thread_semantic_dedupe_agent(config, api_key)
    result = run_agent_sync(
        agent,
        json.dumps(_semantic_dedupe_payload(records), ensure_ascii=False),
        max_turns=max(config.initiative_extractor.max_turns, 1),
        log_llm_payload=log_llm_payload,
    )
    output = result.final_output
    if isinstance(output, InitiativeSemanticDedupeResult):
        return output
    return InitiativeSemanticDedupeResult.model_validate(output)


def _apply_semantic_dedupe_groups(
    records: list[InitiativeExtractionRecord],
    groups: list[InitiativeSemanticDedupeGroup],
    config: AppConfig,
) -> tuple[list[InitiativeExtractionRecord], list[InitiativeReviewItem]]:
    """Merge records using accepted semantic duplicate groups."""
    records_by_id = {record.record_id: record for record in records}
    parent = {record.record_id: record.record_id for record in records}
    threshold = config.initiative_extractor.semantic_dedupe_confidence_threshold
    review_items: list[InitiativeReviewItem] = []

    def find(record_id: str) -> str:
        while parent[record_id] != record_id:
            parent[record_id] = parent[parent[record_id]]
            record_id = parent[record_id]
        return record_id

    for group in groups:
        canonical_id = group.canonical_record_id
        if group.confidence < threshold:
            continue
        if canonical_id not in records_by_id:
            review_items.append(
                InitiativeReviewItem(
                    review_type="semantic_dedupe_invalid_record_id",
                    message="Semantic dedupe returned an unknown canonical record id.",
                    record_id=canonical_id,
                    details={"confidence": group.confidence, "rationale": group.rationale},
                )
            )
            continue
        canonical_root = find(canonical_id)
        for duplicate_id in group.duplicate_record_ids:
            if duplicate_id == canonical_id:
                continue
            if duplicate_id not in records_by_id:
                review_items.append(
                    InitiativeReviewItem(
                        review_type="semantic_dedupe_invalid_record_id",
                        message="Semantic dedupe returned an unknown duplicate record id.",
                        record_id=canonical_id,
                        details={
                            "duplicate_record_id": duplicate_id,
                            "confidence": group.confidence,
                            "rationale": group.rationale,
                        },
                    )
                )
                continue
            parent[find(duplicate_id)] = canonical_root
            review_items.append(
                InitiativeReviewItem(
                    review_type="semantic_duplicate_merged",
                    severity="info",
                    message="Semantic dedupe merged two records that describe the same initiative.",
                    source_document=records_by_id[duplicate_id].source_document,
                    record_id=canonical_id,
                    details={
                        "duplicate_record_id": duplicate_id,
                        "confidence": group.confidence,
                        "rationale": group.rationale,
                    },
                )
            )

    grouped_ids: dict[str, list[str]] = {}
    for record_id in records_by_id:
        grouped_ids.setdefault(find(record_id), []).append(record_id)

    merged_records: list[InitiativeExtractionRecord] = []
    for record in records:
        root_id = find(record.record_id)
        if record.record_id != root_id:
            continue
        merged_record = record
        for duplicate_id in grouped_ids[root_id]:
            if duplicate_id == root_id:
                continue
            merged_record = _merge_record(merged_record, records_by_id[duplicate_id])
        merged_records.append(merged_record)
    return merged_records, review_items


def _semantic_dedupe_records(
    records: list[InitiativeExtractionRecord],
    config: AppConfig,
    api_key: str,
    *,
    log_llm_payload: bool,
) -> tuple[
    list[InitiativeExtractionRecord],
    list[InitiativeSemanticDedupeGroup],
    list[InitiativeReviewItem],
]:
    """Run semantic dedupe over exact-deduped records."""
    if not config.initiative_extractor.semantic_dedupe_enabled or len(records) < 2:
        return records, [], []

    groups: list[InitiativeSemanticDedupeGroup] = []
    review_items: list[InitiativeReviewItem] = []
    for batch in _build_semantic_dedupe_batches(records, config):
        try:
            result = _run_semantic_dedupe_batch(
                batch,
                config,
                api_key,
                log_llm_payload=log_llm_payload,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Semantic initiative dedupe failed for a batch.")
            review_items.append(
                InitiativeReviewItem(
                    review_type="semantic_dedupe_failed",
                    severity="error",
                    message="Semantic initiative dedupe failed for a record batch.",
                    details={"error": str(exc)},
                )
            )
            continue
        groups.extend(result.duplicate_groups)
        for note in result.review_notes:
            review_items.append(
                InitiativeReviewItem(
                    review_type="semantic_dedupe_note",
                    severity="info",
                    message=note,
                )
            )

    merged_records, merge_reviews = _apply_semantic_dedupe_groups(records, groups, config)
    return merged_records, groups, [*review_items, *merge_reviews]


def _dedupe_candidates(
    raw_results: list[InitiativeRawSegmentResult],
) -> tuple[list[InitiativeExtractionRecord], list[InitiativeReviewItem]]:
    """Deduplicate raw candidates into stable initiative records."""
    records_by_key: dict[tuple[str, str, str], InitiativeExtractionRecord] = {}
    review_items: list[InitiativeReviewItem] = []
    for raw in raw_results:
        for candidate in raw.initiatives:
            key = _dedupe_key(candidate, raw.source_document)
            existing = records_by_key.get(key)
            if existing is None:
                record = InitiativeExtractionRecord(
                    initiative=candidate.initiative,
                    document_local_code=candidate.document_local_code,
                    source_quote=candidate.source_quote,
                    source_refs=candidate.source_refs,
                    data_quality_flags=candidate.data_quality_flags,
                    number_context=candidate.number_context,
                    number_deferred=candidate.number_deferred,
                    number_uncertain=candidate.number_uncertain,
                    extraction_notes=candidate.extraction_notes,
                    record_id=_record_id_for(candidate, raw.source_document),
                    source_document=raw.source_document,
                )
                records_by_key[key] = record
                continue
            records_by_key[key] = _merge_record(existing, candidate)
            review_items.append(
                InitiativeReviewItem(
                    review_type="duplicate_merged",
                    severity="info",
                    message="Repeated initiative candidate merged into an existing record.",
                    source_document=raw.source_document,
                    segment_id=raw.segment_id,
                    record_id=existing.record_id,
                    document_local_code=candidate.document_local_code,
                )
            )
    return list(records_by_key.values()), review_items


def _content_has_meta_text(record: InitiativeExtractionRecord) -> bool:
    """Detect extraction-process prose that should not appear in content fields."""
    values = [
        record.initiative.general_description,
        record.initiative.objective_text,
        record.initiative.implementation_text,
        record.initiative.planned_outputs_text,
        record.initiative.delivery_text,
        record.initiative.funding_text,
        record.initiative.timeline_text,
    ]
    text = " ".join(value or "" for value in values).casefold()
    return "extracted source segment" in text or "not present in the extracted" in text


def _build_review_items(
    *,
    segments: list[InitiativeDocumentSegment],
    raw_results: list[InitiativeRawSegmentResult],
    records: list[InitiativeExtractionRecord],
    duplicate_reviews: list[InitiativeReviewItem],
    config: AppConfig,
) -> list[InitiativeReviewItem]:
    """Build coverage and quality review items for one extraction run."""
    review_items = list(duplicate_reviews)
    result_by_segment = {result.segment_id: result for result in raw_results}
    for segment in segments:
        result = result_by_segment.get(segment.segment_id)
        if result is None:
            continue
        if result.status == "error":
            review_items.append(
                InitiativeReviewItem(
                    review_type="segment_extraction_failed",
                    severity="error",
                    message="Segment failed initiative extraction.",
                    source_document=segment.source_document,
                    segment_id=segment.segment_id,
                )
            )
        if result.action_heavy:
            review_items.append(
                InitiativeReviewItem(
                    review_type="action_heavy_segment",
                    severity="info",
                    message="Segment returned more than the configured action-heavy initiative threshold.",
                    source_document=segment.source_document,
                    segment_id=segment.segment_id,
                    details={
                        "extracted_count": len(result.initiatives),
                        "threshold": config.initiative_extractor.action_heavy_initiative_threshold,
                        "extraction_iterations": result.extraction_iterations,
                        "extraction_complete": result.extraction_complete,
                        "stop_reason": result.stop_reason,
                    },
                )
            )
        for flag in result.segment_data_quality_flags:
            review_type = (
                "action_heavy_extraction_flag"
                if flag.startswith("action_heavy_")
                else "source_quality_flag"
            )
            message = (
                f"Segment has action-heavy extraction flag: {flag}"
                if review_type == "action_heavy_extraction_flag"
                else f"Segment has source quality flag: {flag}"
            )
            review_items.append(
                InitiativeReviewItem(
                    review_type=review_type,
                    severity="info",
                    message=message,
                    source_document=segment.source_document,
                    segment_id=segment.segment_id,
                    details={"flag": flag},
                )
            )

    for record in records:
        if _content_has_meta_text(record):
            review_items.append(
                InitiativeReviewItem(
                    review_type="content_contains_extraction_meta_text",
                    message="Content field contains extraction-process prose.",
                    source_document=record.source_document,
                    record_id=record.record_id,
                    document_local_code=record.document_local_code,
                )
            )
        for flag in record.data_quality_flags:
            if flag in SOURCE_QUOTE_FLAGS:
                review_items.append(
                    InitiativeReviewItem(
                        review_type="source_quote_missing_or_invalid",
                        message="Initiative source quote is missing or was not found in the source segment.",
                        source_document=record.source_document,
                        record_id=record.record_id,
                        document_local_code=record.document_local_code,
                        details={"flag": flag},
                    )
                )
                continue
            review_items.append(
                InitiativeReviewItem(
                    review_type="initiative_quality_flag",
                    severity="info",
                    message=f"Initiative has source quality flag: {flag}",
                    source_document=record.source_document,
                    record_id=record.record_id,
                    document_local_code=record.document_local_code,
                    details={"flag": flag},
                )
            )

    return review_items


def _write_run_artifacts(
    *,
    run_dir: Path,
    run_id: str,
    documents: list[Path],
    segments: list[InitiativeDocumentSegment],
    raw_results: list[InitiativeRawSegmentResult],
    exact_records: list[InitiativeExtractionRecord],
    records: list[InitiativeExtractionRecord],
    semantic_groups: list[InitiativeSemanticDedupeGroup],
    review_items: list[InitiativeReviewItem],
    config: AppConfig,
) -> None:
    """Persist all initiative extraction artifacts for a run."""
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "documents": [str(path) for path in documents],
        "model": config.initiative_extractor.model,
        "max_segment_tokens": config.initiative_extractor.max_segment_tokens,
        "segment_overlap_lines": config.initiative_extractor.segment_overlap_lines,
        "prior_initiatives_max_tokens": config.initiative_extractor.prior_initiatives_max_tokens,
        "action_heavy_initiative_threshold": config.initiative_extractor.action_heavy_initiative_threshold,
        "action_heavy_max_followup_calls": config.initiative_extractor.action_heavy_max_followup_calls,
        "semantic_dedupe_enabled": config.initiative_extractor.semantic_dedupe_enabled,
    }
    summary = {
        "run_id": run_id,
        "documents_count": len(documents),
        "segments_count": len(segments),
        "raw_initiatives_count": sum(len(result.initiatives) for result in raw_results),
        "exact_deduped_initiatives_count": len(exact_records),
        "deduped_initiatives_count": len(records),
        "semantic_duplicate_groups_count": len(semantic_groups),
        "semantic_merged_duplicates_count": max(len(exact_records) - len(records), 0),
        "action_heavy_segments_count": sum(1 for result in raw_results if result.action_heavy),
        "action_heavy_followup_iterations_count": sum(
            max(result.extraction_iterations - 1, 0) for result in raw_results
        ),
        "review_items_count": len(review_items),
    }
    write_json(run_dir / "00_source" / "source_manifest.json", manifest, ensure_ascii=False)
    _write_jsonl(run_dir / "01_segments" / "segments.jsonl", segments)
    _write_jsonl(run_dir / "02_raw_extractions" / "raw_segment_extractions.jsonl", raw_results)
    _write_jsonl(run_dir / "03_deduped" / "exact_initiatives.jsonl", _canonical_rows(exact_records))
    _write_jsonl(run_dir / "03_deduped" / "exact_initiative_records.jsonl", exact_records)
    _write_jsonl(run_dir / "03_deduped" / "semantic_duplicate_groups.jsonl", semantic_groups)
    _write_jsonl(run_dir / "03_deduped" / "initiatives.jsonl", _canonical_rows(records))
    _write_jsonl(run_dir / "03_deduped" / "initiative_records.jsonl", records)
    _write_jsonl(run_dir / "04_review" / "review_items.jsonl", review_items)
    write_json(run_dir / "summary.json", summary, ensure_ascii=False)
    (run_dir / "README.md").write_text(
        "\n".join(
            [
                "# Initiative Extraction Run",
                "",
                "This folder contains artifact-first initiative extraction output.",
                "",
                "- `00_source/source_manifest.json`: source documents and run settings.",
                "- `01_segments/segments.jsonl`: ordered line-aware document segments.",
                "- `02_raw_extractions/raw_segment_extractions.jsonl`: per-segment model output.",
                "- `03_deduped/exact_initiatives.jsonl`: canonical v1 initiatives after exact code/title dedupe.",
                "- `03_deduped/exact_initiative_records.jsonl`: exact-deduped pipeline records with generated ids and source quotes.",
                "- `03_deduped/semantic_duplicate_groups.jsonl`: semantic duplicate groups proposed by the LLM.",
                "- `03_deduped/initiatives.jsonl`: final canonical v1 initiative objects.",
                "- `03_deduped/initiative_records.jsonl`: final pipeline records with generated ids and quote-only source citations for TEF mapping.",
                "- `04_review/review_items.jsonl`: coverage and quality review items.",
                "- `summary.json`: run counts.",
                "",
                "No TEF classification or database writes are performed in this step.",
            ]
        ),
        encoding="utf-8",
    )


def extract_initiatives(
    *,
    markdown_path: Path,
    config: AppConfig,
    api_key: str,
    output_root: Path,
    run_id: str | None = None,
    selected_cities: list[str] | None = None,
    max_workers: int | None = None,
    log_llm_payload: bool = False,
) -> InitiativeExtractionRunResult:
    """Run artifact-first initiative extraction over selected markdown documents."""
    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / resolved_run_id
    documents = _discover_markdown_files(markdown_path, selected_cities, config)

    segments: list[InitiativeDocumentSegment] = []
    for path in documents:
        segments.extend(build_document_segments(path, config.initiative_extractor))

    configured_workers = max_workers or config.initiative_extractor.max_workers
    prior_context_enabled = config.initiative_extractor.prior_initiatives_max_tokens > 0
    worker_count = 1 if prior_context_enabled else min(max(configured_workers, 1), max(len(segments), 1))
    raw_results: list[InitiativeRawSegmentResult] = []

    logger.info(
        "Starting initiative extraction run_id=%s documents=%d segments=%d workers=%d",
        resolved_run_id,
        len(documents),
        len(segments),
        worker_count,
    )
    if prior_context_enabled:
        prior_initiatives: list[InitiativeExtractionCandidate] = []
        for segment in segments:
            new_result = _process_segment(
                segment,
                config,
                api_key,
                log_llm_payload=log_llm_payload,
                run_id=resolved_run_id,
                prior_initiatives=prior_initiatives,
            )
            raw_results.append(new_result)
            _extend_prior_initiatives(prior_initiatives, [new_result])
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _process_segment,
                    segment,
                    config,
                    api_key,
                    log_llm_payload=log_llm_payload,
                    run_id=resolved_run_id,
                    prior_initiatives=[],
                )
                for segment in segments
            ]
            for future in as_completed(futures):
                raw_results.append(future.result())

    raw_results.sort(key=lambda result: result.segment_id)
    exact_records, duplicate_reviews = _dedupe_candidates(raw_results)
    exact_records.sort(key=lambda record: record.record_id)
    records, semantic_groups, semantic_reviews = _semantic_dedupe_records(
        exact_records,
        config,
        api_key,
        log_llm_payload=log_llm_payload,
    )
    records.sort(key=lambda record: record.record_id)
    review_items = _build_review_items(
        segments=segments,
        raw_results=raw_results,
        records=records,
        duplicate_reviews=[*duplicate_reviews, *semantic_reviews],
        config=config,
    )

    _write_run_artifacts(
        run_dir=run_dir,
        run_id=resolved_run_id,
        documents=documents,
        segments=segments,
        raw_results=raw_results,
        exact_records=exact_records,
        records=records,
        semantic_groups=semantic_groups,
        review_items=review_items,
        config=config,
    )
    return InitiativeExtractionRunResult(
        run_id=resolved_run_id,
        output_dir=str(run_dir),
        documents_count=len(documents),
        segments_count=len(segments),
        raw_initiatives_count=sum(len(result.initiatives) for result in raw_results),
        deduped_initiatives_count=len(records),
        review_items_count=len(review_items),
    )


__all__ = [
    "build_initiative_extractor_agent",
    "build_initiative_semantic_dedupe_agent",
    "extract_initiatives",
]
