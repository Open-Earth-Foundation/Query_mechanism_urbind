"""Per-segment LLM execution, retries, and dense-segment follow-up handling."""

from __future__ import annotations

import logging
import time

from backend.models import ErrorInfo
from backend.modules.initiative_extractor.models import (
    InitiativeDocumentSegment,
    InitiativeExtractionCandidate,
    InitiativeRawSegmentResult,
    InitiativeSegmentStop,
)
from backend.modules.initiative_extractor.output_parser import (
    _coerce_segment_output,
    _extract_segment_tool_output,
)
from backend.modules.initiative_extractor.records import (
    _dedupe_key,
    _normalize_candidate,
)
from backend.modules.initiative_extractor.segmentation import (
    detect_source_quality_flags,
)
from backend.utils.config import AppConfig
from backend.utils.llm_serialization import (
    count_serialized_tokens_for_llm,
    serialize_for_llm,
)
from backend.utils.retry import (
    RetrySettings,
    compute_retry_delay_seconds,
    log_retry_event,
)

logger = logging.getLogger(__name__)
RETRYABLE_ERROR_NAMES = {
    "APIConnectionError",
    "MaxTurnsExceeded",
    "ModelBehaviorError",
}


def _facade() -> object:
    """Return the compatibility facade module for monkeypatched tests."""
    from backend.modules.initiative_extractor import agent

    return agent


def run_agent_sync(*args: object, **kwargs: object) -> object:
    """Call the facade runner so tests can monkeypatch the public module."""
    return _facade().run_agent_sync(*args, **kwargs)


def _get_thread_agent(config: AppConfig, api_key: str) -> object:
    """Return the facade thread-local extractor agent."""
    return _facade()._get_thread_agent(config, api_key)


def _is_retryable_error(exc: Exception) -> bool:
    """Return whether an extractor failure should be retried."""
    return type(exc).__name__ in RETRYABLE_ERROR_NAMES or (
        isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc)
    )


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
        item_tokens = count_serialized_tokens_for_llm(item)
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
        serialize_for_llm(
            _segment_payload(
                segment,
                prior_initiatives,
                config,
                extraction_mode=extraction_mode,
                already_extracted_scope=already_extracted_scope,
            )
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
            dict.fromkeys(
                [
                    *output.segment_data_quality_flags,
                    *detect_source_quality_flags(segment.content),
                ]
            )
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
        dict.fromkeys(
            [
                *output.segment_data_quality_flags,
                *detect_source_quality_flags(segment.content),
            ]
        )
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
            return _facade()._run_segment_once(
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
                logger.exception(
                    "Initiative extraction failed for segment %s", segment.segment_id
                )
                return InitiativeRawSegmentResult(
                    segment_id=segment.segment_id,
                    source_document=segment.source_document,
                    status="error",
                    segment_data_quality_flags=detect_source_quality_flags(
                        segment.content
                    ),
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
    result = _facade()._run_segment_with_retries(
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
                dict.fromkeys(
                    [*result.segment_data_quality_flags, "action_heavy_segment"]
                )
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
        followup = _facade()._run_segment_with_retries(
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
            dict.fromkeys(
                [
                    *result.segment_data_quality_flags,
                    *followup.segment_data_quality_flags,
                ]
            )
        )
        result.segment_notes = list(
            dict.fromkeys([*result.segment_notes, *followup.segment_notes])
        )

        if followup.status != "success":
            result.segment_data_quality_flags = list(
                dict.fromkeys(
                    [*result.segment_data_quality_flags, "action_heavy_followup_failed"]
                )
            )
            result.error = followup.error
            result.stop_reason = (
                "Action-heavy follow-up failed before model stop signal."
            )
            break
        if followup.extraction_complete:
            result.extraction_complete = True
            result.stop_reason = followup.stop_reason
            break
        if not followup.initiatives:
            result.extraction_complete = True
            result.stop_reason = (
                "Follow-up extraction returned no additional initiatives."
            )
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
            result.stop_reason = (
                "Follow-up extraction returned only already extracted initiatives."
            )
            result.segment_data_quality_flags = list(
                dict.fromkeys(
                    [
                        *result.segment_data_quality_flags,
                        "action_heavy_followup_duplicate_only",
                    ]
                )
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
        result.stop_reason = (
            "Action-heavy follow-up limit reached before model stop signal."
        )
        result.segment_data_quality_flags = list(
            dict.fromkeys(
                [
                    *result.segment_data_quality_flags,
                    "action_heavy_followup_limit_reached",
                ]
            )
        )

    return result
