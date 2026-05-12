"""Semantic duplicate detection and merge orchestration for initiative records."""

from __future__ import annotations

import logging

from backend.modules.initiative_extractor.models import (
    InitiativeExtractionRecord,
    InitiativeReviewItem,
    InitiativeSemanticDedupeGroup,
    InitiativeSemanticDedupeResult,
)
from backend.modules.initiative_extractor.records import _apply_semantic_dedupe_groups
from backend.utils.city_normalization import normalize_city_key
from backend.utils.config import AppConfig
from backend.utils.llm_serialization import (
    count_serialized_tokens_for_llm,
    serialize_for_llm,
)

logger = logging.getLogger(__name__)


def _facade() -> object:
    """Return the compatibility facade module for monkeypatched tests."""
    from backend.modules.initiative_extractor import agent

    return agent


def run_agent_sync(*args: object, **kwargs: object) -> object:
    """Call the facade runner so tests can monkeypatch the public module."""
    return _facade().run_agent_sync(*args, **kwargs)


def _get_thread_semantic_dedupe_agent(config: AppConfig, api_key: str) -> object:
    """Return the facade thread-local semantic dedupe agent."""
    return _facade()._get_thread_semantic_dedupe_agent(config, api_key)


def _semantic_dedupe_payload(
    records: list[InitiativeExtractionRecord],
) -> dict[str, object]:
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

    max_records = max(
        config.initiative_extractor.semantic_dedupe_max_records_per_batch, 1
    )
    max_tokens = max(config.initiative_extractor.semantic_dedupe_max_input_tokens, 1)
    batches: list[list[InitiativeExtractionRecord]] = []
    for scoped_records in records_by_scope.values():
        current: list[InitiativeExtractionRecord] = []
        current_tokens = 0
        for record in sorted(scoped_records, key=lambda item: item.record_id):
            payload = {
                "record_id": record.record_id,
                **record.initiative.model_dump(mode="json"),
            }
            item_tokens = count_serialized_tokens_for_llm(payload)
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
        serialize_for_llm(_semantic_dedupe_payload(records)),
        max_turns=max(config.initiative_extractor.max_turns, 1),
        log_llm_payload=log_llm_payload,
    )
    output = result.final_output
    if isinstance(output, InitiativeSemanticDedupeResult):
        return output
    return InitiativeSemanticDedupeResult.model_validate(output)


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
    """Run semantic dedupe over candidate records."""
    if not config.initiative_extractor.semantic_dedupe_enabled or len(records) < 2:
        return records, [], []

    groups: list[InitiativeSemanticDedupeGroup] = []
    review_items: list[InitiativeReviewItem] = []
    for batch in _build_semantic_dedupe_batches(records, config):
        try:
            result = _facade()._run_semantic_dedupe_batch(
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

    merged_records, merge_reviews = _apply_semantic_dedupe_groups(
        records, groups, config
    )
    return merged_records, groups, [*review_items, *merge_reviews]
