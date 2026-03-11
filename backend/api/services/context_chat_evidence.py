"""Overflow evidence-cache and map-reduce helpers for context chat."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable

from openai import OpenAI

import backend.api.services.context_chat_execution as context_chat_execution
import backend.api.services.context_chat_io as context_chat_io
from backend.api.services.context_chat_planning import CHAT_REF_ID_PATTERN
from backend.api.services.models import ChatContextSource
from backend.api.services.prompts.context_chat import (
    compose_empty_evidence_prompt as _compose_empty_evidence_prompt,
    compose_evidence_map_prompt as _compose_evidence_map_prompt,
    compose_evidence_reduce_prompt as _compose_evidence_reduce_prompt,
    render_evidence_items_block as _render_evidence_items_block,
    render_partial_answers_block as _render_partial_answers_block,
)
from backend.api.services.utils.context_chat import (
    build_messages as _build_messages,
    chat_evidence_cache_path as _chat_evidence_cache_path,
    estimate_messages_tokens as _estimate_messages_tokens,
)
from backend.utils.config import AppConfig
from backend.utils.retry import RetrySettings
from backend.utils.tokenization import count_tokens

logger = logging.getLogger("backend.api.services.context_chat")

CHAT_EVIDENCE_CACHE_SCHEMA_VERSION = 2


def _run_overflow_evidence_map_reduce(
    *,
    prompt_header: str,
    normalized_contexts: list[ChatContextSource],
    normalized_citations: list[dict[str, str]],
    bounded_history: list[dict[str, str]],
    user_content: str,
    effective_token_cap: int,
    config: AppConfig,
    client: OpenAI,
    request_kwargs: dict[str, Any],
    retry_settings: RetrySettings,
    run_id: str | None,
    context_count: int,
) -> str:
    """Answer from compact evidence chunks when the direct prompt would overflow."""
    cache_payload = _load_or_build_evidence_cache(
        run_id=run_id,
        normalized_contexts=normalized_contexts,
        normalized_citations=normalized_citations,
        config=config,
    )
    cache_chunks = _normalize_evidence_cache_chunks(cache_payload)
    evidence_item_count = sum(len(chunk["items"]) for chunk in cache_chunks)
    logger.info(
        "Context chat split mode enabled run_id=%s context_count=%d evidence_items=%d "
        "cache_chunks=%d effective_token_cap=%d",
        run_id,
        context_count,
        evidence_item_count,
        len(cache_chunks),
        effective_token_cap,
    )
    if not cache_chunks:
        logger.info("Context chat split mode empty-evidence fallback run_id=%s", run_id)
        return _run_overflow_empty_evidence_answer(
            prompt_header=prompt_header,
            bounded_history=bounded_history,
            user_content=user_content,
            effective_token_cap=effective_token_cap,
            prompt_token_buffer=config.chat.prompt_token_buffer,
            client=client,
            request_kwargs=request_kwargs,
            retry_settings=retry_settings,
            run_id=run_id,
            context_count=context_count,
        )

    map_history, map_budget = _resolve_prompt_budget(
        prompt_factory=lambda: _compose_evidence_map_prompt(
            prompt_header=prompt_header,
            evidence_block="",
            chunk_index=1,
            total_chunks=1,
        ),
        history=bounded_history,
        user_content=user_content,
        effective_token_cap=effective_token_cap,
        prompt_token_buffer=config.chat.prompt_token_buffer,
        minimum_remaining_budget=max(
            (
                token_count
                for token_count in (
                    chunk.get("token_count")
                    for chunk in cache_chunks
                )
                if isinstance(token_count, int)
            ),
            default=1,
        ),
    )
    evidence_blocks = _resolve_request_evidence_blocks(
        cache_chunks=cache_chunks,
        block_budget=max(map_budget, 1),
    )
    partial_answers: list[str] = []
    total_blocks = len(evidence_blocks)
    logger.info(
        "Context chat split mode map phase run_id=%s map_chunks=%d map_budget=%d",
        run_id,
        total_blocks,
        map_budget,
    )
    for chunk_index, evidence_block in enumerate(evidence_blocks, start=1):
        system_prompt = _compose_evidence_map_prompt(
            prompt_header=prompt_header,
            evidence_block=evidence_block,
            chunk_index=chunk_index,
            total_chunks=total_blocks,
        )
        partial_answers.append(
            context_chat_execution._run_single_pass(
                client=client,
                messages=_build_messages(system_prompt, map_history, user_content),
                request_kwargs=request_kwargs,
                retry_settings=retry_settings,
                run_id=run_id,
                context_count=context_count,
            )
        )
    return _run_reduce_passes(
        partial_answers=partial_answers,
        prompt_header=prompt_header,
        bounded_history=bounded_history,
        user_content=user_content,
        effective_token_cap=effective_token_cap,
        prompt_token_buffer=config.chat.prompt_token_buffer,
        client=client,
        request_kwargs=request_kwargs,
        retry_settings=retry_settings,
        run_id=run_id,
        context_count=context_count,
    )


def _run_overflow_empty_evidence_answer(
    *,
    prompt_header: str,
    bounded_history: list[dict[str, str]],
    user_content: str,
    effective_token_cap: int,
    prompt_token_buffer: int,
    client: OpenAI,
    request_kwargs: dict[str, Any],
    retry_settings: RetrySettings,
    run_id: str | None,
    context_count: int,
) -> str:
    """Use the LLM to explain that no compact evidence is available in overflow mode."""
    working_history, _budget = _resolve_prompt_budget(
        prompt_factory=lambda: _compose_empty_evidence_prompt(prompt_header),
        history=bounded_history,
        user_content=user_content,
        effective_token_cap=effective_token_cap,
        prompt_token_buffer=prompt_token_buffer,
    )
    system_prompt = _compose_empty_evidence_prompt(prompt_header)
    return context_chat_execution._run_single_pass(
        client=client,
        messages=_build_messages(system_prompt, working_history, user_content),
        request_kwargs=request_kwargs,
        retry_settings=retry_settings,
        run_id=run_id,
        context_count=context_count,
    )


def _run_reduce_passes(
    *,
    partial_answers: list[str],
    prompt_header: str,
    bounded_history: list[dict[str, str]],
    user_content: str,
    effective_token_cap: int,
    prompt_token_buffer: int,
    client: OpenAI,
    request_kwargs: dict[str, Any],
    retry_settings: RetrySettings,
    run_id: str | None,
    context_count: int,
) -> str:
    """Recursively merge map outputs until one final answer remains."""
    pending_answers = [answer.strip() for answer in partial_answers if answer.strip()]
    if not pending_answers:
        return _run_overflow_empty_evidence_answer(
            prompt_header=prompt_header,
            bounded_history=bounded_history,
            user_content=user_content,
            effective_token_cap=effective_token_cap,
            prompt_token_buffer=prompt_token_buffer,
            client=client,
            request_kwargs=request_kwargs,
            retry_settings=retry_settings,
            run_id=run_id,
            context_count=context_count,
        )

    stage_index = 1
    while len(pending_answers) > 1:
        reduce_history, reduce_budget = _resolve_prompt_budget(
            prompt_factory=lambda: _compose_evidence_reduce_prompt(
                prompt_header=prompt_header,
                analyses_block="",
                stage_index=stage_index,
                batch_index=1,
                batch_count=1,
            ),
            history=bounded_history,
            user_content=user_content,
            effective_token_cap=effective_token_cap,
            prompt_token_buffer=prompt_token_buffer,
        )
        grouped_blocks = _group_partial_answers(
            partial_answers=pending_answers,
            block_budget=max(reduce_budget, 1),
        )
        next_round: list[str] = []
        total_groups = len(grouped_blocks)
        logger.info(
            "Context chat split mode reduce phase run_id=%s stage=%d input_partials=%d "
            "reduce_batches=%d reduce_budget=%d",
            run_id,
            stage_index,
            len(pending_answers),
            total_groups,
            reduce_budget,
        )
        for batch_index, analyses_block in enumerate(grouped_blocks, start=1):
            system_prompt = _compose_evidence_reduce_prompt(
                prompt_header=prompt_header,
                analyses_block=analyses_block,
                stage_index=stage_index,
                batch_index=batch_index,
                batch_count=total_groups,
            )
            next_round.append(
                context_chat_execution._run_single_pass(
                    client=client,
                    messages=_build_messages(system_prompt, reduce_history, user_content),
                    request_kwargs=request_kwargs,
                    retry_settings=retry_settings,
                    run_id=run_id,
                    context_count=context_count,
                )
            )
        pending_answers = [answer.strip() for answer in next_round if answer.strip()]
        stage_index += 1

    return pending_answers[0]


def _load_or_build_evidence_cache(
    *,
    run_id: str | None,
    normalized_contexts: list[ChatContextSource],
    normalized_citations: list[dict[str, str]],
    config: AppConfig,
) -> dict[str, object]:
    """Load a per-run evidence cache when valid or rebuild it from compact evidence."""
    evidence_items = _resolve_overflow_evidence_items(
        normalized_contexts=normalized_contexts,
        normalized_citations=normalized_citations,
    )
    source_signature = _build_evidence_source_signature(
        normalized_contexts=normalized_contexts,
        evidence_items=evidence_items,
    )
    if run_id:
        cache_path = _chat_evidence_cache_path(config.runs_dir, run_id)
        cached_payload = context_chat_io._read_json_object(cache_path)
        if (
            isinstance(cached_payload, dict)
            and cached_payload.get("schema_version") == CHAT_EVIDENCE_CACHE_SCHEMA_VERSION
            and str(cached_payload.get("source_signature", "")).strip() == source_signature
        ):
            logger.info("Context chat evidence cache hit run_id=%s", run_id)
            return cached_payload

    chunks = _build_evidence_cache_chunks(
        evidence_items=evidence_items,
        block_budget=max(config.chat.multi_pass_chunk_tokens, 1),
    )
    payload: dict[str, object] = {
        "schema_version": CHAT_EVIDENCE_CACHE_SCHEMA_VERSION,
        "source_signature": source_signature,
        "evidence_count": len(evidence_items),
        "chunks": chunks,
    }
    if run_id:
        cache_path = _chat_evidence_cache_path(config.runs_dir, run_id)
        context_chat_io._write_json_object(cache_path, payload)
        logger.info("Context chat evidence cache built run_id=%s chunks=%d", run_id, len(chunks))
    return payload


def _resolve_overflow_evidence_items(
    *,
    normalized_contexts: list[ChatContextSource],
    normalized_citations: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Choose stable compact evidence items for overflow prompts."""
    if normalized_citations:
        return _normalize_evidence_items(normalized_citations)
    return _extract_evidence_items_from_contexts(normalized_contexts)


def _normalize_evidence_items(
    normalized_citations: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Normalize compact evidence items while preserving stable ref order."""
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in normalized_citations:
        ref_id = str(item.get("ref_id", "")).strip()
        if not ref_id or ref_id in seen:
            continue
        seen.add(ref_id)
        items.append(
            {
                "ref_id": ref_id,
                "city_name": str(item.get("city_name", "")).strip(),
                "quote": str(item.get("quote", "")).strip(),
                "partial_answer": str(item.get("partial_answer", "")).strip(),
            }
        )
    return items


def _extract_evidence_items_from_contexts(
    normalized_contexts: list[ChatContextSource],
) -> list[dict[str, str]]:
    """Fallback overflow evidence extraction from stored markdown excerpts."""
    extracted: list[dict[str, str]] = []
    seen: set[str] = set()
    for context in normalized_contexts:
        markdown_payload = context.context_bundle.get("markdown")
        if not isinstance(markdown_payload, dict):
            continue
        raw_excerpts = markdown_payload.get("excerpts")
        if not isinstance(raw_excerpts, list):
            continue
        for raw_excerpt in raw_excerpts:
            if not isinstance(raw_excerpt, dict):
                continue
            ref_id = str(raw_excerpt.get("ref_id", "")).strip()
            if not ref_id or ref_id in seen or not CHAT_REF_ID_PATTERN.fullmatch(ref_id):
                continue
            quote = str(raw_excerpt.get("quote", "")).strip()
            partial_answer = str(raw_excerpt.get("partial_answer", "")).strip()
            if not quote and not partial_answer:
                continue
            extracted.append(
                {
                    "ref_id": ref_id,
                    "city_name": str(raw_excerpt.get("city_name", "")).strip(),
                    "quote": quote,
                    "partial_answer": partial_answer,
                }
            )
            seen.add(ref_id)
    return extracted


def _build_evidence_source_signature(
    *,
    normalized_contexts: list[ChatContextSource],
    evidence_items: list[dict[str, str]],
) -> str:
    """Build a stable cache signature for the active context sources and evidence."""
    payload = {
        "context_ids": [context.run_id for context in normalized_contexts],
        "evidence_items": evidence_items,
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_evidence_cache_chunks(
    cache_payload: dict[str, object],
) -> list[dict[str, object]]:
    """Return cached evidence chunks with computed ids, items, and token counts."""
    chunks = cache_payload.get("chunks")
    if not isinstance(chunks, list):
        return []
    normalized: list[dict[str, object]] = []
    for index, raw_chunk in enumerate(chunks, start=1):
        if not isinstance(raw_chunk, dict):
            continue
        raw_items = raw_chunk.get("items")
        if not isinstance(raw_items, list):
            continue
        items: list[dict[str, str]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            ref_id = str(item.get("ref_id", "")).strip()
            if not ref_id:
                continue
            items.append(
                {
                    "ref_id": ref_id,
                    "city_name": str(item.get("city_name", "")).strip(),
                    "quote": str(item.get("quote", "")).strip(),
                    "partial_answer": str(item.get("partial_answer", "")).strip(),
                }
            )
        if not items:
            continue
        chunk_id = str(raw_chunk.get("chunk_id", "")).strip() or f"chunk_{index}"
        raw_token_count = raw_chunk.get("token_count")
        token_count = (
            raw_token_count
            if isinstance(raw_token_count, int)
            else count_tokens(_render_evidence_items_block(items))
        )
        normalized.append(
            {
                "chunk_id": chunk_id,
                "ref_ids": [item["ref_id"] for item in items],
                "items": items,
                "token_count": token_count,
            }
        )
    return normalized


def _build_evidence_cache_chunks(
    *,
    evidence_items: list[dict[str, str]],
    block_budget: int,
) -> list[dict[str, object]]:
    """Fit oversized evidence items once, then pack them into ordered cache chunks."""
    return _chunk_evidence_items(
        evidence_items=evidence_items,
        block_budget=block_budget,
    )


def _chunk_evidence_items(
    *,
    evidence_items: list[dict[str, str]],
    block_budget: int,
) -> list[dict[str, object]]:
    """Pack prompt-safe evidence items into token-bounded cache chunks."""
    if block_budget <= 0:
        return []
    chunks: list[dict[str, object]] = []
    current_items: list[dict[str, str]] = []
    for item in evidence_items:
        fitted_item = _fit_evidence_item_to_budget(
            item=item,
            block_budget=block_budget,
        )
        candidate_items = current_items + [fitted_item]
        if current_items and count_tokens(_render_evidence_items_block(candidate_items)) > block_budget:
            chunks.append(
                _make_evidence_cache_chunk(
                    chunk_id=f"chunk_{len(chunks) + 1}",
                    items=current_items,
                )
            )
            current_items = [fitted_item]
            continue
        current_items = candidate_items
    if current_items:
        chunks.append(
            _make_evidence_cache_chunk(
                chunk_id=f"chunk_{len(chunks) + 1}",
                items=current_items,
            )
        )
    return chunks


def _fit_evidence_item_to_budget(
    *,
    item: dict[str, str],
    block_budget: int,
) -> dict[str, str]:
    """Trim one evidence item until its rendered block fits the cache budget."""
    fitted_item = {
        "ref_id": str(item.get("ref_id", "")).strip(),
        "city_name": str(item.get("city_name", "")).strip(),
        "quote": str(item.get("quote", "")).strip(),
        "partial_answer": str(item.get("partial_answer", "")).strip(),
    }
    if count_tokens(_render_evidence_items_block([fitted_item])) <= block_budget:
        return fitted_item

    minimal_item = dict(fitted_item)
    minimal_item["quote"] = ""
    minimal_item["partial_answer"] = ""
    if count_tokens(_render_evidence_items_block([minimal_item])) > block_budget:
        raise ValueError(
            f"Evidence item `{fitted_item['ref_id']}` does not fit the cache chunk budget "
            f"({block_budget}) even after trimming quote and partial answer."
        )

    for field_name in ("quote", "partial_answer"):
        fitted_item[field_name] = _fit_evidence_text_field_to_budget(
            item=fitted_item,
            field_name=field_name,
            block_budget=block_budget,
        )
        if count_tokens(_render_evidence_items_block([fitted_item])) <= block_budget:
            return fitted_item

    raise ValueError(
        f"Evidence item `{fitted_item['ref_id']}` does not fit the cache chunk budget "
        f"({block_budget}) after trimming."
    )


def _fit_evidence_text_field_to_budget(
    *,
    item: dict[str, str],
    field_name: str,
    block_budget: int,
) -> str:
    """Return the longest word-trimmed field value that still fits the item budget."""
    raw_value = item.get(field_name, "")
    if not raw_value:
        return ""
    words = raw_value.split()
    best_value = ""
    low = 0
    high = len(words)
    while low <= high:
        midpoint = (low + high) // 2
        candidate_item = dict(item)
        candidate_item[field_name] = _truncate_words(words, midpoint)
        if count_tokens(_render_evidence_items_block([candidate_item])) <= block_budget:
            best_value = candidate_item[field_name]
            low = midpoint + 1
            continue
        high = midpoint - 1
    return best_value


def _truncate_words(words: list[str], word_count: int) -> str:
    """Return a stable truncated text fragment from a word list."""
    if word_count <= 0:
        return ""
    if word_count >= len(words):
        return " ".join(words)
    return f"{' '.join(words[:word_count])} ..."


def _make_evidence_cache_chunk(
    *,
    chunk_id: str,
    items: list[dict[str, str]],
) -> dict[str, object]:
    """Build one cache chunk payload with stable ref ids and token count."""
    rendered_block = _render_evidence_items_block(items)
    return {
        "chunk_id": chunk_id,
        "ref_ids": [item["ref_id"] for item in items],
        "items": items,
        "token_count": count_tokens(rendered_block),
    }


def _resolve_request_evidence_blocks(
    *,
    cache_chunks: list[dict[str, object]],
    block_budget: int,
) -> list[str]:
    """Return request blocks, splitting one oversized cached chunk into two halves once."""
    if block_budget <= 0:
        return []
    request_chunks: list[dict[str, object]] = []
    for chunk in cache_chunks:
        token_count = chunk.get("token_count")
        if isinstance(token_count, int) and token_count <= block_budget:
            request_chunks.append(chunk)
            continue
        request_chunks.extend(
            _split_evidence_cache_chunk_once(
                chunk=chunk,
                block_budget=block_budget,
            )
        )
    return [_render_evidence_items_block(chunk["items"]) for chunk in request_chunks]


def _split_evidence_cache_chunk_once(
    *,
    chunk: dict[str, object],
    block_budget: int,
) -> list[dict[str, object]]:
    """Split one oversized chunk in half once or raise if either half still overflows."""
    items = chunk.get("items")
    chunk_id = str(chunk.get("chunk_id", "")).strip() or "chunk"
    token_count = chunk.get("token_count")
    resolved_token_count = int(token_count) if isinstance(token_count, int) else -1
    if not isinstance(items, list) or len(items) < 2:
        raise ValueError(
            f"Evidence chunk `{chunk_id}` does not fit the map prompt budget "
            f"({resolved_token_count} > {block_budget}) and cannot be split further."
        )

    midpoint = len(items) // 2
    left_chunk = _make_evidence_cache_chunk(
        chunk_id=f"{chunk_id}_a",
        items=items[:midpoint],
    )
    right_chunk = _make_evidence_cache_chunk(
        chunk_id=f"{chunk_id}_b",
        items=items[midpoint:],
    )
    for split_chunk in (left_chunk, right_chunk):
        split_token_count = split_chunk["token_count"]
        if isinstance(split_token_count, int) and split_token_count > block_budget:
            raise ValueError(
                f"Evidence chunk `{chunk_id}` still exceeds the map prompt budget after one half split "
                f"({split_token_count} > {block_budget})."
            )
    return [left_chunk, right_chunk]


def _group_partial_answers(
    *,
    partial_answers: list[str],
    block_budget: int,
) -> list[str]:
    """Group partial answers into token-bounded reduce batches."""
    if block_budget <= 0:
        return []
    groups: list[list[str]] = []
    current_group: list[str] = []
    for answer in partial_answers:
        candidate = current_group + [answer]
        if current_group and count_tokens(_render_partial_answers_block(candidate)) > block_budget:
            groups.append(current_group)
            current_group = [answer]
            continue
        current_group = candidate
    if current_group:
        groups.append(current_group)
    return [_render_partial_answers_block(group) for group in groups]


def _resolve_prompt_budget(
    *,
    prompt_factory: Callable[[], str],
    history: list[dict[str, str]],
    user_content: str,
    effective_token_cap: int,
    prompt_token_buffer: int,
    minimum_remaining_budget: int = 1,
) -> tuple[list[dict[str, str]], int]:
    """Trim history until the prompt has enough remaining budget for context blocks."""
    working_history = list(history)
    while True:
        system_prompt = prompt_factory()
        prompt_tokens = _estimate_messages_tokens(
            _build_messages(system_prompt, working_history, user_content)
        )
        remaining_budget = effective_token_cap - prompt_tokens - prompt_token_buffer
        if (
            (remaining_budget >= minimum_remaining_budget and remaining_budget > 0)
            or not working_history
        ):
            return working_history, remaining_budget
        working_history = working_history[1:]


__all__ = ["CHAT_EVIDENCE_CACHE_SCHEMA_VERSION"]
