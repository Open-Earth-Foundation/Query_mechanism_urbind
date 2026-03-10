"""Build and persist chat-owned follow-up excerpt bundles."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from backend.api.services.city_catalog import list_city_names
from backend.api.services.context_prompt_cache import (
    compute_prompt_context_cache,
    write_prompt_context_cache,
)
from backend.modules.markdown_researcher.agent import extract_markdown_excerpts
from backend.modules.markdown_researcher.services import load_markdown_documents
from backend.modules.orchestrator.agent import refine_research_question
from backend.modules.orchestrator.utils.references import build_markdown_references
from backend.modules.vector_store.retriever import (
    as_markdown_documents,
    list_indexed_city_names,
    retrieve_chunks_for_queries,
)
from backend.utils.city_normalization import format_city_display_name, normalize_city_key
from backend.utils.config import AppConfig
from backend.utils.json_io import write_json
from backend.utils.tokenization import count_tokens

logger = logging.getLogger(__name__)

FollowupSearchStatus = Literal["success", "error"]
CHAT_FOLLOWUP_CITY_UNAVAILABLE = "CHAT_FOLLOWUP_CITY_UNAVAILABLE"
CHAT_FOLLOWUP_SEARCH_FAILED = "CHAT_FOLLOWUP_SEARCH_FAILED"


class CityUnavailableError(ValueError):
    """Raised when the requested follow-up city is not searchable."""


@dataclass(frozen=True)
class ChatFollowupSearchResult:
    """Outcome of one persisted follow-up research attempt."""

    status: FollowupSearchStatus
    bundle_id: str
    target_city: str
    created_at: datetime
    excerpt_count: int
    total_tokens: int
    error_code: str | None = None
    error_message: str | None = None


def run_chat_followup_search(
    *,
    runs_dir: Path,
    run_id: str,
    conversation_id: str,
    turn_index: int,
    question: str,
    target_city: str,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
) -> ChatFollowupSearchResult:
    """Run one follow-up markdown search and persist the resulting bundle."""
    config.enable_sql = False
    created_at = datetime.now(timezone.utc)
    city_name = format_city_display_name(target_city) or target_city.strip()
    city_key = normalize_city_key(city_name)
    bundle_id = build_followup_bundle_id(
        conversation_id=conversation_id,
        turn_index=turn_index,
        city_key=city_key or "unknown",
    )
    bundle_dir = followup_bundle_dir(
        runs_dir=runs_dir,
        run_id=run_id,
        conversation_id=conversation_id,
        bundle_id=bundle_id,
    )

    try:
        _ensure_target_city_available(city_name, config)
        refinement = refine_research_question(
            question=question,
            config=config,
            api_key=api_key,
            selected_cities=[city_name],
            log_llm_payload=log_llm_payload,
        )
        research_question = refinement.research_question.strip() or question.strip()
        retrieval_queries = _dedupe_queries(
            [research_question, *refinement.retrieval_queries]
        )
        documents, retrieval_payload, source_mode = _load_followup_documents(
            research_question=research_question,
            retrieval_queries=retrieval_queries,
            target_city=city_name,
            config=config,
        )
        markdown_result = extract_markdown_excerpts(
            research_question,
            documents,
            config,
            api_key=api_key,
            log_llm_payload=log_llm_payload,
            run_id=run_id,
        )
        return _persist_followup_result(
            bundle_dir=bundle_dir,
            bundle_id=bundle_id,
            run_id=run_id,
            conversation_id=conversation_id,
            config=config,
            created_at=created_at,
            target_city=city_name,
            research_question=research_question,
            retrieval_queries=retrieval_queries,
            source_mode=source_mode,
            retrieval_payload=retrieval_payload,
            markdown_payload=markdown_result.model_dump(),
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.warning(
            "Follow-up search failed run_id=%s conversation_id=%s city=%s error=%s",
            run_id,
            conversation_id,
            city_name,
            exc,
        )
        error_payload = {
            "status": "error",
            "excerpts": [],
            "excerpt_count": 0,
            "selected_city_names": [city_name],
            "inspected_city_names": [],
            "source_mode": "error",
            "error": {
                "code": _classify_followup_error(exc),
                "message": str(exc),
            },
        }
        return _persist_followup_result(
            bundle_dir=bundle_dir,
            bundle_id=bundle_id,
            run_id=run_id,
            conversation_id=conversation_id,
            config=config,
            created_at=created_at,
            target_city=city_name,
            research_question=question.strip(),
            retrieval_queries=[question.strip()],
            source_mode="error",
            retrieval_payload=None,
            markdown_payload=error_payload,
        )


def build_followup_bundle_id(conversation_id: str, turn_index: int, city_key: str) -> str:
    """Build a stable follow-up bundle identifier."""
    conversation_prefix = conversation_id.strip()[:8] or "chat"
    return f"fup_{conversation_prefix}_{turn_index:03d}_{city_key}"


def followup_bundle_dir(
    *, runs_dir: Path, run_id: str, conversation_id: str, bundle_id: str
) -> Path:
    """Return the artifact directory for one follow-up bundle."""
    return runs_dir / run_id / "chat" / conversation_id / "followups" / bundle_id


def _load_followup_documents(
    *,
    research_question: str,
    retrieval_queries: list[str],
    target_city: str,
    config: AppConfig,
) -> tuple[list[dict[str, object]], dict[str, object] | None, str]:
    """Load one-city markdown documents for follow-up extraction."""
    if config.vector_store.enabled:
        chunks, retrieval_meta = retrieve_chunks_for_queries(
            queries=retrieval_queries,
            config=config,
            docs_dir=config.markdown_dir,
            selected_cities=[target_city],
        )
        documents = as_markdown_documents(chunks)
        retrieval_payload = {
            "research_question": research_question,
            "queries": retrieval_queries,
            "selected_cities": [target_city],
            "retrieved_count": len(chunks),
            "meta": retrieval_meta,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "city_name": chunk.city_name,
                    "city_key": str(chunk.metadata.get("city_key", "")),
                    "source_path": chunk.source_path,
                    "heading_path": chunk.heading_path,
                    "block_type": chunk.block_type,
                    "distance": chunk.distance,
                }
                for chunk in chunks
            ],
        }
        return documents, retrieval_payload, "vector_store_retrieval"

    documents = load_markdown_documents(
        config.markdown_dir,
        config.markdown_researcher,
        selected_cities=[target_city],
    )
    return documents, None, "standard_chunking"


def _ensure_target_city_available(target_city: str, config: AppConfig) -> None:
    """Fail fast when the requested city is not present in the active search source."""
    target_key = normalize_city_key(target_city)
    available_names = _list_searchable_city_names(config)
    if not available_names:
        return
    available_keys = {normalize_city_key(name) for name in available_names}
    if target_key in available_keys:
        return

    if config.vector_store.enabled:
        raise CityUnavailableError("Selected city is not available in the vector store index.")
    raise CityUnavailableError("Selected city is not available in markdown documents.")


def _list_searchable_city_names(config: AppConfig) -> list[str]:
    """Return city names that can be searched in the current follow-up mode."""
    try:
        if config.vector_store.enabled:
            return list_indexed_city_names(config)
        return list_city_names(config.markdown_dir)
    except (FileNotFoundError, ValueError):
        return []


def _persist_followup_result(
    *,
    bundle_dir: Path,
    bundle_id: str,
    run_id: str,
    conversation_id: str,
    config: AppConfig,
    created_at: datetime,
    target_city: str,
    research_question: str,
    retrieval_queries: list[str],
    source_mode: str,
    retrieval_payload: dict[str, object] | None,
    markdown_payload: dict[str, Any],
) -> ChatFollowupSearchResult:
    """Persist follow-up artifacts and return a compact result summary."""
    excerpt_records = _coerce_excerpt_records(markdown_payload.get("excerpts"))
    excerpt_count = len(excerpt_records)
    enriched_excerpts = excerpt_records
    references_payload: dict[str, object] = {"references": []}
    if excerpt_records:
        enriched_excerpts, references_payload = build_markdown_references(
            run_id=bundle_id,
            excerpts=excerpt_records,
        )

    markdown_bundle = dict(markdown_payload)
    markdown_bundle["excerpts"] = enriched_excerpts
    markdown_bundle["excerpt_count"] = excerpt_count
    markdown_bundle["selected_city_names"] = [target_city]
    markdown_bundle["inspected_city_names"] = [target_city] if excerpt_count > 0 else []
    markdown_bundle["source_mode"] = source_mode

    context_bundle = {
        "bundle_id": bundle_id,
        "parent_run_id": run_id,
        "conversation_id": conversation_id,
        "source": "chat_followup",
        "created_at": created_at.isoformat(),
        "target_city": target_city,
        "research_question": research_question,
        "retrieval_queries": retrieval_queries,
        "sql": None,
        "final": None,
        "analysis_mode": "aggregate",
        "markdown": markdown_bundle,
    }

    context_bundle_path = bundle_dir / "context_bundle.json"
    markdown_excerpts_path = bundle_dir / "markdown" / "excerpts.json"
    write_json(context_bundle_path, context_bundle, ensure_ascii=False, default=str)
    write_json(
        markdown_excerpts_path,
        {"excerpts": enriched_excerpts},
        ensure_ascii=False,
        default=str,
    )
    write_json(
        bundle_dir / "markdown" / "references.json",
        references_payload,
        ensure_ascii=False,
        default=str,
    )
    if retrieval_payload is not None:
        write_json(
            bundle_dir / "markdown" / "retrieval.json",
            retrieval_payload,
            ensure_ascii=False,
            default=str,
        )

    prompt_context_tokens, prompt_context_kind = compute_prompt_context_cache(
        question=research_question,
        final_document="",
        context_bundle=context_bundle,
        config=config,
    )
    context_bundle = write_prompt_context_cache(
        context_bundle_path=context_bundle_path,
        markdown_excerpts_path=markdown_excerpts_path,
        context_bundle=context_bundle,
        prompt_context_tokens=prompt_context_tokens,
        prompt_context_kind=prompt_context_kind,
    )
    bundle_text = json.dumps(context_bundle, ensure_ascii=False, default=str)
    error_code = _extract_error_code(markdown_bundle.get("error"))
    error_message = _extract_error_message(markdown_bundle.get("error"))
    status: FollowupSearchStatus = "error" if error_code or error_message else "success"
    return ChatFollowupSearchResult(
        status=status,
        bundle_id=bundle_id,
        target_city=target_city,
        created_at=created_at,
        excerpt_count=excerpt_count,
        total_tokens=count_tokens(bundle_text),
        error_code=error_code,
        error_message=error_message,
    )


def _coerce_excerpt_records(value: object) -> list[dict[str, object]]:
    """Normalize raw excerpt payloads into dict records."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dedupe_queries(queries: list[str]) -> list[str]:
    """Normalize retrieval queries while preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        candidate = query.strip()
        if not candidate:
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped[:3] or ["follow-up city question"]


def _classify_followup_error(exc: Exception) -> str:
    """Map follow-up exceptions to stable persisted error codes."""
    if isinstance(exc, CityUnavailableError):
        return CHAT_FOLLOWUP_CITY_UNAVAILABLE
    message = str(exc)
    if message.startswith("Selected city is not available"):
        return CHAT_FOLLOWUP_CITY_UNAVAILABLE
    if message.startswith("Selected cities are not indexed in vector store manifest:"):
        return CHAT_FOLLOWUP_CITY_UNAVAILABLE
    return CHAT_FOLLOWUP_SEARCH_FAILED


def _extract_error_code(value: object) -> str | None:
    """Return the persisted markdown error code when present."""
    if not isinstance(value, dict):
        return None
    code = value.get("code")
    if isinstance(code, str) and code.strip():
        return code.strip()
    return None


def _extract_error_message(value: object) -> str | None:
    """Return the persisted markdown error message when present."""
    if not isinstance(value, dict):
        return None
    message = value.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return None


__all__ = [
    "CHAT_FOLLOWUP_CITY_UNAVAILABLE",
    "CHAT_FOLLOWUP_SEARCH_FAILED",
    "ChatFollowupSearchResult",
    "build_followup_bundle_id",
    "followup_bundle_dir",
    "run_chat_followup_search",
]
