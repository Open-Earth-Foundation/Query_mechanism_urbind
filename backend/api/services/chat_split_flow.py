"""Split-mode queueing and background execution for context chat."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable
from uuid import uuid4

from backend.api.models import (
    ChatMessageJobAcceptedResponse,
    SendChatMessageCompletedResponse,
    SendChatMessageResponse,
)
from backend.api.services.chat_context_loader import (
    load_context_for_run_id,
    load_followup_bundles_by_ids,
)
from backend.api.services.chat_jobs import (
    ChatJobExecutor,
    ChatJobResult,
    StartChatJobCommand,
    build_chat_job_failure_message,
)
from backend.api.services.chat_memory import ChatMemoryStore
from backend.api.services.chat_reply_helpers import answer_from_context_reply, build_chat_sources
from backend.api.services.chat_session_helpers import (
    as_chat_routing,
    as_message,
    as_pending_job,
    as_session_prompt_context_cache,
)
from backend.api.services.run_store import RunStore
from backend.utils.config import load_config

logger = logging.getLogger(__name__)

ChatJobProcessor = Callable[[StartChatJobCommand], ChatJobResult]


def build_chat_job_processor(
    *,
    run_store: RunStore,
    chat_memory_store: ChatMemoryStore,
    config_path: Path,
) -> ChatJobProcessor:
    """Build the queued split-mode worker for long-context chat replies."""

    def _processor(command: StartChatJobCommand) -> ChatJobResult:
        session = chat_memory_store.get_session(command.run_id, command.conversation_id)
        config = load_config(config_path)
        config.enable_sql = False
        loaded_contexts = [
            load_context_for_run_id(run_store, context_run_id, config)
            for context_run_id in command.context_run_ids
        ]
        if not loaded_contexts:
            raise ValueError("No valid run contexts remain for the queued split-mode chat job.")
        loaded_followup_bundles = load_followup_bundles_by_ids(
            run_store=run_store,
            run_id=command.run_id,
            conversation_id=command.conversation_id,
            bundle_ids=command.followup_bundle_ids,
            config=config,
        )
        sources = build_chat_sources(loaded_contexts, loaded_followup_bundles)
        prompt_cache = as_session_prompt_context_cache(
            session,
            context_run_ids=[context.run_id for context in loaded_contexts],
            followup_bundle_ids=[bundle.bundle_id for bundle in loaded_followup_bundles],
        )
        assistant_text, assistant_citations, assistant_citation_warning = (
            answer_from_context_reply(
                run_id=command.run_id,
                original_question=command.original_question,
                sources=sources,
                history=command.history,
                user_content=command.user_content,
                config=config,
                token_cap=command.token_cap,
                api_key_override=command.api_key_override,
                citation_prefix_tokens=(
                    prompt_cache.citation_prefix_tokens if prompt_cache is not None else None
                ),
            )
        )
        return ChatJobResult(
            assistant_content=assistant_text,
            assistant_citations=assistant_citations or None,
            assistant_citation_warning=assistant_citation_warning,
        )

    return _processor


def queue_split_context_chat_job(
    *,
    run_id: str,
    conversation_id: str,
    user_content: str,
    original_question: str,
    effective_user_content: str,
    history: list[dict[str, str]],
    context_run_ids: list[str],
    followup_bundle_ids: list[str],
    token_cap: int,
    assistant_routing: dict[str, object] | None,
    api_key_override: str | None,
    chat_memory_store: ChatMemoryStore,
    chat_job_executor: ChatJobExecutor,
) -> SendChatMessageResponse:
    """Persist one split-mode user turn and queue the assistant answer in the background."""
    job_id = uuid4().hex
    _, pending_job = chat_memory_store.create_pending_job(
        run_id=run_id,
        conversation_id=conversation_id,
        job_id=job_id,
    )
    try:
        _, user_message = chat_memory_store.append_user_message(
            run_id=run_id,
            conversation_id=conversation_id,
            content=user_content,
        )
    except Exception:
        chat_memory_store.clear_pending_job(
            run_id=run_id,
            conversation_id=conversation_id,
            job_id=job_id,
        )
        raise
    command = StartChatJobCommand(
        run_id=run_id,
        conversation_id=conversation_id,
        job_id=job_id,
        job_number=int(pending_job["job_number"]),
        original_question=original_question,
        user_content=effective_user_content,
        history=history,
        context_run_ids=context_run_ids,
        followup_bundle_ids=followup_bundle_ids,
        token_cap=token_cap,
        assistant_routing=assistant_routing,
        api_key_override=api_key_override,
    )
    try:
        chat_job_executor.submit(command)
    except Exception:  # noqa: BLE001
        logger.exception(
            "Context chat split job submission failed run_id=%s conversation_id=%s job_id=%s job_number=%s",
            run_id,
            conversation_id,
            job_id,
            pending_job["job_number"],
        )
        chat_memory_store.clear_pending_job(
            run_id=run_id,
            conversation_id=conversation_id,
            job_id=job_id,
        )
        _, assistant_message = chat_memory_store.append_assistant_message(
            run_id=run_id,
            conversation_id=conversation_id,
            content=build_chat_job_failure_message(),
        )
        return SendChatMessageCompletedResponse(
            mode="completed",
            run_id=run_id,
            conversation_id=conversation_id,
            user_message=as_message(user_message),
            assistant_message=as_message(assistant_message),
        )

    pending_handle = as_pending_job(run_id, conversation_id, pending_job)
    if pending_handle is None:
        raise ValueError("Queued split-mode chat job is missing pending-job metadata.")
    return ChatMessageJobAcceptedResponse(
        mode="queued",
        run_id=run_id,
        conversation_id=conversation_id,
        user_message=as_message(user_message),
        job=pending_handle,
        routing=as_chat_routing(assistant_routing),
    )


__all__ = ["build_chat_job_processor", "queue_split_context_chat_job"]
