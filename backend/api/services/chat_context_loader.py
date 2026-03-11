"""Context loading utilities for chat sessions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import logging

from backend.api.models import ChatContextSummary
from backend.api.services.context_prompt_cache import (
    ensure_prompt_context_cache,
    read_token_sidecar,
    write_token_sidecar,
)
from backend.api.services.models import (
    LoadedChatSource,
    LoadedContext,
    LoadedFollowupBundle,
)
from backend.api.services.run_store import RunRecord, RunStore, SUCCESS_STATUSES
from backend.modules.writer.utils.markdown_helpers import (
    extract_markdown_bundle,
    extract_markdown_excerpts as extract_bundle_excerpts,
)
from backend.utils.config import AppConfig
from backend.utils.tokenization import count_tokens
from backend.api.services.chat_followup_research import followup_bundle_dir
from backend.api.services.context_chat import load_context_bundle, load_final_document

logger = logging.getLogger(__name__)


def resolve_final_output_path(run_store: RunStore, run_id: str, raw_path: Path | None) -> Path:
    """Resolve final output path or raise when missing."""
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


def resolve_context_bundle_path(
    run_store: RunStore, run_id: str, raw_path: Path | None
) -> Path:
    """Resolve context bundle path or raise when missing."""
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


def load_context_for_record(
    run_store: RunStore,
    run_record: RunRecord,
    config: AppConfig,
) -> LoadedContext:
    """Load context material for one completed run."""
    final_path = resolve_final_output_path(
        run_store, run_record.run_id, run_record.final_output_path
    )
    context_path = resolve_context_bundle_path(
        run_store, run_record.run_id, run_record.context_bundle_path
    )
    final_document = load_final_document(final_path)
    context_bundle = load_context_bundle(context_path)
    markdown_excerpts_path = context_path.parent / "markdown" / "excerpts.json"
    context_bundle, prompt_context_tokens, prompt_context_kind, cache_status = (
        ensure_prompt_context_cache(
            context_bundle_path=context_path,
            markdown_excerpts_path=markdown_excerpts_path,
            question=run_record.question,
            final_document=final_document,
            context_bundle=context_bundle,
            config=config,
        )
    )
    logger.info(
        "Context prompt cache %s source_type=run source_id=%s prompt_context_tokens=%d prompt_context_kind=%s",
        cache_status,
        run_record.run_id,
        prompt_context_tokens,
        prompt_context_kind,
    )
    document_tokens = count_tokens(final_document)
    bundle_tokens = count_tokens(context_path.read_text(encoding="utf-8"))
    run_dir = run_store.runs_dir / run_record.run_id
    write_token_sidecar(
        run_dir,
        document_tokens=document_tokens,
        bundle_tokens=bundle_tokens,
        prompt_context_tokens=prompt_context_tokens,
        prompt_context_kind=prompt_context_kind,
    )
    return LoadedContext(
        run_id=run_record.run_id,
        question=run_record.question,
        status=run_record.status,
        started_at=run_record.started_at,
        final_output_path=final_path,
        context_bundle_path=context_path,
        markdown_excerpts_path=markdown_excerpts_path,
        final_document=final_document,
        context_bundle=context_bundle,
        document_tokens=document_tokens,
        bundle_tokens=bundle_tokens,
        prompt_context_tokens=prompt_context_tokens,
        prompt_context_kind=prompt_context_kind,
    )


def load_context_for_run_id(
    run_store: RunStore,
    run_id: str,
    config: AppConfig,
) -> LoadedContext:
    """Load context material for a specific run id."""
    run_record = run_store.get_run(run_id)
    if run_record is None:
        raise ValueError(f"Context run `{run_id}` was not found.")
    if run_record.status not in SUCCESS_STATUSES:
        raise ValueError(
            f"Context run `{run_id}` is not ready (status: `{run_record.status}`)."
        )
    return load_context_for_record(run_store, run_record, config)


def validate_context_run_id(run_store: RunStore, run_id: str) -> None:
    """Check that a run ID is usable as a chat context without reading any files.

    Raises ValueError when the run is missing, not completed, or has no artifacts.
    Only the run record and artifact path existence are checked; no files are read.
    """
    run_record = run_store.get_run(run_id)
    if run_record is None:
        raise ValueError(f"Context run `{run_id}` was not found.")
    if run_record.status not in SUCCESS_STATUSES:
        raise ValueError(
            f"Context run `{run_id}` is not ready (status: `{run_record.status}`)."
        )
    resolve_final_output_path(run_store, run_id, run_record.final_output_path)
    resolve_context_bundle_path(run_store, run_id, run_record.context_bundle_path)


def fast_context_summary(
    run_store: RunStore,
    run_id: str,
    config: AppConfig,
) -> ChatContextSummary:
    """Return a ChatContextSummary for a run using the token sidecar when available.

    Avoids loading the full context bundle unless the sidecar is missing.
    Raises ValueError when the run is not found.
    """
    run_record = run_store.get_run(run_id)
    if run_record is None:
        raise ValueError(f"Context run `{run_id}` was not found.")
    return _fast_context_summary(run_store, run_record, config)


def available_contexts(run_store: RunStore, config: AppConfig) -> list[LoadedContext]:
    """List all completed runs that have usable chat context artifacts."""
    contexts: list[LoadedContext] = []
    for run_record in run_store.list_runs():
        if run_record.status not in SUCCESS_STATUSES:
            continue
        try:
            contexts.append(load_context_for_record(run_store, run_record, config))
        except ValueError:
            continue
    return contexts


def list_available_context_summaries(
    run_store: RunStore, config: AppConfig
) -> list[ChatContextSummary]:
    """Return context summaries for all completed runs using cached token sidecars.

    For runs with a warm sidecar (token_cache.json), no large files are read and no
    tokenization is performed.  For runs with a cold sidecar the full context is loaded
    once and the sidecar is written as a side-effect so subsequent calls are fast.
    """
    summaries: list[ChatContextSummary] = []
    for run_record in run_store.list_runs():
        if run_record.status not in SUCCESS_STATUSES:
            continue
        try:
            summary = _fast_context_summary(run_store, run_record, config)
            summaries.append(summary)
        except ValueError:
            continue
    return summaries


def _fast_context_summary(
    run_store: RunStore,
    run_record: RunRecord,
    config: AppConfig,
) -> ChatContextSummary:
    """Build a ChatContextSummary using the token sidecar when available."""
    run_dir = run_store.runs_dir / run_record.run_id
    final_path = resolve_final_output_path(
        run_store, run_record.run_id, run_record.final_output_path
    )
    context_path = resolve_context_bundle_path(
        run_store, run_record.run_id, run_record.context_bundle_path
    )
    sidecar = read_token_sidecar(run_dir)
    if sidecar is not None:
        logger.debug("Token sidecar hit run_id=%s", run_record.run_id)
        return ChatContextSummary(
            run_id=run_record.run_id,
            question=run_record.question,
            status=run_record.status,
            started_at=run_record.started_at,
            final_output_path=str(final_path),
            context_bundle_path=str(context_path),
            document_tokens=sidecar.document_tokens,
            bundle_tokens=sidecar.bundle_tokens,
            total_tokens=sidecar.document_tokens + sidecar.bundle_tokens,
            prompt_context_tokens=sidecar.prompt_context_tokens,
            prompt_context_kind=sidecar.prompt_context_kind,
        )
    logger.debug("Token sidecar miss run_id=%s — loading full context", run_record.run_id)
    loaded = load_context_for_record(run_store, run_record, config)
    return loaded.to_summary()


def load_followup_bundle(
    *,
    run_store: RunStore,
    run_id: str,
    conversation_id: str,
    bundle_id: str,
    config: AppConfig,
    bundle_meta: dict[str, str] | None = None,
) -> LoadedFollowupBundle:
    """Load one persisted follow-up bundle referenced by the session."""
    bundle_dir = followup_bundle_dir(
        runs_dir=run_store.runs_dir,
        run_id=run_id,
        conversation_id=conversation_id,
        bundle_id=bundle_id,
    )
    context_bundle_path = bundle_dir / "context_bundle.json"
    if not context_bundle_path.exists():
        raise ValueError(f"Follow-up bundle `{bundle_id}` is missing context_bundle.json.")
    context_bundle = load_context_bundle(context_bundle_path)
    markdown_excerpts_path = bundle_dir / "markdown" / "excerpts.json"
    context_bundle, prompt_context_tokens, prompt_context_kind, cache_status = (
        ensure_prompt_context_cache(
            context_bundle_path=context_bundle_path,
            markdown_excerpts_path=markdown_excerpts_path,
            question=str(context_bundle.get("research_question", "")).strip(),
            final_document="",
            context_bundle=context_bundle,
            config=config,
        )
    )
    logger.info(
        "Context prompt cache %s source_type=followup_bundle source_id=%s prompt_context_tokens=%d prompt_context_kind=%s",
        cache_status,
        bundle_id,
        prompt_context_tokens,
        prompt_context_kind,
    )
    bundle_text = context_bundle_path.read_text(encoding="utf-8")
    markdown_bundle = extract_markdown_bundle(context_bundle)
    excerpt_count = len(extract_bundle_excerpts(markdown_bundle))
    fallback_target_city = bundle_meta["target_city"] if bundle_meta is not None else bundle_id
    fallback_created_at = (
        bundle_meta["created_at"] if bundle_meta is not None else datetime.now().isoformat()
    )
    target_city = str(context_bundle.get("target_city", "")).strip() or fallback_target_city
    created_at_raw = str(context_bundle.get("created_at", "")).strip() or fallback_created_at
    try:
        created_at = datetime.fromisoformat(created_at_raw)
    except ValueError as exc:
        raise ValueError(f"Follow-up bundle `{bundle_id}` has invalid created_at.") from exc
    return LoadedFollowupBundle(
        bundle_id=bundle_id,
        target_city=target_city,
        created_at=created_at,
        context_bundle_path=context_bundle_path,
        markdown_excerpts_path=markdown_excerpts_path,
        context_bundle=context_bundle,
        bundle_tokens=count_tokens(bundle_text),
        excerpt_count=excerpt_count,
        prompt_context_tokens=prompt_context_tokens,
        prompt_context_kind=prompt_context_kind,
    )


def load_followup_bundles_by_ids(
    *,
    run_store: RunStore,
    run_id: str,
    conversation_id: str,
    bundle_ids: list[str],
    config: AppConfig,
) -> list[LoadedFollowupBundle]:
    """Load persisted follow-up bundles directly from the stored job snapshot."""
    loaded: list[LoadedFollowupBundle] = []
    for bundle_id in bundle_ids:
        if not isinstance(bundle_id, str) or not bundle_id.strip():
            continue
        loaded.append(
            load_followup_bundle(
                run_store=run_store,
                run_id=run_id,
                conversation_id=conversation_id,
                bundle_id=bundle_id.strip(),
                config=config,
                bundle_meta=None,
            )
        )
    return loaded


__all__ = [
    "LoadedContext",
    "LoadedFollowupBundle",
    "LoadedChatSource",
    "load_context_for_record",
    "load_context_for_run_id",
    "available_contexts",
    "list_available_context_summaries",
    "load_followup_bundle",
    "load_followup_bundles_by_ids",
    "resolve_final_output_path",
    "resolve_context_bundle_path",
]
