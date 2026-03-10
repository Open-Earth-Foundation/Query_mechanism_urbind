"""Persistent prompt-context token cache helpers for saved chat sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from backend.api.services.context_chat import (
    build_citation_catalog_from_contexts,
    estimate_context_window,
    resolve_chat_token_cap,
)
from backend.utils.config import AppConfig
from backend.utils.json_io import read_json_object, write_json

PromptContextKind = Literal["citation_catalog", "serialized_contexts"]
_VALID_PROMPT_CONTEXT_KINDS = {"citation_catalog", "serialized_contexts"}
_TOKEN_SIDECAR_NAME = "token_cache.json"


@dataclass(frozen=True)
class TokenSidecar:
    """All token metrics for one run, stored in a tiny sidecar file."""

    document_tokens: int
    bundle_tokens: int
    prompt_context_tokens: int
    prompt_context_kind: PromptContextKind


def read_token_sidecar(run_dir: Path) -> TokenSidecar | None:
    """Return cached token metrics from the run sidecar when present and valid."""
    payload = read_json_object(run_dir / _TOKEN_SIDECAR_NAME)
    if payload is None:
        return None
    doc = payload.get("document_tokens")
    bundle = payload.get("bundle_tokens")
    prompt = payload.get("prompt_context_tokens")
    kind = payload.get("prompt_context_kind")
    if not isinstance(doc, int) or doc < 0:
        return None
    if not isinstance(bundle, int) or bundle < 0:
        return None
    if not isinstance(prompt, int) or prompt < 0:
        return None
    if kind not in _VALID_PROMPT_CONTEXT_KINDS:
        return None
    return TokenSidecar(
        document_tokens=doc,
        bundle_tokens=bundle,
        prompt_context_tokens=prompt,
        prompt_context_kind=cast(PromptContextKind, kind),
    )


def write_token_sidecar(
    run_dir: Path,
    *,
    document_tokens: int,
    bundle_tokens: int,
    prompt_context_tokens: int,
    prompt_context_kind: PromptContextKind,
) -> None:
    """Persist all token metrics to the run-level sidecar file."""
    write_json(
        run_dir / _TOKEN_SIDECAR_NAME,
        {
            "document_tokens": document_tokens,
            "bundle_tokens": bundle_tokens,
            "prompt_context_tokens": prompt_context_tokens,
            "prompt_context_kind": prompt_context_kind,
        },
        ensure_ascii=False,
        default=str,
    )


def read_prompt_context_cache(
    payload: dict[str, Any] | None,
) -> tuple[int, PromptContextKind] | None:
    """Return cached prompt-context metrics when present and valid."""
    if not isinstance(payload, dict):
        return None
    raw_tokens = payload.get("prompt_context_tokens")
    raw_kind = payload.get("prompt_context_kind")
    if not isinstance(raw_tokens, int) or raw_tokens < 0:
        return None
    if raw_kind not in _VALID_PROMPT_CONTEXT_KINDS:
        return None
    return raw_tokens, cast(PromptContextKind, raw_kind)


def compute_prompt_context_cache(
    *,
    question: str,
    final_document: str,
    context_bundle: dict[str, Any],
    config: AppConfig,
) -> tuple[int, PromptContextKind]:
    """Compute prompt-context metrics using the exact chat estimation path."""
    context_payload = {
        "run_id": "prompt_context_cache_source",
        "question": question,
        "final_document": final_document,
        "context_bundle": context_bundle,
    }
    citation_catalog = build_citation_catalog_from_contexts([context_payload])
    estimate = estimate_context_window(
        original_question=question,
        contexts=[context_payload],
        config=config,
        token_cap=resolve_chat_token_cap(config),
        citation_catalog=citation_catalog,
    )
    resolved_kind = estimate.context_window_kind
    if resolved_kind not in _VALID_PROMPT_CONTEXT_KINDS:
        resolved_kind = "citation_catalog" if citation_catalog else "serialized_contexts"
    return estimate.context_window_tokens or 0, cast(PromptContextKind, resolved_kind)


def write_prompt_context_cache(
    *,
    context_bundle_path: Path,
    markdown_excerpts_path: Path | None,
    context_bundle: dict[str, Any],
    prompt_context_tokens: int,
    prompt_context_kind: PromptContextKind,
) -> dict[str, Any]:
    """Persist prompt-context metrics into bundle artifacts and return updated bundle."""
    updated_bundle = dict(context_bundle)
    updated_bundle["prompt_context_tokens"] = prompt_context_tokens
    updated_bundle["prompt_context_kind"] = prompt_context_kind
    write_json(context_bundle_path, updated_bundle, ensure_ascii=False, default=str)

    if markdown_excerpts_path is not None:
        excerpts_payload = read_json_object(markdown_excerpts_path)
        if not isinstance(excerpts_payload, dict):
            markdown_payload = updated_bundle.get("markdown")
            excerpts_payload = (
                dict(markdown_payload) if isinstance(markdown_payload, dict) else {"excerpts": []}
            )
        excerpts_payload["prompt_context_tokens"] = prompt_context_tokens
        excerpts_payload["prompt_context_kind"] = prompt_context_kind
        write_json(markdown_excerpts_path, excerpts_payload, ensure_ascii=False, default=str)

    return updated_bundle


def ensure_prompt_context_cache(
    *,
    context_bundle_path: Path,
    markdown_excerpts_path: Path | None,
    question: str,
    final_document: str,
    context_bundle: dict[str, Any],
    config: AppConfig,
) -> tuple[dict[str, Any], int, PromptContextKind, Literal["hit", "miss"]]:
    """Load cached prompt-context metrics or compute and persist them once."""
    bundle_cache = read_prompt_context_cache(context_bundle)
    excerpts_payload = (
        read_json_object(markdown_excerpts_path) if markdown_excerpts_path is not None else None
    )
    excerpts_cache = read_prompt_context_cache(excerpts_payload)

    if bundle_cache is not None:
        prompt_context_tokens, prompt_context_kind = bundle_cache
        if excerpts_cache != bundle_cache:
            updated_bundle = write_prompt_context_cache(
                context_bundle_path=context_bundle_path,
                markdown_excerpts_path=markdown_excerpts_path,
                context_bundle=context_bundle,
                prompt_context_tokens=prompt_context_tokens,
                prompt_context_kind=prompt_context_kind,
            )
            return updated_bundle, prompt_context_tokens, prompt_context_kind, "hit"
        return context_bundle, prompt_context_tokens, prompt_context_kind, "hit"

    if excerpts_cache is not None:
        prompt_context_tokens, prompt_context_kind = excerpts_cache
        updated_bundle = write_prompt_context_cache(
            context_bundle_path=context_bundle_path,
            markdown_excerpts_path=markdown_excerpts_path,
            context_bundle=context_bundle,
            prompt_context_tokens=prompt_context_tokens,
            prompt_context_kind=prompt_context_kind,
        )
        return updated_bundle, prompt_context_tokens, prompt_context_kind, "hit"

    prompt_context_tokens, prompt_context_kind = compute_prompt_context_cache(
        question=question,
        final_document=final_document,
        context_bundle=context_bundle,
        config=config,
    )
    updated_bundle = write_prompt_context_cache(
        context_bundle_path=context_bundle_path,
        markdown_excerpts_path=markdown_excerpts_path,
        context_bundle=context_bundle,
        prompt_context_tokens=prompt_context_tokens,
        prompt_context_kind=prompt_context_kind,
    )
    return updated_bundle, prompt_context_tokens, prompt_context_kind, "miss"


__all__ = [
    "PromptContextKind",
    "TokenSidecar",
    "compute_prompt_context_cache",
    "ensure_prompt_context_cache",
    "read_prompt_context_cache",
    "read_token_sidecar",
    "write_prompt_context_cache",
    "write_token_sidecar",
]
