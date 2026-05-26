"""Unit tests for session-level chat prompt cache helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.api.services.chat_session_helpers import (
    as_session_prompt_context_cache,
    build_session_prompt_context_cache,
    session_prompt_context_cache_payload,
)
from backend.api.services.models import (
    ContextWindowEstimate,
    LoadedChatSource,
    SessionPromptContextCache,
)
from backend.utils.config import load_config


def test_session_prompt_context_cache_round_trip_with_full_prefix_data() -> None:
    """Full caches should round-trip without losing prefix-token arrays."""
    cache = SessionPromptContextCache(
        context_run_ids=["run-1"],
        followup_bundle_ids=["bundle-1"],
        mode="direct",
        prompt_context_tokens=123,
        prompt_context_kind="citation_catalog",
        citation_catalog_entry_count=2,
        citation_ref_ids_in_order=["ref_1", "ref_2"],
        citation_prefix_tokens=[40, 85],
    )

    parsed = as_session_prompt_context_cache(
        {"prompt_context_cache": session_prompt_context_cache_payload(cache)},
        context_run_ids=["run-1"],
        followup_bundle_ids=["bundle-1"],
    )

    assert parsed == cache


def test_session_prompt_context_cache_round_trip_without_prefix_data() -> None:
    """Partial caches should round-trip without forcing prefix-token arrays."""
    cache = SessionPromptContextCache(
        context_run_ids=["run-1"],
        followup_bundle_ids=[],
        mode="split",
        prompt_context_tokens=456,
        prompt_context_kind="citation_catalog",
        citation_catalog_entry_count=3,
        citation_ref_ids_in_order=None,
        citation_prefix_tokens=None,
    )

    parsed = as_session_prompt_context_cache(
        {"prompt_context_cache": session_prompt_context_cache_payload(cache)},
        context_run_ids=["run-1"],
        followup_bundle_ids=[],
    )

    assert parsed == cache


def test_build_session_prompt_context_cache_skips_prefix_materialization_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session bootstrap should persist exact cache metrics without prefix arrays."""
    source = LoadedChatSource(
        source_type="run",
        source_id="run-1",
        question="Question",
        final_document="# Final",
        context_bundle={
            "markdown": {
                "excerpts": [
                    {
                        "ref_id": "ref_1",
                        "city_name": "Munich",
                        "quote": "Grounded evidence.",
                        "partial_answer": "Grounded evidence.",
                    }
                ]
            }
        },
    )
    captured_prefix_tokens: list[int] | None = [999]

    def _stub_estimate_context_window(
        original_question: str,
        contexts: list[dict[str, object]],
        config,
        token_cap: int,
        citation_catalog: list[dict[str, str]] | None,
        citation_prefix_tokens: list[int] | None,
    ) -> ContextWindowEstimate:
        nonlocal captured_prefix_tokens
        _ = original_question, contexts, config, token_cap, citation_catalog
        captured_prefix_tokens = citation_prefix_tokens
        return ContextWindowEstimate(
            mode="direct",
            resolved_token_cap=token_cap,
            effective_token_cap=token_cap,
            context_window_kind="citation_catalog",
            context_window_tokens=77,
            fitted_context_window_tokens=77,
            citation_catalog_entry_count=1,
            fitted_citation_entry_count=1,
        )

    monkeypatch.setattr(
        "backend.api.services.chat_session_helpers.build_citation_catalog_token_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Prefix-token cache should stay lazy by default.")
        ),
    )

    cache = build_session_prompt_context_cache(
        original_question="Question",
        sources=[source],
        context_run_ids=["run-1"],
        followup_bundle_ids=[],
        config=load_config(),
        token_cap=220_000,
        build_citation_catalog_fn=lambda _sources: [
            SimpleNamespace(
                ref_id="ref_1",
                city_name="Munich",
                quote="Grounded evidence.",
                partial_answer="Grounded evidence.",
            )
        ],
        estimate_context_window_fn=_stub_estimate_context_window,
    )

    assert captured_prefix_tokens is None
    assert cache.citation_ref_ids_in_order is None
    assert cache.citation_prefix_tokens is None
