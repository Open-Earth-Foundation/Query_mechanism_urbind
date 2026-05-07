"""Tests for the split-phase gap analyst (decompose_fields + detect_city_gaps).

The original ``run_gap_analysis`` is preserved for backward compat and is
exercised by ``tests/test_enrichment_integration.py``.  This file targets
the new functions used by the orchestrator's Phase 0 + Phase 2.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from backend.modules.web_researcher.gap_analysis import (
    decompose_fields,
    detect_city_gaps,
)
from backend.modules.web_researcher.models import (
    FieldClassification,
    FieldDecomposition,
)
from tests.support import build_test_app_config


def _mock_openai_response(payload: dict | str) -> MagicMock:
    """Build a stub of openai.ChatCompletion to inject a given payload."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    response.choices = [choice]
    return response


def _make_config():
    return build_test_app_config(
        enrichment_overrides={
            "enabled": True,
            "model": "openai/gpt-test",
            "temperature": 0.0,
            "reasoning_effort": None,
            "max_fields_per_query": 20,
            "freshness_threshold_days": 730,
        }
    )


# ---------------------------------------------------------------------------
# decompose_fields
# ---------------------------------------------------------------------------


def test_decompose_fields_returns_classified_fields() -> None:
    config = _make_config()
    payload = {
        "query_fields": [
            {
                "field": "public_dc_charger_count",
                "classification": "estimable_numerical",
                "searchable": True,
                "rationale": "Reported by national registries.",
            },
            {
                "field": "depot_charger_count",
                "classification": "derivable_from_ratio",
                "searchable": True,
                "rationale": "Derive from fleet size.",
            },
            {
                "field": "residential_onstreet_charging",
                "classification": "non_estimable",
                "searchable": False,
                "rationale": "Heavily local-context dependent.",
            },
        ],
        "non_estimable_fields": ["residential_onstreet_charging"],
    }

    with patch(
        "backend.modules.web_researcher.gap_analysis.OpenAI"
    ) as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(payload)
        mock_openai_cls.return_value = mock_client

        decomposition = decompose_fields("How many DC chargers?", config, "fake-key")

    assert isinstance(decomposition, FieldDecomposition)
    assert len(decomposition.query_fields) == 3
    classifications = {f.field: f.classification for f in decomposition.query_fields}
    assert classifications["public_dc_charger_count"] == "estimable_numerical"
    assert classifications["residential_onstreet_charging"] == "non_estimable"
    assert decomposition.non_estimable_fields == ["residential_onstreet_charging"]


def test_decompose_fields_returns_empty_on_failure() -> None:
    config = _make_config()
    with patch(
        "backend.modules.web_researcher.gap_analysis.OpenAI"
    ) as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("api dead")
        mock_openai_cls.return_value = mock_client

        decomposition = decompose_fields("question", config, "fake-key")

    assert decomposition.query_fields == []
    assert decomposition.non_estimable_fields == []


def test_decompose_does_not_request_context_bundle() -> None:
    """Decompose runs against the raw question — never a context bundle."""
    config = _make_config()
    payload = {"query_fields": [], "non_estimable_fields": []}

    with patch(
        "backend.modules.web_researcher.gap_analysis.OpenAI"
    ) as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(payload)
        mock_openai_cls.return_value = mock_client

        decompose_fields("a small question", config, "fake-key")

        # Inspect the user prompt that was sent.
        args, kwargs = mock_client.chat.completions.create.call_args
        messages = kwargs.get("messages") or args[0]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "context bundle" not in user_msg.lower()
        assert "research question" not in user_msg.lower()


# ---------------------------------------------------------------------------
# detect_city_gaps
# ---------------------------------------------------------------------------


def _decomposition_for_test() -> FieldDecomposition:
    return FieldDecomposition(
        query_fields=[
            FieldClassification(
                field="public_dc_charger_count",
                classification="estimable_numerical",
                searchable=True,
                rationale="ok",
            ),
            FieldClassification(
                field="residential_onstreet_charging",
                classification="non_estimable",
                searchable=False,
                rationale="local",
            ),
        ],
        non_estimable_fields=["residential_onstreet_charging"],
    )


def _context_bundle() -> dict[str, Any]:
    return {
        "research_question": "infra targets",
        "markdown": {
            "excerpts": [{"city_name": "Dresden", "content": "..."}],
        },
    }


def test_detect_city_gaps_returns_full_manifest() -> None:
    config = _make_config()
    payload = {
        "city_gaps": [
            {
                "city": "Dresden",
                "blank_fields": ["public_dc_charger_count"],
                "stale_flags": [],
                "search_priority": "high",
            }
        ]
    }

    with patch(
        "backend.modules.web_researcher.gap_analysis.OpenAI"
    ) as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(payload)
        mock_openai_cls.return_value = mock_client

        manifest = detect_city_gaps(
            "infra targets",
            _decomposition_for_test(),
            _context_bundle(),
            config,
            "fake-key",
        )

    # Pre-decomposed fields propagate through unchanged.
    fields = {f.field for f in manifest.query_fields}
    assert fields == {"public_dc_charger_count", "residential_onstreet_charging"}
    assert manifest.non_estimable_fields == ["residential_onstreet_charging"]
    # City gaps come from this pass.
    assert len(manifest.city_gaps) == 1
    assert manifest.city_gaps[0].city == "Dresden"
    assert manifest.city_gaps[0].blank_fields == ["public_dc_charger_count"]
    assert manifest.city_gaps[0].search_priority == "high"


def test_detect_city_gaps_short_circuits_without_fields() -> None:
    """Empty decomposition skips the LLM call and returns an empty manifest."""
    config = _make_config()
    empty = FieldDecomposition(query_fields=[], non_estimable_fields=[])

    with patch(
        "backend.modules.web_researcher.gap_analysis.OpenAI"
    ) as mock_openai_cls:
        manifest = detect_city_gaps(
            "q", empty, _context_bundle(), config, "fake-key"
        )
        mock_openai_cls.assert_not_called()

    assert manifest.query_fields == []
    assert manifest.city_gaps == []
    assert manifest.non_estimable_fields == []


def test_detect_city_gaps_returns_empty_gaps_on_failure() -> None:
    config = _make_config()
    with patch(
        "backend.modules.web_researcher.gap_analysis.OpenAI"
    ) as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("api dead")
        mock_openai_cls.return_value = mock_client

        manifest = detect_city_gaps(
            "q",
            _decomposition_for_test(),
            _context_bundle(),
            config,
            "fake-key",
        )

    # Decomposition still propagates so downstream code has fields to work with.
    assert len(manifest.query_fields) == 2
    assert manifest.city_gaps == []


def test_detect_passes_fields_to_prompt() -> None:
    """The detect prompt should include the pre-decomposed fields."""
    config = _make_config()
    payload = {"city_gaps": []}

    with patch(
        "backend.modules.web_researcher.gap_analysis.OpenAI"
    ) as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(payload)
        mock_openai_cls.return_value = mock_client

        detect_city_gaps(
            "q",
            _decomposition_for_test(),
            _context_bundle(),
            config,
            "fake-key",
        )

        _, kwargs = mock_client.chat.completions.create.call_args
        user_msg = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")
        assert "public_dc_charger_count" in user_msg
        assert "Pre-decomposed" in user_msg
