"""Tests for the placeholder external-source pipeline hook."""

from __future__ import annotations

from backend.modules.web_researcher.external_sources import run_external_source_stage
from backend.modules.web_researcher.models import FieldClassification, FieldDecomposition


def test_external_source_stage_returns_empty_result_without_mutating_context() -> None:
    """The cleanup placeholder must not add source-library data yet."""
    decomposition = FieldDecomposition(
        query_fields=[
            FieldClassification(
                field="public_dc_charger_count",
                classification="estimable_numerical",
                searchable=True,
                rationale="Searchable public registry field.",
                scope="municipal",
            )
        ],
        non_estimable_fields=[],
    )
    context_bundle = {"markdown": {"excerpt_count": 2}}
    before = context_bundle.copy()

    result = run_external_source_stage(decomposition, context_bundle)

    assert result.findings == []
    assert context_bundle == before
