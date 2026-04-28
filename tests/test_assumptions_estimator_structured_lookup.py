"""Tests for the estimator's structured-lookup short-circuit (Step 9)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
import yaml

from backend.modules.sources.manifest import load_manifest
from backend.modules.web_researcher.assumptions_estimator import (
    run_assumptions_estimator,
)
from backend.modules.web_researcher.models import (
    EnrichedField,
    FieldClassification,
    GapManifest,
    StructuredLookupResult,
)
from tests.support import build_test_app_config


def _config():
    return build_test_app_config(
        enrichment_overrides={
            "enabled": True,
            "model": "openai/gpt-test",
            "temperature": 0.0,
        }
    )


def _gap_manifest_for_chargers() -> GapManifest:
    return GapManifest(
        query_fields=[
            FieldClassification(
                field="public_dc_charger_count",
                classification="estimable_numerical",
                searchable=True,
                rationale="Reported by national registries.",
            ),
            FieldClassification(
                field="ev_share",
                classification="estimable_numerical",
                searchable=True,
                rationale="Reported by EAFO.",
            ),
        ],
        city_gaps=[],
        non_estimable_fields=[],
    )


def _enriched_fields_to_estimate() -> list[EnrichedField]:
    return [
        EnrichedField(
            city="Dresden",
            field="public_dc_charger_count",
            status="still_missing",
            source="none",
        ),
        EnrichedField(
            city="Dresden",
            field="ev_share",
            status="still_missing",
            source="none",
        ),
    ]


def test_structured_lookup_resolves_field_and_skips_llm() -> None:
    """A field covered by a structured lookup is resolved without calling the LLM."""
    config = _config()
    fake_lookup = StructuredLookupResult(
        source_id="urbind_additional",
        ingestion_id="bnetza_chargers",
        city="Dresden",
        field="public_dc_charger_count",
        value=159,
        unit="stations",
        extra={"source_name": "Bundesnetzagentur Ladesäulenregister"},
    )

    with patch(
        "backend.modules.web_researcher.assumptions_estimator.load_manifest",
        return_value=object(),  # placeholder; real load_manifest replaced below
    ), patch(
        "backend.modules.web_researcher.assumptions_estimator.find_matching_structured_lookups",
        return_value=[fake_lookup],
    ), patch(
        "backend.modules.web_researcher.assumptions_estimator._call_estimator",
        return_value=[],
    ) as mock_call_estimator, patch(
        "backend.modules.web_researcher.assumptions_estimator._check_anchor_sufficiency",
        side_effect=lambda peer, fields: (fields, []),
    ):
        assumptions, non_estimable, saturation = run_assumptions_estimator(
            question="how many DC chargers does Dresden have",
            context_bundle={},
            gap_manifest=_gap_manifest_for_chargers(),
            enriched_fields=_enriched_fields_to_estimate(),
            config=config,
            api_key="k",
        )

    # Dresden's DC charger count was resolved by the lookup.
    dc = next(a for a in assumptions if a.field_name == "public_dc_charger_count")
    assert dc.method_used == "structured_lookup"
    assert dc.confidence == "HIGH"
    assert dc.estimate.low == 159
    assert dc.estimate.mid == 159
    assert dc.estimate.high == 159
    assert "Bundesnetzagentur" in dc.reference_data
    assert dc.is_replaceable is False

    # The LLM estimator was still called for ev_share (not in lookup coverage).
    mock_call_estimator.assert_called()
    estimable_arg = mock_call_estimator.call_args.kwargs["estimable_fields"]
    remaining_fields = {f.field for f in estimable_arg}
    assert "public_dc_charger_count" not in remaining_fields
    assert "ev_share" in remaining_fields
    assert saturation is None
    assert non_estimable == []


def test_structured_lookup_skips_llm_entirely_when_all_resolved() -> None:
    """If every gap is resolved via lookups, the LLM is never invoked."""
    config = _config()
    fake_lookup = StructuredLookupResult(
        source_id="urbind_additional",
        ingestion_id="bnetza_chargers",
        city="Dresden",
        field="public_dc_charger_count",
        value=159,
        unit="stations",
    )

    one_field = [
        EnrichedField(
            city="Dresden",
            field="public_dc_charger_count",
            status="still_missing",
            source="none",
        )
    ]
    one_field_manifest = GapManifest(
        query_fields=[
            FieldClassification(
                field="public_dc_charger_count",
                classification="estimable_numerical",
                searchable=True,
                rationale="x",
            )
        ],
        city_gaps=[],
        non_estimable_fields=[],
    )

    with patch(
        "backend.modules.web_researcher.assumptions_estimator.load_manifest",
        return_value=object(),
    ), patch(
        "backend.modules.web_researcher.assumptions_estimator.find_matching_structured_lookups",
        return_value=[fake_lookup],
    ), patch(
        "backend.modules.web_researcher.assumptions_estimator._call_estimator"
    ) as mock_call_estimator, patch(
        "backend.modules.web_researcher.assumptions_estimator._check_anchor_sufficiency",
        side_effect=lambda peer, fields: (fields, []),
    ):
        assumptions, _non_est, _sat = run_assumptions_estimator(
            question="q",
            context_bundle={},
            gap_manifest=one_field_manifest,
            enriched_fields=one_field,
            config=config,
            api_key="k",
        )

    mock_call_estimator.assert_not_called()
    assert len(assumptions) == 1
    assert assumptions[0].method_used == "structured_lookup"


def test_no_structured_lookup_match_falls_through_to_llm() -> None:
    """A field with no matching lookup goes through the LLM as today."""
    config = _config()

    with patch(
        "backend.modules.web_researcher.assumptions_estimator.load_manifest",
        return_value=object(),
    ), patch(
        "backend.modules.web_researcher.assumptions_estimator.find_matching_structured_lookups",
        return_value=[],
    ), patch(
        "backend.modules.web_researcher.assumptions_estimator._call_estimator",
        return_value=[],
    ) as mock_call_estimator, patch(
        "backend.modules.web_researcher.assumptions_estimator._check_anchor_sufficiency",
        side_effect=lambda peer, fields: (fields, []),
    ):
        run_assumptions_estimator(
            question="q",
            context_bundle={},
            gap_manifest=_gap_manifest_for_chargers(),
            enriched_fields=_enriched_fields_to_estimate(),
            config=config,
            api_key="k",
        )

    mock_call_estimator.assert_called_once()


def test_missing_manifest_skips_lookups_gracefully() -> None:
    """No sources_manifest → estimator continues without lookups."""
    config = _config()

    with patch(
        "backend.modules.web_researcher.assumptions_estimator.load_manifest",
        side_effect=FileNotFoundError("no manifest"),
    ), patch(
        "backend.modules.web_researcher.assumptions_estimator._call_estimator",
        return_value=[],
    ) as mock_call_estimator, patch(
        "backend.modules.web_researcher.assumptions_estimator._check_anchor_sufficiency",
        side_effect=lambda peer, fields: (fields, []),
    ):
        assumptions, _ne, _sat = run_assumptions_estimator(
            question="q",
            context_bundle={},
            gap_manifest=_gap_manifest_for_chargers(),
            enriched_fields=_enriched_fields_to_estimate(),
            config=config,
            api_key="k",
        )

    mock_call_estimator.assert_called_once()
    # No structured lookups produced.
    assert all(a.method_used != "structured_lookup" for a in assumptions)


def test_non_numeric_lookup_value_is_skipped() -> None:
    """A lookup that returns None / non-numeric isn't lifted into an assumption."""
    config = _config()
    fake_lookup = StructuredLookupResult(
        source_id="urbind_additional",
        ingestion_id="bnetza_chargers",
        city="Dresden",
        field="public_dc_charger_count",
        value=None,
        unit="stations",
    )

    with patch(
        "backend.modules.web_researcher.assumptions_estimator.load_manifest",
        return_value=object(),
    ), patch(
        "backend.modules.web_researcher.assumptions_estimator.find_matching_structured_lookups",
        return_value=[fake_lookup],
    ), patch(
        "backend.modules.web_researcher.assumptions_estimator._call_estimator",
        return_value=[],
    ) as mock_call_estimator, patch(
        "backend.modules.web_researcher.assumptions_estimator._check_anchor_sufficiency",
        side_effect=lambda peer, fields: (fields, []),
    ):
        assumptions, _ne, _sat = run_assumptions_estimator(
            question="q",
            context_bundle={},
            gap_manifest=_gap_manifest_for_chargers(),
            enriched_fields=_enriched_fields_to_estimate(),
            config=config,
            api_key="k",
        )

    # No structured lookup record was emitted (value was None).
    assert all(a.method_used != "structured_lookup" for a in assumptions)
    # LLM still runs for the original gaps.
    mock_call_estimator.assert_called_once()
