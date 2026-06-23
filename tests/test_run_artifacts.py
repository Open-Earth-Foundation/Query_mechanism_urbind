"""Tests for the enrichment-audit artifact assembler."""

from __future__ import annotations

import json
from pathlib import Path

from backend.api.services.run_artifacts import build_run_artifacts


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run1"
    enrichment = run / "stage_files" / "008_enrichment"
    assumptions = run / "stage_files" / "010_assumptions"
    extraction = run / "stage_files" / "006_markdown_extraction"

    # Field manifest: classification + scope + rationale per field.
    _write(
        enrichment / "enrichment_bundle.json",
        {
            "field_manifest": {
                "query_fields": [
                    {
                        "field": "public_ac_charger_count",
                        "classification": "estimable_numerical",
                        "scope": "mixed",
                        "rationale": "Charger counts are numeric.",
                    },
                    {
                        "field": "secret_unrelated_metric",
                        "classification": "estimable_numerical",
                        "scope": "mixed",
                        "rationale": "n/a",
                    },
                ]
            },
            "external_no_evidence": [{"city": "Aachen", "field": "secret_unrelated_metric"}],
            "meta": {
                "estimable_count": 2,
                "non_estimable_count": 0,
                "web_findings_count": 5,
                "external_evidence_count": 0,
            },
        },
    )

    # External audit: candidates found for the charger field, none validated.
    _write(
        enrichment / "external_source_search_audit.json",
        {
            "candidates": [
                {"city": "Aachen", "field": "public_ac_charger_count", "title": "EV report", "quote": "charging"}
            ],
            "validated_claims": [],
            "no_evidence": [{"city": "Aachen", "field": "secret_unrelated_metric"}],
        },
    )

    # Excerpts mention "charging" near a number -> shape-mismatch topic.
    _write(
        extraction / "accepted_excerpts.json",
        {"excerpts": [{"text": "The city will deploy 400 new charging stations by 2030."}]},
    )

    # Assumptions: both fields non-estimable.
    _write(
        assumptions / "assumptions_bundle.json",
        {
            "assumptions": [],
            "non_estimable": [
                {
                    "city": "Aachen",
                    "field_name": "public_ac_charger_count",
                    "explanation": "Only 0 peer values available.",
                    "recommendation": "Search registries.",
                },
                {
                    "city": "Aachen",
                    "field_name": "secret_unrelated_metric",
                    "explanation": "Only 0 peer values available.",
                    "recommendation": "n/a",
                },
            ],
        },
    )
    return run


def test_reason_classification_and_break_warning(tmp_path: Path) -> None:
    payload = build_run_artifacts(_make_run_dir(tmp_path))

    by_field = {f["field"]: f for f in payload["fields"]}
    assert by_field["public_ac_charger_count"]["status"] == "non_estimable"
    # Corpus has "charging" + a number, so this is a shape mismatch, not no-data.
    assert by_field["public_ac_charger_count"]["reason"] == "shape_mismatch"
    # No candidates and no matching corpus topic -> genuinely no source data.
    assert by_field["secret_unrelated_metric"]["reason"] == "no_source_data"

    steps = {s["key"]: s for s in payload["enrichment_steps"]}
    # Found web evidence but nothing validated -> the chain breaks here.
    assert steps["external_web"]["warn"]
    assert steps["external_web"]["metrics"]["web_findings"] == 5
    assert steps["external_web"]["metrics"]["validated_evidence"] == 0
    # Rollup counts the reasons.
    breakdown = steps["assumptions"]["metrics"]["reason_breakdown"]
    assert breakdown["shape_mismatch"] == 1
    assert breakdown["no_source_data"] == 1


def test_gap_and_external_detail(tmp_path: Path) -> None:
    payload = build_run_artifacts(_make_run_dir(tmp_path))

    gap = payload["gap_analysis"]
    fields = {f["field"]: f for f in gap["fields"]}
    assert "public_ac_charger_count" in fields
    assert fields["public_ac_charger_count"]["rationale"]  # rationale surfaced

    ext = payload["external_search"]
    # The one candidate (found, not validated) is surfaced as unused.
    assert ext["unused_total"] == 1
    assert ext["unused"][0]["title"] == "EV report"
    # And the no-evidence record is carried through.
    assert {(n["city"], n["field"]) for n in ext["no_evidence"]} == {
        ("Aachen", "secret_unrelated_metric")
    }
