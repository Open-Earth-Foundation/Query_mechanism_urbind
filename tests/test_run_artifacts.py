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
                    {
                        "field": "tram_line_length",
                        "classification": "estimable_numerical",
                        "scope": "mixed",
                        "rationale": "Tram length is numeric.",
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

    # Excerpts mention "charging" and "tram" near numbers -> numeric topics that
    # power the soft extraction-gap hint (the tram one drives a no-source field).
    _write(
        extraction / "accepted_excerpts.json",
        {
            "excerpts": [
                {"text": "The city will deploy 400 new charging stations by 2030."},
                {"text": "Plans add 12 km of new tram line across the centre."},
            ]
        },
    )

    # Assumptions: all three fields non-estimable.
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
                {
                    "city": "Aachen",
                    "field_name": "tram_line_length",
                    "explanation": "Only 0 peer values available.",
                    "recommendation": "Check transit plan.",
                },
            ],
        },
    )
    return run


def test_reason_classification_and_break_warning(tmp_path: Path) -> None:
    payload = build_run_artifacts(_make_run_dir(tmp_path))

    by_field = {f["field"]: f for f in payload["fields"]}
    assert by_field["public_ac_charger_count"]["status"] == "non_estimable"
    # One candidate was found (not validated) -> authoritative reason is
    # found_not_validated. The extraction-gap hint is scoped to no_source_data,
    # so even though the corpus mentions "charging" it does NOT show here.
    assert by_field["public_ac_charger_count"]["reason"] == "found_not_validated"
    assert by_field["public_ac_charger_count"]["reason_hint"] is None
    assert by_field["public_ac_charger_count"]["reason_hint_evidence"] == []
    # No candidates and no matching corpus topic -> genuinely no source data,
    # and no hint (nothing in the corpus to point at).
    assert by_field["secret_unrelated_metric"]["reason"] == "no_source_data"
    assert by_field["secret_unrelated_metric"]["reason_hint"] is None
    # No source data AND the corpus mentions "tram" near a figure -> the soft
    # extraction-gap hint fires, carrying the matched snippet as evidence.
    tram = by_field["tram_line_length"]
    assert tram["reason"] == "no_source_data"
    assert tram["reason_hint"] == "Possible extraction gap"
    assert tram["reason_hint_evidence"]
    assert any("tram" in snippet.lower() for snippet in tram["reason_hint_evidence"])

    steps = {s["key"]: s for s in payload["enrichment_steps"]}
    # Found web evidence but nothing validated -> the chain breaks here.
    assert steps["external_web"]["warn"]
    assert steps["external_web"]["metrics"]["web_findings"] == 5
    assert steps["external_web"]["metrics"]["validated_evidence"] == 0
    # Rollup counts the authoritative reason codes (shape_mismatch is gone).
    breakdown = steps["assumptions"]["metrics"]["reason_breakdown"]
    assert breakdown["found_not_validated"] == 1
    assert breakdown["no_source_data"] == 2
    assert "shape_mismatch" not in breakdown

    # Both artifacts read cleanly -> healthy, not degraded.
    assert payload["artifact_health"]["enrichment_bundle"] == "ok"
    assert payload["artifact_health"]["assumptions_bundle"] == "ok"
    assert payload["degraded"] is False


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


def test_stage_details_surface_web_research_without_external_audit(tmp_path: Path) -> None:
    run = tmp_path / "web_only_run"
    enrichment = run / "stage_files" / "008_enrichment"
    _write(
        enrichment / "enrichment_bundle.json",
        {
            "field_manifest": {"query_fields": []},
            "gap_manifest": {"city_gaps": []},
            "enriched_fields": [],
            "web_findings": [
                {
                    "city": "Aachen",
                    "field": "battery_storage_mwh",
                    "value": 0,
                    "unit": "MWh",
                    "source_url": "https://example.test/battery",
                    "source_tier": "open",
                }
            ],
            "external_evidence": [],
            "external_resolutions": [],
            "external_no_evidence": [],
            "freshness_results": [
                {
                    "city": "Aachen",
                    "field": "battery_storage_mwh",
                    "classification": "uncertain",
                    "web_value": "0",
                }
            ],
            "meta": {
                "estimable_count": 0,
                "non_estimable_count": 0,
                "web_findings_count": 1,
                "external_evidence_count": 0,
                "elapsed_seconds": 1.0,
            },
        },
    )
    _write(
        enrichment / "web_research_audit.json",
        {
            "outputs": {
                "search_batches": [],
                "national_findings": [],
                "comparative_findings": [],
                "serper_billing_summary": {
                    "planned_search_query_count": 2,
                    "actual_serper_call_count": 8,
                    "successful_serper_call_count": 8,
                    "tier1_site_call_count": 6,
                    "open_call_count": 2,
                    "skipped_open_call_count": 0,
                    "estimated_max_serper_call_count": 12,
                },
            },
            "metrics": {
                "search_batch_count": 1,
                "search_query_count": 2,
                "web_finding_count": 1,
                "freshness_result_count": 1,
            },
        },
    )

    payload = build_run_artifacts(run)

    web = payload["stage_details"]["enrichment"]["web_research"]
    freshness = payload["stage_details"]["enrichment"]["freshness"]
    external = payload["stage_details"]["enrichment"]["external_sources"]
    assert web["executed"] is True
    assert web["search_query_count"] == 2
    assert web["planned_search_query_count"] == 2
    assert web["actual_serper_call_count"] == 8
    assert web["tier1_site_call_count"] == 6
    assert web["open_call_count"] == 2
    assert web["web_finding_count"] == 1
    assert web["findings"][0]["source_url"] == "https://example.test/battery"
    assert freshness["executed"] is True
    assert freshness["classification_counts"] == {"uncertain": 1}
    assert external["executed"] is False
    assert external["validated_count"] == 0
    assert payload["artifact_health"]["web_research_audit"] == "ok"


def test_manifest_alias_takes_precedence_over_glob(tmp_path: Path) -> None:
    run = tmp_path / "aliased_run"
    # Write the assumptions bundle in a NON-conventional location that the
    # stage-glob fallback would never find, and point the manifest alias at it.
    relocated = run / "relocated" / "assumptions_bundle.json"
    _write(
        relocated,
        {
            "assumptions": [],
            "non_estimable": [
                {"city": "Aachen", "field_name": "aliased_field", "explanation": "x"}
            ],
        },
    )
    _write(
        run / "manifest.json",
        {"aliases": {"assumptions_assumptions_bundle": {"path": "relocated/assumptions_bundle.json"}}},
    )

    payload = build_run_artifacts(run)

    # Resolved purely via the manifest alias — no stage_files/*assumptions* dir.
    assert payload["artifact_health"]["assumptions_bundle"] == "ok"
    assert {f["field"] for f in payload["fields"]} == {"aliased_field"}


def test_manifest_alias_resolves_enrichment_bundle_without_glob(tmp_path: Path) -> None:
    run = tmp_path / "aliased_enrichment_run"
    relocated = run / "relocated" / "enrichment_bundle.json"
    _write(
        relocated,
        {
            "field_manifest": {
                "query_fields": [
                    {
                        "field": "aliased_energy_target",
                        "classification": "estimable_numerical",
                        "scope": "city",
                        "rationale": "Energy target is numeric.",
                    }
                ]
            },
            "meta": {"estimable_count": 1, "non_estimable_count": 0},
        },
    )
    _write(
        run / "manifest.json",
        {
            "aliases": {
                "enrichment_bundle": {"path": "relocated/enrichment_bundle.json"}
            }
        },
    )

    payload = build_run_artifacts(run)

    assert payload["artifact_health"]["enrichment_bundle"] == "ok"
    assert payload["gap_analysis"]["fields"] == [
        {
            "field": "aliased_energy_target",
            "classification": "estimable_numerical",
            "scope": "city",
            "rationale": "Energy target is numeric.",
        }
    ]


def test_manifest_alias_resolves_accepted_excerpts_without_glob(tmp_path: Path) -> None:
    run = tmp_path / "aliased_excerpts_run"
    _write(
        run / "relocated" / "assumptions_bundle.json",
        {
            "assumptions": [],
            "non_estimable": [
                {
                    "city": "Aachen",
                    "field_name": "tram_line_length",
                    "explanation": "No direct source found.",
                }
            ],
        },
    )
    _write(
        run / "relocated" / "accepted_excerpts.json",
        {
            "excerpts": [
                {"text": "Plans add 12 km of new tram line across the centre."}
            ]
        },
    )
    _write(
        run / "manifest.json",
        {
            "aliases": {
                "assumptions_assumptions_bundle": {
                    "path": "relocated/assumptions_bundle.json"
                },
                "markdown_excerpts": {"path": "relocated/accepted_excerpts.json"},
            }
        },
    )

    payload = build_run_artifacts(run)
    field = payload["fields"][0]

    assert payload["artifact_health"]["accepted_excerpts"] == "ok"
    assert field["reason"] == "no_source_data"
    assert field["reason_hint"] == "Possible extraction gap"
    assert any("tram" in snippet.lower() for snippet in field["reason_hint_evidence"])


def test_manifest_present_missing_alias_does_not_use_glob_fallback(
    tmp_path: Path,
) -> None:
    run = _make_run_dir(tmp_path)
    _write(run / "manifest.json", {"aliases": {}})

    payload = build_run_artifacts(run)

    assert set(payload["artifact_health"].values()) == {"missing"}
    assert payload["fields"] == []
    assert payload["gap_analysis"]["fields"] == []


def test_unreadable_artifact_marks_degraded(tmp_path: Path) -> None:
    run = _make_run_dir(tmp_path)
    # Corrupt the assumptions bundle: present on disk but not valid JSON.
    bundle = run / "stage_files" / "010_assumptions" / "assumptions_bundle.json"
    bundle.write_text("{ not json", encoding="utf-8")

    payload = build_run_artifacts(run)

    # A parse failure is surfaced as "unreadable" (not "missing") and flips the
    # top-level degraded flag, so consumers can distinguish it from a disabled
    # stage that simply never wrote the file.
    assert payload["artifact_health"]["assumptions_bundle"] == "unreadable"
    assert payload["degraded"] is True
    # Other artifacts still read fine.
    assert payload["artifact_health"]["enrichment_bundle"] == "ok"


def test_missing_artifact_is_not_degraded(tmp_path: Path) -> None:
    # An empty run dir (no stage files at all) is "missing" everywhere, which is
    # the normal disabled-stage case — not a degradation.
    payload = build_run_artifacts(tmp_path / "empty_run")

    assert set(payload["artifact_health"].values()) == {"missing"}
    assert payload["degraded"] is False
