import json
from pathlib import Path

from backend.modules.initiative_extractor.models import InitiativeExtractionRecord
from backend.modules.tef_mapper.benchmark import (
    classify_mapping_set,
    compare_mapping_run_to_source_truth,
    source_truth_row_to_initiative_record,
    write_comparison_artifacts,
)


def _mapping(
    target_id: str,
    target_path: str,
    *,
    target_type: str = "transition_element",
    is_primary: bool = True,
) -> dict[str, object]:
    """Build a minimal source-truth or mapper final mapping row."""
    return {
        "target_type": target_type,
        "target_id": target_id,
        "target_path": target_path,
        "confidence": 0.9,
        "is_primary": is_primary,
        "needs_review": False,
        "rationale": f"Maps to {target_id}.",
    }


def _source_row(
    record_id: str,
    local_code: str,
    mappings: list[dict[str, object]],
) -> dict[str, object]:
    """Build a minimal mapped Krakow source-truth row."""
    return {
        "record_id": record_id,
        "city": "Krakow",
        "source_document": "Krakow.md",
        "local_code": local_code,
        "initiative_name": f"Initiative {local_code}",
        "general_description": f"{local_code} general description.",
        "objective_text": f"{local_code} objective.",
        "planned_outputs_text": f"{local_code} planned output.",
        "implementation_text": "City implementation.",
        "delivery_text": "Municipal delivery.",
        "funding_text": "No funding details.",
        "timeline_text": "2025-2030.",
        "numbers": {"current": {}, "planned": {"cost": "PLN 1"}},
        "source_refs": [
            {
                "source_document": "Krakow.md",
                "source_path": "documents/Krakow.md",
                "start_line": 10,
                "end_line": 20,
            }
        ],
        "primary_tef_mapping": next(mapping for mapping in mappings if mapping["is_primary"]),
        "tef_mappings": mappings,
        "review_items": [],
    }


def _fresh_row(
    record_id: str,
    local_code: str,
    mapping: dict[str, object],
) -> dict[str, object]:
    """Build a minimal final mapping artifact row."""
    return {
        "initiative_record_id": record_id,
        "city": "Krakow",
        "source_document": "Krakow.md",
        "document_local_code": local_code,
        "initiative_name": f"Initiative {local_code}",
        "sector_route": {},
        "subsector_routes": [],
        "mapper_version": "test",
        "tef_source_version": "test",
        "extraction_run_id": "test",
        **mapping,
    }


def test_classify_mapping_set_assigns_clean_p1_p2_and_p3() -> None:
    """Mapping-set classifier should separate clean, P1, P2, and P3 cases."""
    primary = _mapping("target-a", "1-sector/1a-branch")
    secondary = _mapping("target-b", "1-sector/1a-branch", is_primary=False)
    different = _mapping("target-c", "2-sector/2a-branch")

    clean = classify_mapping_set([primary], [primary])
    assert clean.status == "clean_match"
    assert clean.priority is None

    p3 = classify_mapping_set([primary, secondary], [primary])
    assert p3.status == "nonprimary_mapping_set_drift"
    assert p3.priority == "P3"

    p2 = classify_mapping_set(
        [primary, secondary],
        [
            {**secondary, "is_primary": True},
            {**primary, "is_primary": False},
        ],
    )
    assert p2.status == "source_primary_demoted"
    assert p2.priority == "P2"

    p1 = classify_mapping_set([primary], [different])
    assert p1.status == "primary_target_mismatch"
    assert p1.priority == "P1"


def test_compare_mapping_run_to_source_truth_writes_p1_p2_p3_report(tmp_path: Path) -> None:
    """Benchmark comparison should emit P1-P3 issue counts and detailed artifacts."""
    source_rows = [
        _source_row("krakow:clean", "CLEAN", [_mapping("target-a", "1-sector/1a")]),
        _source_row(
            "krakow:p3",
            "P3",
            [
                _mapping("target-a", "1-sector/1a"),
                _mapping("target-b", "1-sector/1a", is_primary=False),
            ],
        ),
        _source_row(
            "krakow:p2",
            "P2",
            [
                _mapping("target-a", "1-sector/1a"),
                _mapping("target-b", "1-sector/1a", is_primary=False),
            ],
        ),
        _source_row("krakow:p1", "P1", [_mapping("target-a", "1-sector/1a")]),
    ]
    source_truth_path = tmp_path / "source_truth.json"
    source_truth_path.write_text(
        json.dumps({"metadata": {}, "initiatives": source_rows}),
        encoding="utf-8",
    )
    mapping_run_dir = tmp_path / "mapping_run"
    final_rows = [
        _fresh_row("krakow:clean", "CLEAN", _mapping("target-a", "1-sector/1a")),
        _fresh_row("krakow:p3", "P3", _mapping("target-a", "1-sector/1a")),
        _fresh_row(
            "krakow:p2",
            "P2",
            _mapping("target-b", "1-sector/1a", is_primary=True),
        ),
        _fresh_row("krakow:p2", "P2", _mapping("target-a", "1-sector/1a", is_primary=False)),
        _fresh_row("krakow:p1", "P1", _mapping("target-c", "2-sector/2a")),
    ]
    final_path = mapping_run_dir / "05_final_mappings" / "final_mappings.jsonl"
    final_path.parent.mkdir(parents=True)
    final_path.write_text(
        "\n".join(json.dumps(row) for row in final_rows) + "\n",
        encoding="utf-8",
    )

    comparison = compare_mapping_run_to_source_truth(
        source_truth_path=source_truth_path,
        mapping_run_dir=mapping_run_dir,
    )

    assert comparison.summary["clean_primary_and_set_matches"] == 1
    assert comparison.summary["issues_marked"] == 3
    assert comparison.summary["priority_counts"] == {"P3": 1, "P2": 1, "P1": 1}
    assert [issue["priority"] for issue in comparison.issues] == ["P1", "P2", "P3"]
    assert all(issue["initiative_about"] for issue in comparison.issues)

    issues_path, report_path = write_comparison_artifacts(
        comparison,
        tmp_path / "comparison",
    )
    assert issues_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "P1 means" in report_text
    assert "What the initiative is about" in report_text


def test_source_truth_row_to_initiative_record_preserves_input_shape() -> None:
    """Source-truth rows should convert into mapper-ready initiative records."""
    row = _source_row("krakow:test", "TEST", [_mapping("target-a", "1-sector/1a")])

    record = source_truth_row_to_initiative_record(row)

    assert isinstance(record, InitiativeExtractionRecord)
    assert record.record_id == "krakow:test"
    assert record.initiative.city == "Krakow"
    assert record.document_local_code == "TEST"
    assert record.initiative.numbers.planned["cost"] == "PLN 1"
    assert record.source_refs[0].segment_id == "krakow:test:source:1"
