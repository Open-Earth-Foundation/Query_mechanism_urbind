import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.modules.calculator import agent as calculator_agent
from backend.modules.calculator.models import (
    CalculationCategory,
    CalculationPlan,
    CalculationRecord,
    CalculationWorkerOutput,
)
from backend.utils.config import AppConfig
from tests.support import build_test_app_config


def _build_test_config(tmp_path: Path) -> AppConfig:
    return build_test_app_config(
        runs_dir=tmp_path / "output",
        markdown_dir=tmp_path / "documents",
        calculator_overrides={"max_workers": 1, "max_passes_per_category": 3},
    )


def _build_category(*, year_policy: str = "ignore_year") -> CalculationCategory:
    return CalculationCategory(
        category_key="total_ev_cars",
        label="Total EV Cars",
        description="Observed EV car counts.",
        operation="sum",
        preferred_unit="vehicles",
        year_policy=year_policy,
        inclusion_rule="Include explicit EV car counts.",
        exclusion_rule="Exclude percentages and total fleets.",
    )


def _build_record(
    *,
    city: str,
    value: float,
    unit: str = "vehicles",
    year: int | None = None,
    record_role: str = "atomic",
    ref_id: str = "ref_1",
    note: str = "record",
) -> CalculationRecord:
    return CalculationRecord(
        category_key="total_ev_cars",
        city=city,
        value=value,
        unit=unit,
        note=note,
        ref_ids=[ref_id],
        source_chunk_ids=["chunk_1"],
        year=year,
        record_role=record_role,
    )


def test_calculation_plan_allows_zero_categories() -> None:
    plan = CalculationPlan(categories=[], note="No numeric categories found.")

    assert plan.categories == []
    assert plan.note == "No numeric categories found."


def test_calculation_plan_rejects_duplicate_category_keys() -> None:
    category = _build_category()

    with pytest.raises(ValidationError):
        CalculationPlan(categories=[category, category], note="duplicate")


def test_aggregate_category_records_separates_current_target_and_non_additive() -> None:
    summary = calculator_agent.aggregate_category_records(
        categories=[_build_category()],
        category_records={
            "total_ev_cars": [
                _build_record(
                    city="Aachen",
                    value=10,
                    ref_id="ref_1",
                    note="atomic 1",
                ),
                _build_record(
                    city="Munich",
                    value=20,
                    ref_id="ref_2",
                    note="atomic 2",
                ),
                _build_record(
                    city="Berlin",
                    value=50,
                    record_role="reported_total",
                    ref_id="ref_3",
                    note="reported total",
                ),
                _build_record(
                    city="Porto",
                    value=40,
                    record_role="target",
                    ref_id="ref_4",
                    note="target",
                ),
            ]
        },
        selected_city_names=["Aachen", "Munich", "Berlin", "Porto", "Leipzig"],
    )

    assert summary.status == "success"
    assert summary.category_count == 1
    group = summary.categories[0].groups[0]
    assert group.current_total == pytest.approx(30.0)
    assert group.target_total == pytest.approx(40.0)
    assert group.current_record_count == 2
    assert group.target_record_count == 1
    assert group.cities_with_current_records == ["Aachen", "Munich"]
    assert group.cities_with_target_records == ["Porto"]
    assert group.cities_with_only_non_additive_records == ["Berlin"]
    assert group.cities_with_no_usable_records == ["Leipzig"]
    assert len(group.non_additive_records) == 1
    assert group.ref_ids == ["ref_1", "ref_2", "ref_3", "ref_4"]

    category = summary.categories[0]
    assert category.current_record_count == 2
    assert category.target_record_count == 1


def test_aggregate_category_records_respects_year_policy() -> None:
    summary = calculator_agent.aggregate_category_records(
        categories=[_build_category(year_policy="separate_by_year")],
        category_records={
            "total_ev_cars": [
                _build_record(
                    city="Aachen",
                    value=10,
                    year=2025,
                    ref_id="ref_1",
                    note="2025",
                ),
                _build_record(
                    city="Munich",
                    value=20,
                    year=2030,
                    ref_id="ref_2",
                    note="2030",
                ),
            ]
        },
        selected_city_names=["Aachen", "Munich"],
    )

    years = {group.year for group in summary.categories[0].groups}
    assert years == {2025, 2030}
    for group in summary.categories[0].groups:
        assert group.target_total == pytest.approx(0.0)


def test_run_calculator_stage_writes_pass_artifacts_and_stops_on_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path)
    context_bundle = {
        "research_question": "What is the total EV car count?",
        "markdown": {
            "selected_city_names": ["Aachen", "Munich"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Aachen",
                    "quote": "Aachen reports 10 EVs.",
                    "partial_answer": "Aachen reports 10 EVs.",
                    "source_chunk_ids": ["chunk_1"],
                },
                {
                    "ref_id": "ref_2",
                    "city_name": "Munich",
                    "quote": "Munich reports 20 EVs.",
                    "partial_answer": "Munich reports 20 EVs.",
                    "source_chunk_ids": ["chunk_2"],
                },
            ],
        },
    }
    category = _build_category()

    monkeypatch.setattr(
        calculator_agent,
        "plan_categories",
        lambda **_kwargs: CalculationPlan(categories=[category], note="One category."),
    )

    responses = [
        CalculationWorkerOutput(
            status="records",
            category_key="total_ev_cars",
            records=[
                _build_record(city="Aachen", value=10, ref_id="ref_1", note="Aachen"),
                _build_record(city="Munich", value=20, ref_id="ref_2", note="Munich"),
            ],
            note="Two records found.",
        ),
        CalculationWorkerOutput(
            status="done",
            category_key="total_ev_cars",
            records=[],
            note="No new records remain.",
        ),
    ]

    monkeypatch.setattr(
        calculator_agent,
        "extract_category_records",
        lambda **_kwargs: responses.pop(0),
    )

    summary = calculator_agent.run_calculator_stage(
        question="What is the total EV car count?",
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        base_dir=tmp_path / "run-1",
    )

    category_dir = tmp_path / "run-1" / "calculator" / "categories" / "total_ev_cars"
    assert summary.status == "success"
    assert (tmp_path / "run-1" / "calculator" / "plan.json").exists()
    assert (tmp_path / "run-1" / "calculator" / "manifest.json").exists()
    assert (tmp_path / "run-1" / "calculator" / "summary.json").exists()
    assert (category_dir / "pass_1.json").exists()
    assert (category_dir / "pass_2.json").exists()
    assert not (category_dir / "pass_3.json").exists()
    assert (category_dir / "records.json").exists()

    manifest_payload = json.loads(
        (
            tmp_path / "run-1" / "calculator" / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest_payload["categories"][0]["pass_count"] == 2
    assert manifest_payload["categories"][0]["stop_reason"] == "worker_done"


def test_run_calculator_stage_preserves_multiple_records_for_one_city(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path)
    context_bundle = {
        "markdown": {
            "selected_city_names": ["Aachen"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Aachen",
                    "quote": "Aachen reports 10 EVs.",
                    "partial_answer": "Aachen reports 10 EVs.",
                    "source_chunk_ids": ["chunk_1"],
                }
            ],
        }
    }
    category = _build_category()

    monkeypatch.setattr(
        calculator_agent,
        "plan_categories",
        lambda **_kwargs: CalculationPlan(categories=[category], note="One category."),
    )
    monkeypatch.setattr(
        calculator_agent,
        "extract_category_records",
        lambda **_kwargs: CalculationWorkerOutput(
            status="done",
            category_key="total_ev_cars",
            records=[],
            note="done immediately",
        ),
    )

    summary = calculator_agent.aggregate_category_records(
        categories=[category],
        category_records={
            "total_ev_cars": [
                _build_record(
                    city="Aachen",
                    value=10,
                    ref_id="ref_1",
                    note="candidate one",
                ),
                _build_record(
                    city="Aachen",
                    value=12,
                    ref_id="ref_2",
                    note="candidate two",
                ),
            ]
        },
        selected_city_names=["Aachen"],
    )

    assert len(summary.categories[0].records) == 2


def test_run_calculator_stage_marks_partial_when_one_category_worker_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path)
    first_category = _build_category()
    second_category = CalculationCategory(
        category_key="buses_added",
        label="Buses Added",
        description="Observed bus additions.",
        operation="sum",
        preferred_unit="buses",
        year_policy="ignore_year",
        inclusion_rule="Include explicit new buses added.",
        exclusion_rule="Exclude targets and fleet totals.",
    )
    context_bundle = {
        "markdown": {
            "selected_city_names": ["Aachen", "Munich"],
            "excerpts": [],
        }
    }

    monkeypatch.setattr(
        calculator_agent,
        "plan_categories",
        lambda **_kwargs: CalculationPlan(
            categories=[first_category, second_category],
            note="Two categories.",
        ),
    )

    def _stub_run_category_worker(**kwargs):
        category = kwargs["category"]
        category_dir = kwargs["category_dir"]
        if category.category_key == "buses_added":
            raise RuntimeError("worker crashed")

        record = _build_record(city="Aachen", value=10, ref_id="ref_1", note="Aachen")
        pass_path = category_dir / "pass_1.json"
        records_path = category_dir / "records.json"
        calculator_agent.write_json(
            pass_path,
            {
                "pass_index": 1,
                "category": category.model_dump(),
                "status": "done",
                "note": "complete",
                "record_count": 1,
                "records": [record.model_dump()],
                "excerpt_ref_ids": ["ref_1"],
            },
            ensure_ascii=False,
            default=str,
        )
        calculator_agent.write_json(
            records_path,
            {
                "category": category.model_dump(),
                "status": "success",
                "note": "complete",
                "record_count": 1,
                "records": [record.model_dump()],
            },
            ensure_ascii=False,
            default=str,
        )
        return calculator_agent._CategoryExecution(
            category=category,
            status="success",
            note="complete",
            pass_count=1,
            stop_reason="worker_done",
            records=[record],
            pass_files=[pass_path],
            records_path=records_path,
        )

    monkeypatch.setattr(
        calculator_agent,
        "_run_category_worker",
        _stub_run_category_worker,
    )

    summary = calculator_agent.run_calculator_stage(
        question="How many EV cars and buses were added?",
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        base_dir=tmp_path / "run-2",
    )

    statuses = {
        category_summary.category.category_key: category_summary.status
        for category_summary in summary.categories
    }
    manifest_payload = json.loads(
        (
            tmp_path / "run-2" / "calculator" / "manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert summary.status == "partial"
    assert statuses == {
        "total_ev_cars": "success",
        "buses_added": "error",
    }
    assert (
        tmp_path
        / "run-2"
        / "calculator"
        / "categories"
        / "buses_added"
        / "records.json"
    ).exists()
    assert manifest_payload["categories"][0]["status"] in {"error", "success"}
    assert {item["category_key"]: item["status"] for item in manifest_payload["categories"]} == statuses


def test_run_calculator_stage_returns_empty_summary_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path)
    config.calculator.enabled = False
    context_bundle = {
        "markdown": {
            "selected_city_names": ["Aachen", "Munich"],
            "excerpts": [],
        }
    }

    def _fail_if_called(**_kwargs):
        raise AssertionError("planner should not run when calculator is disabled")

    monkeypatch.setattr(calculator_agent, "plan_categories", _fail_if_called)

    summary = calculator_agent.run_calculator_stage(
        question="What is the EV total?",
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        base_dir=tmp_path / "run-disabled",
    )

    manifest_payload = json.loads(
        (
            tmp_path / "run-disabled" / "calculator" / "manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert summary.status == "empty"
    assert summary.note == "Calculator stage disabled by feature flag."
    assert summary.category_count == 0
    assert summary.categories == []
    assert manifest_payload["calculator_enabled"] is False
