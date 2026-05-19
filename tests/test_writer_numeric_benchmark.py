from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.benchmarks.writer_numeric.models import (
    WriterMetricExtraction,
    WriterNumberExtraction,
    WriterNumericBenchmarkDataset,
)
from backend.benchmarks.writer_numeric.runner import (
    apply_pipeline_mode,
    normalize_metric_value,
    resolve_requested_modes,
    run_writer_numeric_benchmark,
    select_benchmark_cases,
)
from backend.utils.config import load_config
from backend.utils.json_io import write_json
from backend.utils.paths import create_run_paths


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal config file for benchmark runner tests."""
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "orchestrator:",
                "  model: test-model",
                "markdown_researcher:",
                "  model: test-model",
                "  chunk_overlap_tokens: 2000",
                "  batch_max_chunks: 32",
                "  max_workers: 4",
                "  request_backoff_base_seconds: 0.5",
                "  request_backoff_max_seconds: 2.0",
                "writer:",
                "  model: test-model",
                "chat:",
                "  model: test-model",
                "  provider_timeout_seconds: 60.0",
                "  followup_router_max_excerpts_per_source: 50",
                "assumptions_reviewer:",
                "  model: test-model",
                "benchmark_fact_judge:",
                "  model: test-model",
                "benchmark_number_extractor:",
                "  model: test-model",
                "retry:",
                "  max_attempts: 2",
                "  backoff_base_seconds: 1.0",
                "  backoff_max_seconds: 2.0",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _sample_dataset_payload() -> dict[str, object]:
    """Return one small benchmark fixture for runner tests."""
    return {
        "version": 1,
        "default_mode": "ccc_only",
        "cases": [
            {
                "case_id": "sample_case",
                "question": "Answer the question and then provide a Numeric summary.",
                "selected_cities": ["Krakow"],
                "baseline_metrics": [
                    {
                        "metric_id": "coverage_count",
                        "label": "Coverage count",
                        "unit": "cities",
                        "expected_value": 1,
                        "components": [
                            {
                                "label": "Krakow",
                                "value": 1,
                                "source_note": "Manual baseline component.",
                            }
                        ],
                    },
                    {
                        "metric_id": "combined_public_charging_total",
                        "label": "Combined public charging total",
                        "unit": "stations",
                        "expected_value": 150,
                        "components": [
                            {
                                "label": "Krakow",
                                "value": 150,
                                "source_note": "Manual baseline component.",
                            }
                        ],
                    },
                    {
                        "metric_id": "largest_public_charging_value",
                        "label": "Largest public charging value",
                        "unit": "stations",
                        "expected_value": 150,
                        "components": [
                            {
                                "label": "Krakow",
                                "value": 150,
                                "source_note": "Manual baseline component.",
                            }
                        ],
                        "display_metadata": {"selected_city": "Krakow"},
                    },
                ],
            }
        ],
    }


def test_writer_numeric_dataset_rejects_duplicate_case_ids() -> None:
    """Dataset validation rejects duplicate case identifiers."""
    payload = _sample_dataset_payload()
    payload["cases"] = [payload["cases"][0], payload["cases"][0]]

    with pytest.raises(ValueError, match="Duplicate case_id"):
        WriterNumericBenchmarkDataset.model_validate(payload)


def test_writer_numeric_dataset_rejects_dynamic_scope_tokens() -> None:
    """Cases must freeze explicit city lists instead of dynamic placeholders."""
    payload = _sample_dataset_payload()
    payload["cases"][0]["selected_cities"] = ["all_cities"]

    with pytest.raises(ValueError, match="explicit city list"):
        WriterNumericBenchmarkDataset.model_validate(payload)


def test_writer_numeric_dataset_rejects_empty_baseline_metrics() -> None:
    """Cases must include at least one baseline metric."""
    payload = _sample_dataset_payload()
    payload["cases"][0]["baseline_metrics"] = []

    with pytest.raises(ValueError, match="baseline_metrics"):
        WriterNumericBenchmarkDataset.model_validate(payload)


def test_select_benchmark_cases_excludes_optional_cases_by_default() -> None:
    """Optional fixture cases should require an explicit include flag."""
    payload = _sample_dataset_payload()
    optional_case = json.loads(json.dumps(payload["cases"][0]))
    optional_case["case_id"] = "optional_case"
    optional_case["requires_explicit_include"] = True
    payload["notes"] = [
        "Optional all-cities cases can take a long time and consume many tokens."
    ]
    payload["cases"].append(optional_case)
    dataset = WriterNumericBenchmarkDataset.model_validate(payload)

    default_cases = select_benchmark_cases(dataset, include_optional_cases=False)
    all_cases = select_benchmark_cases(dataset, include_optional_cases=True)

    assert [case.case_id for case in default_cases] == ["sample_case"]
    assert [case.case_id for case in all_cases] == ["sample_case", "optional_case"]


def test_resolve_requested_modes_and_apply_pipeline_mode(tmp_path: Path) -> None:
    """Runner mode helpers force enrichment flags exactly as intended."""
    config = load_config(_write_config(tmp_path))

    assert resolve_requested_modes("ccc_only") == ["ccc_only"]
    assert resolve_requested_modes("both") == ["ccc_only", "full_pipeline"]

    ccc_config = apply_pipeline_mode(config, "ccc_only", tmp_path / "ccc")
    assert ccc_config.runs_dir == tmp_path / "ccc"
    assert ccc_config.enrichment.enabled is False
    assert ccc_config.enrichment.external_source_search_enabled is False
    assert ccc_config.enrichment.web_research_enabled is False

    full_config = apply_pipeline_mode(config, "full_pipeline", tmp_path / "full")
    assert full_config.runs_dir == tmp_path / "full"
    assert full_config.enrichment.enabled is True
    assert full_config.enrichment.external_source_search_enabled is True
    assert full_config.enrichment.web_research_enabled is True


@pytest.mark.parametrize(
    ("raw_value", "unit", "expected"),
    [
        ("PLN 6 500 000", "PLN", "6500000"),
        ("EUR 1,444,000", "EUR", "1444000"),
        ("12,5%", "%", "12.5"),
        (25, "cities", "25"),
        (None, "cities", None),
    ],
)
def test_normalize_metric_value_handles_numeric_formats(
    raw_value: object,
    unit: str,
    expected: str | None,
) -> None:
    """Normalization strips units and separators before deterministic comparison."""
    assert normalize_metric_value(raw_value, unit) == expected


def test_run_writer_numeric_benchmark_writes_report_only_artifacts(tmp_path: Path) -> None:
    """Mismatches and missing values are reported without aborting the benchmark."""
    config = load_config(_write_config(tmp_path))
    benchmark_file = tmp_path / "writer_numeric_fixture.json"
    benchmark_file.write_text(
        json.dumps(_sample_dataset_payload(), indent=2),
        encoding="utf-8",
    )

    def fake_pipeline_runner(**kwargs: object) -> object:
        run_id = str(kwargs["run_id"])
        runner_config = kwargs["config"]
        question = str(kwargs["question"])
        paths = create_run_paths(
            runner_config.runs_dir,
            run_id,
            runner_config.orchestrator.context_bundle_name,
        )
        paths.base_dir.mkdir(parents=True, exist_ok=True)
        paths.markdown_dir.mkdir(parents=True, exist_ok=True)
        paths.final_output.write_text(
            (
                "| Metric | Value |\n"
                "| --- | --- |\n"
                "| coverage_count | 1 |\n"
                "| combined_public_charging_total | 999 |\n\n"
                "Numeric summary\n"
                "coverage_count: 1\n"
                "combined_public_charging_total: 999\n"
            ),
            encoding="utf-8",
        )
        write_json(
            paths.context_bundle,
            {
                "research_question": question,
                "selected_cities": kwargs["selected_cities"],
            },
            ensure_ascii=False,
        )
        return paths

    def fake_extractor(**kwargs: object) -> WriterNumberExtraction:
        return WriterNumberExtraction(
            metrics=[
                WriterMetricExtraction(
                    metric_id="coverage_count",
                    found=True,
                    raw_snippet="coverage_count: 1",
                    normalized_value=1,
                    unit="cities",
                    notes="Match from numeric summary.",
                ),
                WriterMetricExtraction(
                    metric_id="combined_public_charging_total",
                    found=True,
                    raw_snippet="combined_public_charging_total: 999",
                    normalized_value=999,
                    unit="stations",
                    notes="Intentional mismatch for report-only coverage.",
                ),
                WriterMetricExtraction(
                    metric_id="largest_public_charging_value",
                    found=False,
                    raw_snippet=None,
                    normalized_value=None,
                    unit="stations",
                    notes="Intentional omission for missing coverage.",
                ),
            ]
        )

    report = run_writer_numeric_benchmark(
        benchmark_file=benchmark_file,
        output_dir=tmp_path / "benchmark_output",
        benchmark_id="writer_numeric_test",
        requested_mode="both",
        config=config,
        api_key="test-api-key",
        pipeline_runner=fake_pipeline_runner,
        extractor=fake_extractor,
    )

    benchmark_root = tmp_path / "benchmark_output" / "writer_numeric_test"
    assert report.summary.output_count == 2
    assert report.summary.metric_count == 6
    assert report.summary.match_count == 2
    assert report.summary.mismatch_count == 2
    assert report.summary.missing_count == 2
    assert (benchmark_root / "benchmark_summary.json").exists()
    assert (benchmark_root / "benchmark_report.md").exists()
    assert (
        benchmark_root / "runs" / "sample_case__ccc_only" / "extracted_numbers.json"
    ).exists()
    assert (
        benchmark_root / "runs" / "sample_case__full_pipeline" / "extracted_numbers.json"
    ).exists()
