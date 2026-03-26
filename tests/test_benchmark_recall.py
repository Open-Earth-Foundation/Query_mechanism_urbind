from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from backend.api.models import SourceChunkItem
from backend.benchmarks.gold_recall.models import FactJudgeDecision
from backend.benchmarks.gold_recall.runner import (
    load_gold_benchmark_dataset,
    run_recall_benchmark,
)
from backend.scripts import benchmark_recall as benchmark_recall_script


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_GOLD_FILE = FIXTURES_DIR / "benchmark_recall" / "sample_gold.json"
REAL_GOLD_FILE = FIXTURES_DIR / "benchmark_gold.json"


def test_load_gold_benchmark_dataset_rejects_duplicate_case_ids(
    tmp_path: Path,
) -> None:
    gold_file = tmp_path / "bad_gold.json"
    gold_file.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "case_id": "dup",
                        "question": "Q1",
                        "gold_chunk_ids": ["chunk-1"],
                        "gold_facts": ["fact-1"],
                        "gold_city": ["Aachen"],
                    },
                    {
                        "case_id": "dup",
                        "question": "Q2",
                        "gold_chunk_ids": ["chunk-2"],
                        "gold_facts": ["fact-2"],
                        "gold_city": ["Munich"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate case_id"):
        load_gold_benchmark_dataset(gold_file)


def test_load_gold_benchmark_dataset_preserves_empty_selected_cities(
    tmp_path: Path,
) -> None:
    gold_file = tmp_path / "all_cities_gold.json"
    gold_file.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "case_id": "all_cities",
                        "question": "Cross-city question",
                        "gold_chunk_ids": ["chunk-1"],
                        "gold_facts": ["fact-1"],
                        "gold_city": ["Aachen"],
                        "selected_cities": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = load_gold_benchmark_dataset(gold_file)
    case = dataset.cases[0]
    assert case.selected_cities == []
    assert case.resolved_selected_cities() == []


def test_run_recall_benchmark_scores_cached_run(tmp_path: Path) -> None:
    def _fake_judge(**kwargs) -> FactJudgeDecision:
        stage = kwargs["stage_label"]
        fact = kwargs["fact"]
        yes_pairs = {
            ("stage_b", "Sample City plans 500 rooftop solar installations by 2030."),
            ("stage_b", "Sample City will expand district heating to 12,000 households."),
            ("stage_c", "Sample City plans 500 rooftop solar installations by 2030."),
            ("stage_c", "Sample City allocated EUR 2 million for retrofit grants."),
        }
        verdict = "YES" if (stage, fact) in yes_pairs else "NO"
        return FactJudgeDecision(verdict=verdict, rationale=f"{stage}:{verdict}")

    report = run_recall_benchmark(
        benchmark_id="bench_sample",
        gold_file=SAMPLE_GOLD_FILE,
        output_dir=tmp_path / "output",
        config_path=Path("llm_config.yaml"),
        api_key_override="test-key",
        judge_func=_fake_judge,
    )

    assert report.summary.case_count == 1
    result = report.results[0]
    assert result.case_id == "sample_case"
    assert result.used_cached_run is True
    assert result.stage_a.retrieval_recall == pytest.approx(0.5)
    assert result.stage_a.retrieval_precision == pytest.approx(2.0 / 3.0)
    assert result.stage_a.mrr == pytest.approx(0.5)
    assert result.stage_a.delivery_recall == pytest.approx(0.75)
    assert result.stage_a.delivery_precision == pytest.approx(0.75)
    assert result.stage_a.seed_hit_count == 1
    assert result.stage_a.fallback_top_up_hit_count == 1
    assert result.stage_a.neighbor_only_hit_count == 1
    assert result.stage_a.miss_count == 1
    assert result.stage_b.extraction_recall == pytest.approx(0.5)
    assert result.stage_b.fact_extraction_rate == pytest.approx(2.0 / 3.0)
    assert result.stage_c.end_to_end_fact_recall == pytest.approx(2.0 / 3.0)
    assert result.stage_c.citation_coverage == pytest.approx(0.5)
    assert result.loss_waterfall.gold_chunk_count == 4
    assert result.loss_waterfall.seed_hit_chunk_count == 2
    assert result.loss_waterfall.delivery_hit_chunk_count == 3
    assert result.loss_waterfall.stage_b_fact_hit_count == 2
    assert result.loss_waterfall.stage_c_fact_hit_count == 2
    assert {item.chunk_id: item.bucket for item in result.chunk_diagnostics} == {
        "chunk-seed-1": "seed_hit",
        "chunk-fallback-1": "fallback_top_up_hit",
        "chunk-neighbor-1": "neighbor_only_hit",
        "chunk-miss-1": "miss",
    }
    assert (tmp_path / "output" / "bench_sample" / "benchmark_report.json").exists()
    assert (tmp_path / "output" / "bench_sample" / "benchmark_report.md").exists()


def test_run_recall_benchmark_rejects_cached_question_mismatch(
    tmp_path: Path,
) -> None:
    gold_file = tmp_path / "mismatch_gold.json"
    gold_file.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "case_id": "sample_case",
                        "question": "A different question",
                        "gold_chunk_ids": ["chunk-seed-1"],
                        "gold_facts": ["fact-1"],
                        "gold_city": ["Sample City"],
                        "cached_run_dir": str(
                            FIXTURES_DIR / "benchmark_recall" / "sample_run_1"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Cached run question mismatch"):
        run_recall_benchmark(
            benchmark_id="bench_mismatch",
            gold_file=gold_file,
            output_dir=tmp_path / "output",
            config_path=Path("llm_config.yaml"),
            api_key_override="test-key",
            judge_func=lambda **_kwargs: FactJudgeDecision(
                verdict="NO",
                rationale="not used",
            ),
        )


def test_run_recall_benchmark_uses_gold_chunk_text_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gold_file = tmp_path / "text_fallback_gold.json"
    gold_file.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "case_id": "sample_case",
                        "question": "What does Sample City plan for solar, retrofits, and district heating?",
                        "gold_chunk_ids": [
                            "gold-text-seed",
                            "gold-text-fallback",
                            "gold-text-neighbor",
                            "gold-text-miss",
                        ],
                        "gold_chunk_texts": [
                            "Solar canonical chunk text",
                            "Retrofit canonical chunk text",
                            "District heating canonical chunk text",
                            "Missing canonical chunk text",
                        ],
                        "gold_facts": [
                            "Sample City plans 500 rooftop solar installations by 2030.",
                            "Sample City allocated EUR 2 million for retrofit grants.",
                            "Sample City will expand district heating to 12,000 households.",
                        ],
                        "gold_city": ["Sample City"],
                        "cached_run_dir": str(
                            FIXTURES_DIR / "benchmark_recall" / "sample_run_1"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def _fake_load_source_chunks(*_args, **kwargs) -> list[SourceChunkItem]:
        content_by_chunk_id = {
            "chunk-seed-1": "Solar canonical chunk text",
            "chunk-fallback-1": "Retrofit canonical chunk text",
            "chunk-neighbor-1": "District heating canonical chunk text",
            "chunk-nongold-1": "Non-gold chunk text",
        }
        chunk_ids = kwargs["chunk_ids"]
        return [
            SourceChunkItem(chunk_id=chunk_id, content=content_by_chunk_id[chunk_id])
            for chunk_id in chunk_ids
            if chunk_id in content_by_chunk_id
        ]

    monkeypatch.setattr(
        "backend.benchmarks.gold_recall.runner.load_source_chunks",
        _fake_load_source_chunks,
    )

    def _fake_judge(**kwargs) -> FactJudgeDecision:
        stage = kwargs["stage_label"]
        fact = kwargs["fact"]
        yes_pairs = {
            ("stage_b", "Sample City plans 500 rooftop solar installations by 2030."),
            ("stage_b", "Sample City will expand district heating to 12,000 households."),
            ("stage_c", "Sample City plans 500 rooftop solar installations by 2030."),
            ("stage_c", "Sample City allocated EUR 2 million for retrofit grants."),
        }
        verdict = "YES" if (stage, fact) in yes_pairs else "NO"
        return FactJudgeDecision(verdict=verdict, rationale=f"{stage}:{verdict}")

    report = run_recall_benchmark(
        benchmark_id="bench_text_fallback",
        gold_file=gold_file,
        output_dir=tmp_path / "output",
        config_path=Path("llm_config.yaml"),
        api_key_override="test-key",
        judge_func=_fake_judge,
    )

    result = report.results[0]
    assert result.stage_a.retrieval_recall == pytest.approx(0.5)
    assert result.stage_a.delivery_recall == pytest.approx(0.75)
    assert result.stage_b.extraction_recall == pytest.approx(0.5)
    assert result.stage_c.citation_coverage == pytest.approx(0.5)
    assert result.loss_waterfall.seed_hit_chunk_count == 2
    assert result.loss_waterfall.delivery_hit_chunk_count == 3
    assert {item.chunk_id: item.bucket for item in result.chunk_diagnostics} == {
        "gold-text-seed": "seed_hit",
        "gold-text-fallback": "fallback_top_up_hit",
        "gold-text-neighbor": "neighbor_only_hit",
        "gold-text-miss": "miss",
    }


def test_run_recall_benchmark_uses_gold_chunk_excerpt_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gold_file = tmp_path / "excerpt_fallback_gold.json"
    gold_file.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "case_id": "sample_case",
                        "question": "What does Sample City plan for solar, retrofits, and district heating?",
                        "gold_chunk_ids": [
                            "gold-text-seed",
                            "gold-text-fallback",
                            "gold-text-neighbor",
                            "gold-text-miss",
                        ],
                        "gold_chunk_texts": [
                            "Solar canonical excerpt",
                            "Retrofit canonical excerpt",
                            "District heating canonical excerpt",
                            "Missing canonical excerpt",
                        ],
                        "gold_facts": [
                            "Sample City plans 500 rooftop solar installations by 2030.",
                            "Sample City allocated EUR 2 million for retrofit grants.",
                            "Sample City will expand district heating to 12,000 households.",
                        ],
                        "gold_city": ["Sample City"],
                        "cached_run_dir": str(
                            FIXTURES_DIR / "benchmark_recall" / "sample_run_1"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def _fake_load_source_chunks(*_args, **kwargs) -> list[SourceChunkItem]:
        content_by_chunk_id = {
            "chunk-seed-1": "Prefix Solar canonical excerpt suffix",
            "chunk-fallback-1": "Start Retrofit canonical excerpt end",
            "chunk-neighbor-1": "District heating canonical excerpt surrounded",
            "chunk-nongold-1": "Non-gold chunk text",
        }
        chunk_ids = kwargs["chunk_ids"]
        return [
            SourceChunkItem(chunk_id=chunk_id, content=content_by_chunk_id[chunk_id])
            for chunk_id in chunk_ids
            if chunk_id in content_by_chunk_id
        ]

    monkeypatch.setattr(
        "backend.benchmarks.gold_recall.runner.load_source_chunks",
        _fake_load_source_chunks,
    )

    def _fake_judge(**kwargs) -> FactJudgeDecision:
        stage = kwargs["stage_label"]
        fact = kwargs["fact"]
        yes_pairs = {
            ("stage_b", "Sample City plans 500 rooftop solar installations by 2030."),
            ("stage_b", "Sample City will expand district heating to 12,000 households."),
            ("stage_c", "Sample City plans 500 rooftop solar installations by 2030."),
            ("stage_c", "Sample City allocated EUR 2 million for retrofit grants."),
        }
        verdict = "YES" if (stage, fact) in yes_pairs else "NO"
        return FactJudgeDecision(verdict=verdict, rationale=f"{stage}:{verdict}")

    report = run_recall_benchmark(
        benchmark_id="bench_excerpt_fallback",
        gold_file=gold_file,
        output_dir=tmp_path / "output",
        config_path=Path("llm_config.yaml"),
        api_key_override="test-key",
        judge_func=_fake_judge,
    )

    result = report.results[0]
    assert result.stage_a.retrieval_recall == pytest.approx(0.5)
    assert result.stage_a.delivery_recall == pytest.approx(0.75)
    assert result.stage_b.extraction_recall == pytest.approx(0.5)
    assert result.stage_c.citation_coverage == pytest.approx(0.5)
    assert result.loss_waterfall.seed_hit_chunk_count == 2
    assert result.loss_waterfall.delivery_hit_chunk_count == 3
    assert {item.chunk_id: item.bucket for item in result.chunk_diagnostics} == {
        "gold-text-seed": "seed_hit",
        "gold-text-fallback": "fallback_top_up_hit",
        "gold-text-neighbor": "neighbor_only_hit",
        "gold-text-miss": "miss",
    }


def test_benchmark_recall_script_passes_cli_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):
        captured.update(kwargs)

        class _Report:
            benchmark_id = "bench_script"
            output_dir = str(tmp_path / "output" / "bench_script")

        return _Report()

    monkeypatch.setattr(benchmark_recall_script, "run_recall_benchmark", _fake_runner)
    monkeypatch.setattr(benchmark_recall_script, "setup_logger", lambda: None)
    monkeypatch.setattr(
        benchmark_recall_script,
        "parse_args",
        lambda: argparse.Namespace(
            gold_file="tests/fixtures/benchmark_gold.json",
            benchmark_id="bench_script",
            output_dir=str(tmp_path / "output"),
            config="llm_config.yaml",
            case_id=["case-a"],
            run_live=True,
            log_llm_payload=False,
        ),
    )

    benchmark_recall_script.main()

    assert captured["benchmark_id"] == "bench_script"
    assert captured["gold_file"] == Path("tests/fixtures/benchmark_gold.json")
    assert captured["output_dir"] == tmp_path / "output"
    assert captured["config_path"] == Path("llm_config.yaml")
    assert captured["selected_case_ids"] == ["case-a"]
    assert captured["run_live"] is True
    assert captured["log_llm_payload"] is False


def test_real_gold_fixture_contains_expected_case_count() -> None:
    dataset = load_gold_benchmark_dataset(REAL_GOLD_FILE)
    assert len(dataset.cases) == 9
