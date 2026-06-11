from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.api.models import SourceChunkItem
from backend.benchmarks.gold_recall.models import FactJudgeDecision
from backend.benchmarks.gold_recall.runner import (
    load_gold_benchmark_dataset,
    run_recall_benchmark,
)
from backend.utils.artifact_writer import ArtifactWriter


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REAL_GOLD_FILE = FIXTURES_DIR / "benchmark_gold.json"
SAMPLE_QUESTION = "What does Sample City plan for solar, retrofits, and district heating?"
SAMPLE_GOLD_CHUNK_IDS = [
    "chunk-seed-1",
    "chunk-fallback-1",
    "chunk-neighbor-1",
    "chunk-miss-1",
]
SAMPLE_GOLD_FACTS = [
    "Sample City plans 500 rooftop solar installations by 2030.",
    "Sample City allocated EUR 2 million for retrofit grants.",
    "Sample City will expand district heating to 12,000 households.",
]
SAMPLE_RETRIEVAL_PAYLOAD: dict[str, Any] = {
    "queries": [SAMPLE_QUESTION],
    "selected_cities": ["Sample City"],
    "retrieved_count": 4,
    "meta": {
        "queries": [SAMPLE_QUESTION],
        "cities": ["sample_city"],
        "per_city": [
            {
                "city_key": "sample_city",
                "city_name": "Sample City",
                "retrieved_unique_chunks": 3,
                "query_stats": [
                    {
                        "query_id": "q1",
                        "query": SAMPLE_QUESTION,
                        "qualified_chunks": 3,
                    }
                ],
            }
        ],
        "max_distance": 0.8,
        "fallback_min_chunks_per_city_query": 3,
        "max_chunks_per_city_query": 10,
        "min_chunks_per_city": 3,
        "context_window_chunks": 1,
        "table_context_window_chunks": 1,
        "seed_retrieved_total_chunks": 3,
        "neighbor_expanded_total_chunks": 1,
        "retrieved_total_chunks": 4,
    },
    "seed_chunks": [
        {
            "chunk_id": "chunk-nongold-1",
            "city_name": "Sample City",
            "city_key": "sample_city",
            "source_path": "documents/Sample_City.md",
            "heading_path": "Energy > Overview",
            "block_type": "paragraph",
            "distance": 0.08,
            "chunk_index": 1,
            "provenance": {
                "origin": "seed",
                "selection_mode": "distance_qualified",
                "seed_rank": 1,
                "seed_query_ids": ["q1"],
                "expanded_from_chunk_ids": [],
            },
        },
        {
            "chunk_id": "chunk-seed-1",
            "city_name": "Sample City",
            "city_key": "sample_city",
            "source_path": "documents/Sample_City.md",
            "heading_path": "Energy > Solar",
            "block_type": "paragraph",
            "distance": 0.12,
            "chunk_index": 2,
            "provenance": {
                "origin": "seed",
                "selection_mode": "distance_qualified",
                "seed_rank": 2,
                "seed_query_ids": ["q1"],
                "expanded_from_chunk_ids": [],
            },
        },
        {
            "chunk_id": "chunk-fallback-1",
            "city_name": "Sample City",
            "city_key": "sample_city",
            "source_path": "documents/Sample_City.md",
            "heading_path": "Buildings > Retrofits",
            "block_type": "paragraph",
            "distance": 0.42,
            "chunk_index": 7,
            "provenance": {
                "origin": "seed",
                "selection_mode": "fallback_top_up",
                "seed_rank": 3,
                "seed_query_ids": ["q1"],
                "expanded_from_chunk_ids": [],
            },
        },
    ],
    "chunks": [
        {
            "chunk_id": "chunk-nongold-1",
            "city_name": "Sample City",
            "city_key": "sample_city",
            "source_path": "documents/Sample_City.md",
            "heading_path": "Energy > Overview",
            "block_type": "paragraph",
            "distance": 0.08,
            "chunk_index": 1,
            "provenance": {
                "origin": "seed",
                "selection_mode": "distance_qualified",
                "seed_rank": 1,
                "seed_query_ids": ["q1"],
                "expanded_from_chunk_ids": [],
            },
        },
        {
            "chunk_id": "chunk-seed-1",
            "city_name": "Sample City",
            "city_key": "sample_city",
            "source_path": "documents/Sample_City.md",
            "heading_path": "Energy > Solar",
            "block_type": "paragraph",
            "distance": 0.12,
            "chunk_index": 2,
            "provenance": {
                "origin": "seed",
                "selection_mode": "distance_qualified",
                "seed_rank": 2,
                "seed_query_ids": ["q1"],
                "expanded_from_chunk_ids": [],
            },
        },
        {
            "chunk_id": "chunk-neighbor-1",
            "city_name": "Sample City",
            "city_key": "sample_city",
            "source_path": "documents/Sample_City.md",
            "heading_path": "Energy > District heating",
            "block_type": "paragraph",
            "distance": 0.12,
            "chunk_index": 3,
            "provenance": {
                "origin": "neighbor",
                "selection_mode": "neighbor_context",
                "seed_rank": None,
                "seed_query_ids": [],
                "expanded_from_chunk_ids": ["chunk-seed-1"],
            },
        },
        {
            "chunk_id": "chunk-fallback-1",
            "city_name": "Sample City",
            "city_key": "sample_city",
            "source_path": "documents/Sample_City.md",
            "heading_path": "Buildings > Retrofits",
            "block_type": "paragraph",
            "distance": 0.42,
            "chunk_index": 7,
            "provenance": {
                "origin": "seed",
                "selection_mode": "fallback_top_up",
                "seed_rank": 3,
                "seed_query_ids": ["q1"],
                "expanded_from_chunk_ids": [],
            },
        },
    ],
}
SAMPLE_EXCERPTS_PAYLOAD: dict[str, Any] = {
    "status": "success",
    "excerpts": [
        {
            "quote": "Sample City plans 500 rooftop solar installations by 2030.",
            "city_name": "Sample City",
            "city_key": "sample_city",
            "partial_answer": "Sample City plans 500 rooftop solar installations by 2030.",
            "source_chunk_ids": ["chunk-seed-1"],
            "ref_id": "ref_1",
        },
        {
            "quote": "District heating will reach 12,000 households.",
            "city_name": "Sample City",
            "city_key": "sample_city",
            "partial_answer": "Sample City will expand district heating to 12,000 households.",
            "source_chunk_ids": ["chunk-neighbor-1"],
            "ref_id": "ref_3",
        },
    ],
    "decision_audit": {
        "accepted_total": 3,
        "rejected_total": 0,
        "unresolved_total": 0,
        "invariant_ok": True,
        "status": "complete",
    },
    "retrieval_mode": "vector_store_retrieval",
    "analysis_mode": "aggregate",
    "retrieval_queries": [SAMPLE_QUESTION],
    "inspected_cities": ["sample_city"],
    "inspected_city_names": ["Sample City"],
    "selected_cities": ["sample_city"],
    "selected_city_names": ["Sample City"],
    "excerpt_count": 2,
}
SAMPLE_REFERENCES_PAYLOAD: dict[str, Any] = {
    "run_id": "sample_case",
    "reference_count": 3,
    "references": [
        {
            "ref_id": "ref_1",
            "excerpt_index": 0,
            "city_name": "Sample City",
            "quote": "Sample City plans 500 rooftop solar installations by 2030.",
            "partial_answer": "Sample City plans 500 rooftop solar installations by 2030.",
            "source_chunk_ids": ["chunk-seed-1"],
        },
        {
            "ref_id": "ref_2",
            "excerpt_index": 1,
            "city_name": "Sample City",
            "quote": "Retrofit grants total EUR 2 million.",
            "partial_answer": "Sample City allocated EUR 2 million for retrofit grants.",
            "source_chunk_ids": ["chunk-fallback-1"],
        },
        {
            "ref_id": "ref_3",
            "excerpt_index": 2,
            "city_name": "Sample City",
            "quote": "District heating will reach 12,000 households.",
            "partial_answer": "Sample City will expand district heating to 12,000 households.",
            "source_chunk_ids": ["chunk-neighbor-1"],
        },
    ],
}
SAMPLE_FINAL_TEXT = """## Summary

Sample City plans 500 rooftop solar installations by 2030. [ref_1]

The city also allocated EUR 2 million for retrofit grants. [ref_2]
"""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON payload to disk."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_gold_file(
    path: Path,
    *,
    gold_chunk_ids: list[str] | None = None,
    gold_chunk_alternatives: list[list[dict[str, str]]] | None = None,
    gold_chunk_texts: list[str] | None = None,
    extra_case_fields: dict[str, Any] | None = None,
) -> None:
    """Write one single-case gold benchmark fixture."""
    case_payload: dict[str, Any] = {
        "case_id": "sample_case",
        "question": SAMPLE_QUESTION,
        "gold_chunk_ids": list(gold_chunk_ids or SAMPLE_GOLD_CHUNK_IDS),
        "gold_facts": list(SAMPLE_GOLD_FACTS),
        "gold_city": ["Sample City"],
        "selected_cities": ["Sample City"],
    }
    if gold_chunk_alternatives is not None:
        case_payload["gold_chunk_alternatives"] = [
            [dict(item) for item in group] for group in gold_chunk_alternatives
        ]
    if gold_chunk_texts is not None:
        case_payload["gold_chunk_texts"] = list(gold_chunk_texts)
    if extra_case_fields:
        case_payload.update(extra_case_fields)

    _write_json(path, {"version": 1, "cases": [case_payload]})


def _write_sample_run_artifacts(run_dir: Path) -> None:
    """Write the minimal artifact set required by the recall benchmark."""
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = ArtifactWriter(run_dir, run_dir.name)
    writer.write_stage_file(
        "retrieval",
        "retrieval.json",
        SAMPLE_RETRIEVAL_PAYLOAD,
        alias="retrieval",
    )
    writer.write_stage_file(
        "markdown_extraction",
        "excerpts.json",
        SAMPLE_EXCERPTS_PAYLOAD,
        alias="markdown_excerpts",
    )
    writer.write_stage_file(
        "markdown_extraction",
        "references.json",
        SAMPLE_REFERENCES_PAYLOAD,
        alias="references",
    )
    (run_dir / "final.md").write_text(SAMPLE_FINAL_TEXT, encoding="utf-8")
    writer.register_file("final_output", run_dir / "final.md")
    writer.write_manifest()


def _make_sample_run_pipeline():
    """Build a fake live pipeline that writes deterministic benchmark artifacts."""

    def _fake_run_pipeline(
        *,
        question: str,
        config,
        run_id: str,
        log_llm_payload: bool,
        selected_cities: list[str],
    ) -> SimpleNamespace:
        assert question == SAMPLE_QUESTION
        assert log_llm_payload is False
        assert selected_cities == ["Sample City"]
        run_dir = Path(config.runs_dir) / run_id
        _write_sample_run_artifacts(run_dir)
        return SimpleNamespace(base_dir=run_dir)

    return _fake_run_pipeline


def _fake_judge(**kwargs) -> FactJudgeDecision:
    """Return deterministic fact-judge responses for the sample benchmark case."""
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


def test_load_gold_benchmark_dataset_rejects_legacy_cached_run_dir(
    tmp_path: Path,
) -> None:
    gold_file = tmp_path / "legacy_gold.json"
    _write_gold_file(
        gold_file,
        extra_case_fields={"cached_run_dir": "legacy-run-dir"},
    )

    with pytest.raises(ValueError, match="cached_run_dir"):
        load_gold_benchmark_dataset(gold_file)


def test_load_gold_benchmark_dataset_rejects_misaligned_chunk_alternatives(
    tmp_path: Path,
) -> None:
    gold_file = tmp_path / "bad_alternatives_gold.json"
    _write_gold_file(
        gold_file,
        gold_chunk_alternatives=[
            [{"chunk_id": "chunk-alt-1", "chunk_text": "Alternative text"}]
        ],
    )

    with pytest.raises(ValueError, match="gold_chunk_alternatives"):
        load_gold_benchmark_dataset(gold_file)


def test_run_recall_benchmark_scores_live_run(tmp_path: Path) -> None:
    gold_file = tmp_path / "sample_gold.json"
    _write_gold_file(gold_file)

    report = run_recall_benchmark(
        benchmark_id="bench_sample",
        gold_file=gold_file,
        output_dir=tmp_path / "output",
        config_path=Path("llm_config.yaml"),
        api_key_override="test-key",
        judge_func=_fake_judge,
        run_pipeline_func=_make_sample_run_pipeline(),
    )

    assert report.summary.case_count == 1
    result = report.results[0]
    assert result.case_id == "sample_case"
    assert result.run_dir == str(
        tmp_path / "output" / "bench_sample" / "runs" / "sample_case"
    )
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


def test_run_recall_benchmark_uses_gold_chunk_text_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gold_file = tmp_path / "text_fallback_gold.json"
    _write_gold_file(
        gold_file,
        gold_chunk_texts=[
            "Solar canonical chunk text",
            "Retrofit canonical chunk text",
            "District heating canonical chunk text",
            "Missing canonical chunk text",
        ],
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

    report = run_recall_benchmark(
        benchmark_id="bench_text_fallback",
        gold_file=gold_file,
        output_dir=tmp_path / "output",
        config_path=Path("llm_config.yaml"),
        api_key_override="test-key",
        judge_func=_fake_judge,
        run_pipeline_func=_make_sample_run_pipeline(),
    )

    result = report.results[0]
    assert result.stage_a.retrieval_recall == pytest.approx(0.5)
    assert result.stage_a.delivery_recall == pytest.approx(0.75)
    assert result.stage_b.extraction_recall == pytest.approx(0.5)
    assert result.stage_c.citation_coverage == pytest.approx(0.5)
    assert result.loss_waterfall.seed_hit_chunk_count == 2
    assert result.loss_waterfall.delivery_hit_chunk_count == 3
    assert {item.chunk_id: item.bucket for item in result.chunk_diagnostics} == {
        "chunk-seed-1": "seed_hit",
        "chunk-fallback-1": "fallback_top_up_hit",
        "chunk-neighbor-1": "neighbor_only_hit",
        "chunk-miss-1": "miss",
    }


def test_run_recall_benchmark_accepts_gold_chunk_alternatives(
    tmp_path: Path,
) -> None:
    gold_file = tmp_path / "alternative_ids_gold.json"
    _write_gold_file(
        gold_file,
        gold_chunk_ids=[
            "chunk-seed-1",
            "chunk-canonical-fallback",
            "chunk-neighbor-1",
            "chunk-miss-1",
        ],
        gold_chunk_alternatives=[
            [],
            [
                {
                    "chunk_id": "chunk-fallback-1",
                    "chunk_text": "Retrofit canonical chunk text",
                }
            ],
            [],
            [],
        ],
    )

    report = run_recall_benchmark(
        benchmark_id="bench_alternative_ids",
        gold_file=gold_file,
        output_dir=tmp_path / "output",
        config_path=Path("llm_config.yaml"),
        api_key_override="test-key",
        judge_func=_fake_judge,
        run_pipeline_func=_make_sample_run_pipeline(),
    )

    result = report.results[0]
    assert result.stage_a.retrieval_recall == pytest.approx(0.5)
    assert result.stage_a.delivery_recall == pytest.approx(0.75)
    assert result.stage_c.citation_coverage == pytest.approx(0.5)
    assert result.loss_waterfall.seed_hit_chunk_count == 2
    assert result.loss_waterfall.delivery_hit_chunk_count == 3
    assert {item.chunk_id: item.matched_chunk_id for item in result.chunk_diagnostics} == {
        "chunk-seed-1": "chunk-seed-1",
        "chunk-canonical-fallback": "chunk-fallback-1",
        "chunk-neighbor-1": "chunk-neighbor-1",
        "chunk-miss-1": None,
    }
    assert {
        item.chunk_id: item.selection_mode for item in result.chunk_diagnostics
    } == {
        "chunk-seed-1": "distance_qualified",
        "chunk-canonical-fallback": "fallback_top_up",
        "chunk-neighbor-1": "neighbor_context",
        "chunk-miss-1": None,
    }


def test_run_recall_benchmark_uses_alternative_chunk_text_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gold_file = tmp_path / "alternative_text_fallback_gold.json"
    _write_gold_file(
        gold_file,
        gold_chunk_ids=[
            "chunk-seed-1",
            "chunk-canonical-fallback",
            "chunk-neighbor-1",
            "chunk-miss-1",
        ],
        gold_chunk_alternatives=[
            [],
            [
                {
                    "chunk_id": "chunk-alternative-canonical",
                    "chunk_text": "Retrofit canonical chunk text",
                }
            ],
            [],
            [],
        ],
        gold_chunk_texts=[
            "Solar canonical chunk text",
            "Primary canonical text that will not match",
            "District heating canonical chunk text",
            "Missing canonical chunk text",
        ],
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

    report = run_recall_benchmark(
        benchmark_id="bench_alternative_text_fallback",
        gold_file=gold_file,
        output_dir=tmp_path / "output",
        config_path=Path("llm_config.yaml"),
        api_key_override="test-key",
        judge_func=_fake_judge,
        run_pipeline_func=_make_sample_run_pipeline(),
    )

    result = report.results[0]
    assert result.stage_a.retrieval_recall == pytest.approx(0.5)
    assert result.loss_waterfall.seed_hit_chunk_count == 2
    assert {item.chunk_id: item.matched_chunk_id for item in result.chunk_diagnostics} == {
        "chunk-seed-1": "chunk-seed-1",
        "chunk-canonical-fallback": "chunk-fallback-1",
        "chunk-neighbor-1": "chunk-neighbor-1",
        "chunk-miss-1": None,
    }
    assert {
        item.chunk_id: item.selection_mode for item in result.chunk_diagnostics
    }["chunk-canonical-fallback"] == "text_fallback:fallback_top_up"


def test_run_recall_benchmark_uses_gold_chunk_excerpt_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gold_file = tmp_path / "excerpt_fallback_gold.json"
    _write_gold_file(
        gold_file,
        gold_chunk_texts=[
            "Solar canonical excerpt",
            "Retrofit canonical excerpt",
            "District heating canonical excerpt",
            "Missing canonical excerpt",
        ],
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

    report = run_recall_benchmark(
        benchmark_id="bench_excerpt_fallback",
        gold_file=gold_file,
        output_dir=tmp_path / "output",
        config_path=Path("llm_config.yaml"),
        api_key_override="test-key",
        judge_func=_fake_judge,
        run_pipeline_func=_make_sample_run_pipeline(),
    )

    result = report.results[0]
    assert result.stage_a.retrieval_recall == pytest.approx(0.5)
    assert result.stage_a.delivery_recall == pytest.approx(0.75)
    assert result.stage_b.extraction_recall == pytest.approx(0.5)
    assert result.stage_c.citation_coverage == pytest.approx(0.5)
    assert result.loss_waterfall.seed_hit_chunk_count == 2
    assert result.loss_waterfall.delivery_hit_chunk_count == 3
    assert {item.chunk_id: item.bucket for item in result.chunk_diagnostics} == {
        "chunk-seed-1": "seed_hit",
        "chunk-fallback-1": "fallback_top_up_hit",
        "chunk-neighbor-1": "neighbor_only_hit",
        "chunk-miss-1": "miss",
    }


def test_run_recall_benchmark_rejects_missing_live_artifacts(tmp_path: Path) -> None:
    gold_file = tmp_path / "sample_gold.json"
    _write_gold_file(gold_file)

    def _fake_run_pipeline(
        *,
        question: str,
        config,
        run_id: str,
        log_llm_payload: bool,
        selected_cities: list[str],
    ) -> SimpleNamespace:
        del question, log_llm_payload, selected_cities
        run_dir = Path(config.runs_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(base_dir=run_dir)

    with pytest.raises(ValueError, match="Expected a JSON object"):
        run_recall_benchmark(
            benchmark_id="bench_missing_artifacts",
            gold_file=gold_file,
            output_dir=tmp_path / "output",
            config_path=Path("llm_config.yaml"),
            api_key_override="test-key",
            judge_func=lambda **_kwargs: FactJudgeDecision(
                verdict="NO",
                rationale="not used",
            ),
            run_pipeline_func=_fake_run_pipeline,
        )


def test_real_gold_fixture_contains_expected_case_count() -> None:
    dataset = load_gold_benchmark_dataset(REAL_GOLD_FILE)
    assert len(dataset.cases) == 9
