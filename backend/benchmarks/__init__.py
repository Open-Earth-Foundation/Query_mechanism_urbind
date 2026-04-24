"""Public benchmark exports loaded lazily to avoid heavy import side effects."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "BenchmarkMarkdownConfig": (
        "backend.benchmarks.runner",
        "BenchmarkMarkdownConfig",
    ),
    "BenchmarkModeConfig": (
        "backend.benchmarks.runner",
        "BenchmarkModeConfig",
    ),
    "BenchmarkQuestionResult": (
        "backend.benchmarks.runner",
        "BenchmarkQuestionResult",
    ),
    "BenchmarkReport": (
        "backend.benchmarks.runner",
        "BenchmarkReport",
    ),
    "FactJudgeDecision": (
        "backend.benchmarks.gold_recall.models",
        "FactJudgeDecision",
    ),
    "FactPresenceJudgement": (
        "backend.benchmarks.gold_recall.models",
        "FactPresenceJudgement",
    ),
    "GoldBenchmarkCase": (
        "backend.benchmarks.gold_recall.models",
        "GoldBenchmarkCase",
    ),
    "GoldBenchmarkDataset": (
        "backend.benchmarks.gold_recall.models",
        "GoldBenchmarkDataset",
    ),
    "LossWaterfall": (
        "backend.benchmarks.gold_recall.models",
        "LossWaterfall",
    ),
    "RecallBenchmarkCaseResult": (
        "backend.benchmarks.gold_recall.models",
        "RecallBenchmarkCaseResult",
    ),
    "RecallBenchmarkReport": (
        "backend.benchmarks.gold_recall.models",
        "RecallBenchmarkReport",
    ),
    "RecallBenchmarkSummary": (
        "backend.benchmarks.gold_recall.models",
        "RecallBenchmarkSummary",
    ),
    "RetrievalChunkDiagnostic": (
        "backend.benchmarks.gold_recall.models",
        "RetrievalChunkDiagnostic",
    ),
    "StageARetrievalMetrics": (
        "backend.benchmarks.gold_recall.models",
        "StageARetrievalMetrics",
    ),
    "StageBExtractionMetrics": (
        "backend.benchmarks.gold_recall.models",
        "StageBExtractionMetrics",
    ),
    "StageCWriterMetrics": (
        "backend.benchmarks.gold_recall.models",
        "StageCWriterMetrics",
    ),
    "build_fact_judge_agent": (
        "backend.benchmarks.gold_recall.judge",
        "build_fact_judge_agent",
    ),
    "judge_fact_presence": (
        "backend.benchmarks.gold_recall.judge",
        "judge_fact_presence",
    ),
    "load_gold_benchmark_dataset": (
        "backend.benchmarks.gold_recall.runner",
        "load_gold_benchmark_dataset",
    ),
    "run_recall_benchmark": (
        "backend.benchmarks.gold_recall.runner",
        "run_recall_benchmark",
    ),
    "run_retrieval_strategy_benchmark": (
        "backend.benchmarks.runner",
        "run_retrieval_strategy_benchmark",
    ),
}


def __getattr__(name: str):
    """Resolve public benchmark exports on first access."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    return getattr(module, attribute_name)


__all__ = list(_EXPORTS)
