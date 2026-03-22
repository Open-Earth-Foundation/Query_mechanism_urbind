from __future__ import annotations

import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.benchmarks.gold_recall.judge import FACT_JUDGE_MODEL, judge_fact_presence
from backend.benchmarks.gold_recall.models import (
    FactPresenceJudgement,
    GoldBenchmarkCase,
    GoldBenchmarkDataset,
    LossWaterfall,
    RecallBenchmarkCaseResult,
    RecallBenchmarkReport,
    RecallBenchmarkSummary,
    RetrievalChunkDiagnostic,
    StageARetrievalMetrics,
    StageBExtractionMetrics,
    StageCWriterMetrics,
)
from backend.modules.orchestrator.module import run_pipeline
from backend.modules.writer.utils.markdown_helpers import extract_cited_ref_ids
from backend.utils.config import AppConfig, get_openrouter_api_key, load_config
from backend.utils.json_io import read_json_object, write_json

logger = logging.getLogger(__name__)


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a bounded ratio, or 0.0 when the denominator is empty."""
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean or 0.0 for an empty list."""
    if not values:
        return 0.0
    return statistics.fmean(values)


def load_gold_benchmark_dataset(path: Path) -> GoldBenchmarkDataset:
    """Load and validate the gold benchmark input file."""
    payload = read_json_object(path)
    if payload is None:
        raise FileNotFoundError(f"Gold benchmark JSON not found or invalid: {path}")
    return GoldBenchmarkDataset.model_validate(payload)


def _select_cases(
    dataset: GoldBenchmarkDataset,
    selected_case_ids: list[str],
) -> list[GoldBenchmarkCase]:
    """Filter the dataset down to the requested case ids when provided."""
    if not selected_case_ids:
        return list(dataset.cases)

    requested = {case_id.strip() for case_id in selected_case_ids if case_id.strip()}
    selected = [case for case in dataset.cases if case.case_id in requested]
    missing = sorted(requested - {case.case_id for case in selected})
    if missing:
        raise ValueError("Unknown --case-id values: " + ", ".join(missing))
    return selected


def _resolve_cached_run_dir(gold_file: Path, raw_run_dir: str) -> Path:
    """Resolve a cached run directory relative to the gold file when needed."""
    candidate = Path(raw_run_dir)
    if candidate.is_absolute():
        return candidate
    return (gold_file.parent / candidate).resolve()


def _require_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object or raise a validation error."""
    payload = read_json_object(path)
    if payload is None:
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def _require_text(path: Path) -> str:
    """Read one UTF-8 text file or raise when it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return path.read_text(encoding="utf-8")


def _extract_cached_question(run_dir: Path) -> str:
    """Read the original question stored in a cached run directory."""
    research_question_path = run_dir / "research_question.json"
    if research_question_path.exists():
        payload = _require_json_object(research_question_path)
        original_question = str(payload.get("original_question", "")).strip()
        if original_question:
            return original_question

    run_log_path = run_dir / "run.json"
    if run_log_path.exists():
        payload = _require_json_object(run_log_path)
        inputs = payload.get("inputs")
        if isinstance(inputs, dict):
            original_question = str(inputs.get("original_question", "")).strip()
            if original_question:
                return original_question

    raise ValueError(
        f"Could not validate cached question for run directory: {run_dir}"
    )


def _validate_cached_run(case: GoldBenchmarkCase, run_dir: Path) -> None:
    """Validate cached-run artifacts and question matching before scoring."""
    if not run_dir.exists():
        raise FileNotFoundError(f"Cached run directory not found: {run_dir}")

    cached_question = _extract_cached_question(run_dir)
    if cached_question != case.question:
        raise ValueError(
            f"Cached run question mismatch for case_id={case.case_id}: "
            f"{cached_question!r} != {case.question!r}"
        )

    required_paths = [
        run_dir / "markdown" / "retrieval.json",
        run_dir / "markdown" / "excerpts.json",
        run_dir / "markdown" / "references.json",
        run_dir / "final.md",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required cached artifact not found: {path}")


def _validate_retrieval_payload(payload: dict[str, Any], case_id: str) -> None:
    """Validate that retrieval.json contains the strict seed-layer artifact data."""
    seed_chunks = payload.get("seed_chunks")
    chunks = payload.get("chunks")
    if not isinstance(seed_chunks, list):
        raise ValueError(
            f"retrieval.json for case_id={case_id} is missing seed_chunks[]."
        )
    if not isinstance(chunks, list):
        raise ValueError(f"retrieval.json for case_id={case_id} is missing chunks[].")

    for seed_chunk in seed_chunks:
        if not isinstance(seed_chunk, dict):
            raise ValueError(f"Invalid seed chunk payload in case_id={case_id}.")
        provenance = seed_chunk.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(
                f"Seed chunk provenance missing in retrieval.json for case_id={case_id}."
            )
        if provenance.get("origin") != "seed":
            raise ValueError(
                f"Seed chunk origin must be 'seed' for case_id={case_id}."
            )
        if not isinstance(provenance.get("seed_rank"), int):
            raise ValueError(
                f"Seed chunk is missing seed_rank for case_id={case_id}."
            )


def _build_chunk_index(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index persisted retrieval chunk objects by chunk id."""
    index: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        index[chunk_id] = chunk
    return index


def _build_reference_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index persisted reference entries by ref id."""
    references = payload.get("references", [])
    index: dict[str, dict[str, Any]] = {}
    if not isinstance(references, list):
        return index
    for reference in references:
        if not isinstance(reference, dict):
            continue
        ref_id = str(reference.get("ref_id", "")).strip()
        if not ref_id:
            continue
        index[ref_id] = reference
    return index


def _collect_excerpt_source_ids(excerpts_payload: dict[str, Any]) -> set[str]:
    """Collect the union of source chunk ids cited by markdown excerpts."""
    excerpts = excerpts_payload.get("excerpts", [])
    source_chunk_ids: set[str] = set()
    if not isinstance(excerpts, list):
        return source_chunk_ids
    for excerpt in excerpts:
        if not isinstance(excerpt, dict):
            continue
        raw_ids = excerpt.get("source_chunk_ids", [])
        if not isinstance(raw_ids, list):
            continue
        for raw_chunk_id in raw_ids:
            if not isinstance(raw_chunk_id, str):
                continue
            chunk_id = raw_chunk_id.strip()
            if chunk_id:
                source_chunk_ids.add(chunk_id)
    return source_chunk_ids


def _build_stage_b_candidate_text(excerpts_payload: dict[str, Any]) -> str:
    """Build the text judged for markdown-stage fact extraction."""
    excerpts = excerpts_payload.get("excerpts", [])
    if not isinstance(excerpts, list):
        return ""

    parts: list[str] = []
    for excerpt in excerpts:
        if not isinstance(excerpt, dict):
            continue
        partial_answer = str(excerpt.get("partial_answer", "")).strip()
        quote = str(excerpt.get("quote", "")).strip()
        if partial_answer:
            parts.append(partial_answer)
        if quote:
            parts.append(quote)
    return "\n\n".join(parts)


def _judge_stage_facts(
    *,
    stage: str,
    question: str,
    gold_facts: list[str],
    candidate_text: str,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool,
    judge_func,
) -> list[FactPresenceJudgement]:
    """Run the fact judge once per gold fact for the requested stage."""
    judgements: list[FactPresenceJudgement] = []
    for fact in gold_facts:
        decision = judge_func(
            question=question,
            stage_label=stage,
            fact=fact,
            candidate_text=candidate_text,
            config=config,
            api_key=api_key,
            log_llm_payload=log_llm_payload,
        )
        judgements.append(
            FactPresenceJudgement(
                stage=stage,
                fact=fact,
                verdict=decision.verdict,
                rationale=decision.rationale,
            )
        )
    return judgements


def _build_chunk_diagnostics(
    *,
    gold_chunk_ids: list[str],
    seed_index: dict[str, dict[str, Any]],
    delivery_index: dict[str, dict[str, Any]],
) -> list[RetrievalChunkDiagnostic]:
    """Classify every gold chunk as seed-hit, neighbor-only, fallback, or miss."""
    diagnostics: list[RetrievalChunkDiagnostic] = []
    for chunk_id in gold_chunk_ids:
        seed_chunk = seed_index.get(chunk_id)
        delivery_chunk = delivery_index.get(chunk_id)
        if seed_chunk is not None:
            provenance = seed_chunk.get("provenance", {})
            selection_mode = (
                str(provenance.get("selection_mode", "")).strip() or None
            )
            bucket = (
                "fallback_top_up_hit"
                if selection_mode == "fallback_top_up"
                else "seed_hit"
            )
            seed_rank = provenance.get("seed_rank")
            diagnostics.append(
                RetrievalChunkDiagnostic(
                    chunk_id=chunk_id,
                    bucket=bucket,
                    seed_rank=seed_rank if isinstance(seed_rank, int) else None,
                    selection_mode=selection_mode,
                )
            )
            continue
        if delivery_chunk is not None:
            provenance = delivery_chunk.get("provenance", {})
            selection_mode = (
                str(provenance.get("selection_mode", "")).strip() or None
            )
            diagnostics.append(
                RetrievalChunkDiagnostic(
                    chunk_id=chunk_id,
                    bucket="neighbor_only_hit",
                    selection_mode=selection_mode,
                )
            )
            continue
        diagnostics.append(RetrievalChunkDiagnostic(chunk_id=chunk_id, bucket="miss"))
    return diagnostics


def _build_case_result(
    *,
    case: GoldBenchmarkCase,
    run_dir: Path,
    retrieval_payload: dict[str, Any],
    excerpts_payload: dict[str, Any],
    references_payload: dict[str, Any],
    final_text: str,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool,
    judge_func,
    used_cached_run: bool,
) -> RecallBenchmarkCaseResult:
    """Compute all recall/precision metrics for one benchmark case."""
    _validate_retrieval_payload(retrieval_payload, case.case_id)

    seed_index = _build_chunk_index(list(retrieval_payload.get("seed_chunks", [])))
    delivery_index = _build_chunk_index(list(retrieval_payload.get("chunks", [])))
    gold_chunk_ids = list(case.gold_chunk_ids)
    gold_chunk_set = set(gold_chunk_ids)

    seed_hit_ids = gold_chunk_set & set(seed_index.keys())
    delivery_hit_ids = gold_chunk_set & set(delivery_index.keys())
    seed_precision_hits = seed_hit_ids
    delivery_precision_hits = gold_chunk_set & set(delivery_index.keys())

    matching_seed_ranks: list[int] = []
    for chunk_id in seed_hit_ids:
        provenance = seed_index[chunk_id].get("provenance", {})
        seed_rank = provenance.get("seed_rank")
        if isinstance(seed_rank, int):
            matching_seed_ranks.append(seed_rank)

    chunk_diagnostics = _build_chunk_diagnostics(
        gold_chunk_ids=gold_chunk_ids,
        seed_index=seed_index,
        delivery_index=delivery_index,
    )
    stage_a = StageARetrievalMetrics(
        retrieval_recall=_safe_ratio(len(seed_hit_ids), len(gold_chunk_ids)),
        retrieval_precision=_safe_ratio(
            len(seed_precision_hits),
            len(seed_index),
        ),
        mrr=(1.0 / float(min(matching_seed_ranks))) if matching_seed_ranks else 0.0,
        delivery_recall=_safe_ratio(len(delivery_hit_ids), len(gold_chunk_ids)),
        delivery_precision=_safe_ratio(
            len(delivery_precision_hits),
            len(delivery_index),
        ),
        seed_hit_count=sum(
            1 for diagnostic in chunk_diagnostics if diagnostic.bucket == "seed_hit"
        ),
        neighbor_only_hit_count=sum(
            1
            for diagnostic in chunk_diagnostics
            if diagnostic.bucket == "neighbor_only_hit"
        ),
        fallback_top_up_hit_count=sum(
            1
            for diagnostic in chunk_diagnostics
            if diagnostic.bucket == "fallback_top_up_hit"
        ),
        miss_count=sum(
            1 for diagnostic in chunk_diagnostics if diagnostic.bucket == "miss"
        ),
    )

    excerpt_source_ids = _collect_excerpt_source_ids(excerpts_payload)
    stage_b_text = _build_stage_b_candidate_text(excerpts_payload)
    stage_b_judgements = _judge_stage_facts(
        stage="stage_b",
        question=case.question,
        gold_facts=list(case.gold_facts),
        candidate_text=stage_b_text,
        config=config,
        api_key=api_key,
        log_llm_payload=log_llm_payload,
        judge_func=judge_func,
    )
    stage_b_fact_hit_count = sum(
        1 for judgement in stage_b_judgements if judgement.is_hit
    )
    stage_b = StageBExtractionMetrics(
        extraction_recall=_safe_ratio(
            len(excerpt_source_ids & gold_chunk_set),
            len(gold_chunk_ids),
        ),
        fact_extraction_rate=_safe_ratio(
            stage_b_fact_hit_count,
            len(case.gold_facts),
        ),
    )

    stage_c_judgements = _judge_stage_facts(
        stage="stage_c",
        question=case.question,
        gold_facts=list(case.gold_facts),
        candidate_text=final_text,
        config=config,
        api_key=api_key,
        log_llm_payload=log_llm_payload,
        judge_func=judge_func,
    )
    stage_c_fact_hit_count = sum(
        1 for judgement in stage_c_judgements if judgement.is_hit
    )
    reference_index = _build_reference_index(references_payload)
    cited_ref_ids = extract_cited_ref_ids(final_text)
    cited_source_chunk_ids: set[str] = set()
    for ref_id in cited_ref_ids:
        reference = reference_index.get(ref_id)
        if reference is None:
            continue
        raw_chunk_ids = reference.get("source_chunk_ids", [])
        if not isinstance(raw_chunk_ids, list):
            continue
        for raw_chunk_id in raw_chunk_ids:
            if not isinstance(raw_chunk_id, str):
                continue
            chunk_id = raw_chunk_id.strip()
            if chunk_id:
                cited_source_chunk_ids.add(chunk_id)
    stage_c = StageCWriterMetrics(
        end_to_end_fact_recall=_safe_ratio(
            stage_c_fact_hit_count,
            len(case.gold_facts),
        ),
        citation_coverage=_safe_ratio(
            len(cited_source_chunk_ids & gold_chunk_set),
            len(gold_chunk_ids),
        ),
    )

    return RecallBenchmarkCaseResult(
        case_id=case.case_id,
        question=case.question,
        gold_city=list(case.gold_city),
        selected_cities=case.resolved_selected_cities(),
        used_cached_run=used_cached_run,
        run_dir=str(run_dir),
        retrieval_path=str(run_dir / "markdown" / "retrieval.json"),
        excerpts_path=str(run_dir / "markdown" / "excerpts.json"),
        references_path=str(run_dir / "markdown" / "references.json"),
        final_output_path=str(run_dir / "final.md"),
        stage_a=stage_a,
        stage_b=stage_b,
        stage_c=stage_c,
        loss_waterfall=LossWaterfall(
            gold_chunk_count=len(gold_chunk_ids),
            seed_hit_chunk_count=len(seed_hit_ids),
            delivery_hit_chunk_count=len(delivery_hit_ids),
            stage_b_fact_hit_count=stage_b_fact_hit_count,
            stage_c_fact_hit_count=stage_c_fact_hit_count,
        ),
        chunk_diagnostics=chunk_diagnostics,
        stage_b_judgements=stage_b_judgements,
        stage_c_judgements=stage_c_judgements,
    )


def _build_summary(results: list[RecallBenchmarkCaseResult]) -> RecallBenchmarkSummary:
    """Build the aggregate metric summary for the full report."""
    return RecallBenchmarkSummary(
        case_count=len(results),
        retrieval_recall_mean=_mean(
            [result.stage_a.retrieval_recall for result in results]
        ),
        retrieval_precision_mean=_mean(
            [result.stage_a.retrieval_precision for result in results]
        ),
        mrr_mean=_mean([result.stage_a.mrr for result in results]),
        delivery_recall_mean=_mean(
            [result.stage_a.delivery_recall for result in results]
        ),
        delivery_precision_mean=_mean(
            [result.stage_a.delivery_precision for result in results]
        ),
        extraction_recall_mean=_mean(
            [result.stage_b.extraction_recall for result in results]
        ),
        fact_extraction_rate_mean=_mean(
            [result.stage_b.fact_extraction_rate for result in results]
        ),
        end_to_end_fact_recall_mean=_mean(
            [result.stage_c.end_to_end_fact_recall for result in results]
        ),
        citation_coverage_mean=_mean(
            [result.stage_c.citation_coverage for result in results]
        ),
    )


def _render_report_markdown(report: RecallBenchmarkReport) -> str:
    """Render a concise human-readable benchmark report."""
    lines = [
        "# Recall Benchmark",
        "",
        f"- Benchmark ID: {report.benchmark_id}",
        f"- Gold file: {report.gold_file}",
        f"- Judge model: {report.judge_model}",
        f"- Cases: {report.summary.case_count}",
        "",
        "## Aggregate Summary",
        "",
        "| Metric | Mean |",
        "| --- | ---: |",
        f"| Retrieval recall | {report.summary.retrieval_recall_mean:.3f} |",
        f"| Retrieval precision | {report.summary.retrieval_precision_mean:.3f} |",
        f"| MRR | {report.summary.mrr_mean:.3f} |",
        f"| Delivery recall | {report.summary.delivery_recall_mean:.3f} |",
        f"| Delivery precision | {report.summary.delivery_precision_mean:.3f} |",
        f"| Extraction recall | {report.summary.extraction_recall_mean:.3f} |",
        f"| Fact extraction rate | {report.summary.fact_extraction_rate_mean:.3f} |",
        f"| End-to-end fact recall | {report.summary.end_to_end_fact_recall_mean:.3f} |",
        f"| Citation coverage | {report.summary.citation_coverage_mean:.3f} |",
        "",
        "## Per Case",
        "",
    ]

    for result in report.results:
        lines.extend(
            [
                f"### {result.case_id}",
                "",
                f"- Question: {result.question}",
                f"- Run dir: {result.run_dir}",
                f"- Cached run: {'yes' if result.used_cached_run else 'no'}",
                (
                    "- Waterfall: "
                    f"gold_chunks={result.loss_waterfall.gold_chunk_count}, "
                    f"seed_hits={result.loss_waterfall.seed_hit_chunk_count}, "
                    f"delivery_hits={result.loss_waterfall.delivery_hit_chunk_count}, "
                    f"stage_b_fact_hits={result.loss_waterfall.stage_b_fact_hit_count}, "
                    f"stage_c_fact_hits={result.loss_waterfall.stage_c_fact_hit_count}"
                ),
                (
                    "- Metrics: "
                    f"retrieval_recall={result.stage_a.retrieval_recall:.3f}, "
                    f"retrieval_precision={result.stage_a.retrieval_precision:.3f}, "
                    f"mrr={result.stage_a.mrr:.3f}, "
                    f"delivery_recall={result.stage_a.delivery_recall:.3f}, "
                    f"extraction_recall={result.stage_b.extraction_recall:.3f}, "
                    f"fact_recall={result.stage_c.end_to_end_fact_recall:.3f}, "
                    f"citation_coverage={result.stage_c.citation_coverage:.3f}"
                ),
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def run_recall_benchmark(
    *,
    benchmark_id: str,
    gold_file: Path,
    output_dir: Path,
    config_path: Path,
    selected_case_ids: list[str] | None = None,
    run_live: bool = False,
    log_llm_payload: bool = False,
    api_key_override: str | None = None,
    judge_func=judge_fact_presence,
    run_pipeline_func=run_pipeline,
) -> RecallBenchmarkReport:
    """Execute the rigorous recall benchmark against the gold dataset."""
    dataset = load_gold_benchmark_dataset(gold_file)
    cases = _select_cases(dataset, selected_case_ids or [])
    benchmark_root = output_dir / benchmark_id
    benchmark_root.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    api_key = (
        api_key_override.strip()
        if isinstance(api_key_override, str) and api_key_override.strip()
        else get_openrouter_api_key()
    )

    results: list[RecallBenchmarkCaseResult] = []
    for case in cases:
        logger.info("Recall benchmark case_id=%s", case.case_id)
        if case.cached_run_dir and not run_live:
            run_dir = _resolve_cached_run_dir(gold_file, case.cached_run_dir)
            _validate_cached_run(case, run_dir)
            used_cached_run = True
        else:
            live_config = config.model_copy(deep=True)
            live_config.runs_dir = benchmark_root / "runs"
            run_paths = run_pipeline_func(
                question=case.question,
                config=live_config,
                run_id=case.case_id,
                log_llm_payload=log_llm_payload,
                selected_cities=case.resolved_selected_cities(),
            )
            run_dir = run_paths.base_dir
            used_cached_run = False

        retrieval_payload = _require_json_object(run_dir / "markdown" / "retrieval.json")
        excerpts_payload = _require_json_object(run_dir / "markdown" / "excerpts.json")
        references_payload = _require_json_object(
            run_dir / "markdown" / "references.json"
        )
        final_text = _require_text(run_dir / "final.md")
        results.append(
            _build_case_result(
                case=case,
                run_dir=run_dir,
                retrieval_payload=retrieval_payload,
                excerpts_payload=excerpts_payload,
                references_payload=references_payload,
                final_text=final_text,
                config=config,
                api_key=api_key,
                log_llm_payload=log_llm_payload,
                judge_func=judge_func,
                used_cached_run=used_cached_run,
            )
        )

    report = RecallBenchmarkReport(
        benchmark_id=benchmark_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        output_dir=str(benchmark_root),
        gold_file=str(gold_file),
        judge_model=FACT_JUDGE_MODEL,
        results=results,
        summary=_build_summary(results),
    )
    write_json(benchmark_root / "benchmark_report.json", report.model_dump())
    (benchmark_root / "benchmark_report.md").write_text(
        _render_report_markdown(report),
        encoding="utf-8",
    )
    return report


__all__ = [
    "load_gold_benchmark_dataset",
    "run_recall_benchmark",
]
