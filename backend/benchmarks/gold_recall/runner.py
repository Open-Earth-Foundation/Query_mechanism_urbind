from __future__ import annotations

import logging
import statistics
import unicodedata
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.api.services.source_chunks import load_source_chunks
from backend.benchmarks.gold_recall.judge import judge_fact_presence
from backend.benchmarks.gold_recall.models import (
    FactJudgeDecision,
    FactPresenceJudgement,
    GoldBenchmarkCase,
    GoldChunkAlternative,
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
from backend.modules.writer.utils.markdown_helpers import extract_cited_ref_ids
from backend.utils.config import AppConfig, get_openrouter_api_key, load_config
from backend.utils.json_io import read_json_object, write_json
from backend.utils.paths import RunPaths

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


def _normalize_chunk_text(value: str) -> str:
    """Normalize chunk text for canonical fallback matching."""
    normalized = unicodedata.normalize("NFKC", value)
    collapsed = " ".join(normalized.split())
    return collapsed.casefold()


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


def _load_chunk_text_map(
    *,
    run_dir: Path,
    config: AppConfig,
    chunk_ids: set[str],
) -> dict[str, str]:
    """Resolve normalized text for the requested chunk ids."""
    normalized_ids = {chunk_id.strip() for chunk_id in chunk_ids if chunk_id.strip()}
    if not normalized_ids:
        return {}

    try:
        chunks = load_source_chunks(
            run_dir=run_dir,
            markdown_dir=config.markdown_dir,
            config=config,
            chunk_ids=sorted(normalized_ids),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to resolve chunk text for benchmark scoring: %s", exc)
        return {}

    text_map: dict[str, str] = {}
    for chunk in chunks:
        if not chunk.content.strip():
            continue
        text_map[chunk.chunk_id] = _normalize_chunk_text(chunk.content)
    return text_map


def _build_text_match_index(text_map: dict[str, str]) -> dict[str, list[str]]:
    """Index chunk ids by normalized chunk text."""
    index: dict[str, list[str]] = {}
    for chunk_id, normalized_text in text_map.items():
        if not normalized_text:
            continue
        index.setdefault(normalized_text, []).append(chunk_id)
    return index


def _find_containing_text_match(
    normalized_text: str,
    target_text_map: dict[str, str],
) -> str | None:
    """Return the first chunk whose normalized text contains the canonical text."""
    if not normalized_text:
        return None
    for chunk_id, target_text in target_text_map.items():
        if normalized_text in target_text:
            return chunk_id
    return None


def _find_direct_chunk_match(
    candidate_chunk_ids: list[str],
    target_index: dict[str, dict[str, Any]],
) -> str | None:
    """Return the first accepted chunk id that exists in the target index."""
    for candidate_chunk_id in candidate_chunk_ids:
        if candidate_chunk_id in target_index:
            return candidate_chunk_id
    return None


def _collect_match_texts(
    *,
    gold_chunk_text: str | None,
    gold_chunk_alternatives: list[GoldChunkAlternative],
) -> list[str]:
    """Return de-duplicated canonical texts that may satisfy one gold slot."""
    candidate_texts: list[str] = []
    seen: set[str] = set()

    for candidate_text in [
        gold_chunk_text,
        *[alternative.chunk_text for alternative in gold_chunk_alternatives],
    ]:
        if not candidate_text:
            continue
        normalized_text = _normalize_chunk_text(candidate_text)
        if normalized_text in seen:
            continue
        seen.add(normalized_text)
        candidate_texts.append(candidate_text)
    return candidate_texts


def _match_gold_chunks(
    *,
    gold_chunk_ids: list[str],
    gold_chunk_alternatives: list[list[GoldChunkAlternative]],
    gold_chunk_texts: list[str] | None,
    target_index: dict[str, dict[str, Any]],
    target_text_index: dict[str, list[str]],
    target_text_map: dict[str, str],
) -> dict[int, tuple[str, str]]:
    """Match gold chunks to retrieval chunks by id or fallback text containment."""
    matches: dict[int, tuple[str, str]] = {}
    for idx, gold_chunk_id in enumerate(gold_chunk_ids):
        direct_match = _find_direct_chunk_match(
            [gold_chunk_id, *[alt.chunk_id for alt in gold_chunk_alternatives[idx]]],
            target_index,
        )
        if direct_match is not None:
            matches[idx] = (direct_match, "direct_id")
            continue
        canonical_text = gold_chunk_texts[idx] if gold_chunk_texts is not None else None
        for candidate_text in _collect_match_texts(
            gold_chunk_text=canonical_text,
            gold_chunk_alternatives=gold_chunk_alternatives[idx],
        ):
            normalized_text = _normalize_chunk_text(candidate_text)
            candidate_ids = target_text_index.get(normalized_text, [])
            if candidate_ids:
                matches[idx] = (candidate_ids[0], "text_fallback")
                break
            containing_match = _find_containing_text_match(
                normalized_text,
                target_text_map,
            )
            if containing_match is not None:
                matches[idx] = (containing_match, "text_fallback")
                break
    return matches


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
    judge_func: Callable[..., FactJudgeDecision],
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
    gold_chunk_alternatives: list[list[GoldChunkAlternative]],
    gold_chunk_texts: list[str] | None,
    seed_index: dict[str, dict[str, Any]],
    delivery_index: dict[str, dict[str, Any]],
    seed_text_index: dict[str, list[str]],
    delivery_text_index: dict[str, list[str]],
    seed_text_map: dict[str, str],
    delivery_text_map: dict[str, str],
) -> list[RetrievalChunkDiagnostic]:
    """Classify every gold chunk as seed-hit, neighbor-only, fallback, or miss."""
    diagnostics: list[RetrievalChunkDiagnostic] = []
    seed_matches = _match_gold_chunks(
        gold_chunk_ids=gold_chunk_ids,
        gold_chunk_alternatives=gold_chunk_alternatives,
        gold_chunk_texts=gold_chunk_texts,
        target_index=seed_index,
        target_text_index=seed_text_index,
        target_text_map=seed_text_map,
    )
    delivery_matches = _match_gold_chunks(
        gold_chunk_ids=gold_chunk_ids,
        gold_chunk_alternatives=gold_chunk_alternatives,
        gold_chunk_texts=gold_chunk_texts,
        target_index=delivery_index,
        target_text_index=delivery_text_index,
        target_text_map=delivery_text_map,
    )

    for idx, chunk_id in enumerate(gold_chunk_ids):
        matched_seed = seed_matches.get(idx)
        matched_delivery = delivery_matches.get(idx)
        matched_seed_id = matched_seed[0] if matched_seed else None
        matched_delivery_id = matched_delivery[0] if matched_delivery else None
        seed_chunk = seed_index.get(matched_seed_id) if matched_seed_id else None
        delivery_chunk = (
            delivery_index.get(matched_delivery_id) if matched_delivery_id else None
        )
        if seed_chunk is not None:
            provenance = seed_chunk.get("provenance", {})
            raw_selection_mode = (
                str(provenance.get("selection_mode", "")).strip() or None
            )
            bucket = (
                "fallback_top_up_hit"
                if raw_selection_mode == "fallback_top_up"
                else "seed_hit"
            )
            selection_mode = raw_selection_mode
            if matched_seed is not None and matched_seed[1] == "text_fallback":
                selection_mode = (
                    f"text_fallback:{raw_selection_mode}"
                    if raw_selection_mode
                    else "text_fallback"
                )
            seed_rank = provenance.get("seed_rank")
            diagnostics.append(
                RetrievalChunkDiagnostic(
                    chunk_id=chunk_id,
                    bucket=bucket,
                    matched_chunk_id=matched_seed_id,
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
            if matched_delivery is not None and matched_delivery[1] == "text_fallback":
                selection_mode = "text_fallback"
            diagnostics.append(
                RetrievalChunkDiagnostic(
                    chunk_id=chunk_id,
                    bucket="neighbor_only_hit",
                    matched_chunk_id=matched_delivery_id,
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
    judge_func: Callable[..., FactJudgeDecision],
) -> RecallBenchmarkCaseResult:
    """Compute all recall/precision metrics for one benchmark case."""
    _validate_retrieval_payload(retrieval_payload, case.case_id)

    seed_index = _build_chunk_index(list(retrieval_payload.get("seed_chunks", [])))
    delivery_index = _build_chunk_index(list(retrieval_payload.get("chunks", [])))
    gold_chunk_ids = list(case.gold_chunk_ids)
    gold_chunk_alternatives = case.resolved_gold_chunk_alternatives()
    gold_chunk_texts = list(case.gold_chunk_texts) if case.gold_chunk_texts else None

    delivery_chunk_ids_for_text = set(delivery_index.keys())
    excerpt_source_ids = _collect_excerpt_source_ids(excerpts_payload)
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

    chunk_text_map = _load_chunk_text_map(
        run_dir=run_dir,
        config=config,
        chunk_ids=set(seed_index.keys())
        | delivery_chunk_ids_for_text
        | excerpt_source_ids
        | cited_source_chunk_ids,
    )
    seed_text_map = {
        chunk_id: text for chunk_id, text in chunk_text_map.items() if chunk_id in seed_index
    }
    delivery_text_map = {
        chunk_id: text
        for chunk_id, text in chunk_text_map.items()
        if chunk_id in delivery_index
    }
    excerpt_text_map = {
        chunk_id: text
        for chunk_id, text in chunk_text_map.items()
        if chunk_id in excerpt_source_ids
    }
    cited_text_map = {
        chunk_id: text
        for chunk_id, text in chunk_text_map.items()
        if chunk_id in cited_source_chunk_ids
    }
    seed_text_index = _build_text_match_index(seed_text_map)
    delivery_text_index = _build_text_match_index(delivery_text_map)
    excerpt_text_index = _build_text_match_index(excerpt_text_map)
    cited_text_index = _build_text_match_index(cited_text_map)

    seed_matches = _match_gold_chunks(
        gold_chunk_ids=gold_chunk_ids,
        gold_chunk_alternatives=gold_chunk_alternatives,
        gold_chunk_texts=gold_chunk_texts,
        target_index=seed_index,
        target_text_index=seed_text_index,
        target_text_map=seed_text_map,
    )
    delivery_matches = _match_gold_chunks(
        gold_chunk_ids=gold_chunk_ids,
        gold_chunk_alternatives=gold_chunk_alternatives,
        gold_chunk_texts=gold_chunk_texts,
        target_index=delivery_index,
        target_text_index=delivery_text_index,
        target_text_map=delivery_text_map,
    )
    excerpt_matches = _match_gold_chunks(
        gold_chunk_ids=gold_chunk_ids,
        gold_chunk_alternatives=gold_chunk_alternatives,
        gold_chunk_texts=gold_chunk_texts,
        target_index={chunk_id: {} for chunk_id in excerpt_source_ids},
        target_text_index=excerpt_text_index,
        target_text_map=excerpt_text_map,
    )
    cited_matches = _match_gold_chunks(
        gold_chunk_ids=gold_chunk_ids,
        gold_chunk_alternatives=gold_chunk_alternatives,
        gold_chunk_texts=gold_chunk_texts,
        target_index={chunk_id: {} for chunk_id in cited_source_chunk_ids},
        target_text_index=cited_text_index,
        target_text_map=cited_text_map,
    )

    matching_seed_ranks: list[int] = []
    for chunk_id, _match_strategy in seed_matches.values():
        provenance = seed_index[chunk_id].get("provenance", {})
        seed_rank = provenance.get("seed_rank")
        if isinstance(seed_rank, int):
            matching_seed_ranks.append(seed_rank)

    chunk_diagnostics = _build_chunk_diagnostics(
        gold_chunk_ids=gold_chunk_ids,
        gold_chunk_alternatives=gold_chunk_alternatives,
        gold_chunk_texts=gold_chunk_texts,
        seed_index=seed_index,
        delivery_index=delivery_index,
        seed_text_index=seed_text_index,
        delivery_text_index=delivery_text_index,
        seed_text_map=seed_text_map,
        delivery_text_map=delivery_text_map,
    )
    stage_a = StageARetrievalMetrics(
        retrieval_recall=_safe_ratio(len(seed_matches), len(gold_chunk_ids)),
        retrieval_precision=_safe_ratio(
            len(seed_matches),
            len(seed_index),
        ),
        mrr=(1.0 / float(min(matching_seed_ranks))) if matching_seed_ranks else 0.0,
        delivery_recall=_safe_ratio(len(delivery_matches), len(gold_chunk_ids)),
        delivery_precision=_safe_ratio(
            len(delivery_matches),
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
            len(excerpt_matches),
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
    stage_c = StageCWriterMetrics(
        end_to_end_fact_recall=_safe_ratio(
            stage_c_fact_hit_count,
            len(case.gold_facts),
        ),
        citation_coverage=_safe_ratio(
            len(cited_matches),
            len(gold_chunk_ids),
        ),
    )

    return RecallBenchmarkCaseResult(
        case_id=case.case_id,
        question=case.question,
        gold_city=list(case.gold_city),
        selected_cities=case.resolved_selected_cities(),
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
            seed_hit_chunk_count=len(seed_matches),
            delivery_hit_chunk_count=len(delivery_matches),
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
    log_llm_payload: bool = False,
    api_key_override: str | None = None,
    judge_func: Callable[..., FactJudgeDecision] = judge_fact_presence,
    run_pipeline_func: Callable[..., RunPaths] | None = None,
) -> RecallBenchmarkReport:
    """Execute the rigorous recall benchmark against the gold dataset."""
    dataset = load_gold_benchmark_dataset(gold_file)
    cases = _select_cases(dataset, selected_case_ids or [])
    benchmark_root = output_dir / benchmark_id
    benchmark_root.mkdir(parents=True, exist_ok=True)

    if run_pipeline_func is None:
        from backend.modules.orchestrator.module import run_pipeline

        run_pipeline_func = run_pipeline

    config = load_config(config_path)
    api_key = (
        api_key_override.strip()
        if isinstance(api_key_override, str) and api_key_override.strip()
        else get_openrouter_api_key()
    )

    results: list[RecallBenchmarkCaseResult] = []
    for case in cases:
        logger.info("Recall benchmark case_id=%s", case.case_id)
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
            )
        )

    report = RecallBenchmarkReport(
        benchmark_id=benchmark_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        output_dir=str(benchmark_root),
        gold_file=str(gold_file),
        judge_model=config.benchmark_fact_judge.model,
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
