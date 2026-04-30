"""Facade and orchestration entrypoint for artifact-first initiative extraction."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from backend.modules.initiative_extractor.artifacts import _write_run_artifacts
from backend.modules.initiative_extractor.llm import (
    _get_thread_agent,
    _get_thread_semantic_dedupe_agent,
    build_initiative_extractor_agent,
    build_initiative_semantic_dedupe_agent,
    run_agent_sync,
)
from backend.modules.initiative_extractor.models import (
    InitiativeDocumentSegment,
    InitiativeExtractionCandidate,
    InitiativeRawSegmentResult,
    InitiativeExtractionRunResult,
)
from backend.modules.initiative_extractor.output_parser import (
    _coerce_segment_output_payload,
)
from backend.modules.initiative_extractor.records import (
    _apply_semantic_dedupe_groups,
    _build_candidate_records,
    _extend_prior_initiatives,
    _normalize_candidate,
)
from backend.modules.initiative_extractor.review import _build_review_items
from backend.modules.initiative_extractor.segmentation import build_document_segments
from backend.modules.initiative_extractor.segment_runner import (
    _process_segment,
    _run_segment_once,
    _run_segment_with_retries,
    _segment_payload,
)
from backend.modules.initiative_extractor.semantic_dedupe import (
    _run_semantic_dedupe_batch,
    _semantic_dedupe_payload,
    _semantic_dedupe_records,
)
from backend.utils.city_normalization import normalize_city_key
from backend.utils.config import AppConfig
from backend.utils.markdown_files import list_markdown_files

logger = logging.getLogger(__name__)


def _discover_markdown_files(
    markdown_path: Path,
    selected_cities: list[str] | None,
    config: AppConfig,
) -> list[Path]:
    """Discover selected markdown files without scanning directory subfolders."""
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown path not found: {markdown_path}")
    files = (
        [markdown_path]
        if markdown_path.is_file()
        else list_markdown_files(markdown_path)
    )
    if selected_cities:
        requested = {normalize_city_key(city) for city in selected_cities}
        files = [path for path in files if normalize_city_key(path.stem) in requested]
    selected_files: list[Path] = []
    for path in files:
        if path.stat().st_size <= config.initiative_extractor.max_file_bytes:
            selected_files.append(path)
        else:
            logger.warning(
                "Skipping oversized markdown file during initiative extraction: %s",
                path,
            )
    if len(selected_files) > config.initiative_extractor.max_files:
        logger.warning(
            "Limiting initiative extraction to the first %d markdown files.",
            config.initiative_extractor.max_files,
        )
    return selected_files[: config.initiative_extractor.max_files]


def extract_initiatives(
    *,
    markdown_path: Path,
    config: AppConfig,
    api_key: str,
    output_root: Path,
    run_id: str | None = None,
    selected_cities: list[str] | None = None,
    max_workers: int | None = None,
    log_llm_payload: bool = False,
) -> InitiativeExtractionRunResult:
    """Run artifact-first initiative extraction over selected markdown documents."""
    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / resolved_run_id
    documents = _discover_markdown_files(markdown_path, selected_cities, config)

    segments: list[InitiativeDocumentSegment] = []
    for path in documents:
        segments.extend(build_document_segments(path, config.initiative_extractor))

    configured_workers = max_workers or config.initiative_extractor.max_workers
    prior_context_enabled = config.initiative_extractor.prior_initiatives_max_tokens > 0
    worker_count = (
        1
        if prior_context_enabled
        else min(max(configured_workers, 1), max(len(segments), 1))
    )
    raw_results: list[InitiativeRawSegmentResult] = []

    logger.info(
        "Starting initiative extraction run_id=%s documents=%d segments=%d workers=%d",
        resolved_run_id,
        len(documents),
        len(segments),
        worker_count,
    )
    if prior_context_enabled:
        prior_initiatives: list[InitiativeExtractionCandidate] = []
        for segment in segments:
            new_result = _process_segment(
                segment,
                config,
                api_key,
                log_llm_payload=log_llm_payload,
                run_id=resolved_run_id,
                prior_initiatives=prior_initiatives,
            )
            raw_results.append(new_result)
            _extend_prior_initiatives(prior_initiatives, [new_result])
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _process_segment,
                    segment,
                    config,
                    api_key,
                    log_llm_payload=log_llm_payload,
                    run_id=resolved_run_id,
                    prior_initiatives=[],
                )
                for segment in segments
            ]
            for future in as_completed(futures):
                raw_results.append(future.result())

    raw_results.sort(key=lambda result: result.segment_id)
    candidate_records = _build_candidate_records(raw_results)
    candidate_records.sort(key=lambda record: record.record_id)
    records, semantic_groups, semantic_reviews = _semantic_dedupe_records(
        candidate_records,
        config,
        api_key,
        log_llm_payload=log_llm_payload,
    )
    records.sort(key=lambda record: record.record_id)
    review_items = _build_review_items(
        segments=segments,
        raw_results=raw_results,
        records=records,
        duplicate_reviews=semantic_reviews,
        config=config,
    )

    _write_run_artifacts(
        run_dir=run_dir,
        run_id=resolved_run_id,
        documents=documents,
        segments=segments,
        raw_results=raw_results,
        candidate_records=candidate_records,
        records=records,
        semantic_groups=semantic_groups,
        review_items=review_items,
        config=config,
    )
    return InitiativeExtractionRunResult(
        run_id=resolved_run_id,
        output_dir=str(run_dir),
        documents_count=len(documents),
        segments_count=len(segments),
        raw_initiatives_count=sum(len(result.initiatives) for result in raw_results),
        deduped_initiatives_count=len(records),
        review_items_count=len(review_items),
    )


__all__ = [
    "_apply_semantic_dedupe_groups",
    "_build_candidate_records",
    "_build_review_items",
    "_coerce_segment_output_payload",
    "_get_thread_agent",
    "_get_thread_semantic_dedupe_agent",
    "_normalize_candidate",
    "_process_segment",
    "_run_segment_once",
    "_run_segment_with_retries",
    "_run_semantic_dedupe_batch",
    "_segment_payload",
    "_semantic_dedupe_payload",
    "_semantic_dedupe_records",
    "build_initiative_extractor_agent",
    "build_initiative_semantic_dedupe_agent",
    "extract_initiatives",
    "run_agent_sync",
]
