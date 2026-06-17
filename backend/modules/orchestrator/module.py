from __future__ import annotations

import inspect
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Literal

from backend.modules.markdown_researcher.agent import extract_markdown_excerpts
from backend.modules.markdown_researcher.models import MarkdownResearchResult
from backend.modules.markdown_researcher.services import (
    build_city_batches,
    load_markdown_documents,
    resolve_batch_input_token_limit,
    split_documents_by_city,
)
from backend.modules.orchestrator.utils import (
    attach_run_file_logger,
    build_markdown_metrics,
    build_markdown_references,
    build_retrieval_metrics,
    build_source_chunk_index,
    handle_write_decision,
    handle_task_error,
)
from backend.modules.orchestrator.utils.error_handlers import (
    detach_run_file_logger,
)
from backend.modules.web_researcher.module import run_enrichment_pipeline
from backend.modules.vector_store.indexer import update_markdown_index
from backend.modules.vector_store.retriever import (
    as_markdown_documents,
    build_retrieval_artifact,
    retrieve_chunks_for_queries,
)
from backend.modules.writer.agent import write_markdown
from backend.modules.writer.models import WriterOutput
from backend.services.progress_tracker import ProgressTracker
from backend.services.run_logger import RunLogger
from backend.utils.config import AppConfig, get_openrouter_api_key
from backend.utils.city_normalization import format_city_stem, normalize_city_key
from backend.utils.paths import RunPaths, build_run_id, create_run_paths
from backend.utils.run_snapshot import (
    build_code_snapshot,
    build_config_snapshot,
    build_documents_snapshot,
    build_execution_snapshot,
    build_vector_store_snapshot,
)
from backend.utils.tokenization import count_tokens

logger = logging.getLogger(__name__)


def _write_context_handoff(
    *,
    run_logger: RunLogger,
    stage_name: str,
    snapshot_filename: str,
    progress: ProgressTracker,
    progress_label: str,
) -> dict[str, str]:
    """Persist one immutable full-context handoff snapshot."""
    context_snapshot_path = run_logger.write_stage_file(
        stage_name,
        snapshot_filename,
        deepcopy(run_logger.context_bundle),
        alias=f"{stage_name}_context_snapshot",
    )
    outputs: dict[str, str] = {
        "context_bundle_snapshot": run_logger.artifact_label(context_snapshot_path),
    }
    progress.add_item(stage_name, progress_label)
    progress.complete_step(stage_name)
    return outputs


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Return unique non-empty values while preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _build_retrieval_queries(
    *queries: str | None,
    limit: int = 3,
) -> list[str]:
    """Normalize, de-duplicate, and cap retrieval queries while preserving order."""
    normalized: list[str] = []
    seen: set[str] = set()
    for query in queries:
        if query is None:
            continue
        candidate = query.strip()
        if not candidate:
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(candidate)
        if len(normalized) >= limit:
            break
    return normalized


def _collect_markdown_decision_artifacts(
    markdown_chunks: list[dict[str, object]],
    markdown_result: MarkdownResearchResult,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build rejected chunk decisions and a run-level decision audit."""

    batch_failures_payload = [
        failure.model_dump() for failure in markdown_result.batch_failures
    ]

    retrieved_ids = _dedupe_preserve_order(
        [str(document.get("chunk_id", "")).strip() for document in markdown_chunks]
    )
    retrieved_set = set(retrieved_ids)

    accepted_ids = _dedupe_preserve_order(markdown_result.accepted_chunk_ids)
    rejected_ids = _dedupe_preserve_order(markdown_result.rejected_chunk_ids)
    unresolved_ids = _dedupe_preserve_order(markdown_result.unresolved_chunk_ids)
    accepted_set = set(accepted_ids)
    rejected_set = set(rejected_ids)
    unresolved_set = set(unresolved_ids)

    overlap_decision_ids = {
        chunk_id
        for chunk_id in accepted_set
        if chunk_id in rejected_set or chunk_id in unresolved_set
    } | {chunk_id for chunk_id in rejected_set if chunk_id in unresolved_set}

    unknown_decision_ids = _dedupe_preserve_order(
        [
            chunk_id
            for chunk_id in accepted_ids + rejected_ids + unresolved_ids
            if chunk_id not in retrieved_set
        ]
    )
    unknown_decision_set = set(unknown_decision_ids)

    excerpt_source_ids: list[str] = []
    for excerpt in markdown_result.excerpts:
        excerpt_source_ids.extend(
            [
                source_chunk_id.strip()
                for source_chunk_id in excerpt.source_chunk_ids
                if source_chunk_id.strip()
            ]
        )
    unknown_excerpt_source_ids = _dedupe_preserve_order(
        [
            source_chunk_id
            for source_chunk_id in excerpt_source_ids
            if source_chunk_id not in accepted_set
        ]
    )

    decided_valid_ids = {
        chunk_id
        for chunk_id in accepted_set | rejected_set | unresolved_set
        if chunk_id in retrieved_set and chunk_id not in unknown_decision_set
    }
    missing_chunk_ids = [
        chunk_id for chunk_id in retrieved_ids if chunk_id not in decided_valid_ids
    ]

    city_by_chunk_id: dict[str, str] = {}
    for chunk in markdown_chunks:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        city_key = str(chunk.get("city_key", "")).strip()
        if not city_key:
            city_name = str(chunk.get("city_name", "")).strip()
            city_key = normalize_city_key(city_name) if city_name else ""
        city_by_chunk_id[chunk_id] = city_key or "unknown"

    rejected_by_city: dict[str, list[str]] = {}
    for chunk_id in rejected_ids:
        city_key = city_by_chunk_id.get(chunk_id, "unknown")
        rejected_by_city.setdefault(city_key, []).append(chunk_id)

    invariant_ok = not (
        overlap_decision_ids
        or unknown_decision_ids
        or missing_chunk_ids
        or unknown_excerpt_source_ids
    )
    artifact_status = (
        "complete"
        if invariant_ok and not markdown_result.batch_failures and not unresolved_ids
        else "partial"
    )

    rejected_artifact = {
        "status": artifact_status,
        "rejected_chunk_ids": rejected_ids,
        "rejected_by_city": rejected_by_city,
        "counts": {
            "rejected": len(rejected_ids),
        },
    }
    audit_artifact = {
        "status": artifact_status,
        "retrieved_total": len(retrieved_ids),
        "accepted_total": len(accepted_ids),
        "rejected_total": len(rejected_ids),
        "unresolved_total": len(unresolved_ids),
        "invariant_ok": invariant_ok,
        "missing_chunk_ids": missing_chunk_ids,
        "unknown_decision_ids": unknown_decision_ids,
        "unknown_excerpt_source_ids": unknown_excerpt_source_ids,
        "overlap_decision_ids": sorted(overlap_decision_ids),
        "batch_failures": batch_failures_payload,
    }
    return rejected_artifact, audit_artifact


def _write_input_snapshots(
    *,
    run_logger: RunLogger,
    config: AppConfig,
    config_path: Path,
    requested_run_id: str,
    resolved_run_id: str,
    invocation_command: str | None,
    selected_cities: list[str] | None,
) -> tuple[dict[str, object], dict[str, str]]:
    """Persist stage-001 reproducibility snapshots and the compact overview."""
    repo_root = Path(__file__).resolve().parents[3]
    execution_snapshot = build_execution_snapshot(
        argv=sys.argv,
        cwd=Path.cwd(),
        config_path=config_path,
        requested_run_id=requested_run_id,
        resolved_run_id=resolved_run_id,
        invocation_command=invocation_command,
    )
    code_snapshot = build_code_snapshot(repo_root)
    config_snapshot = build_config_snapshot(config, config_path)
    vector_store_snapshot = build_vector_store_snapshot(config)
    documents_snapshot = build_documents_snapshot(config.markdown_dir, selected_cities)

    execution_path = run_logger.write_stage_file(
        "input_snapshot",
        "execution_snapshot.json",
        execution_snapshot,
        alias="execution_snapshot",
    )
    code_path = run_logger.write_stage_file(
        "input_snapshot",
        "code_snapshot.json",
        code_snapshot,
        alias="code_snapshot",
    )
    config_snapshot_path = run_logger.write_stage_file(
        "input_snapshot",
        "config_snapshot.json",
        config_snapshot,
        alias="config_snapshot",
    )
    vector_store_path = run_logger.write_stage_file(
        "input_snapshot",
        "vector_store_snapshot.json",
        vector_store_snapshot,
        alias="vector_store_snapshot",
    )
    documents_path = run_logger.write_stage_file(
        "input_snapshot",
        "documents_snapshot.json",
        documents_snapshot,
        alias="documents_snapshot",
    )

    snapshot_summary: dict[str, object] = {
        "execution": {
            "invocation_command": execution_snapshot.get("invocation_command"),
            "config_path": execution_snapshot.get("config_path"),
            "requested_run_id": requested_run_id,
            "resolved_run_id": resolved_run_id,
        },
        "code": {
            "git_commit": code_snapshot.get("git_commit"),
            "git_branch": code_snapshot.get("git_branch"),
            "git_dirty": code_snapshot.get("git_dirty"),
            "changed_file_count": len(code_snapshot.get("changed_files", [])),
        },
        "config": {
            "config_file_hash": config_snapshot.get("config_file_hash"),
            "snapshot_hash": config_snapshot.get("snapshot_hash"),
        },
        "vector_store": {
            "enabled": vector_store_snapshot.get("enabled"),
            "collection_name": vector_store_snapshot.get("collection_name"),
            "index_manifest_hash": vector_store_snapshot.get("index_manifest_hash"),
            "manifest_summary": vector_store_snapshot.get("manifest_summary"),
        },
        "documents": {
            "markdown_dir": documents_snapshot.get("markdown_dir"),
            "file_count": documents_snapshot.get("file_count"),
            "summary": documents_snapshot.get("summary"),
            "snapshot_hash": documents_snapshot.get("snapshot_hash"),
        },
    }
    snapshot_artifacts = {
        "execution_snapshot": run_logger.artifact_label(execution_path),
        "code_snapshot": run_logger.artifact_label(code_path),
        "config_snapshot": run_logger.artifact_label(config_snapshot_path),
        "vector_store_snapshot": run_logger.artifact_label(vector_store_path),
        "documents_snapshot": run_logger.artifact_label(documents_path),
    }
    run_logger.write_input_snapshot_stage(
        snapshot_summary=snapshot_summary,
        snapshot_artifacts=snapshot_artifacts,
    )
    return snapshot_summary, snapshot_artifacts


def _refresh_vector_store_snapshot(
    *,
    run_logger: RunLogger,
    config: AppConfig,
    snapshot_summary: dict[str, object],
    snapshot_artifacts: dict[str, str],
    update_stats: object | None = None,
    selected_cities: list[str] | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    """Refresh the vector-store snapshot after an index update changes its manifest."""
    vector_store_snapshot = build_vector_store_snapshot(
        config,
        update_stats=update_stats,
        selected_cities=selected_cities,
    )
    vector_store_path = run_logger.write_stage_file(
        "input_snapshot",
        "vector_store_snapshot.json",
        vector_store_snapshot,
        alias="vector_store_snapshot",
    )
    updated_summary = dict(snapshot_summary)
    updated_summary["vector_store"] = {
        "enabled": vector_store_snapshot.get("enabled"),
        "collection_name": vector_store_snapshot.get("collection_name"),
        "index_manifest_hash": vector_store_snapshot.get("index_manifest_hash"),
        "manifest_summary": vector_store_snapshot.get("manifest_summary"),
        "auto_update": vector_store_snapshot.get("auto_update"),
    }
    updated_artifacts = dict(snapshot_artifacts)
    updated_artifacts["vector_store_snapshot"] = run_logger.artifact_label(vector_store_path)
    run_logger.write_input_snapshot_stage(
        snapshot_summary=updated_summary,
        snapshot_artifacts=updated_artifacts,
    )
    return updated_summary, updated_artifacts


def run_pipeline(
    question: str,
    config: AppConfig,
    run_id: str | None = None,
    log_llm_payload: bool = True,
    selected_cities: list[str] | None = None,
    analysis_mode: Literal["aggregate", "city_by_city"] = "aggregate",
    query_mode: Literal["standard", "dev"] = "standard",
    query_2: str | None = None,
    query_3: str | None = None,
    api_key_override: str | None = None,
    markdown_func: Callable[..., MarkdownResearchResult] = extract_markdown_excerpts,
    writer_func: Callable[..., WriterOutput] = write_markdown,
    config_path: Path | None = None,
    invocation_command: str | None = None,
    vector_update_docs_dir: Path | None = None,
) -> RunPaths:
    """
    Run the multi-agent document builder pipeline.

    Orchestrates retrieval-query preparation, markdown extraction, and final writing.

    Args:
        question: User question to answer
        config: Application configuration
        run_id: Optional run identifier
        log_llm_payload: Whether to log full LLM request/response payloads
        selected_cities: Optional list of city names to limit markdown document loading
        analysis_mode: Writer synthesis mode ("aggregate" | "city_by_city")
        query_mode: Retrieval query mode label persisted for run diagnostics
        query_2: Optional second direct retrieval query
        query_3: Optional third direct retrieval query
        api_key_override: Optional per-run API key override
        markdown_func: Markdown extraction function (default: extract_markdown_excerpts)
        writer_func: Document writing function (default: write_markdown)
        config_path: Optional path to the resolved config file for snapshot logging
        invocation_command: Optional human-readable command string for reruns
        vector_update_docs_dir: Optional canonical markdown directory used for vector
            index auto-updates. This can differ from ``config.markdown_dir`` when a
            run uses a selected-city copy for extraction.

    Returns:
        Run paths containing output artifacts

    Raises:
        Exception: Any unexpected exception from the write phase is re-raised after
            ``run_logger.finalize("failed")`` and log handler teardown have run,
            so that ``error_log.txt`` and ``api_state.json`` are always written on
            failure.
    """
    api_key = (
        api_key_override.strip()
        if isinstance(api_key_override, str) and api_key_override.strip()
        else get_openrouter_api_key()
    )
    requested_run_id = run_id or build_run_id()
    paths = create_run_paths(
        config.runs_dir, requested_run_id, config.orchestrator.context_bundle_name
    )
    run_id_value = paths.base_dir.name
    run_logger = RunLogger(paths, question)
    run_logger.update_analysis_mode(analysis_mode)
    run_logger.update_requested_city_scope(selected_cities)
    run_logger.record_artifact("context_bundle", paths.context_bundle)
    run_log_handler = attach_run_file_logger(paths.base_dir)
    progress = ProgressTracker(paths.base_dir)

    progress.start_step("input_snapshot", "Capturing run inputs")
    progress.start_step("query_preparation", "Preparing retrieval queries")
    canonical_research_query = question.strip() or question
    retrieval_queries = _build_retrieval_queries(
        canonical_research_query,
        query_2,
        query_3,
    ) or [canonical_research_query]
    logger.info(
        "Using verbatim retrieval queries for run_id=%s query_mode=%s query_count=%d",
        run_id_value,
        query_mode,
        len(retrieval_queries),
    )

    run_logger.update_query_inputs(
        original_question=question,
        canonical_research_query=canonical_research_query,
        retrieval_queries=retrieval_queries,
        query_mode=query_mode,
        write_stage_detail=False,
    )
    snapshot_summary, snapshot_artifacts = _write_input_snapshots(
        run_logger=run_logger,
        config=config,
        config_path=config_path or Path("llm_config.yaml"),
        requested_run_id=requested_run_id,
        resolved_run_id=run_id_value,
        invocation_command=invocation_command,
        selected_cities=selected_cities,
    )
    progress.add_item(
        "input_snapshot",
        f"Resolved run id: {run_id_value}",
    )
    if selected_cities:
        progress.add_item(
            "input_snapshot",
            f"Selected cities planned: {', '.join(selected_cities)}",
        )
    progress.complete_step("input_snapshot")
    run_logger.write_stage_file(
        "query_preparation",
        "research_question.json",
        {
            "original_question": question,
            "query_mode": query_mode,
            "canonical_research_query": canonical_research_query,
            "retrieval_queries": retrieval_queries,
            "retrieval_query_1": retrieval_queries[0] if len(retrieval_queries) >= 1 else None,
            "retrieval_query_2": retrieval_queries[1] if len(retrieval_queries) >= 2 else None,
            "retrieval_query_3": retrieval_queries[2] if len(retrieval_queries) >= 3 else None,
        },
        alias="research_question",
    )
    run_logger.write_query_preparation_stage(
        original_question=question,
        canonical_research_query=canonical_research_query,
        retrieval_queries=retrieval_queries,
        query_mode=query_mode,
    )
    progress.add_item("query_preparation", f"Primary retrieval query: {canonical_research_query}")
    for i, rq in enumerate(retrieval_queries, 1):
        progress.add_item("query_preparation", f"Retrieval query {i}: {rq}")
    progress.complete_step("query_preparation")

    def _run_initial_markdown() -> dict[str, object]:
        markdown_source_mode = "standard_chunking"
        markdown_chunks: list[dict[str, object]]
        if config.vector_store.enabled:
            markdown_source_mode = "vector_store_retrieval"
            if config.vector_store.auto_update_on_run:
                update_docs_dir = vector_update_docs_dir or config.markdown_dir
                update_stats = update_markdown_index(
                    config=config,
                    docs_dir=update_docs_dir,
                    selected_cities=selected_cities,
                    dry_run=False,
                )
                logger.info(
                    "Vector index auto-update finished run_id=%s changed=%d unchanged=%d "
                    "deleted=%d chunks=%d",
                    run_id_value,
                    update_stats.files_changed,
                    update_stats.files_unchanged,
                    update_stats.files_deleted,
                    update_stats.chunks_created,
                )
                nonlocal snapshot_summary, snapshot_artifacts
                snapshot_summary, snapshot_artifacts = _refresh_vector_store_snapshot(
                    run_logger=run_logger,
                    config=config,
                    snapshot_summary=snapshot_summary,
                    snapshot_artifacts=snapshot_artifacts,
                    update_stats=update_stats,
                    selected_cities=selected_cities,
                )
            retrieval_kwargs: dict[str, object] = {
                "queries": retrieval_queries,
                "config": config,
                "docs_dir": config.markdown_dir,
                "selected_cities": selected_cities,
            }
            retriever_signature = inspect.signature(retrieve_chunks_for_queries)
            if "run_id" in retriever_signature.parameters:
                retrieval_kwargs["run_id"] = run_id_value
            retrieved_chunks, retrieval_meta = retrieve_chunks_for_queries(
                **retrieval_kwargs
            )
            markdown_chunks = as_markdown_documents(retrieved_chunks)
            retrieval_payload = build_retrieval_artifact(
                queries=retrieval_queries,
                selected_cities=selected_cities,
                final_chunks=retrieved_chunks,
                retrieval_meta=retrieval_meta,
            )
            retrieval_path = run_logger.write_stage_file(
                "retrieval",
                "retrieval.json",
                retrieval_payload,
                alias="retrieval",
            )
            run_logger.write_stage_detail(
                "retrieval",
                {
                    "inputs": {
                        "queries": retrieval_queries,
                        "selected_cities": selected_cities or [],
                        "vector_store_enabled": True,
                        "retrieval_max_distance": config.vector_store.retrieval_max_distance,
                        "retrieval_max_chunks_per_city_query": (
                            config.vector_store.retrieval_max_chunks_per_city_query
                        ),
                    },
                    "outputs": {
                        "retrieval_artifact": run_logger.artifact_label(retrieval_path),
                        "retrieved_count": retrieval_payload.get("retrieved_count"),
                        "meta": retrieval_payload.get("meta"),
                    },
                    "metrics": build_retrieval_metrics(retrieval_payload),
                },
            )
        else:
            markdown_chunks = load_markdown_documents(
                config.markdown_dir,
                config.markdown_researcher,
                selected_cities=selected_cities,
            )
        logger.info(
            "run_id=%s markdown_source_mode=%s",
            run_id_value,
            markdown_source_mode,
        )
        run_logger.record_markdown_inputs(
            markdown_dir=config.markdown_dir,
            selected_cities_planned=selected_cities,
            markdown_chunks=markdown_chunks,
            markdown_source_mode=markdown_source_mode,
            analysis_mode=analysis_mode,
        )
        documents_by_city = split_documents_by_city(markdown_chunks)
        batch_max_chunks = int(max(config.markdown_researcher.batch_max_chunks, 1))
        batch_token_limit = int(resolve_batch_input_token_limit(config))
        batch_plan = build_city_batches(
            documents_by_city=documents_by_city,
            max_batch_input_tokens=batch_token_limit,
            max_batch_chunks=batch_max_chunks,
        )
        batches_payload = {
            "batch_max_chunks": batch_max_chunks,
            "batch_max_input_tokens": batch_token_limit,
            "cities": sorted(documents_by_city.keys()),
            "batches": [
                {
                    "city_name": city_name,
                    "batch_index": batch_index,
                    "chunk_count": len(batch),
                    "estimated_tokens": sum(
                        max(count_tokens(str(item.get("content", ""))), 0)
                        for item in batch
                    ),
                    "chunks": [
                        {
                            "chunk_id": str(item.get("chunk_id", "")),
                            "path": str(item.get("path", "")),
                            "chunk_index": item.get("chunk_index"),
                            "distance": item.get("distance"),
                        }
                        for item in batch
                    ],
                }
                for city_name, batch_index, batch in batch_plan
            ],
        }
        batches_path = run_logger.write_stage_file(
            "markdown_batching",
            "batches.json",
            batches_payload,
            alias="markdown_batches",
        )
        source_chunk_index = build_source_chunk_index(batches_payload)
        source_chunk_index_path = run_logger.write_stage_file(
            "markdown_batching",
            "source_chunk_index.json",
            source_chunk_index,
            alias="source_chunk_index",
        )
        run_logger.write_stage_detail(
            "markdown_batching",
            {
                "inputs": {
                    "batch_max_chunks": batch_max_chunks,
                    "batch_max_input_tokens": batch_token_limit,
                    "markdown_source_mode": markdown_source_mode,
                },
                "outputs": {
                    "batches": run_logger.artifact_label(batches_path),
                    "source_chunk_index": run_logger.artifact_label(
                        source_chunk_index_path
                    ),
                    "cities": batches_payload["cities"],
                },
                "metrics": {
                    "batch_count": len(batches_payload["batches"]),
                    "source_chunk_count": source_chunk_index["source_chunk_count"],
                },
            },
        )
        markdown_kwargs: dict[str, object] = {
            "log_llm_payload": log_llm_payload,
        }
        markdown_signature = inspect.signature(markdown_func)
        if "run_id" in markdown_signature.parameters:
            markdown_kwargs["run_id"] = run_id_value
        markdown_result = markdown_func(
            canonical_research_query,
            markdown_chunks,
            config,
            api_key,
            **markdown_kwargs,
        )
        return {
            "markdown_chunks": markdown_chunks,
            "result": markdown_result,
            "retrieval_queries": retrieval_queries,
            "markdown_source_mode": markdown_source_mode,
        }

    progress.start_step("markdown_research", "Searching markdown documents")

    try:
        markdown_payload = _run_initial_markdown()
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        return handle_task_error("markdown", exc, run_logger, run_log_handler, paths)

    if not markdown_payload:
        run_logger.finalize("failed", finish_reason="markdown_extraction_failed")
        detach_run_file_logger(run_log_handler)
        return paths

    # Process and log initial markdown results
    markdown_chunks = markdown_payload["markdown_chunks"]
    markdown_result = markdown_payload["result"]
    if isinstance(markdown_result, MarkdownResearchResult):
        markdown_bundle = markdown_result.model_dump()
        rejected_artifact, decision_audit_artifact = (
            _collect_markdown_decision_artifacts(markdown_chunks, markdown_result)
        )
        rejected_chunks_path = run_logger.write_stage_file(
            "markdown_extraction",
            "rejected_chunks.json",
            rejected_artifact,
            alias="markdown_rejected_chunks",
        )
        decision_audit_path = run_logger.write_stage_file(
            "markdown_extraction",
            "decision_audit.json",
            decision_audit_artifact,
            alias="markdown_decision_audit",
        )
        logger.info(
            "markdown_decision_audit accepted=%d rejected=%d unresolved=%d invariant_ok=%s",
            int(decision_audit_artifact["accepted_total"]),
            int(decision_audit_artifact["rejected_total"]),
            int(decision_audit_artifact["unresolved_total"]),
            bool(decision_audit_artifact["invariant_ok"]),
        )
        markdown_bundle["decision_audit"] = {
            "accepted_total": decision_audit_artifact["accepted_total"],
            "rejected_total": decision_audit_artifact["rejected_total"],
            "unresolved_total": decision_audit_artifact["unresolved_total"],
            "invariant_ok": decision_audit_artifact["invariant_ok"],
            "status": decision_audit_artifact["status"],
        }
        source_mode = str(
            markdown_payload.get("markdown_source_mode", "standard_chunking")
        )
        inspected_cities = sorted(
            {
                normalize_city_key(str(document.get("city_key", "")).strip())
                for document in markdown_chunks
                if normalize_city_key(str(document.get("city_key", "")).strip())
            }
        )
        # Display names for evidence header (e.g. "Aachen" not "aachen")
        key_to_name: dict[str, str] = {}
        for document in markdown_chunks:
            key = normalize_city_key(str(document.get("city_key", "")).strip())
            name = document.get("city_name")
            if key and key not in key_to_name:
                key_to_name[key] = (
                    format_city_stem(str(name).strip())
                    if name
                    else format_city_stem(key)
                )
        inspected_city_names = [
            key_to_name[k] for k in inspected_cities if k in key_to_name
        ]
        selected_city_keys = sorted(
            {
                normalize_city_key(city)
                for city in (selected_cities or [])
                if isinstance(city, str) and city.strip()
            }
        )
        selected_city_names = [
            key_to_name.get(key, format_city_stem(key))
            for key in selected_city_keys
        ]
        context_bundle = run_logger.context_bundle
        context_bundle["city_scope_mode"] = (
            "selected_cities" if selected_city_keys else "all_cities"
        )
        context_bundle["selected_cities"] = selected_city_keys
        context_bundle["selected_city_names"] = selected_city_names
        context_bundle["inspected_cities"] = inspected_cities
        context_bundle["inspected_city_names"] = inspected_city_names
        if selected_city_keys:
            context_bundle["selected_city_names"] = selected_city_names
        excerpts = markdown_bundle.get("excerpts", [])
        if isinstance(excerpts, list):
            excerpt_entries = [
                excerpt for excerpt in excerpts if isinstance(excerpt, dict)
            ]
            enriched_excerpts, _references_payload = build_markdown_references(
                run_id=run_id_value,
                excerpts=excerpt_entries,
            )
            markdown_bundle["excerpts"] = enriched_excerpts
            markdown_bundle["excerpt_count"] = len(enriched_excerpts)
        else:
            markdown_bundle["excerpts"] = []
            markdown_bundle["excerpt_count"] = 0

        accepted_excerpts_payload = {
            "excerpts": markdown_bundle["excerpts"],
            "excerpt_count": markdown_bundle["excerpt_count"],
        }
        markdown_excerpts_path = run_logger.write_stage_file(
            "markdown_extraction",
            "accepted_excerpts.json",
            accepted_excerpts_payload,
            alias="markdown_excerpts",
        )
        run_logger.update_markdown_bundle(markdown_bundle)
        run_logger.write_stage_detail(
            "markdown_extraction",
            {
                "inputs": {
                    "source_mode": source_mode,
                    "analysis_mode": analysis_mode,
                    "retrieval_queries": retrieval_queries,
                    "city_scope_mode": context_bundle.get("city_scope_mode"),
                    "selected_cities": context_bundle.get("selected_cities", []),
                },
                "outputs": {
                    "markdown_excerpts": run_logger.artifact_label(
                        markdown_excerpts_path
                    ),
                    "accepted_excerpts": run_logger.artifact_label(
                        markdown_excerpts_path
                    ),
                    "rejected_chunks": run_logger.artifact_label(
                        rejected_chunks_path
                    ),
                    "decision_audit": run_logger.artifact_label(
                        decision_audit_path
                    ),
                    "selected_cities": context_bundle.get("selected_cities", []),
                    "selected_city_names": context_bundle.get("selected_city_names", []),
                    "inspected_cities": context_bundle.get("inspected_cities", []),
                    "inspected_city_names": context_bundle.get(
                        "inspected_city_names",
                        [],
                    ),
                },
                "metrics": build_markdown_metrics(
                    markdown_chunks=markdown_chunks,
                    markdown_bundle=markdown_bundle,
                    rejected_artifact=rejected_artifact,
                    decision_audit_artifact=decision_audit_artifact,
                ),
            },
        )
        progress.add_item(
            "markdown_research",
            f"{len(markdown_chunks)} chunks from {len(inspected_cities)} cities",
        )
        progress.add_item(
            "markdown_research",
            f"{markdown_bundle.get('excerpt_count', 0)} excerpts extracted",
        )
        progress.complete_step("markdown_research")
        if config.markdown_researcher.strict_decision_audit and not bool(
            decision_audit_artifact["invariant_ok"]
        ):
            run_logger.record_decision(
                {
                    "code": "MARKDOWN_DECISION_AUDIT_FAILED",
                    "message": "Strict markdown decision audit failed.",
                    "decision_audit": decision_audit_artifact,
                }
            )
            run_logger.finalize(
                "failed", finish_reason="markdown_decision_audit_failed"
            )
            detach_run_file_logger(run_log_handler)
            return paths
        if markdown_result.status == "error":
            run_logger.record_decision(markdown_result.model_dump())
            run_logger.finalize("failed", finish_reason="markdown_result_error")
            detach_run_file_logger(run_log_handler)
            return paths
    else:
        run_logger.finalize("failed", finish_reason="markdown_extraction_failed")
        detach_run_file_logger(run_log_handler)
        return paths

    # Freeze the full runtime context after the markdown pipeline completes.
    progress.start_step("markdown_context_handoff", "Freezing markdown context handoff")
    context_bundle = run_logger.context_bundle
    context_bundle["analysis_mode"] = analysis_mode
    run_logger.write_context_bundle()
    markdown_payload = (
        context_bundle.get("markdown")
        if isinstance(context_bundle.get("markdown"), dict)
        else None
    )
    markdown_handoff_outputs = _write_context_handoff(
        run_logger=run_logger,
        stage_name="markdown_context_handoff",
        snapshot_filename="context_bundle_after_markdown.json",
        progress=progress,
        progress_label="Markdown context snapshot written",
    )
    run_logger.write_stage_detail(
        "markdown_context_handoff",
        {
            "inputs": {"analysis_mode": analysis_mode},
            "outputs": markdown_handoff_outputs,
            "metrics": {
                "markdown_excerpt_count": markdown_payload.get("excerpt_count", 0)
                if isinstance(markdown_payload, dict)
                else 0,
                "context_bundle_top_level_keys": len(context_bundle),
            },
        },
    )

    # --- Enrichment layer (gap analysis + web research + assumptions modelling) ---
    if config.enrichment.enabled:
        context_bundle = run_enrichment_pipeline(
            question=question,
            context_bundle=context_bundle,
            base_dir=paths.base_dir,
            run_logger=run_logger,
            config=config,
            api_key=api_key,
            progress=progress,
        )
        run_logger.context_bundle = context_bundle
        run_logger.write_context_bundle()
    # --- END enrichment ---

    progress.start_step("writer", "Generating final document")
    try:
        result = handle_write_decision(
            question,
            context_bundle,
            paths=paths,
            run_logger=run_logger,
            run_log_handler=run_log_handler,
            config=config,
            api_key=api_key,
            log_llm_payload=log_llm_payload,
            writer_func=writer_func,
            progress=progress,
        )
        progress.add_item("writer", "Document written")
        progress.complete_step("writer")
        return result if result is not None else paths
    except Exception:
        logger.exception(
            "Unexpected error during write decision for run_id=%s", run_id_value
        )
        progress.complete_step("writer", status="error")
        run_logger.finalize("failed", finish_reason="writer_unexpected_error")
        detach_run_file_logger(run_log_handler)
        raise


__all__ = ["run_pipeline"]
