from __future__ import annotations

import logging
import sqlite3
import inspect
from pathlib import Path
from typing import Callable, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg

from backend.modules.markdown_researcher.agent import extract_markdown_excerpts
from backend.modules.markdown_researcher.models import MarkdownResearchResult
from backend.modules.markdown_researcher.services import (
    build_city_batches,
    load_markdown_documents,
    resolve_batch_input_token_limit,
    split_documents_by_city,
)
from backend.modules.orchestrator.agent import refine_research_question
from backend.modules.orchestrator.models import (
    ResearchQuestionRefinement,
)
from backend.modules.orchestrator.utils import (
    attach_run_file_logger,
    build_markdown_references,
    collect_identifiers,
    fetch_city_list,
    handle_write_decision,
    handle_task_error,
    run_sql_rounds,
)
from backend.modules.orchestrator.utils.error_handlers import (
    detach_run_file_logger,
)
from backend.modules.orchestrator.utils.io import (
    write_json,
)
from backend.modules.sql_researcher.agent import plan_sql_queries
from backend.modules.sql_researcher.models import SqlQueryPlan
from backend.modules.sql_researcher.services import (
    build_sql_research_result,
    cap_results,
)
from backend.modules.vector_store.indexer import update_markdown_index
from backend.modules.vector_store.retriever import (
    as_markdown_documents,
    retrieve_chunks_for_queries,
)
from backend.modules.writer.agent import write_markdown
from backend.modules.writer.models import WriterOutput
from backend.services.db_client import DbClient, get_db_client
from backend.services.run_logger import RunLogger
from backend.services.schema_registry import load_schema
from backend.utils.config import AppConfig, get_openrouter_api_key
from backend.utils.city_normalization import format_city_stem, normalize_city_key
from backend.utils.paths import RunPaths, build_run_id, create_run_paths
from backend.utils.tokenization import count_tokens

logger = logging.getLogger(__name__)


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
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Build accepted/rejected id artifacts and a run-level decision audit."""

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

    accepted_by_city: dict[str, list[str]] = {}
    for chunk_id in accepted_ids:
        city_key = city_by_chunk_id.get(chunk_id, "unknown")
        accepted_by_city.setdefault(city_key, []).append(chunk_id)

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

    accepted_artifact = {
        "status": artifact_status,
        "accepted_chunk_ids": accepted_ids,
        "accepted_by_city": accepted_by_city,
        "counts": {
            "accepted": len(accepted_ids),
        },
    }
    rejected_artifact = {
        "status": artifact_status,
        "rejected_chunk_ids": rejected_ids,
        "rejected_by_city": rejected_by_city,
        "counts": {
            "rejected": len(rejected_ids),
        },
    }
    audit_artifact = {
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
    return accepted_artifact, rejected_artifact, audit_artifact


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
    sql_plan_func: Callable[..., SqlQueryPlan] = plan_sql_queries,
    markdown_func: Callable[..., MarkdownResearchResult] = extract_markdown_excerpts,
    refine_question_func: Callable[
        ..., ResearchQuestionRefinement
    ] = refine_research_question,
    writer_func: Callable[..., WriterOutput] = write_markdown,
) -> RunPaths:
    """
    Run the multi-agent document builder pipeline.

    Orchestrates SQL research (when enabled), markdown extraction, and final writing.

    Args:
        question: User question to answer
        config: Application configuration
        run_id: Optional run identifier
        log_llm_payload: Whether to log full LLM request/response payloads
        selected_cities: Optional list of city names to limit markdown document loading
        analysis_mode: Writer synthesis mode ("aggregate" | "city_by_city")
        query_mode: Retrieval query mode ("standard" uses refinement, "dev" uses direct inputs)
        query_2: Optional second direct retrieval query used in dev mode
        query_3: Optional third direct retrieval query used in dev mode
        api_key_override: Optional per-run API key override
        sql_plan_func: SQL planning function (default: plan_sql_queries)
        markdown_func: Markdown extraction function (default: extract_markdown_excerpts)
        refine_question_func: Question refinement function (default: refine_research_question)
        writer_func: Document writing function (default: write_markdown)

    Returns:
        Run paths containing output artifacts

    Raises:
        Exception: Any unexpected exception from the write phase is re-raised after
            ``run_logger.finalize("failed")`` and log handler teardown have run, so
            that ``error_log.txt`` and ``run.json`` are always written on failure.
    """
    api_key = (
        api_key_override.strip()
        if isinstance(api_key_override, str) and api_key_override.strip()
        else get_openrouter_api_key()
    )
    run_id_value = run_id or build_run_id()
    paths = create_run_paths(
        config.runs_dir, run_id_value, config.orchestrator.context_bundle_name
    )
    run_logger = RunLogger(paths, question)
    run_logger.update_analysis_mode(analysis_mode)
    run_logger.record_artifact("context_bundle", paths.context_bundle)
    run_log_handler = attach_run_file_logger(paths.base_dir)

    canonical_research_query = question.strip() or question
    retrieval_queries: list[str]
    if query_mode == "dev":
        retrieval_queries = _build_retrieval_queries(
            canonical_research_query,
            query_2,
            query_3,
        ) or [canonical_research_query]
        logger.info(
            "Using direct retrieval queries for run_id=%s query_count=%d",
            run_id_value,
            len(retrieval_queries),
        )
    else:
        # Standard mode keeps the current research-question refinement flow.
        retrieval_queries = [canonical_research_query]
        try:
            refinement = refine_question_func(
                question,
                config,
                api_key,
                selected_cities=selected_cities,
                log_llm_payload=log_llm_payload,
            )
            candidate = refinement.research_question.strip()
            if candidate:
                canonical_research_query = candidate
            else:
                logger.warning(
                    "Research question refinement returned empty output; using original question."
                )
            retrieval_queries = _build_retrieval_queries(
                canonical_research_query,
                *refinement.retrieval_queries,
            ) or [canonical_research_query]
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            logger.warning(
                "Research question refinement failed; using original question. error=%s",
                exc,
            )

    run_logger.update_query_inputs(
        original_question=question,
        canonical_research_query=canonical_research_query,
        retrieval_queries=retrieval_queries,
        query_mode=query_mode,
    )
    write_json(
        paths.research_question,
        {
            "original_question": question,
            "query_mode": query_mode,
            "canonical_research_query": canonical_research_query,
            "retrieval_queries": retrieval_queries,
            "retrieval_query_1": retrieval_queries[0] if len(retrieval_queries) >= 1 else None,
            "retrieval_query_2": retrieval_queries[1] if len(retrieval_queries) >= 2 else None,
            "retrieval_query_3": retrieval_queries[2] if len(retrieval_queries) >= 3 else None,
        },
    )
    run_logger.record_artifact("research_question", paths.research_question)

    sql_enabled = config.enable_sql
    schema_summary: dict = {}
    identifiers: list[str] = []
    city_names: list[str] = []
    db_client: DbClient | None = None

    if sql_enabled:
        # Load schema and extract identifiers
        models_dir = Path(__file__).resolve().parents[2] / "db_models"
        schema_summary = load_schema(models_dir).model_dump()
        write_json(paths.schema_summary, schema_summary)
        run_logger.record_artifact("schema_summary", paths.schema_summary)
        identifiers = collect_identifiers(schema_summary)

        # Get database client and city list
        db_client = get_db_client(config.source_db_path, config.source_db_url)
        try:
            city_names = fetch_city_list(db_client)
        except (sqlite3.OperationalError, psycopg.DatabaseError, OSError) as e:
            logger.warning("Failed to fetch city list: %s", e)
        write_json(paths.city_list, city_names)
        run_logger.record_artifact("city_list", paths.city_list)
    else:
        logger.info("SQL disabled; running without database lookups")

    # Run initial SQL and markdown in parallel
    def _run_initial_sql() -> dict[str, object]:
        sql_plan = sql_plan_func(
            canonical_research_query,
            schema_summary,
            city_names,
            config,
            api_key,
            per_city_focus=True,
            log_llm_payload=log_llm_payload,
        )
        if sql_plan.status == "error":
            return {"status": "error", "plan": sql_plan}
        sql_plan, full_results, sql_rounds = run_sql_rounds(
            canonical_research_query,
            sql_plan,
            db_client,
            config,
            api_key,
            log_llm_payload,
            schema_summary,
            city_names,
            identifiers,
            sql_plan_func,
            write_json,
            run_logger,
            paths,
        )
        capped_results, total_tokens, truncated = cap_results(
            full_results, config.sql_researcher.max_result_tokens
        )
        return {
            "status": "ok",
            "plan": sql_plan,
            "full_results": full_results,
            "sql_rounds": sql_rounds,
            "capped_results": capped_results,
            "total_tokens": total_tokens,
            "truncated": truncated,
        }

    def _run_initial_markdown() -> dict[str, object]:
        markdown_source_mode = "standard_chunking"
        markdown_chunks: list[dict[str, object]]
        if config.vector_store.enabled:
            markdown_source_mode = "vector_store_retrieval"
            if config.vector_store.auto_update_on_run:
                update_markdown_index(
                    config=config,
                    docs_dir=config.markdown_dir,
                    selected_cities=selected_cities,
                    dry_run=False,
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
            retrieval_payload = {
                "queries": retrieval_queries,
                "selected_cities": selected_cities or [],
                "retrieved_count": len(retrieved_chunks),
                "meta": retrieval_meta,
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "city_name": chunk.city_name,
                        "city_key": str(chunk.metadata.get("city_key", "")),
                        "source_path": chunk.source_path,
                        "heading_path": chunk.heading_path,
                        "block_type": chunk.block_type,
                        "distance": chunk.distance,
                    }
                    for chunk in retrieved_chunks
                ],
            }
            retrieval_path = paths.markdown_dir / "retrieval.json"
            write_json(retrieval_path, retrieval_payload)
            run_logger.record_artifact("markdown_retrieval", retrieval_path)
        else:
            markdown_chunks = load_markdown_documents(
                config.markdown_dir,
                config.markdown_researcher,
                selected_cities=selected_cities,
            )
        logger.info(
            "run_id=%s markdown_source_mode=%s",
            run_id,
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
        batches_path = paths.markdown_dir / "batches.json"
        write_json(batches_path, batches_payload)
        run_logger.record_artifact("markdown_batches", batches_path)
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

    sql_payload: dict[str, object] | None = None
    markdown_payload: dict[str, object] | None = None

    with ThreadPoolExecutor(max_workers=2 if sql_enabled else 1) as executor:
        futures = {}
        if sql_enabled:
            futures[executor.submit(_run_initial_sql)] = "sql"
        futures[executor.submit(_run_initial_markdown)] = "markdown"
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                payload = future.result()
                if task_name == "sql":
                    sql_payload = payload
                else:
                    markdown_payload = payload
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                return handle_task_error(
                    task_name, exc, run_logger, run_log_handler, paths
                )

    if sql_enabled and not sql_payload:
        run_logger.finalize("failed", finish_reason="sql_execution_failed")
        detach_run_file_logger(run_log_handler)
        return paths
    if not markdown_payload:
        run_logger.finalize("failed", finish_reason="markdown_extraction_failed")
        detach_run_file_logger(run_log_handler)
        return paths

    if sql_enabled and sql_payload and sql_payload.get("status") == "error":
        sql_plan_obj = sql_payload.get("plan")
        if isinstance(sql_plan_obj, SqlQueryPlan):
            run_logger.record_decision(sql_plan_obj.model_dump())
        run_logger.finalize("failed", finish_reason="sql_plan_error")
        detach_run_file_logger(run_log_handler)
        return paths

    if sql_enabled and sql_payload:
        # Process and log initial SQL results
        sql_plan: SqlQueryPlan = sql_payload["plan"]  # type: ignore
        full_results: list = sql_payload["full_results"]  # type: ignore
        sql_rounds: list[dict[str, object]] = sql_payload["sql_rounds"]  # type: ignore
        capped_results: list = sql_payload["capped_results"]  # type: ignore
        total_tokens: int = sql_payload["total_tokens"]  # type: ignore
        truncated: bool = sql_payload["truncated"]  # type: ignore

        sql_rounds_path = paths.sql_dir / "rounds.json"
        write_json(sql_rounds_path, sql_rounds)
        run_logger.record_artifact("sql_rounds", sql_rounds_path)
        write_json(paths.sql_queries, sql_plan.model_dump())
        run_logger.record_artifact("sql_queries", paths.sql_queries)

        full_payload = [result.model_dump() for result in full_results]
        capped_payload = [result.model_dump() for result in capped_results]
        write_json(paths.sql_results_full, full_payload)
        write_json(paths.sql_results, capped_payload)
        run_logger.record_artifact("sql_results_full", paths.sql_results_full)
        run_logger.record_artifact("sql_results", paths.sql_results)

        sql_result = build_sql_research_result(
            queries=sql_plan.queries,
            results=capped_results,
            total_tokens=total_tokens,
            truncated=truncated,
        )
        run_logger.update_sql_bundle(sql_result.model_dump())

    # Process and log initial markdown results
    markdown_chunks = markdown_payload["markdown_chunks"]
    markdown_result = markdown_payload["result"]
    if isinstance(markdown_result, MarkdownResearchResult):
        markdown_bundle = markdown_result.model_dump()
        (
            accepted_artifact,
            rejected_artifact,
            decision_audit_artifact,
        ) = _collect_markdown_decision_artifacts(markdown_chunks, markdown_result)
        write_json(paths.markdown_accepted_excerpts, accepted_artifact)
        write_json(paths.markdown_rejected_excerpts, rejected_artifact)
        write_json(paths.markdown_decision_audit, decision_audit_artifact)
        run_logger.record_artifact(
            "markdown_accepted_excerpts",
            paths.markdown_accepted_excerpts,
        )
        run_logger.record_artifact(
            "markdown_rejected_excerpts",
            paths.markdown_rejected_excerpts,
        )
        run_logger.record_artifact(
            "markdown_decision_audit",
            paths.markdown_decision_audit,
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
            "status": accepted_artifact["status"],
        }
        source_mode = str(
            markdown_payload.get("markdown_source_mode", "standard_chunking")
        )
        markdown_bundle["retrieval_mode"] = source_mode
        markdown_bundle["analysis_mode"] = analysis_mode
        if source_mode == "vector_store_retrieval":
            markdown_bundle["retrieval_queries"] = markdown_payload.get(
                "retrieval_queries",
                [],
            )
        inspected_cities = sorted(
            {
                normalize_city_key(str(document.get("city_key", "")).strip())
                for document in markdown_chunks
                if normalize_city_key(str(document.get("city_key", "")).strip())
            }
        )
        markdown_bundle["inspected_cities"] = inspected_cities
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
        markdown_bundle["inspected_city_names"] = [
            key_to_name[k] for k in inspected_cities if k in key_to_name
        ]
        selected_city_keys = sorted(
            {
                normalize_city_key(city)
                for city in (selected_cities or [])
                if isinstance(city, str) and city.strip()
            }
        )
        markdown_bundle["selected_cities"] = selected_city_keys
        if selected_city_keys:
            markdown_bundle["selected_city_names"] = [
                key_to_name.get(key, format_city_stem(key))
                for key in selected_city_keys
            ]
        else:
            markdown_bundle["selected_city_names"] = markdown_bundle[
                "inspected_city_names"
            ]
        excerpts = markdown_bundle.get("excerpts", [])
        if isinstance(excerpts, list):
            excerpt_entries = [
                excerpt for excerpt in excerpts if isinstance(excerpt, dict)
            ]
            enriched_excerpts, references_payload = build_markdown_references(
                run_id=run_id_value,
                excerpts=excerpt_entries,
            )
            markdown_bundle["excerpts"] = enriched_excerpts
            markdown_bundle["excerpt_count"] = len(enriched_excerpts)
            write_json(paths.markdown_references, references_payload)
            run_logger.record_artifact(
                "markdown_references",
                paths.markdown_references,
            )
        else:
            markdown_bundle["excerpts"] = []
            markdown_bundle["excerpt_count"] = 0

        write_json(paths.markdown_excerpts, markdown_bundle)
        run_logger.record_artifact("markdown_excerpts", paths.markdown_excerpts)
        run_logger.update_markdown_bundle(markdown_bundle)
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

    # Write final output directly from the prepared context bundle.
    context_bundle = run_logger.context_bundle
    context_bundle["analysis_mode"] = analysis_mode
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
        )
        return result if result is not None else paths
    except Exception:
        logger.exception(
            "Unexpected error during write decision for run_id=%s", run_id_value
        )
        run_logger.finalize("failed", finish_reason="writer_unexpected_error")
        detach_run_file_logger(run_log_handler)
        raise


__all__ = ["run_pipeline"]
