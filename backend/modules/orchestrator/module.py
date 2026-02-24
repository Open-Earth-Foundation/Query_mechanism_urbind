from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Callable
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
from backend.utils.paths import RunPaths, build_run_id, create_run_paths
from backend.utils.tokenization import count_tokens

logger = logging.getLogger(__name__)


def run_pipeline(
    question: str,
    config: AppConfig,
    run_id: str | None = None,
    log_llm_payload: bool = True,
    selected_cities: list[str] | None = None,
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
        sql_plan_func: SQL planning function (default: plan_sql_queries)
        markdown_func: Markdown extraction function (default: extract_markdown_excerpts)
        refine_question_func: Question refinement function (default: refine_research_question)
        writer_func: Document writing function (default: write_markdown)

    Returns:
        Run paths containing output artifacts
    """
    api_key = get_openrouter_api_key()
    run_id_value = run_id or build_run_id()
    paths = create_run_paths(
        config.runs_dir, run_id_value, config.orchestrator.context_bundle_name
    )
    run_logger = RunLogger(paths, question)
    run_logger.record_artifact("context_bundle", paths.context_bundle)
    run_log_handler = attach_run_file_logger(paths.base_dir)

    # First step in the chain: rewrite the user question into a research-ready form.
    canonical_research_query = question
    retrieval_queries: list[str] = [canonical_research_query]
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
        retrieval_query_candidates = [query.strip() for query in refinement.retrieval_queries]
        combined_queries = [canonical_research_query] + [
            candidate_query
            for candidate_query in retrieval_query_candidates
            if candidate_query
        ]
        deduped_queries: list[str] = []
        seen_queries: set[str] = set()
        for query_candidate in combined_queries:
            key = query_candidate.casefold()
            if key in seen_queries:
                continue
            seen_queries.add(key)
            deduped_queries.append(query_candidate)
            if len(deduped_queries) >= 3:
                break
        retrieval_queries = deduped_queries or [canonical_research_query]
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        logger.warning(
            "Research question refinement failed; using original question. error=%s",
            exc,
        )

    run_logger.update_research_question(canonical_research_query)
    write_json(
        paths.research_question,
        {
            "original_question": question,
            "canonical_research_query": canonical_research_query,
            "retrieval_queries": retrieval_queries,
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
            retrieved_chunks, retrieval_meta = retrieve_chunks_for_queries(
                queries=retrieval_queries,
                config=config,
                docs_dir=config.markdown_dir,
                selected_cities=selected_cities,
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
                        max(count_tokens(str(item.get("content", ""))), 0) for item in batch
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
        markdown_result = markdown_func(
            canonical_research_query,
            markdown_chunks,
            config,
            api_key,
            log_llm_payload=log_llm_payload,
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
        source_mode = str(markdown_payload.get("markdown_source_mode", "standard_chunking"))
        markdown_bundle["retrieval_mode"] = source_mode
        if source_mode == "vector_store_retrieval":
            markdown_bundle["retrieval_queries"] = markdown_payload.get(
                "retrieval_queries",
                [],
            )
        inspected_cities = sorted(
            {
                str(document.get("city_key", "")).strip()
                for document in markdown_chunks
                if str(document.get("city_key", "")).strip()
            }
        )
        markdown_bundle["inspected_cities"] = inspected_cities
        # Display names for evidence header (e.g. "Aachen" not "aachen")
        key_to_name: dict[str, str] = {}
        for document in markdown_chunks:
            key = str(document.get("city_key", "")).strip()
            name = document.get("city_name")
            if key and key not in key_to_name:
                key_to_name[key] = str(name).strip() if name else key
        markdown_bundle["inspected_city_names"] = [
            key_to_name[k] for k in inspected_cities if k in key_to_name
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


__all__ = ["run_pipeline"]
