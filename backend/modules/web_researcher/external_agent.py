"""LLM harness for governed external Markdown source research."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from agents import Agent, AgentOutputSchema, function_tool
from agents.exceptions import MaxTurnsExceeded

from backend.modules.web_researcher.external_resolver import resolve_external_evidence
from backend.modules.web_researcher.external_sources import (
    EXTERNAL_SOURCE_SEARCH_AUDIT_FILENAME,
    ExternalSearchSession,
    ExternalSourceToolError,
    build_external_search_limits,
    try_load_external_source_registry,
)
from backend.modules.web_researcher.models import (
    EvidenceCandidate,
    EvidenceCandidateInput,
    ExternalEvidenceClaim,
    ExternalEvidenceResolution,
    ExternalSourceAgentResult,
    GapManifest,
    NoEvidenceRecord,
)
from backend.services.agents import (
    build_model_settings,
    build_openrouter_model,
    run_agent_sync,
)
from backend.services.llm_observability import LlmCallContext, LlmCallRecorder
from backend.utils.artifact_writer import stage_file_dir_name
from backend.utils.config import AppConfig
from backend.utils.prompts import load_prompt

logger = logging.getLogger(__name__)


def run_external_source_enrichment(
    *,
    question: str,
    context_bundle: dict[str, Any],
    gap_manifest: GapManifest,
    base_dir: Path,
    config: AppConfig,
    api_key: str,
    run_id: str | None = None,
    llm_recorder: LlmCallRecorder | None = None,
) -> tuple[
    list[ExternalEvidenceClaim],
    list[ExternalEvidenceResolution],
    list[NoEvidenceRecord],
    list[dict[str, object]],
    dict[str, object],
]:
    """Run the external-source LLM tool loop and resolve evidence decisions."""
    if not config.enrichment.external_source_search_enabled:
        return [], [], [], [], {}

    registry = try_load_external_source_registry(config.enrichment.external_source_dir)
    if registry is None or not gap_manifest.city_gaps:
        return [], [], [], [], {}

    session = ExternalSearchSession(
        run_id=run_id or base_dir.name,
        registry=registry,
        limits=build_external_search_limits(config),
        artifact_dir=base_dir / "stage_files" / stage_file_dir_name("enrichment"),
    )
    agent = build_external_source_research_agent(config, api_key, session)
    finalizer = build_external_source_finalizer_agent(config, api_key)
    all_claims: list[ExternalEvidenceClaim] = []
    all_rejected_claims: list[dict[str, object]] = []
    all_no_evidence: list[NoEvidenceRecord] = []
    candidate_index: dict[str, EvidenceCandidate] = {}
    max_turn_exceeded_count = 0
    fallback_finalization_count = 0

    for task in _iter_external_research_tasks(gap_manifest, context_bundle, question):
        session.set_active_task(str(task["city"]), str(task["field"]))
        agent_output: ExternalSourceAgentResult | None = None
        fallback_attempted = False
        try:
            result = run_agent_sync(
                agent,
                json.dumps(task, ensure_ascii=False),
                max_turns=config.enrichment.max_turns,
                llm_recorder=llm_recorder,
                llm_call_context=LlmCallContext(
                    stage_name="enrichment",
                    stage_family="enrichment",
                    agent="external_source_researcher",
                    call_kind="external_source_research",
                    model=config.enrichment.model,
                    metadata={
                        "city": task.get("city"),
                        "field": task.get("field"),
                    },
                ),
            )
            if isinstance(result.final_output, ExternalSourceAgentResult):
                agent_output = result.final_output
            else:
                logger.warning(
                    "External source research returned unexpected output for city=%s field=%s",
                    task["city"],
                    task["field"],
                )
        except MaxTurnsExceeded:
            max_turn_exceeded_count += 1
            logger.warning(
                "External source research exceeded max turns for city=%s field=%s; "
                "trying expanded-hit finalization.",
                task["city"],
                task["field"],
            )
            fallback_finalization_count += 1
            fallback_attempted = True
            agent_output = _finalize_from_expanded_hits(
                task=task,
                session=session,
                finalizer=finalizer,
                llm_recorder=llm_recorder,
                model=config.enrichment.model,
            )
        except Exception:
            logger.warning(
                "External source research failed for city=%s field=%s",
                task["city"],
                task["field"],
                exc_info=True,
            )
            fallback_finalization_count += 1
            fallback_attempted = True
            agent_output = _finalize_from_expanded_hits(
                task=task,
                session=session,
                finalizer=finalizer,
                llm_recorder=llm_recorder,
                model=config.enrichment.model,
            )

        if agent_output is None:
            continue
        if not agent_output.claims and not fallback_attempted:
            fallback_finalization_count += 1
            fallback_output = _finalize_from_expanded_hits(
                task=task,
                session=session,
                finalizer=finalizer,
                llm_recorder=llm_recorder,
                model=config.enrichment.model,
            )
            if fallback_output is not None and fallback_output.claims:
                agent_output = fallback_output

        candidate_index.update(
            {candidate.candidate_id: candidate for candidate in session.evidence_candidates()}
        )
        task_validated_claims, task_rejected_claims = _validated_claims(
            agent_output.claims,
            candidate_index,
        )
        all_claims.extend(task_validated_claims)
        all_rejected_claims.extend(task_rejected_claims)
        all_no_evidence.extend(agent_output.no_evidence)

    existing_no_evidence_ids = {record.record_id for record in all_no_evidence}
    all_no_evidence.extend(
        record
        for record in session.no_evidence_records()
        if record.record_id not in existing_no_evidence_ids
    )
    validated_claims, rejected_claims = _validated_claims(all_claims, candidate_index)
    rejected_claims = [*all_rejected_claims, *rejected_claims]
    deduped_claims, duplicate_claims = _dedupe_claims(validated_claims)
    rejected_claims.extend(duplicate_claims)
    ccc_values = _extract_external_ccc_values(context_bundle)
    resolutions = resolve_external_evidence(
        gap_manifest.city_gaps,
        deduped_claims,
        all_no_evidence,
        ccc_values=ccc_values,
    )
    audit_payload = _build_external_search_audit_payload(
        session=session,
        gap_manifest=gap_manifest,
        validated_claims=deduped_claims,
        resolutions=resolutions,
        rejected_claims=rejected_claims,
        max_turn_exceeded_count=max_turn_exceeded_count,
        fallback_finalization_count=fallback_finalization_count,
    )
    return deduped_claims, resolutions, all_no_evidence, session.tool_call_log(), audit_payload


def build_external_source_research_agent(
    config: AppConfig,
    api_key: str,
    session: ExternalSearchSession,
) -> Agent:
    """Build an agent that can only use controlled external-source tools."""
    model = build_openrouter_model(
        config.enrichment.model,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )
    settings = build_model_settings(
        config.enrichment.temperature,
        config.enrichment.max_output_tokens,
        reasoning_effort=config.enrichment.reasoning_effort,
    )
    settings.parallel_tool_calls = False

    def _has_active_task_hits(_ctx: Any, _agent: Any) -> bool:
        """Enable anchored tools only after the active task has regex hits."""
        return session.has_hits_for_active_task()

    @function_tool(strict_mode=False)
    def get_tag_options() -> dict[str, Any]:
        """Return available metadata values for external-source filters."""
        return _tool_result(session.get_tag_options)

    @function_tool(strict_mode=False)
    def list_candidate_sources(
        cities: list[str] | None = None,
        countries: list[str] | None = None,
        verticals: list[str] | None = None,
        tef_sectors: list[str] | None = None,
        source_types: list[str] | None = None,
        publication_year_min: int | None = None,
        publication_year_max: int | None = None,
        max_files: int = 50,
    ) -> dict[str, Any]:
        """Return source summaries matching metadata filters."""
        return _tool_result(
            lambda: session.list_candidate_sources(
                cities=cities,
                countries=countries,
                verticals=verticals,
                tef_sectors=tef_sectors,
                source_types=source_types,
                publication_year_min=publication_year_min,
                publication_year_max=publication_year_max,
                max_files=max_files,
            )
        )

    @function_tool(strict_mode=False)
    def regex_search(
        pattern: str,
        cities: list[str] | None = None,
        countries: list[str] | None = None,
        verticals: list[str] | None = None,
        tef_sectors: list[str] | None = None,
        source_types: list[str] | None = None,
        case_sensitive: bool = False,
        context_words: int | None = None,
        context_lines: int | None = None,
        max_matches: int | None = None,
    ) -> dict[str, Any]:
        """Run a validated regex over scoped external Markdown sources."""
        return _tool_result(
            lambda: session.regex_search(
                pattern=pattern,
                cities=cities,
                countries=countries,
                verticals=verticals,
                tef_sectors=tef_sectors,
                source_types=source_types,
                case_sensitive=case_sensitive,
                context_words=context_words,
                context_lines=context_lines,
                max_matches=max_matches,
            )
        )

    @function_tool(strict_mode=False, is_enabled=_has_active_task_hits)
    def expand_hits(
        hit_ids: list[str],
        context_words: int | None = None,
        context_lines: int | None = None,
    ) -> dict[str, Any]:
        """Expand prior hit IDs from this run."""
        return _tool_result(session.expand_hits, hit_ids, context_words, context_lines)

    @function_tool(strict_mode=False, is_enabled=_has_active_task_hits)
    def add_evidence_candidates(candidates: list[Any]) -> dict[str, Any]:
        """Save selected hits into the evidence basket."""
        parsed = _parse_candidate_inputs(candidates, session)
        return _tool_result(session.add_evidence_candidates, parsed)

    @function_tool(strict_mode=False)
    def list_evidence_candidates() -> dict[str, Any]:
        """Return evidence candidates already saved in this run."""
        return _tool_result(session.list_evidence_candidates)

    @function_tool(strict_mode=False)
    def mark_no_evidence_found(
        city: str,
        field: str,
        searched_source_ids: list[str],
        search_summary: str,
    ) -> dict[str, Any]:
        """Record that relevant sources were searched without usable evidence."""
        return _tool_result(
            session.mark_no_evidence_found,
            city,
            field,
            searched_source_ids,
            search_summary,
        )

    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "external_source_researcher_system.md"
    )
    return Agent(
        name="External Source Researcher",
        instructions=load_prompt(prompt_path),
        model=model,
        model_settings=settings,
        tools=[
            get_tag_options,
            list_candidate_sources,
            regex_search,
            expand_hits,
            add_evidence_candidates,
            list_evidence_candidates,
            mark_no_evidence_found,
        ],
        output_type=AgentOutputSchema(
            ExternalSourceAgentResult,
            strict_json_schema=False,
        ),
    )


def build_external_source_finalizer_agent(config: AppConfig, api_key: str) -> Agent:
    """Build a no-tool agent that finalizes claims from saved candidates."""
    model = build_openrouter_model(
        config.enrichment.model,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )
    settings = build_model_settings(
        config.enrichment.temperature,
        config.enrichment.max_output_tokens,
        reasoning_effort=config.enrichment.reasoning_effort,
    )
    settings.parallel_tool_calls = False
    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "external_source_finalizer_system.md"
    )
    return Agent(
        name="External Source Finalizer",
        instructions=load_prompt(prompt_path),
        model=model,
        model_settings=settings,
        tools=[],
        output_type=AgentOutputSchema(
            ExternalSourceAgentResult,
            strict_json_schema=False,
        ),
    )


def _finalize_from_expanded_hits(
    *,
    task: dict[str, object],
    session: ExternalSearchSession,
    finalizer: Agent,
    llm_recorder: LlmCallRecorder | None = None,
    model: str | None = None,
) -> ExternalSourceAgentResult | None:
    """Ask a compact finalizer to extract claims from already-expanded hits."""
    session.stage_expanded_hits_for_active_task()
    candidates = session.evidence_candidates_for_active_task()
    if not candidates:
        return None
    payload = {
        "task": task,
        "evidence_candidates": [
            candidate.model_dump(mode="json") for candidate in candidates
        ],
    }
    try:
        result = run_agent_sync(
            finalizer,
            json.dumps(payload, ensure_ascii=False),
            max_turns=2,
            llm_recorder=llm_recorder,
            llm_call_context=LlmCallContext(
                stage_name="enrichment",
                stage_family="enrichment",
                agent="external_source_finalizer",
                call_kind="external_source_finalization",
                model=model,
                metadata={
                    "city": task.get("city"),
                    "field": task.get("field"),
                    "candidate_count": len(candidates),
                },
            ),
        )
    except Exception:
        logger.warning(
            "External source finalizer failed for city=%s field=%s",
            task["city"],
            task["field"],
            exc_info=True,
        )
        return None
    if isinstance(result.final_output, ExternalSourceAgentResult):
        return result.final_output
    logger.warning(
        "External source finalizer returned unexpected output for city=%s field=%s",
        task["city"],
        task["field"],
    )
    return None


def _tool_result(function: Any, *args: Any) -> dict[str, Any]:
    """Run a tool function and convert errors into structured payloads."""
    try:
        result = function(*args)
    except ExternalSourceToolError as exc:
        return exc.to_dict()
    except Exception as exc:  # noqa: BLE001
        return {
            "error": {
                "code": "INVALID_TOOL_INPUT",
                "message": str(exc),
            }
        }
    if isinstance(result, list):
        return {"items": [_serialize_tool_item(item) for item in result]}
    return _serialize_tool_item(result)


def _serialize_tool_item(item: Any) -> dict[str, Any]:
    """Serialize Pydantic models and mappings for tool responses."""
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(item, dict):
        return item
    return {"value": item}


def _parse_candidate_inputs(
    candidates: list[Any],
    session: ExternalSearchSession,
) -> list[EvidenceCandidateInput]:
    """Parse model-provided candidate inputs with task-scoped fallbacks."""
    city, field = session.active_task()
    parsed: list[EvidenceCandidateInput] = []
    for candidate in candidates:
        if isinstance(candidate, str):
            parsed.append(
                EvidenceCandidateInput(
                    hit_id=candidate,
                    city=city,
                    field=field,
                    reason="Selected by the external-source researcher as relevant evidence.",
                    confidence=0.7,
                )
            )
            continue
        if isinstance(candidate, dict):
            candidate_data = {
                "hit_id": candidate.get("hit_id"),
                "city": candidate.get("city") or city,
                "field": candidate.get("field") or field,
                "reason": candidate.get("reason")
                or "Selected by the external-source researcher as relevant evidence.",
                "confidence": candidate.get("confidence", 0.7),
            }
            parsed.append(EvidenceCandidateInput.model_validate(candidate_data))
    return parsed


def _iter_external_research_tasks(
    gap_manifest: GapManifest,
    context_bundle: dict[str, Any],
    question: str,
) -> list[dict[str, object]]:
    """Build one compact LLM task per city-field gap."""
    tasks: list[dict[str, object]] = []
    ccc_context = _extract_external_ccc_context(context_bundle)
    for city_gap in gap_manifest.city_gaps:
        fields = sorted(set(city_gap.blank_fields) | set(city_gap.stale_flags))
        for field in fields:
            if field in city_gap.blank_fields:
                field_status = "blank"
            elif field in city_gap.stale_flags:
                field_status = "stale"
            else:
                field_status = "unknown"
            field_terms = _field_terms(field)
            tasks.append(
                {
                    "question": (
                        f"Find governed external-source evidence for {city_gap.city} "
                        f"field `{field}` only."
                    ),
                    "original_question": question,
                    "city": city_gap.city,
                    "field": field,
                    "field_terms": field_terms,
                    "field_years": _field_years(field),
                    "field_unit_terms": _field_unit_terms(field_terms),
                    "field_status": field_status,
                    "ccc_context": ccc_context.get(
                        (city_gap.city.casefold(), field.casefold()),
                        "",
                    ),
                }
            )
    return tasks


def _field_terms(field: str) -> list[str]:
    """Split a field name into lowercase search terms."""
    return [
        token
        for token in re.split(r"[^a-z0-9]+", field.casefold())
        if len(token) > 1
    ]


def _field_years(field: str) -> list[str]:
    """Extract target years embedded in a field name."""
    return re.findall(r"(?:19|20)\d{2}", field)


def _field_unit_terms(field_terms: list[str]) -> list[str]:
    """Infer unit hints from field-name tokens."""
    units: list[str] = []
    if "eur" in field_terms or "euro" in field_terms:
        units.extend(["EUR", "euro"])
    if "pln" in field_terms:
        units.extend(["PLN"])
    if "percent" in field_terms or "reduction" in field_terms:
        units.extend(["%", "percent", "proc"])
    if "ev" in field_terms or "charger" in field_terms or "chargers" in field_terms:
        units.extend(["charger", "charging", "station", "stacji", "ladowania"])
    return units


def _extract_external_ccc_context(
    context_bundle: dict[str, Any],
) -> dict[tuple[str, str], str]:
    """Extract optional CCC context snippets keyed by city and field."""
    enrichment = context_bundle.get("enrichment")
    if not isinstance(enrichment, dict):
        return {}
    raw_context = enrichment.get("external_ccc_context")
    if not isinstance(raw_context, list):
        return {}

    indexed: dict[tuple[str, str], str] = {}
    for item in raw_context:
        if not isinstance(item, dict):
            continue
        city = str(item.get("city", "")).strip()
        field = str(item.get("field", "")).strip()
        context = str(item.get("context", "")).strip()
        if city and field and context:
            indexed[(city.casefold(), field.casefold())] = context
    return indexed


def _extract_external_ccc_values(
    context_bundle: dict[str, Any],
) -> dict[tuple[str, str], str | float | int | None]:
    """Extract any structured CCC values available to the external resolver."""
    enrichment = context_bundle.get("enrichment")
    if not isinstance(enrichment, dict):
        return {}

    indexed: dict[tuple[str, str], str | float | int | None] = {}
    for item in _iter_context_records(enrichment.get("external_ccc_context")):
        ccc_value = _extract_ccc_value(item)
        if ccc_value is not None:
            _index_ccc_value(indexed, item, ccc_value)

    for item in _iter_context_records(enrichment.get("freshness_results")):
        ccc_value = item.get("ccc_value")
        if ccc_value is not None:
            _index_ccc_value(indexed, item, ccc_value)
    return indexed


def _iter_context_records(value: object) -> list[dict[str, Any]]:
    """Return dictionary records from a list-like context payload."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _extract_ccc_value(item: dict[str, Any]) -> str | float | int | None:
    """Read the first structured CCC value field present in a context record."""
    for key in ("ccc_value", "value", "ccc_value_extracted", "known_value"):
        value = item.get(key)
        if value is not None:
            return value
    return None


def _index_ccc_value(
    indexed: dict[tuple[str, str], str | float | int | None],
    item: dict[str, Any],
    ccc_value: str | float | int | None,
) -> None:
    """Store a structured CCC value when the record has a city-field key."""
    city = str(item.get("city", "")).strip()
    field = str(item.get("field", "")).strip()
    if city and field:
        indexed[(city.casefold(), field.casefold())] = ccc_value


def _validated_claims(
    claims: list[ExternalEvidenceClaim],
    candidates: dict[str, EvidenceCandidate],
) -> tuple[list[ExternalEvidenceClaim], list[dict[str, object]]]:
    """Keep only claims backed by saved evidence candidates."""
    validated: list[ExternalEvidenceClaim] = []
    rejected: list[dict[str, object]] = []
    for claim in claims:
        if not claim.candidate_id:
            logger.warning("External claim skipped because candidate_id is missing.")
            rejected.append(
                {
                    "city": claim.city,
                    "field": claim.field,
                    "candidate_id": None,
                    "rejection_reason": "missing_candidate_id",
                    "claim": claim.model_dump(mode="json"),
                }
            )
            continue
        candidate = candidates.get(claim.candidate_id)
        if candidate is None:
            logger.warning(
                "External claim skipped because candidate_id=%s is unknown.",
                claim.candidate_id,
            )
            rejected.append(
                {
                    "city": claim.city,
                    "field": claim.field,
                    "candidate_id": claim.candidate_id,
                    "rejection_reason": "unknown_candidate_id",
                    "claim": claim.model_dump(mode="json"),
                }
            )
            continue
        if candidate.city.casefold() != claim.city.casefold() or (
            candidate.field.casefold() != claim.field.casefold()
        ):
            logger.warning(
                "External claim skipped because candidate_id=%s belongs to %s/%s, not %s/%s.",
                claim.candidate_id,
                candidate.city,
                candidate.field,
                claim.city,
                claim.field,
            )
            rejected.append(
                {
                    "city": claim.city,
                    "field": claim.field,
                    "candidate_id": claim.candidate_id,
                    "rejection_reason": "candidate_city_field_mismatch",
                    "claim": claim.model_dump(mode="json"),
                    "candidate": candidate.model_dump(mode="json"),
                }
            )
            continue
        updated_claim = claim.model_copy(
            update={
                "source_id": candidate.source_id,
                "source_type": candidate.source_type,
                "publication_year": candidate.publication_year,
                "line_start": candidate.line_start,
                "line_end": candidate.line_end,
                "quote": candidate.quote,
                "source_url": candidate.source_url,
            }
        )
        if not _claim_contains_field_requirements(updated_claim):
            logger.warning(
                "External claim skipped because it does not satisfy field year/unit "
                "requirements for field=%s candidate_id=%s.",
                claim.field,
                claim.candidate_id,
            )
            rejected.append(
                {
                    "city": claim.city,
                    "field": claim.field,
                    "candidate_id": claim.candidate_id,
                    "rejection_reason": "field_requirements_not_satisfied",
                    "claim": updated_claim.model_dump(mode="json"),
                    "candidate": candidate.model_dump(mode="json"),
                }
            )
            continue
        validated.append(updated_claim)
    return validated, rejected


def _claim_contains_field_requirements(claim: ExternalEvidenceClaim) -> bool:
    """Require explicit year/unit evidence when the field name asks for it."""
    field_terms = _field_terms(claim.field)
    haystack = f"{claim.value} {claim.unit or ''} {claim.quote}".casefold()
    if any(year not in haystack for year in _field_years(claim.field)):
        if not _allows_implicit_infrastructure_target_year(field_terms, haystack):
            return False
    unit_terms = _field_unit_terms(field_terms)
    if unit_terms and not any(term.casefold() in haystack for term in unit_terms):
        return False
    return True


def _allows_implicit_infrastructure_target_year(field_terms: list[str], haystack: str) -> bool:
    """Allow target-year context for infrastructure milestones stated as program timing."""
    if "target" not in field_terms:
        return False
    if not {"ev", "charger", "chargers", "station", "stations"} & set(field_terms):
        return False
    has_quantity = bool(re.search(r"\d", haystack))
    has_program_timing = any(
        term in haystack
        for term in (
            "program",
            "first five",
            "pierwszych",
            "latach",
            "new",
            "nowych",
            "co najmniej",
            "at least",
        )
    )
    return has_quantity and has_program_timing


def _dedupe_claims(
    claims: list[ExternalEvidenceClaim],
) -> tuple[list[ExternalEvidenceClaim], list[dict[str, object]]]:
    """Keep the highest-confidence claim per city-field-source-line tuple."""
    indexed: dict[tuple[str, str, str, int, int], ExternalEvidenceClaim] = {}
    duplicates: list[dict[str, object]] = []
    for claim in claims:
        key = (
            claim.city.casefold(),
            claim.field.casefold(),
            claim.source_id,
            claim.line_start,
            claim.line_end,
        )
        existing = indexed.get(key)
        if existing is None:
            indexed[key] = claim
            continue
        if claim.confidence > existing.confidence:
            duplicates.append(
                {
                    "city": existing.city,
                    "field": existing.field,
                    "candidate_id": existing.candidate_id,
                    "rejection_reason": "lower_confidence_duplicate",
                    "claim": existing.model_dump(mode="json"),
                }
            )
            indexed[key] = claim
            continue
        duplicates.append(
            {
                "city": claim.city,
                "field": claim.field,
                "candidate_id": claim.candidate_id,
                "rejection_reason": "lower_confidence_duplicate",
                "claim": claim.model_dump(mode="json"),
            }
        )
    return list(indexed.values()), duplicates


def _build_external_search_audit_payload(
    *,
    session: ExternalSearchSession,
    gap_manifest: GapManifest,
    validated_claims: list[ExternalEvidenceClaim],
    resolutions: list[ExternalEvidenceResolution],
    rejected_claims: list[dict[str, object]],
    max_turn_exceeded_count: int,
    fallback_finalization_count: int,
) -> dict[str, object]:
    """Build the final external-source search audit payload."""
    candidates = session.evidence_candidates()
    tool_calls = session.tool_call_log()
    no_evidence = session.no_evidence_records()
    searched_city_fields = _gap_city_fields(gap_manifest)
    resolved_keys = {
        (resolution.city.casefold(), resolution.field.casefold()) for resolution in resolutions
    }
    used_candidate_ids = {
        claim.candidate_id for claim in validated_claims if isinstance(claim.candidate_id, str)
    }
    unused_candidates = [
        candidate.model_dump(mode="json")
        for candidate in candidates
        if candidate.candidate_id not in used_candidate_ids
    ]
    unresolved_searched_city_fields = [
        item
        for item in searched_city_fields
        if (str(item["city"]).casefold(), str(item["field"]).casefold()) not in resolved_keys
    ]
    return {
        "run_id": session.run_id,
        "audit_file": EXTERNAL_SOURCE_SEARCH_AUDIT_FILENAME,
        "searched_city_fields": searched_city_fields,
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "validated_claims": [claim.model_dump(mode="json") for claim in validated_claims],
        "rejected_claims": rejected_claims,
        "unused_candidates": unused_candidates,
        "no_evidence": [record.model_dump(mode="json") for record in no_evidence],
        "resolutions": [resolution.model_dump(mode="json") for resolution in resolutions],
        "tool_calls": tool_calls,
        "unresolved_searched_city_fields": unresolved_searched_city_fields,
        "metrics": {
            "searched_city_field_count": len(searched_city_fields),
            "candidate_count": len(candidates),
            "validated_claim_count": len(validated_claims),
            "rejected_claim_count": len(rejected_claims),
            "unused_candidate_count": len(unused_candidates),
            "no_evidence_count": len(no_evidence),
            "resolution_count": len(resolutions),
            "tool_call_count": len(tool_calls),
            "unresolved_searched_city_field_count": len(unresolved_searched_city_fields),
            "max_turn_exceeded_count": max_turn_exceeded_count,
            "fallback_finalization_count": fallback_finalization_count,
        },
    }


def _gap_city_fields(gap_manifest: GapManifest) -> list[dict[str, str]]:
    """Return unique city-field pairs searched by the external-source stage."""
    pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for city_gap in gap_manifest.city_gaps:
        for field in [*city_gap.blank_fields, *city_gap.stale_flags]:
            key = (city_gap.city.casefold(), field.casefold())
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"city": city_gap.city, "field": field})
    return pairs


__all__ = [
    "build_external_source_finalizer_agent",
    "build_external_source_research_agent",
    "run_external_source_enrichment",
]
