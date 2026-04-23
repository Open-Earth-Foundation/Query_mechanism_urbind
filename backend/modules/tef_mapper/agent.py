from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from backend.models import ErrorInfo
from backend.modules.initiative_extractor.models import InitiativeExtractionRecord
from backend.modules.tef_mapper.catalog import TefCatalog
from backend.modules.tef_mapper.models import (
    TefFinalMappingRecord,
    TefInitiativeMappingResult,
    TefMappingReviewItem,
    TefMappingRunResult,
    TefSectorRoute,
    TefSectorRouteRecord,
    TefSubsectorRoute,
    TefSubsectorRouteRecord,
    TefTransitionMapping,
    TefTransitionMappingRecord,
    TefTransitionMatch,
)
from backend.modules.tef_mapper.numeric_rollup import write_numeric_rollup_artifacts
from backend.modules.tef_mapper.rendering import (
    initiative_payload,
    json_input,
    transition_candidate_payload,
)
from backend.utils.city_normalization import normalize_city_key, normalize_city_keys
from backend.utils.config import AppConfig
from backend.utils.json_io import read_json_object, write_json
from backend.utils.prompts import load_prompt
from backend.utils.retry import RetrySettings, call_with_retries

logger = logging.getLogger(__name__)
_thread_local = threading.local()

MAPPER_VERSION = "tef_mapper_v1_json_staged"
SECTOR_PROMPT = Path("backend/prompts/tef_mapper_sector_router_system.md")
SUBSECTOR_PROMPT = Path("backend/prompts/tef_mapper_subsector_router_system.md")
TRANSITION_PROMPT = Path("backend/prompts/tef_mapper_transition_mapper_system.md")
RETRYABLE_ERROR_NAMES = {
    "APIConnectionError",
    "MaxTurnsExceeded",
    "ModelBehaviorError",
    "ValidationError",
}
STAGE_TOOL_NAMES = {
    "sector": "submit_tef_sector_route",
    "subsector": "submit_tef_subsector_route",
    "transition": "submit_tef_transition_mapping",
}


def run_agent_sync(*args: Any, **kwargs: Any) -> Any:
    """Lazy wrapper so tests can monkeypatch LLM execution without importing Agents SDK."""
    from backend.services.agents import run_agent_sync as run_sync

    return run_sync(*args, **kwargs)


def _write_jsonl(path: Path, rows: list[object]) -> None:
    """Write model or dictionary rows as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in rows:
        payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
        lines.append(json.dumps(payload, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows from disk."""
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(payload)
    return rows


def _load_initiatives(path: Path) -> list[InitiativeExtractionRecord]:
    """Load deduplicated initiative extraction records from JSONL."""
    return [InitiativeExtractionRecord.model_validate(row) for row in _read_jsonl(path)]


def _is_retryable_error(exc: Exception) -> bool:
    """Return whether a mapper failure should be retried."""
    return type(exc).__name__ in RETRYABLE_ERROR_NAMES or (
        isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc)
    )


def _get_field(value: object, key: str) -> object:
    """Read a field from a dict-like or object-like SDK payload."""
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _extract_stage_tool_output(
    result: object,
    tool_name: str,
    output_model: type[TefSectorRoute] | type[TefSubsectorRoute] | type[TefTransitionMapping],
) -> TefSectorRoute | TefSubsectorRoute | TefTransitionMapping | None:
    """Extract structured tool-call arguments from the Agents SDK raw response."""
    raw_responses = list(getattr(result, "raw_responses", []) or [])
    for response in reversed(raw_responses):
        output_items = _get_field(response, "output")
        if not isinstance(output_items, list):
            continue
        for item in reversed(output_items):
            if _get_field(item, "type") != "function_call":
                continue
            if _get_field(item, "name") != tool_name:
                continue
            arguments = _get_field(item, "arguments")
            if not isinstance(arguments, str):
                continue
            return output_model.model_validate(json.loads(arguments))
    return None


def _coerce_stage_output(
    output: object,
    output_model: type[TefSectorRoute] | type[TefSubsectorRoute] | type[TefTransitionMapping],
) -> TefSectorRoute | TefSubsectorRoute | TefTransitionMapping:
    """Coerce final output into the expected TEF stage model."""
    if isinstance(output, output_model):
        return output
    if isinstance(output, str) and output.strip().startswith("{"):
        output = json.loads(output)
    return output_model.model_validate(output)


def _build_model(config: AppConfig, api_key: str) -> object:
    """Build the configured OpenRouter-backed model for the TEF mapper."""
    from backend.services.agents import build_openrouter_model

    return build_openrouter_model(
        config.tef_mapper.model,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )


def _build_settings(config: AppConfig) -> object:
    """Build shared model settings for TEF mapper stages."""
    from backend.services.agents import build_model_settings

    return build_model_settings(
        config.tef_mapper.temperature,
        config.tef_mapper.max_output_tokens,
        reasoning_effort=config.tef_mapper.reasoning_effort,
    )


def build_sector_router_agent(config: AppConfig, api_key: str) -> object:
    """Build the sector-router agent and load only the sector prompt."""
    from agents import Agent, function_tool

    settings = _build_settings(config)
    settings.tool_choice = "submit_tef_sector_route"
    settings.parallel_tool_calls = False

    @function_tool(strict_mode=False)
    def submit_tef_sector_route(
        sector: str,
        confidence: float,
        needs_review: bool,
        rationale: str,
        alternatives: list[dict[str, Any]] | None = None,
    ) -> TefSectorRoute:
        return TefSectorRoute.model_validate(
            {
                "sector": sector,
                "confidence": confidence,
                "needs_review": needs_review,
                "rationale": rationale,
                "alternatives": alternatives or [],
            }
        )

    return Agent(
        name="TEF Sector Router",
        instructions=load_prompt(SECTOR_PROMPT),
        model=_build_model(config, api_key),
        model_settings=settings,
        tools=[submit_tef_sector_route],
        tool_use_behavior="stop_on_first_tool",
    )


def build_subsector_router_agent(config: AppConfig, api_key: str) -> object:
    """Build the subsector-router agent and load only the subsector prompt."""
    from agents import Agent, AgentOutputSchema, function_tool

    settings = _build_settings(config)
    settings.tool_choice = "submit_tef_subsector_route"
    settings.parallel_tool_calls = False

    @function_tool(strict_mode=False)
    def submit_tef_subsector_route(
        selected_path: str,
        confidence: float,
        needs_review: bool,
        rationale: str,
        alternatives: list[dict[str, Any]] | None = None,
    ) -> TefSubsectorRoute:
        return TefSubsectorRoute.model_validate(
            {
                "selected_path": selected_path,
                "confidence": confidence,
                "needs_review": needs_review,
                "rationale": rationale,
                "alternatives": alternatives or [],
            }
        )

    return Agent(
        name="TEF Subsector Router",
        instructions=load_prompt(SUBSECTOR_PROMPT),
        model=_build_model(config, api_key),
        model_settings=settings,
        tools=[submit_tef_subsector_route],
        output_type=AgentOutputSchema(TefSubsectorRoute, strict_json_schema=False),
        tool_use_behavior="stop_on_first_tool",
    )


def build_transition_mapper_agent(config: AppConfig, api_key: str) -> object:
    """Build the transition-mapper agent and load only the transition prompt."""
    from agents import Agent, AgentOutputSchema, function_tool

    settings = _build_settings(config)
    settings.tool_choice = "submit_tef_transition_mapping"
    settings.parallel_tool_calls = False

    @function_tool(strict_mode=False)
    def submit_tef_transition_mapping(
        needs_review: bool,
        matches: list[dict[str, Any]] | None = None,
    ) -> TefTransitionMapping:
        return TefTransitionMapping.model_validate(
            {
                "needs_review": needs_review,
                "matches": matches or [],
            }
        )

    return Agent(
        name="TEF Transition Mapper",
        instructions=load_prompt(TRANSITION_PROMPT),
        model=_build_model(config, api_key),
        model_settings=settings,
        tools=[submit_tef_transition_mapping],
        output_type=AgentOutputSchema(TefTransitionMapping, strict_json_schema=False),
        tool_use_behavior="stop_on_first_tool",
    )


def _get_thread_agent(stage: str, config: AppConfig, api_key: str) -> object:
    """Return a thread-local stage agent."""
    cache = getattr(_thread_local, "tef_mapper_agents", None)
    if cache is None:
        cache = {}
        _thread_local.tef_mapper_agents = cache
    if stage not in cache:
        if stage == "sector":
            cache[stage] = build_sector_router_agent(config, api_key)
        elif stage == "subsector":
            cache[stage] = build_subsector_router_agent(config, api_key)
        elif stage == "transition":
            cache[stage] = build_transition_mapper_agent(config, api_key)
        else:
            raise ValueError(f"Unsupported TEF mapper stage: {stage}")
    return cache[stage]


def _retry_settings(config: AppConfig) -> RetrySettings:
    """Build bounded retry settings from shared app config."""
    return RetrySettings.bounded(
        max_attempts=config.retry.max_attempts,
        backoff_base_seconds=config.retry.backoff_base_seconds,
        backoff_max_seconds=config.retry.backoff_max_seconds,
    )


def _run_stage(
    *,
    stage: str,
    payload: dict[str, Any],
    output_model: type[TefSectorRoute] | type[TefSubsectorRoute] | type[TefTransitionMapping],
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool,
    run_id: str,
    initiative_record_id: str,
) -> TefSectorRoute | TefSubsectorRoute | TefTransitionMapping:
    """Run one LLM stage with retries and structured output validation."""
    agent = _get_thread_agent(stage, config, api_key)

    def call() -> TefSectorRoute | TefSubsectorRoute | TefTransitionMapping:
        result = run_agent_sync(
            agent,
            json_input(payload),
            max_turns=max(config.tef_mapper.max_turns, 1),
            log_llm_payload=log_llm_payload,
        )
        return _extract_stage_tool_output(
            result,
            STAGE_TOOL_NAMES[stage],
            output_model,
        ) or _coerce_stage_output(result.final_output, output_model)

    return call_with_retries(
        call,
        operation=f"tef_mapper.{stage}",
        retry_settings=_retry_settings(config),
        should_retry=_is_retryable_error,
        run_id=run_id,
        context={"initiative_record_id": initiative_record_id},
    )


def _build_sector_payload(
    record: InitiativeExtractionRecord,
    catalog: TefCatalog,
) -> dict[str, Any]:
    """Build the sector pass payload without subsectors or Transition Elements."""
    return {
        "initiative": initiative_payload(record),
        "sectors": catalog.sector_cards(),
    }


def _build_subsector_payload(
    record: InitiativeExtractionRecord,
    catalog: TefCatalog,
    parent_path: str,
) -> dict[str, Any]:
    """Build the category-routing payload with only direct child categories."""
    return {
        "initiative": initiative_payload(record),
        "selected_category": catalog.category_payload(parent_path),
        "candidate_subcategories": catalog.subsector_cards(parent_path),
    }


def _build_transition_payload(
    record: InitiativeExtractionRecord,
    catalog: TefCatalog,
    selected_path: str,
) -> dict[str, Any]:
    """Build the transition pass payload with only direct Transition Elements."""
    candidates = catalog.transition_elements(selected_path)
    return {
        "initiative": initiative_payload(record),
        "selected_category": catalog.category_payload(selected_path),
        "candidate_transition_elements": transition_candidate_payload(candidates),
    }


def _close_alternatives(alternatives: list[Any], confidence: float, delta: float) -> bool:
    """Return whether any alternative is close enough to require review."""
    return any((confidence - item.confidence) <= delta for item in alternatives)


def _route_needs_review(
    *,
    needs_review: bool,
    confidence: float,
    alternatives: list[Any],
    config: AppConfig,
) -> bool:
    """Apply shared confidence and close-alternative review rules."""
    return (
        needs_review
        or confidence < config.tef_mapper.review_confidence_threshold
        or _close_alternatives(alternatives, confidence, config.tef_mapper.close_alternative_delta)
    )


def _review_item(
    review_type: str,
    message: str,
    record: InitiativeExtractionRecord,
    *,
    severity: Literal["info", "warning", "error"] = "warning",
    target_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> TefMappingReviewItem:
    """Build one manual-review item for an initiative."""
    return TefMappingReviewItem(
        review_type=review_type,
        severity=severity,
        message=message,
        initiative_record_id=record.record_id,
        source_document=record.source_document,
        target_id=target_id,
        details=details or {},
    )


def _validate_sector_route(route: TefSectorRoute, catalog: TefCatalog) -> None:
    """Validate sector route against the current sector catalog slice."""
    expected_path = catalog.sector_path(route.sector)
    if route.selected_path != expected_path:
        raise ValueError(
            f"Sector route selected_path {route.selected_path!r} does not match "
            f"sector {route.sector!r} path {expected_path!r}."
        )
    for alternative in route.alternatives:
        expected_alternative_path = catalog.sector_path(alternative.sector)
        if alternative.path != expected_alternative_path:
            raise ValueError(
                f"Sector alternative path {alternative.path!r} does not match "
                f"sector {alternative.sector!r} path {expected_alternative_path!r}."
            )


def _hydrate_sector_route(route: TefSectorRoute, catalog: TefCatalog) -> TefSectorRoute:
    """Assign catalog paths for the selected sector and sector alternatives."""
    alternatives = [
        alternative.model_copy(update={"path": catalog.sector_path(alternative.sector)})
        for alternative in route.alternatives
    ]
    return route.model_copy(
        update={
            "selected_path": catalog.sector_path(route.sector),
            "alternatives": alternatives,
        }
    )


def _validate_subsector_route(route: TefSubsectorRoute, candidate_paths: set[str]) -> None:
    """Validate subsector route against the direct child subsector slice."""
    if route.selected_path not in candidate_paths:
        raise ValueError(f"Unknown subsector selected_path: {route.selected_path}")
    for alternative in route.alternatives:
        if alternative.path not in candidate_paths:
            raise ValueError(f"Unknown subsector alternative path: {alternative.path}")


def _direct_candidate_for_path(path: str, candidate_paths: set[str]) -> str | None:
    """Return the direct candidate path matching a returned path or its descendant."""
    if path in candidate_paths:
        return path
    matches = [candidate for candidate in candidate_paths if path.startswith(f"{candidate}/")]
    if len(matches) == 1:
        return matches[0]
    return None


def _normalize_subsector_route(
    route: TefSubsectorRoute,
    candidate_paths: set[str],
) -> tuple[TefSubsectorRoute, list[str]]:
    """Normalize descendant paths back to the current direct-candidate slice."""
    selected_path = _direct_candidate_for_path(route.selected_path, candidate_paths)
    if selected_path is None:
        raise ValueError(f"Unknown subsector selected_path: {route.selected_path}")

    notes: list[str] = []
    if selected_path != route.selected_path:
        notes.append(
            f"Normalized selected_path {route.selected_path!r} to direct child {selected_path!r}."
        )

    alternatives = []
    for alternative in route.alternatives:
        alternative_path = _direct_candidate_for_path(alternative.path, candidate_paths)
        if alternative_path is None:
            notes.append(f"Discarded invalid alternative path {alternative.path!r}.")
            continue
        if alternative_path != alternative.path:
            notes.append(
                f"Normalized alternative path {alternative.path!r} to direct child "
                f"{alternative_path!r}."
            )
        alternatives.append(alternative.model_copy(update={"path": alternative_path}))

    normalized_route = route.model_copy(
        update={"selected_path": selected_path, "alternatives": alternatives}
    )
    return normalized_route, notes


def _validate_transition_mapping(
    mapping: TefTransitionMapping,
    candidate_tef_ids: set[str],
) -> None:
    """Validate transition mapping against the direct candidate Transition Elements."""
    for match in mapping.matches:
        if match.tef_id not in candidate_tef_ids:
            raise ValueError(f"Unknown Transition Element tef_id: {match.tef_id}")


def _final_mapping_base(
    record: InitiativeExtractionRecord,
    sector_route: TefSectorRoute,
    subsector_routes: list[TefSubsectorRouteRecord],
    catalog: TefCatalog,
    extraction_run_id: str | None,
) -> dict[str, Any]:
    """Build shared final mapping fields."""
    return {
        "initiative_record_id": record.record_id,
        "city": record.initiative.city,
        "source_document": record.source_document,
        "document_local_code": record.document_local_code,
        "initiative_name": record.initiative.initiative_name,
        "source_quote": record.source_quote,
        "sector_route": sector_route.model_dump(mode="json"),
        "subsector_routes": [
            route_record.model_dump(mode="json") for route_record in subsector_routes
        ],
        "mapper_version": MAPPER_VERSION,
        "tef_source_version": catalog.source_version,
        "extraction_run_id": extraction_run_id,
    }


def _build_transition_final_mappings(
    *,
    record: InitiativeExtractionRecord,
    sector_route: TefSectorRoute,
    subsector_routes: list[TefSubsectorRouteRecord],
    selected_path: str,
    matches: list[TefTransitionMatch],
    route_review: bool,
    min_transition_confidence: float,
    catalog: TefCatalog,
    extraction_run_id: str | None,
) -> list[TefFinalMappingRecord]:
    """Build final mapping rows for Transition Element matches."""
    base = _final_mapping_base(
        record,
        sector_route,
        subsector_routes,
        catalog,
        extraction_run_id,
    )
    return [
        TefFinalMappingRecord(
            **base,
            target_type="transition_element",
            target_id=match.tef_id,
            target_path=selected_path,
            confidence=match.confidence,
            is_primary=match.is_primary,
            needs_review=route_review or match.confidence < min_transition_confidence,
            rationale=match.rationale,
        )
        for match in matches
    ]


def _build_subcategory_final_mapping(
    *,
    record: InitiativeExtractionRecord,
    sector_route: TefSectorRoute,
    subsector_routes: list[TefSubsectorRouteRecord],
    selected_path: str,
    confidence: float,
    rationale: str,
    catalog: TefCatalog,
    extraction_run_id: str | None,
) -> TefFinalMappingRecord:
    """Build a reviewed final mapping row that targets the selected TEF subcategory."""
    base = _final_mapping_base(
        record,
        sector_route,
        subsector_routes,
        catalog,
        extraction_run_id,
    )
    return TefFinalMappingRecord(
        **base,
        target_type="subcategory",
        target_id=selected_path,
        target_path=selected_path,
        confidence=confidence,
        is_primary=True,
        needs_review=True,
        rationale=rationale,
    )


def _map_one_initiative(
    *,
    record: InitiativeExtractionRecord,
    catalog: TefCatalog,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool,
    run_id: str,
    extraction_run_id: str | None,
) -> TefInitiativeMappingResult:
    """Run all required TEF mapping passes for one initiative."""
    review_items: list[TefMappingReviewItem] = []
    try:
        sector_payload = _build_sector_payload(record, catalog)
        sector_route = _run_stage(
            stage="sector",
            payload=sector_payload,
            output_model=TefSectorRoute,
            config=config,
            api_key=api_key,
            log_llm_payload=log_llm_payload,
            run_id=run_id,
            initiative_record_id=record.record_id,
        )
        assert isinstance(sector_route, TefSectorRoute)
        sector_route = _hydrate_sector_route(sector_route, catalog)
        _validate_sector_route(sector_route, catalog)
        sector_route_record = TefSectorRouteRecord(
            initiative_record_id=record.record_id,
            source_document=record.source_document,
            status="success",
            route=sector_route,
        )
        route_review = _route_needs_review(
            needs_review=sector_route.needs_review,
            confidence=sector_route.confidence,
            alternatives=sector_route.alternatives,
            config=config,
        )
        if route_review:
            review_items.append(
                _review_item(
                    "sector_route_needs_review",
                    "Sector routing confidence or alternatives require manual review.",
                    record,
                    details={"route": sector_route.model_dump(mode="json")},
                )
            )
        for flag in record.data_quality_flags:
            review_items.append(
                _review_item(
                    "source_quality_flag",
                    f"Initiative extraction has source quality flag: {flag}",
                    record,
                    severity="info",
                    details={"flag": flag},
                )
            )

        current_path = sector_route.selected_path
        subsector_records: list[TefSubsectorRouteRecord] = []
        selected_confidence = sector_route.confidence
        selected_rationale = sector_route.rationale

        while True:
            children = catalog.child_subsectors(current_path)
            if children:
                subsector_payload = _build_subsector_payload(record, catalog, current_path)
                candidate_paths = {child.path for child in children}
                subsector_route = _run_stage(
                    stage="subsector",
                    payload=subsector_payload,
                    output_model=TefSubsectorRoute,
                    config=config,
                    api_key=api_key,
                    log_llm_payload=log_llm_payload,
                    run_id=run_id,
                    initiative_record_id=record.record_id,
                )
                assert isinstance(subsector_route, TefSubsectorRoute)
                subsector_route, normalization_notes = _normalize_subsector_route(
                    subsector_route,
                    candidate_paths,
                )
                _validate_subsector_route(subsector_route, candidate_paths)
                if normalization_notes:
                    review_items.append(
                        _review_item(
                            "subsector_route_path_normalized",
                            "Subsector route returned a descendant or invalid path for the current pass.",
                            record,
                            target_id=subsector_route.selected_path,
                            details={
                                "parent_path": current_path,
                                "candidate_paths": sorted(candidate_paths),
                                "notes": normalization_notes,
                            },
                        )
                    )
                subsector_record = TefSubsectorRouteRecord(
                    initiative_record_id=record.record_id,
                    source_document=record.source_document,
                    parent_path=current_path,
                    candidate_paths=sorted(candidate_paths),
                    status="success",
                    route=subsector_route,
                )
                subsector_records.append(subsector_record)
                selected_confidence = subsector_route.confidence
                selected_rationale = subsector_route.rationale
                if _route_needs_review(
                    needs_review=subsector_route.needs_review,
                    confidence=subsector_route.confidence,
                    alternatives=subsector_route.alternatives,
                    config=config,
                ):
                    route_review = True
                    review_items.append(
                        _review_item(
                            "subsector_route_needs_review",
                            "Subsector routing confidence or alternatives require manual review.",
                            record,
                            target_id=subsector_route.selected_path,
                            details={"route": subsector_route.model_dump(mode="json")},
                        )
                    )
                current_path = subsector_route.selected_path
                continue

            direct_transitions = catalog.transition_elements(current_path)
            if direct_transitions:
                transition_payload = _build_transition_payload(record, catalog, current_path)
                transition_mapping = _run_stage(
                    stage="transition",
                    payload=transition_payload,
                    output_model=TefTransitionMapping,
                    config=config,
                    api_key=api_key,
                    log_llm_payload=log_llm_payload,
                    run_id=run_id,
                    initiative_record_id=record.record_id,
                )
                assert isinstance(transition_mapping, TefTransitionMapping)
                candidate_tef_ids = {item.tef_id for item in direct_transitions}
                _validate_transition_mapping(transition_mapping, candidate_tef_ids)
                transition_record = TefTransitionMappingRecord(
                    initiative_record_id=record.record_id,
                    source_document=record.source_document,
                    selected_path=current_path,
                    candidate_tef_ids=sorted(candidate_tef_ids),
                    status="success",
                    mapping=transition_mapping,
                )
                transition_review = route_review or transition_mapping.needs_review
                if not transition_mapping.matches:
                    review_items.append(
                        _review_item(
                            "no_transition_match",
                            "Transition mapper returned no TEF Transition Element matches.",
                            record,
                            target_id=current_path,
                        )
                    )
                if transition_mapping.needs_review:
                    review_items.append(
                        _review_item(
                            "transition_mapping_needs_review",
                            "Transition mapper marked this initiative for manual review.",
                            record,
                            target_id=current_path,
                            details={"mapping": transition_mapping.model_dump(mode="json")},
                        )
                    )
                for match in transition_mapping.matches:
                    if match.confidence < config.tef_mapper.min_transition_confidence:
                        review_items.append(
                            _review_item(
                                "low_transition_confidence",
                                "Transition Element match confidence is below the configured threshold.",
                                record,
                                target_id=match.tef_id,
                                details={"confidence": match.confidence},
                            )
                        )
                final_mappings = _build_transition_final_mappings(
                    record=record,
                    sector_route=sector_route,
                    subsector_routes=subsector_records,
                    selected_path=current_path,
                    matches=transition_mapping.matches,
                    route_review=transition_review,
                    min_transition_confidence=config.tef_mapper.min_transition_confidence,
                    catalog=catalog,
                    extraction_run_id=extraction_run_id,
                )
                if not final_mappings:
                    final_mappings = [
                        _build_subcategory_final_mapping(
                            record=record,
                            sector_route=sector_route,
                            subsector_routes=subsector_records,
                            selected_path=current_path,
                            confidence=selected_confidence,
                            rationale=(
                                f"{selected_rationale} The transition mapper returned no "
                                "exact Transition Element match, so the initiative is mapped "
                                "to the selected TEF category."
                            ),
                            catalog=catalog,
                            extraction_run_id=extraction_run_id,
                        )
                    ]
                return TefInitiativeMappingResult(
                    initiative_record_id=record.record_id,
                    source_document=record.source_document,
                    status="success",
                    sector_route_record=sector_route_record,
                    subsector_route_records=subsector_records,
                    transition_mapping_record=transition_record,
                    final_mappings=final_mappings,
                    review_items=review_items,
                )

            final_mapping = _build_subcategory_final_mapping(
                record=record,
                sector_route=sector_route,
                subsector_routes=subsector_records,
                selected_path=current_path,
                confidence=selected_confidence,
                rationale=(
                    f"{selected_rationale} The TEF catalog has no Transition "
                    "Elements for this category, so the initiative is mapped to "
                    "the selected category."
                ),
                catalog=catalog,
                extraction_run_id=extraction_run_id,
            )
            review_items.append(
                _review_item(
                    "subcategory_final_without_transitions",
                    "Selected TEF category has no Transition Elements; mapped to the category.",
                    record,
                    target_id=current_path,
                )
            )
            return TefInitiativeMappingResult(
                initiative_record_id=record.record_id,
                source_document=record.source_document,
                status="success",
                sector_route_record=sector_route_record,
                subsector_route_records=subsector_records,
                final_mappings=[final_mapping],
                review_items=review_items,
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("TEF mapping failed for initiative %s", record.record_id)
        error = ErrorInfo(
            code="TEF_MAPPING_FAILED",
            message="TEF mapping failed for this initiative.",
            details=[str(exc)],
        )
        return TefInitiativeMappingResult(
            initiative_record_id=record.record_id,
            source_document=record.source_document,
            status="error",
            review_items=[
                _review_item(
                    "initiative_mapping_failed",
                    "TEF mapping failed for this initiative.",
                    record,
                    severity="error",
                    details={"error": str(exc)},
                )
            ],
            error=error,
        )


def _extract_run_id(extraction_run_dir: Path | None) -> str | None:
    """Read extraction run id from source manifest when available."""
    if extraction_run_dir is None:
        return None
    manifest = read_json_object(extraction_run_dir / "00_source" / "source_manifest.json")
    if manifest and isinstance(manifest.get("run_id"), str):
        return str(manifest["run_id"])
    return extraction_run_dir.name


def _resolve_initiatives_path(
    extraction_run_dir: Path | None,
    initiatives_jsonl: Path | None,
) -> Path:
    """Resolve the initiative extraction records JSONL input path."""
    if initiatives_jsonl is not None:
        return initiatives_jsonl
    if extraction_run_dir is None:
        raise ValueError("Either extraction_run_dir or initiatives_jsonl is required.")
    return extraction_run_dir / "03_deduped" / "initiative_records.jsonl"


def _filter_initiatives(
    records: list[InitiativeExtractionRecord],
    selected_cities: list[str] | None,
    limit: int | None,
) -> list[InitiativeExtractionRecord]:
    """Apply city and limit filters to input initiative records."""
    city_keys = set(normalize_city_keys(selected_cities))
    filtered = [
        record
        for record in records
        if not city_keys or normalize_city_key(record.initiative.city) in city_keys
    ]
    if limit is not None:
        return filtered[: max(limit, 0)]
    return filtered


def _write_run_artifacts(
    *,
    run_dir: Path,
    run_id: str,
    initiatives_path: Path,
    extraction_run_id: str | None,
    input_records: list[InitiativeExtractionRecord],
    results: list[TefInitiativeMappingResult],
    config: AppConfig,
    catalog: TefCatalog,
) -> None:
    """Persist all TEF mapping artifacts for a run."""
    sector_records = [
        result.sector_route_record
        for result in results
        if result.sector_route_record is not None
    ]
    subsector_records = [
        record
        for result in results
        for record in result.subsector_route_records
    ]
    transition_records = [
        result.transition_mapping_record
        for result in results
        if result.transition_mapping_record is not None
    ]
    final_mappings = [
        mapping
        for result in results
        for mapping in result.final_mappings
    ]
    review_items = [
        item
        for result in results
        for item in result.review_items
    ]
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "initiatives_path": str(initiatives_path),
        "extraction_run_id": extraction_run_id,
        "tef_catalog_root": str(catalog.root),
        "tef_source_version": catalog.source_version,
        "mapper_version": MAPPER_VERSION,
        "model": config.tef_mapper.model,
        "review_confidence_threshold": config.tef_mapper.review_confidence_threshold,
    }
    summary = {
        "run_id": run_id,
        "initiatives_count": len(input_records),
        "mapped_initiatives_count": sum(1 for result in results if result.final_mappings),
        "final_mappings_count": len(final_mappings),
        "review_items_count": len(review_items),
        "error_count": sum(1 for result in results if result.status == "error"),
    }
    write_json(run_dir / "00_source" / "source_manifest.json", manifest, ensure_ascii=False)
    _write_jsonl(run_dir / "01_inputs" / "initiatives.jsonl", input_records)
    _write_jsonl(run_dir / "02_sector_routes" / "sector_routes.jsonl", sector_records)
    _write_jsonl(run_dir / "03_subsector_routes" / "subsector_routes.jsonl", subsector_records)
    _write_jsonl(
        run_dir / "04_transition_mappings" / "transition_mappings.jsonl",
        transition_records,
    )
    _write_jsonl(run_dir / "05_final_mappings" / "final_mappings.jsonl", final_mappings)
    _write_jsonl(run_dir / "06_review" / "review_items.jsonl", review_items)
    write_numeric_rollup_artifacts(
        run_dir=run_dir,
        run_id=run_id,
        extraction_run_id=extraction_run_id,
        initiative_records=input_records,
        final_mappings=final_mappings,
    )
    write_json(run_dir / "summary.json", summary, ensure_ascii=False)
    (run_dir / "README.md").write_text(
        "\n".join(
            [
                "# TEF Mapping Run",
                "",
                "This folder contains JSON-only staged TEF mapping artifacts.",
                "",
                "- `00_source/source_manifest.json`: source run and mapper settings.",
                "- `01_inputs/initiatives.jsonl`: initiative extraction records mapped in this run, including source quotes for traceability.",
                "- `02_sector_routes/sector_routes.jsonl`: sector-routing outputs.",
                "- `03_subsector_routes/subsector_routes.jsonl`: recursive subsector-routing outputs.",
                "- `04_transition_mappings/transition_mappings.jsonl`: Transition Element mapper outputs.",
                "- `05_final_mappings/final_mappings.jsonl`: durable final mappings with copied source quotes.",
                "- `06_review/review_items.jsonl`: manual-review flags.",
                "- `07_numeric_facts/initiative_numeric_facts.jsonl`: clean v1 initiative numbers joined to TEF mappings with copied source quotes.",
                "- `08_tef_groups/tef_grouped_initiatives.jsonl`: initiatives grouped by TEF target with copied source quotes.",
                "- `08_tef_groups/tef_metric_rollups.json`: additive metric rollups by TEF target.",
                "- `summary.json`: run counts.",
                "",
                "No database writes or LLM ambiguity-review pass are performed.",
            ]
        ),
        encoding="utf-8",
    )


def map_initiatives_to_tef(
    *,
    config: AppConfig,
    api_key: str,
    tef_catalog_dir: Path,
    output_root: Path,
    extraction_run_dir: Path | None = None,
    initiatives_jsonl: Path | None = None,
    run_id: str | None = None,
    selected_cities: list[str] | None = None,
    limit: int | None = None,
    max_workers: int | None = None,
    log_llm_payload: bool = False,
) -> TefMappingRunResult:
    """Run JSON-only staged TEF mapping over extracted initiative records."""
    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / resolved_run_id
    initiatives_path = _resolve_initiatives_path(extraction_run_dir, initiatives_jsonl)
    extraction_run_id = _extract_run_id(extraction_run_dir)
    catalog = TefCatalog(tef_catalog_dir)
    records = _filter_initiatives(
        _load_initiatives(initiatives_path),
        selected_cities,
        limit,
    )
    configured_workers = max_workers or config.tef_mapper.max_workers
    worker_count = min(max(configured_workers, 1), max(len(records), 1))
    logger.info(
        "Starting TEF mapping run_id=%s initiatives=%d workers=%d",
        resolved_run_id,
        len(records),
        worker_count,
    )

    results: list[TefInitiativeMappingResult] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _map_one_initiative,
                record=record,
                catalog=catalog,
                config=config,
                api_key=api_key,
                log_llm_payload=log_llm_payload,
                run_id=resolved_run_id,
                extraction_run_id=extraction_run_id,
            )
            for record in records
        ]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda result: result.initiative_record_id)
    _write_run_artifacts(
        run_dir=run_dir,
        run_id=resolved_run_id,
        initiatives_path=initiatives_path,
        extraction_run_id=extraction_run_id,
        input_records=records,
        results=results,
        config=config,
        catalog=catalog,
    )
    final_mappings_count = sum(len(result.final_mappings) for result in results)
    review_items_count = sum(len(result.review_items) for result in results)
    return TefMappingRunResult(
        run_id=resolved_run_id,
        output_dir=str(run_dir),
        initiatives_count=len(records),
        mapped_initiatives_count=sum(1 for result in results if result.final_mappings),
        final_mappings_count=final_mappings_count,
        review_items_count=review_items_count,
    )


__all__ = [
    "MAPPER_VERSION",
    "build_sector_router_agent",
    "build_subsector_router_agent",
    "build_transition_mapper_agent",
    "map_initiatives_to_tef",
]
