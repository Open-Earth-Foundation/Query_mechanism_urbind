"""LLM planning and extraction plus deterministic aggregation for calculator runs."""

from __future__ import annotations

import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from agents import Agent, function_tool
from agents.exceptions import MaxTurnsExceeded

from backend.modules.calculator.models import (
    CalculationCategory,
    CalculationCategorySummary,
    CalculationGroupSummary,
    CalculationPlan,
    CalculationRecord,
    CalculationRunSummary,
    CalculationWorkerOutput,
)
from backend.services.agents import (
    build_model_settings,
    build_openrouter_model,
    run_agent_sync,
)
from backend.utils.city_normalization import (
    dedupe_city_labels,
    format_city_display_name,
    normalize_city_key,
)
from backend.utils.config import AppConfig
from backend.utils.json_io import write_json
from backend.utils.prompts import load_prompt

logger = logging.getLogger(__name__)

_NUMERIC_PATTERN = re.compile(r"\d")
_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_CATEGORY_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "into",
    "this",
    "have",
    "has",
    "had",
    "year",
    "years",
    "city",
    "cities",
    "total",
    "totals",
    "across",
    "amount",
    "amounts",
    "value",
    "values",
    "count",
    "counts",
    "group",
    "groups",
    "metric",
    "metrics",
}
_FINANCIAL_CATEGORY_TOKENS = {
    "budget",
    "budgets",
    "capex",
    "cost",
    "costs",
    "fund",
    "funding",
    "funds",
    "investment",
    "investments",
    "spend",
    "spending",
    "subsidy",
    "subsidies",
}
_PLANNER_MAX_EXCERPTS = 200
_CATEGORY_MAX_CANDIDATE_EXCERPTS = 300


@dataclass
class _CategoryExecution:
    """Internal execution payload for one calculator category worker."""

    category: CalculationCategory
    status: str
    note: str
    pass_count: int
    stop_reason: str
    records: list[CalculationRecord]
    pass_files: list[Path]
    records_path: Path
    error_message: str | None = None


def _calculator_prompts_dir() -> Path:
    """Return the prompts directory used by calculator agents."""
    return Path(__file__).resolve().parents[2] / "prompts"


def _normalize_excerpt_records(
    context_bundle: dict[str, object],
) -> list[dict[str, object]]:
    """Return normalized excerpt dictionaries from the markdown bundle."""
    markdown_payload = context_bundle.get("markdown")
    if not isinstance(markdown_payload, dict):
        return []
    excerpts = markdown_payload.get("excerpts")
    if not isinstance(excerpts, list):
        return []
    return [excerpt for excerpt in excerpts if isinstance(excerpt, dict)]


def _selected_city_names(context_bundle: dict[str, object]) -> list[str]:
    """Return the selected city names relevant to calculator coverage."""
    markdown_payload = context_bundle.get("markdown")
    if not isinstance(markdown_payload, dict):
        return []
    selected = markdown_payload.get("selected_city_names")
    if isinstance(selected, list):
        return dedupe_city_labels([str(item) for item in selected if str(item).strip()])
    inspected = markdown_payload.get("inspected_city_names")
    if isinstance(inspected, list):
        return dedupe_city_labels([str(item) for item in inspected if str(item).strip()])
    excerpt_names = [
        str(excerpt.get("city_name", "")).strip()
        for excerpt in _normalize_excerpt_records(context_bundle)
        if str(excerpt.get("city_name", "")).strip()
    ]
    return dedupe_city_labels(excerpt_names)


def _excerpt_has_numeric_signal(excerpt: dict[str, object]) -> bool:
    """Return whether the excerpt contains any obvious numeric signal."""
    quote = str(excerpt.get("quote", ""))
    partial_answer = str(excerpt.get("partial_answer", ""))
    return bool(_NUMERIC_PATTERN.search(f"{quote} {partial_answer}"))


def _planner_excerpts(context_bundle: dict[str, object]) -> list[dict[str, object]]:
    """Return compact planner excerpts prioritized for numeric planning."""
    excerpts = [
        {
            "ref_id": str(excerpt.get("ref_id", "")).strip(),
            "city_name": str(excerpt.get("city_name", "")).strip(),
            "quote": str(excerpt.get("quote", "")).strip(),
            "partial_answer": str(excerpt.get("partial_answer", "")).strip(),
            "source_chunk_ids": list(excerpt.get("source_chunk_ids", []))
            if isinstance(excerpt.get("source_chunk_ids"), list)
            else [],
        }
        for excerpt in _normalize_excerpt_records(context_bundle)
        if _excerpt_has_numeric_signal(excerpt)
    ]
    return excerpts[:_PLANNER_MAX_EXCERPTS]


def _category_tokens(category: CalculationCategory) -> set[str]:
    """Return compact category tokens for excerpt scoring."""
    source_text = " ".join(
        [
            category.category_key.replace("_", " "),
            category.label,
            category.description,
            category.inclusion_rule,
            category.preferred_unit,
        ]
    ).lower()
    return {
        token
        for token in _WORD_PATTERN.findall(source_text)
        if len(token) >= 3 and token not in _CATEGORY_STOPWORDS
    }


def _score_category_excerpt(
    excerpt: dict[str, object],
    category_tokens: set[str],
) -> tuple[int, str, str]:
    """Return a deterministic sort key for category excerpt ranking."""
    body = (
        f"{excerpt.get('quote', '')} {excerpt.get('partial_answer', '')}"
    ).strip()
    body_tokens = set(_WORD_PATTERN.findall(body.lower()))
    overlap = len(category_tokens & body_tokens)
    city_name = str(excerpt.get("city_name", "")).strip().casefold()
    ref_id = str(excerpt.get("ref_id", "")).strip()
    return overlap, city_name, ref_id


def _slice_category_excerpts(
    category: CalculationCategory,
    context_bundle: dict[str, object],
    max_passes: int,
) -> list[list[dict[str, object]]]:
    """Split category-relevant excerpts into deterministic pass slices."""
    numeric_excerpts = [
        excerpt
        for excerpt in _normalize_excerpt_records(context_bundle)
        if _excerpt_has_numeric_signal(excerpt)
    ]
    if not numeric_excerpts or max_passes <= 0:
        return []
    category_tokens = _category_tokens(category)
    scored = sorted(
        numeric_excerpts,
        key=lambda excerpt: _score_category_excerpt(excerpt, category_tokens),
        reverse=True,
    )
    if category_tokens:
        positive_overlap = [
            excerpt
            for excerpt in scored
            if _score_category_excerpt(excerpt, category_tokens)[0] > 0
        ]
        if positive_overlap:
            scored = positive_overlap
    limited = scored[:_CATEGORY_MAX_CANDIDATE_EXCERPTS]
    chunk_size = max(math.ceil(len(limited) / max_passes), 1)
    return [
        limited[index : index + chunk_size]
        for index in range(0, len(limited), chunk_size)
    ]


def build_calculator_planner_agent(config: AppConfig, api_key: str) -> Agent:
    """Build the calculator category planner agent."""
    instructions = load_prompt(
        _calculator_prompts_dir() / "calculator_plan_system.md"
    )
    model = build_openrouter_model(
        config.calculator.model,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )
    settings = build_model_settings(
        config.calculator.temperature,
        config.calculator.max_output_tokens,
        reasoning_effort=config.calculator.reasoning_effort,
    )
    settings.tool_choice = "submit_calculation_plan"
    settings.parallel_tool_calls = False

    @function_tool(strict_mode=True)
    def submit_calculation_plan(result: CalculationPlan) -> CalculationPlan:
        """Return a structured calculator category plan unchanged."""
        return result

    return Agent(
        name="Calculator Planner",
        instructions=instructions,
        model=model,
        model_settings=settings,
        tools=[submit_calculation_plan],
        output_type=CalculationPlan,
        tool_use_behavior="stop_on_first_tool",
    )


def build_calculator_worker_agent(config: AppConfig, api_key: str) -> Agent:
    """Build the calculator category worker agent."""
    instructions = load_prompt(
        _calculator_prompts_dir() / "calculator_worker_system.md"
    )
    model = build_openrouter_model(
        config.calculator.model,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )
    settings = build_model_settings(
        config.calculator.temperature,
        config.calculator.max_output_tokens,
        reasoning_effort=config.calculator.reasoning_effort,
    )
    settings.tool_choice = "submit_calculation_worker_output"
    settings.parallel_tool_calls = False

    @function_tool(strict_mode=True)
    def submit_calculation_worker_output(
        result: CalculationWorkerOutput,
    ) -> CalculationWorkerOutput:
        """Return one structured calculator worker output unchanged."""
        return result

    return Agent(
        name="Calculator Worker",
        instructions=instructions,
        model=model,
        model_settings=settings,
        tools=[submit_calculation_worker_output],
        output_type=CalculationWorkerOutput,
        tool_use_behavior="stop_on_first_tool",
    )


def plan_categories(
    question: str,
    context_bundle: dict[str, object],
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
) -> CalculationPlan:
    """Return up to ten calculation categories derived from question and excerpts."""
    planner = build_calculator_planner_agent(config, api_key)
    payload = {
        "question": question,
        "research_question": str(context_bundle.get("research_question", "")).strip(),
        "selected_cities": _selected_city_names(context_bundle),
        "excerpts": _planner_excerpts(context_bundle),
    }
    result = run_agent_sync(
        planner,
        json.dumps(payload, ensure_ascii=False),
        max_turns=config.calculator.max_turns,
        log_llm_payload=log_llm_payload,
    )
    output = result.final_output
    if not isinstance(output, CalculationPlan):
        raise ValueError("Calculator planner did not return a structured plan.")
    if len(output.categories) > config.calculator.max_categories:
        output.categories = output.categories[: config.calculator.max_categories]
    return _normalize_calculation_plan(output)


def extract_category_records(
    question: str,
    context_bundle: dict[str, object],
    category: CalculationCategory,
    previous_records: list[CalculationRecord],
    excerpts: list[dict[str, object]],
    pass_index: int,
    max_passes: int,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
) -> CalculationWorkerOutput:
    """Extract category-specific numeric records from excerpt evidence."""
    worker = build_calculator_worker_agent(config, api_key)
    payload = {
        "question": question,
        "research_question": str(context_bundle.get("research_question", "")).strip(),
        "category": category.model_dump(),
        "selected_cities": _selected_city_names(context_bundle),
        "pass_index": pass_index,
        "max_passes": max_passes,
        "previous_records": [record.model_dump() for record in previous_records],
        "excerpts": excerpts,
    }
    result = run_agent_sync(
        worker,
        json.dumps(payload, ensure_ascii=False),
        max_turns=config.calculator.max_turns,
        log_llm_payload=log_llm_payload,
    )
    output = result.final_output
    if isinstance(output, CalculationWorkerOutput):
        return output
    raise ValueError("Calculator worker did not return structured records.")


def _record_identity_key(record: CalculationRecord) -> tuple[object, ...]:
    """Return a deterministic identity key for exact duplicate records."""
    return (
        record.category_key,
        normalize_city_key(record.city),
        record.value,
        record.unit.casefold(),
        record.note.casefold(),
        record.year,
        record.record_role,
        tuple(record.ref_ids),
        tuple(record.source_chunk_ids),
    )


def _dedupe_records(records: list[CalculationRecord]) -> list[CalculationRecord]:
    """Return exact-duplicate records only once while preserving distinct candidates."""
    deduped: list[CalculationRecord] = []
    seen: set[tuple[object, ...]] = set()
    for record in records:
        key = _record_identity_key(record)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _normalize_unit(unit: str) -> str:
    """Normalize obvious unit aliases for deterministic grouping."""
    normalized = unit.strip().casefold()
    if not normalized:
        return ""
    replacements = {
        "€": "eur",
        "euro": "eur",
        "euros": "eur",
        "%": "percent",
        "per cent": "percent",
    }
    replacements["\u20ac"] = "eur"
    normalized = replacements.get(normalized, normalized)
    million_aliases = {
        "million eur",
        "eur million",
        "m eur",
        "meur",
        "eur m",
    }
    billion_aliases = {
        "billion eur",
        "eur billion",
        "bn eur",
        "eur bn",
        "beur",
    }
    if normalized in million_aliases:
        return "eur_million"
    if normalized in billion_aliases:
        return "eur_billion"
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def _category_semantic_tokens(category: CalculationCategory) -> set[str]:
    """Return semantic category tokens used for deterministic plan normalization."""
    source_text = " ".join(
        [
            category.category_key.replace("_", " "),
            category.label,
            category.description,
            category.inclusion_rule,
            category.exclusion_rule,
        ]
    ).lower()
    return set(_WORD_PATTERN.findall(source_text))


def _should_sum_reported_total_into_target(category: CalculationCategory) -> bool:
    """Return whether reported totals should contribute to target aggregation."""
    if category.sum_reported_total_into_target or category.operation != "sum":
        return category.sum_reported_total_into_target
    if not _normalize_unit(category.preferred_unit).startswith("eur"):
        return False
    return bool(_category_semantic_tokens(category) & _FINANCIAL_CATEGORY_TOKENS)


def _normalize_calculation_plan(plan: CalculationPlan) -> CalculationPlan:
    """Return planner categories with deterministic aggregation semantics applied."""
    normalized_categories = [
        category.model_copy(
            update={
                "sum_reported_total_into_target": _should_sum_reported_total_into_target(
                    category
                )
            }
        )
        for category in plan.categories
    ]
    return plan.model_copy(update={"categories": normalized_categories})


def _record_sort_key(record: CalculationRecord) -> tuple[str, int, str, float, str]:
    """Return a deterministic ordering key for extracted records."""
    return (
        normalize_city_key(record.city),
        record.year or -1,
        _normalize_unit(record.unit),
        float(record.value),
        record.record_role,
    )


def _record_uses_year_bucket(
    category: CalculationCategory,
    record: CalculationRecord,
) -> int | None:
    """Return the grouping year bucket for a record and category."""
    if category.year_policy == "separate_by_year":
        return record.year
    return None


def _current_roles_for_category(category: CalculationCategory) -> set[str]:
    """Return the current-observed record roles allowed for a category."""
    _ = category
    return {"atomic"}


def _target_roles_for_category(category: CalculationCategory) -> set[str]:
    """Return the target/planned record roles allowed for a category."""
    target_roles = {"target"}
    if category.sum_reported_total_into_target:
        target_roles.add("reported_total")
    return target_roles


def aggregate_category_records(
    *,
    categories: list[CalculationCategory],
    category_records: dict[str, list[CalculationRecord]],
    selected_city_names: list[str],
    category_statuses: dict[str, str] | None = None,
    category_notes: dict[str, str] | None = None,
) -> CalculationRunSummary:
    """Merge category records and compute deterministic grouped totals."""
    status_map = category_statuses or {}
    note_map = category_notes or {}
    normalized_selected_names = dedupe_city_labels(selected_city_names)
    selected_city_keys = {normalize_city_key(city) for city in normalized_selected_names}
    category_summaries: list[CalculationCategorySummary] = []

    for category in categories:
        records = _dedupe_records(category_records.get(category.category_key, []))
        records = sorted(records, key=_record_sort_key)
        current_roles = _current_roles_for_category(category)
        target_roles = _target_roles_for_category(category)
        grouped_records: dict[tuple[str, int | None], dict[str, list[CalculationRecord]]] = {}

        for record in records:
            group_key = (
                _normalize_unit(record.unit) or category.preferred_unit,
                _record_uses_year_bucket(category, record),
            )
            bucket = grouped_records.setdefault(
                group_key,
                {"current": [], "target": [], "non_additive": []},
            )
            if record.record_role in current_roles:
                bucket["current"].append(record)
            elif record.record_role in target_roles:
                bucket["target"].append(record)
            else:
                bucket["non_additive"].append(record)

        group_summaries: list[CalculationGroupSummary] = []
        for (normalized_unit, year_bucket), grouped in grouped_records.items():
            current_terms = grouped["current"]
            target_terms = grouped["target"]
            non_additive_records = grouped["non_additive"]
            group_records = current_terms + target_terms + non_additive_records
            current_total_decimal = sum(
                (Decimal(str(record.value)) for record in current_terms),
                start=Decimal("0"),
            )
            target_total_decimal = sum(
                (Decimal(str(record.value)) for record in target_terms),
                start=Decimal("0"),
            )
            current_city_keys = {
                normalize_city_key(record.city) for record in current_terms
            }
            target_city_keys = {
                normalize_city_key(record.city) for record in target_terms
            }
            non_additive_city_keys = {
                normalize_city_key(record.city) for record in non_additive_records
            }
            only_non_additive = sorted(
                non_additive_city_keys - current_city_keys - target_city_keys,
            )
            no_usable = sorted(
                selected_city_keys
                - current_city_keys
                - target_city_keys
                - non_additive_city_keys,
            )
            ref_ids = sorted(
                {
                    ref_id
                    for record in group_records
                    for ref_id in record.ref_ids
                }
            )
            source_chunk_ids = sorted(
                {
                    chunk_id
                    for record in group_records
                    for chunk_id in record.source_chunk_ids
                }
            )
            display_unit = category.preferred_unit
            for candidate_records in (
                current_terms,
                target_terms,
                non_additive_records,
            ):
                if candidate_records:
                    display_unit = candidate_records[0].unit
                    break
            group_summaries.append(
                CalculationGroupSummary(
                    normalized_unit=normalized_unit,
                    display_unit=display_unit,
                    year=year_bucket,
                    current_total=float(current_total_decimal),
                    current_terms=current_terms,
                    target_total=float(target_total_decimal),
                    target_terms=target_terms,
                    non_additive_records=non_additive_records,
                    current_record_count=len(current_terms),
                    target_record_count=len(target_terms),
                    current_city_count=len(current_city_keys),
                    target_city_count=len(target_city_keys),
                    selected_city_count=len(normalized_selected_names),
                    current_coverage_ratio=(
                        f"{len(current_city_keys)}/{len(normalized_selected_names)}"
                    ),
                    target_coverage_ratio=(
                        f"{len(target_city_keys)}/{len(normalized_selected_names)}"
                    ),
                    cities_with_current_records=[
                        format_city_display_name(city_key)
                        for city_key in sorted(current_city_keys)
                    ],
                    cities_with_target_records=[
                        format_city_display_name(city_key)
                        for city_key in sorted(target_city_keys)
                    ],
                    cities_with_only_non_additive_records=[
                        format_city_display_name(city_key) for city_key in only_non_additive
                    ],
                    cities_with_no_usable_records=[
                        format_city_display_name(city_key) for city_key in no_usable
                    ],
                    ref_ids=ref_ids,
                    source_chunk_ids=source_chunk_ids,
                )
            )

        current_record_count = sum(
            1 for record in records if record.record_role in current_roles
        )
        target_record_count = sum(
            1 for record in records if record.record_role in target_roles
        )
        category_status = status_map.get(
            category.category_key,
            "success" if records else "empty",
        )
        category_note = note_map.get(category.category_key, "")
        category_summaries.append(
            CalculationCategorySummary(
                category=category,
                status=category_status,
                note=category_note,
                record_count=len(records),
                current_record_count=current_record_count,
                target_record_count=target_record_count,
                records=records,
                groups=group_summaries,
            )
        )

    if not category_summaries:
        return CalculationRunSummary(
            status="empty",
            note="Calculator planner returned no categories.",
            selected_city_names=normalized_selected_names,
            category_count=0,
            categories=[],
        )

    if any(category.status in {"partial", "error"} for category in category_summaries):
        run_status = "partial"
    elif all(category.status == "empty" for category in category_summaries):
        run_status = "empty"
    else:
        run_status = "success"

    return CalculationRunSummary(
        status=run_status,
        note="Calculator stage completed.",
        selected_city_names=normalized_selected_names,
        category_count=len(category_summaries),
        categories=category_summaries,
    )


def _run_category_worker(
    *,
    question: str,
    context_bundle: dict[str, object],
    category: CalculationCategory,
    config: AppConfig,
    api_key: str,
    category_dir: Path,
    log_llm_payload: bool,
) -> _CategoryExecution:
    """Run all calculator passes for one category and persist pass artifacts."""
    max_passes = max(config.calculator.max_passes_per_category, 1)
    excerpt_slices = _slice_category_excerpts(category, context_bundle, max_passes)
    records: list[CalculationRecord] = []
    pass_files: list[Path] = []
    stop_reason = "max_passes_reached"
    final_status = "empty"
    final_note = ""
    error_message: str | None = None

    for pass_index in range(1, max_passes + 1):
        excerpts = (
            excerpt_slices[pass_index - 1]
            if pass_index - 1 < len(excerpt_slices)
            else []
        )
        if not excerpts:
            pass_output = CalculationWorkerOutput(
                status="done",
                category_key=category.category_key,
                note="No remaining relevant excerpts for this category.",
            )
        else:
            try:
                pass_output = extract_category_records(
                    question=question,
                    context_bundle=context_bundle,
                    category=category,
                    previous_records=records,
                    excerpts=excerpts,
                    pass_index=pass_index,
                    max_passes=max_passes,
                    config=config,
                    api_key=api_key,
                    log_llm_payload=log_llm_payload,
                )
            except MaxTurnsExceeded as exc:
                pass_output = CalculationWorkerOutput(
                    status="done",
                    category_key=category.category_key,
                    note="Calculator worker exceeded max turns and stopped early.",
                )
                error_message = str(exc)
                final_status = "partial"
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Calculator worker failed for category %s pass %d",
                    category.category_key,
                    pass_index,
                )
                pass_output = CalculationWorkerOutput(
                    status="done",
                    category_key=category.category_key,
                    note="Calculator worker failed and stopped early.",
                )
                error_message = str(exc)
                final_status = "partial"

        pass_path = category_dir / f"pass_{pass_index}.json"
        pass_payload = {
            "pass_index": pass_index,
            "category": category.model_dump(),
            "status": pass_output.status,
            "note": pass_output.note,
            "record_count": len(pass_output.records),
            "records": [record.model_dump() for record in pass_output.records],
            "excerpt_ref_ids": [
                str(excerpt.get("ref_id", "")).strip() for excerpt in excerpts
            ],
        }
        write_json(pass_path, pass_payload, ensure_ascii=False, default=str)
        pass_files.append(pass_path)

        if pass_output.status == "records":
            records.extend(pass_output.records)
            final_status = "success" if records else final_status
            final_note = pass_output.note
            continue

        stop_reason = "worker_done"
        if final_status == "empty" and records:
            final_status = "success"
        final_note = pass_output.note
        break
    else:
        if not records and final_status != "partial":
            final_status = "empty"

    records = _dedupe_records(records)
    if records and final_status == "empty":
        final_status = "success"
    records_path = category_dir / "records.json"
    write_json(
        records_path,
        {
            "category": category.model_dump(),
            "status": final_status,
            "note": final_note,
            "record_count": len(records),
            "records": [record.model_dump() for record in records],
        },
        ensure_ascii=False,
        default=str,
    )
    return _CategoryExecution(
        category=category,
        status=final_status,
        note=final_note,
        pass_count=len(pass_files),
        stop_reason=stop_reason,
        records=records,
        pass_files=pass_files,
        records_path=records_path,
        error_message=error_message,
    )


def run_calculator_stage(
    question: str,
    context_bundle: dict[str, object],
    config: AppConfig,
    api_key: str,
    *,
    base_dir: Path,
    log_llm_payload: bool = False,
) -> CalculationRunSummary:
    """Plan, extract, aggregate, and persist calculator artifacts for one run."""
    calculator_dir = base_dir / "calculator"
    categories_dir = calculator_dir / "categories"
    calculator_dir.mkdir(parents=True, exist_ok=True)
    categories_dir.mkdir(parents=True, exist_ok=True)
    selected_city_names = _selected_city_names(context_bundle)
    plan_path = calculator_dir / "plan.json"
    manifest_path = calculator_dir / "manifest.json"
    summary_path = calculator_dir / "summary.json"

    if not config.calculator.enabled:
        summary = CalculationRunSummary(
            status="empty",
            note="Calculator stage disabled by feature flag.",
            selected_city_names=selected_city_names,
            category_count=0,
            categories=[],
        )
        write_json(
            plan_path,
            {"categories": [], "note": "Calculator stage disabled by feature flag."},
            ensure_ascii=False,
            default=str,
        )
        write_json(summary_path, summary.model_dump(), ensure_ascii=False, default=str)
        write_json(
            manifest_path,
            {
                "status": "empty",
                "note": summary.note,
                "calculator_enabled": False,
                "category_count": 0,
                "categories": [],
                "artifacts": {
                    "plan": str(plan_path),
                    "summary": str(summary_path),
                },
            },
            ensure_ascii=False,
            default=str,
        )
        return summary

    try:
        plan = plan_categories(
            question=question,
            context_bundle=context_bundle,
            config=config,
            api_key=api_key,
            log_llm_payload=log_llm_payload,
        )
        write_json(plan_path, plan.model_dump(), ensure_ascii=False, default=str)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Calculator planner failed")
        summary = CalculationRunSummary(
            status="error",
            note=f"Calculator planner failed: {exc}",
            selected_city_names=selected_city_names,
            category_count=0,
            categories=[],
        )
        write_json(
            plan_path,
            {"categories": [], "note": f"Planner failed: {exc}"},
            ensure_ascii=False,
            default=str,
        )
        write_json(summary_path, summary.model_dump(), ensure_ascii=False, default=str)
        write_json(
            manifest_path,
            {
                "status": "error",
                "note": summary.note,
                "category_count": 0,
                "categories": [],
                "artifacts": {
                    "plan": str(plan_path),
                    "summary": str(summary_path),
                },
            },
            ensure_ascii=False,
            default=str,
        )
        return summary

    if not plan.categories:
        summary = CalculationRunSummary(
            status="empty",
            note=plan.note or "Calculator planner returned no categories.",
            selected_city_names=selected_city_names,
            category_count=0,
            categories=[],
        )
        write_json(summary_path, summary.model_dump(), ensure_ascii=False, default=str)
        write_json(
            manifest_path,
            {
                "status": "empty",
                "note": summary.note,
                "category_count": 0,
                "categories": [],
                "artifacts": {
                    "plan": str(plan_path),
                    "summary": str(summary_path),
                },
            },
            ensure_ascii=False,
            default=str,
        )
        return summary

    category_results: list[_CategoryExecution] = []
    max_workers = min(
        max(config.calculator.max_workers, 1),
        max(len(plan.categories), 1),
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_category_worker,
                question=question,
                context_bundle=context_bundle,
                category=category,
                config=config,
                api_key=api_key,
                category_dir=categories_dir / category.category_key,
                log_llm_payload=log_llm_payload,
            ): category
            for category in plan.categories
        }
        for future in as_completed(futures):
            category = futures[future]
            try:
                category_results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Calculator worker crashed for category %s",
                    category.category_key,
                )
                records_path = categories_dir / category.category_key / "records.json"
                write_json(
                    records_path,
                    {
                        "category": category.model_dump(),
                        "status": "error",
                        "note": "Calculator worker crashed before completing.",
                        "record_count": 0,
                        "records": [],
                    },
                    ensure_ascii=False,
                    default=str,
                )
                category_results.append(
                    _CategoryExecution(
                        category=category,
                        status="error",
                        note="Calculator worker crashed before completing.",
                        pass_count=0,
                        stop_reason="worker_exception",
                        records=[],
                        pass_files=[],
                        records_path=records_path,
                        error_message=str(exc),
                    )
                )

    category_results.sort(key=lambda item: item.category.category_key)
    summary = aggregate_category_records(
        categories=plan.categories,
        category_records={
            item.category.category_key: item.records for item in category_results
        },
        selected_city_names=selected_city_names,
        category_statuses={
            item.category.category_key: item.status for item in category_results
        },
        category_notes={
            item.category.category_key: item.note for item in category_results
        },
    )
    write_json(summary_path, summary.model_dump(), ensure_ascii=False, default=str)
    write_json(
        manifest_path,
        {
            "status": summary.status,
            "note": summary.note,
            "category_count": len(plan.categories),
            "categories": [
                {
                    "category_key": item.category.category_key,
                    "status": item.status,
                    "note": item.note,
                    "pass_count": item.pass_count,
                    "stop_reason": item.stop_reason,
                    "pass_files": [str(path) for path in item.pass_files],
                    "records_path": str(item.records_path),
                    "error_message": item.error_message,
                }
                for item in category_results
            ],
            "artifacts": {
                "plan": str(plan_path),
                "summary": str(summary_path),
            },
        },
        ensure_ascii=False,
        default=str,
    )
    return summary


__all__ = [
    "aggregate_category_records",
    "build_calculator_planner_agent",
    "build_calculator_worker_agent",
    "extract_category_records",
    "plan_categories",
    "run_calculator_stage",
]
