from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.modules.initiative_extractor.models import InitiativeExtractionRecord, JsonValue
from backend.modules.tef_mapper.models import TefFinalMappingRecord, TefTargetType
from backend.utils.config import AppConfig
from backend.utils.json_io import read_json_object, write_json
from backend.utils.llm_serialization import serialize_for_llm
from backend.utils.prompts import load_prompt

NumericFactBucket = Literal["current", "planned"]
NumericMetricType = Literal[
    "emissions",
    "capacity",
    "energy",
    "cost",
    "rate",
    "time",
    "count",
    "other",
]
NumericNormalizedUnit = Literal[
    "tCO2e/year",
    "tCO2e",
    "MW",
    "MWh",
    "EUR",
    "PLN",
    "percent_or_fraction",
    "count",
]
NumericUnitRaw = Literal["tco2e_per_year", "tco2e", "mw", "mwh", "eur", "pln"]
NumericAggregationMethod = Literal["sum", "none"]
NumericUnitClassificationMethod = Literal["llm", "rule"]
NumericUnitClassifier = Callable[
    ["NumericUnitClassificationInput"],
    "NumericUnitClassification",
]

logger = logging.getLogger(__name__)

NUMERIC_UNIT_CLASSIFIER_PROMPT = Path("backend/prompts/tef_numeric_unit_classifier_system.md")
ADDITIVE_METRIC_TYPES = {"capacity", "cost", "emissions", "energy", "count"}
NORMALIZED_UNITS_BY_METRIC: dict[NumericMetricType, set[NumericNormalizedUnit | None]] = {
    "emissions": {"tCO2e/year", "tCO2e"},
    "capacity": {"MW"},
    "energy": {"MWh"},
    "cost": {"EUR", "PLN"},
    "rate": {"percent_or_fraction"},
    "time": {None},
    "count": {"count"},
    "other": {None},
}
NORMALIZED_UNIT_BY_RAW: dict[NumericUnitRaw, NumericNormalizedUnit] = {
    "tco2e_per_year": "tCO2e/year",
    "tco2e": "tCO2e",
    "mw": "MW",
    "mwh": "MWh",
    "eur": "EUR",
    "pln": "PLN",
}
RAW_UNIT_BY_NORMALIZED: dict[NumericNormalizedUnit, NumericUnitRaw] = {
    normalized_unit: unit_raw
    for unit_raw, normalized_unit in NORMALIZED_UNIT_BY_RAW.items()
}


def run_agent_sync(*args: Any, **kwargs: Any) -> Any:
    """Lazy wrapper so tests can monkeypatch LLM execution without importing Agents SDK."""
    from backend.services.agents import run_agent_sync as run_sync

    return run_sync(*args, **kwargs)


class NumericUnitClassificationInput(BaseModel):
    """One numeric field plus source context for unit classification."""

    model_config = ConfigDict(extra="forbid")

    number_key_raw: str
    value_raw: JsonValue
    value_number: float | None = None
    source_number_bucket: NumericFactBucket
    initiative_name: str
    initiative_text: str | None = None
    source_quote: str | None = None


class NumericUnitClassification(BaseModel):
    """Constrained metric/unit classification for one extracted numeric field."""

    model_config = ConfigDict(extra="forbid")

    metric_type: NumericMetricType
    normalized_unit: NumericNormalizedUnit | None = None
    unit_raw: NumericUnitRaw | None = None
    aggregation_method: NumericAggregationMethod
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool
    rationale: str
    method: NumericUnitClassificationMethod = "llm"

    @model_validator(mode="after")
    def _validate_unit_contract(self) -> "NumericUnitClassification":
        """Reject invented unit values or additive decisions that cannot be safely summed."""
        allowed_units = NORMALIZED_UNITS_BY_METRIC[self.metric_type]
        if self.normalized_unit not in allowed_units:
            raise ValueError(
                f"normalized_unit {self.normalized_unit!r} is invalid for {self.metric_type!r}"
            )
        expected_unit_raw = RAW_UNIT_BY_NORMALIZED.get(self.normalized_unit)
        if self.unit_raw != expected_unit_raw:
            raise ValueError("unit_raw must match normalized_unit")
        if self.aggregation_method == "sum" and not _is_additive(
            self.metric_type,
            self.normalized_unit,
        ):
            raise ValueError("aggregation_method=sum requires an additive metric and known unit")
        return self


class TefNumericFactRecord(BaseModel):
    """One extracted initiative number joined to one TEF final mapping."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str
    extraction_run_id: str | None = None
    tef_mapping_run_id: str
    initiative_record_id: str
    city: str
    source_document: str
    document_local_code: str | None = None
    initiative_name: str
    source_quote: str | None = None
    target_type: TefTargetType
    target_id: str
    target_path: str
    mapping_confidence: float
    is_primary_mapping: bool
    mapping_needs_review: bool
    source_number_bucket: NumericFactBucket
    number_key_raw: str
    value_raw: JsonValue
    value_number: float | None = None
    value_text: str | None = None
    metric_key: str
    metric_type: NumericMetricType
    unit_raw: NumericUnitRaw | None = None
    normalized_unit: NumericNormalizedUnit | None = None
    aggregation_method: NumericAggregationMethod
    unit_classification_method: NumericUnitClassificationMethod
    unit_classification_confidence: float
    unit_classification_rationale: str
    include_in_default_rollup: bool
    aggregation_weight: float
    numeric_quality_flags: list[str] = Field(default_factory=list)
    needs_review: bool
    review_reasons: list[str] = Field(default_factory=list)


class TefGroupedInitiative(BaseModel):
    """One initiative listed under a TEF target group."""

    model_config = ConfigDict(extra="forbid")

    initiative_record_id: str
    city: str
    source_document: str
    initiative_name: str
    source_quote: str | None = None
    mapping_confidence: float
    is_primary_mapping: bool
    mapping_needs_review: bool
    numeric_fact_ids: list[str] = Field(default_factory=list)


class TefGroupedInitiativesRecord(BaseModel):
    """One TEF target group with mapped initiatives and available metric keys."""

    model_config = ConfigDict(extra="forbid")

    group_id: str
    target_type: TefTargetType
    target_id: str
    target_path: str
    initiative_count: int
    primary_initiative_count: int
    needs_review_count: int
    initiatives: list[TefGroupedInitiative] = Field(default_factory=list)
    available_metric_keys: list[str] = Field(default_factory=list)
    review_summary: dict[str, int] = Field(default_factory=dict)


def _write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    """Write Pydantic rows as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row.model_dump(mode="json"), ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows from disk."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(payload)
    return rows


def load_initiative_records(path: Path) -> list[InitiativeExtractionRecord]:
    """Load pipeline initiative records that wrap the clean v1 initiative object."""
    return [InitiativeExtractionRecord.model_validate(row) for row in _read_jsonl(path)]


def load_final_mappings(path: Path) -> list[TefFinalMappingRecord]:
    """Load TEF final mapping rows."""
    return [TefFinalMappingRecord.model_validate(row) for row in _read_jsonl(path)]


def resolve_initiative_records_path(
    *,
    tef_run_dir: Path,
    extraction_run_dir: Path | None = None,
    initiative_records_jsonl: Path | None = None,
) -> Path:
    """Resolve the initiative record sidecar for numeric rollups."""
    if initiative_records_jsonl is not None:
        return initiative_records_jsonl
    if extraction_run_dir is not None:
        return extraction_run_dir / "03_deduped" / "initiative_records.jsonl"
    return tef_run_dir / "01_inputs" / "initiatives.jsonl"


def _parse_number(value: JsonValue) -> float | None:
    """Parse a scalar number without guessing units or currencies."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    compact = re.sub(r"(?<=\d)[\s,](?=\d{3}\b)", "", text)
    compact = compact.replace(",", ".") if compact.count(",") == 1 and "." not in compact else compact
    if not re.fullmatch(r"[-+]?\d+(\.\d+)?", compact):
        return None
    return float(compact)


def _metric_key(number_key: str) -> str:
    """Build a stable metric key from the extracted number key."""
    metric_key = number_key.lower().strip()
    for suffix in ("_approx", "_estimated", "_estimate"):
        if metric_key.endswith(suffix):
            metric_key = metric_key[: -len(suffix)]
    return metric_key


def _unit_for_key(
    number_key: str,
) -> tuple[NumericMetricType, NumericNormalizedUnit | None, NumericUnitRaw | None]:
    """Infer coarse unit metadata from the extracted number key."""
    key = number_key.lower()
    if "tco2e_per_year" in key or "tco2_per_year" in key or "_t_per_year" in key:
        return "emissions", "tCO2e/year", "tco2e_per_year"
    if "tco2e" in key or "tco2" in key or "tonnes_co2" in key:
        return "emissions", "tCO2e", "tco2e"
    if key.endswith("_mw") or "_mw_" in key:
        return "capacity", "MW", "mw"
    if key.endswith("_mwh") or "_mwh_" in key:
        return "energy", "MWh", "mwh"
    if "_eur" in key or key.endswith("eur"):
        return "cost", "EUR", "eur"
    if "_pln" in key or key.endswith("pln"):
        return "cost", "PLN", "pln"
    if "percent" in key or "share" in key or "rate" in key or "fraction" in key:
        return "rate", "percent_or_fraction", None
    if "year" in key or "date" in key or "deadline" in key:
        return "time", None, None
    if "count" in key or "number" in key or key.endswith("_total"):
        return "count", "count", None
    return "other", None, None


def _is_additive(
    metric_type: NumericMetricType,
    normalized_unit: NumericNormalizedUnit | None,
) -> bool:
    """Return whether a metric can be summed in default rollups."""
    return metric_type in ADDITIVE_METRIC_TYPES and normalized_unit is not None


def classify_numeric_unit_with_rules(
    payload: NumericUnitClassificationInput,
) -> NumericUnitClassification:
    """Classify a numeric field using the deterministic key-based fallback rules."""
    metric_type, normalized_unit, unit_raw = _unit_for_key(payload.number_key_raw)
    aggregation_method: NumericAggregationMethod = (
        "sum" if _is_additive(metric_type, normalized_unit) else "none"
    )
    return NumericUnitClassification(
        metric_type=metric_type,
        normalized_unit=normalized_unit,
        unit_raw=unit_raw,
        aggregation_method=aggregation_method,
        confidence=0.70 if metric_type != "other" else 0.30,
        needs_review=False,
        rationale="Rule-based fallback inferred the metric metadata from the numeric key.",
        method="rule",
    )


def _get_field(value: object, key: str) -> object:
    """Read a field from a dict-like or object-like SDK payload."""
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _extract_numeric_unit_tool_output(result: object) -> NumericUnitClassification | None:
    """Extract numeric classifier tool-call arguments from an Agents SDK result."""
    raw_responses = list(getattr(result, "raw_responses", []) or [])
    for response in reversed(raw_responses):
        output_items = _get_field(response, "output")
        if not isinstance(output_items, list):
            continue
        for item in reversed(output_items):
            if _get_field(item, "type") != "function_call":
                continue
            if _get_field(item, "name") != "submit_numeric_unit_classification":
                continue
            arguments = _get_field(item, "arguments")
            if isinstance(arguments, str):
                return NumericUnitClassification.model_validate(json.loads(arguments))
    return None


def _coerce_numeric_unit_output(output: object) -> NumericUnitClassification:
    """Coerce final LLM output into the numeric classification model."""
    if isinstance(output, NumericUnitClassification):
        return output
    if isinstance(output, str) and output.strip().startswith("{"):
        output = json.loads(output)
    return NumericUnitClassification.model_validate(output)


def build_numeric_unit_classifier_agent(config: AppConfig, api_key: str) -> object:
    """Build the numeric unit classifier agent with constrained tool output."""
    from agents import Agent, function_tool
    from backend.services.agents import build_model_settings, build_openrouter_model

    settings = build_model_settings(
        config.tef_mapper.temperature,
        config.tef_mapper.max_output_tokens,
        reasoning_effort=config.tef_mapper.reasoning_effort,
    )
    settings.tool_choice = "submit_numeric_unit_classification"
    settings.parallel_tool_calls = False

    @function_tool(strict_mode=False)
    def submit_numeric_unit_classification(
        metric_type: NumericMetricType,
        aggregation_method: NumericAggregationMethod,
        confidence: float,
        needs_review: bool,
        rationale: str,
        normalized_unit: NumericNormalizedUnit | None = None,
        unit_raw: NumericUnitRaw | None = None,
    ) -> NumericUnitClassification:
        return NumericUnitClassification(
            metric_type=metric_type,
            normalized_unit=normalized_unit,
            unit_raw=unit_raw,
            aggregation_method=aggregation_method,
            confidence=confidence,
            needs_review=needs_review,
            rationale=rationale,
            method="llm",
        )

    model = build_openrouter_model(
        config.tef_mapper.model,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )
    return Agent(
        name="TEF Numeric Unit Classifier",
        instructions=load_prompt(NUMERIC_UNIT_CLASSIFIER_PROMPT),
        model=model,
        model_settings=settings,
        tools=[submit_numeric_unit_classification],
        tool_use_behavior="stop_on_first_tool",
    )


def _llm_numeric_unit_classifier(
    *,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool,
) -> NumericUnitClassifier:
    """Build a callable that classifies numeric units through the LLM."""
    agent = build_numeric_unit_classifier_agent(config, api_key)

    def classify(payload: NumericUnitClassificationInput) -> NumericUnitClassification:
        result = run_agent_sync(
            agent,
            serialize_for_llm(payload),
            max_turns=max(config.tef_mapper.max_turns, 1),
            log_llm_payload=log_llm_payload,
        )
        classification = _extract_numeric_unit_tool_output(result) or _coerce_numeric_unit_output(
            result.final_output
        )
        return classification.model_copy(update={"method": "llm"})

    return classify


def build_numeric_unit_classifier(
    *,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
) -> NumericUnitClassifier:
    """Return the configured LLM-backed numeric unit classifier."""
    return _llm_numeric_unit_classifier(
        config=config,
        api_key=api_key,
        log_llm_payload=log_llm_payload,
    )


def _numeric_quality_flags(number_key: str, value_number: float | None) -> list[str]:
    """Return simple quality flags for one extracted numeric field."""
    flags: list[str] = []
    key = number_key.lower()
    if "approx" in key or "estimated" in key or "estimate" in key:
        flags.append("approximate_value")
    if value_number is None:
        flags.append("non_numeric_value")
    return flags


def _classification_input(
    *,
    record: InitiativeExtractionRecord,
    bucket: NumericFactBucket,
    number_key: str,
    value_raw: JsonValue,
    value_number: float | None,
) -> NumericUnitClassificationInput:
    """Build the LLM/rule classifier payload for one numeric field."""
    initiative = record.initiative
    initiative_text = "\n".join(
        part
        for part in (
            initiative.general_description,
            initiative.objective_text,
            initiative.implementation_text,
            initiative.planned_outputs_text,
            initiative.delivery_text,
            initiative.funding_text,
            initiative.timeline_text,
        )
        if part
    )
    return NumericUnitClassificationInput(
        number_key_raw=number_key,
        value_raw=value_raw,
        value_number=value_number,
        source_number_bucket=bucket,
        initiative_name=initiative.initiative_name,
        initiative_text=initiative_text or None,
        source_quote=record.source_quote,
    )


def _resolve_numeric_unit_classification(
    payload: NumericUnitClassificationInput,
    unit_classifier: NumericUnitClassifier | None,
) -> NumericUnitClassification:
    """Resolve metric metadata with the optional LLM classifier and rule fallback."""
    if unit_classifier is None:
        return classify_numeric_unit_with_rules(payload)
    try:
        return unit_classifier(payload)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Falling back to rule-based numeric unit classification for number_key=%s",
            payload.number_key_raw,
            exc_info=True,
        )
        fallback = classify_numeric_unit_with_rules(payload)
        return fallback.model_copy(
            update={
                "confidence": min(fallback.confidence, 0.40),
                "needs_review": True,
                "rationale": "LLM unit classification failed; rule-based fallback was used.",
            }
        )


def _review_reasons(
    *,
    mapping: TefFinalMappingRecord,
    value_number: float | None,
    classification: NumericUnitClassification,
    include_in_default_rollup: bool,
    quality_flags: list[str],
) -> list[str]:
    """Explain why a numeric fact needs review or is not in default totals."""
    reasons: list[str] = []
    if mapping.needs_review:
        reasons.append("mapping_needs_review")
    if not mapping.is_primary:
        reasons.append("non_primary_mapping")
    if value_number is None:
        reasons.append("non_numeric_value")
    if classification.aggregation_method == "none":
        reasons.append("not_additive")
    if classification.needs_review:
        reasons.append("numeric_unit_needs_review")
    if not include_in_default_rollup:
        reasons.append("excluded_from_default_rollup")
    reasons.extend(quality_flags)
    return list(dict.fromkeys(reasons))


def _fact_id(
    *,
    mapping: TefFinalMappingRecord,
    bucket: NumericFactBucket,
    number_key: str,
) -> str:
    """Build a deterministic pipeline-generated numeric fact id."""
    return ":".join(
        [
            mapping.initiative_record_id,
            bucket,
            number_key,
            mapping.target_type,
            mapping.target_id,
        ]
    )


def build_numeric_facts(
    *,
    run_id: str,
    extraction_run_id: str | None,
    initiative_records: list[InitiativeExtractionRecord],
    final_mappings: list[TefFinalMappingRecord],
    unit_classifier: NumericUnitClassifier | None = None,
) -> list[TefNumericFactRecord]:
    """Join final mappings to clean v1 initiative numbers and emit fact rows."""
    records_by_id = {record.record_id: record for record in initiative_records}
    classifications: dict[tuple[str, NumericFactBucket, str], NumericUnitClassification] = {}
    facts: list[TefNumericFactRecord] = []
    for mapping in final_mappings:
        record = records_by_id.get(mapping.initiative_record_id)
        if record is None:
            continue
        for bucket, numbers in (
            ("current", record.initiative.numbers.current),
            ("planned", record.initiative.numbers.planned),
        ):
            for number_key, value_raw in numbers.items():
                value_number = _parse_number(value_raw)
                classification_key = (record.record_id, bucket, number_key)
                classification = classifications.get(classification_key)
                if classification is None:
                    classification = _resolve_numeric_unit_classification(
                        _classification_input(
                            record=record,
                            bucket=bucket,
                            number_key=number_key,
                            value_raw=value_raw,
                            value_number=value_number,
                        ),
                        unit_classifier,
                    )
                    classifications[classification_key] = classification
                include = (
                    mapping.is_primary
                    and value_number is not None
                    and classification.aggregation_method == "sum"
                )
                quality_flags = _numeric_quality_flags(number_key, value_number)
                if classification.method == "rule" and unit_classifier is not None:
                    quality_flags.append("unit_classification_fallback")
                review_reasons = _review_reasons(
                    mapping=mapping,
                    value_number=value_number,
                    classification=classification,
                    include_in_default_rollup=include,
                    quality_flags=quality_flags,
                )
                facts.append(
                    TefNumericFactRecord(
                        fact_id=_fact_id(mapping=mapping, bucket=bucket, number_key=number_key),
                        extraction_run_id=extraction_run_id,
                        tef_mapping_run_id=run_id,
                        initiative_record_id=mapping.initiative_record_id,
                        city=record.initiative.city,
                        source_document=record.source_document,
                        document_local_code=record.document_local_code,
                        initiative_name=record.initiative.initiative_name,
                        source_quote=record.source_quote,
                        target_type=mapping.target_type,
                        target_id=mapping.target_id,
                        target_path=mapping.target_path,
                        mapping_confidence=mapping.confidence,
                        is_primary_mapping=mapping.is_primary,
                        mapping_needs_review=mapping.needs_review,
                        source_number_bucket=bucket,
                        number_key_raw=number_key,
                        value_raw=value_raw,
                        value_number=value_number,
                        value_text=value_raw if isinstance(value_raw, str) else None,
                        metric_key=_metric_key(number_key),
                        metric_type=classification.metric_type,
                        unit_raw=classification.unit_raw,
                        normalized_unit=classification.normalized_unit,
                        aggregation_method=classification.aggregation_method,
                        unit_classification_method=classification.method,
                        unit_classification_confidence=classification.confidence,
                        unit_classification_rationale=classification.rationale,
                        include_in_default_rollup=include,
                        aggregation_weight=1.0 if include else 0.0,
                        numeric_quality_flags=quality_flags,
                        needs_review=bool(review_reasons),
                        review_reasons=review_reasons,
                    )
                )
    return facts


def build_grouped_initiatives(
    final_mappings: list[TefFinalMappingRecord],
    facts: list[TefNumericFactRecord],
) -> list[TefGroupedInitiativesRecord]:
    """Group final mappings and numeric fact ids by TEF target."""
    fact_ids_by_mapping: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    metric_keys_by_target: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for fact in facts:
        mapping_key = (fact.initiative_record_id, fact.target_type, fact.target_id)
        target_key = (fact.target_type, fact.target_id, fact.target_path)
        fact_ids_by_mapping[mapping_key].append(fact.fact_id)
        metric_keys_by_target[target_key].add(fact.metric_key)

    mappings_by_target: dict[tuple[str, str, str], list[TefFinalMappingRecord]] = defaultdict(list)
    for mapping in final_mappings:
        mappings_by_target[(mapping.target_type, mapping.target_id, mapping.target_path)].append(mapping)

    groups: list[TefGroupedInitiativesRecord] = []
    for (target_type, target_id, target_path), mappings in sorted(mappings_by_target.items()):
        initiatives = [
            TefGroupedInitiative(
                initiative_record_id=mapping.initiative_record_id,
                city=mapping.city,
                source_document=mapping.source_document,
                initiative_name=mapping.initiative_name,
                source_quote=mapping.source_quote,
                mapping_confidence=mapping.confidence,
                is_primary_mapping=mapping.is_primary,
                mapping_needs_review=mapping.needs_review,
                numeric_fact_ids=sorted(
                    fact_ids_by_mapping[
                        (mapping.initiative_record_id, mapping.target_type, mapping.target_id)
                    ]
                ),
            )
            for mapping in mappings
        ]
        groups.append(
            TefGroupedInitiativesRecord(
                group_id=f"{target_type}:{target_id}",
                target_type=target_type,
                target_id=target_id,
                target_path=target_path,
                initiative_count=len({mapping.initiative_record_id for mapping in mappings}),
                primary_initiative_count=len(
                    {mapping.initiative_record_id for mapping in mappings if mapping.is_primary}
                ),
                needs_review_count=len(
                    {mapping.initiative_record_id for mapping in mappings if mapping.needs_review}
                ),
                initiatives=initiatives,
                available_metric_keys=sorted(
                    metric_keys_by_target[(target_type, target_id, target_path)]
                ),
                review_summary={
                    "mapping_needs_review": sum(1 for mapping in mappings if mapping.needs_review),
                    "non_primary_mapping": sum(1 for mapping in mappings if not mapping.is_primary),
                },
            )
        )
    return groups


def build_metric_rollups(
    *,
    run_id: str,
    extraction_run_id: str | None,
    facts: list[TefNumericFactRecord],
) -> dict[str, Any]:
    """Aggregate default numeric totals by TEF target and compatible metric."""
    included = [fact for fact in facts if fact.include_in_default_rollup and fact.value_number is not None]
    grouped: dict[tuple[str, str, str, str, str | None], list[TefNumericFactRecord]] = defaultdict(list)
    for fact in included:
        grouped[
            (
                fact.target_type,
                fact.target_id,
                fact.target_path,
                fact.metric_key,
                fact.normalized_unit,
            )
        ].append(fact)

    rollups: list[dict[str, Any]] = []
    for (
        target_type,
        target_id,
        target_path,
        metric_key,
        normalized_unit,
    ), group_facts in sorted(grouped.items()):
        breakdown: dict[str, dict[str, Any]] = {}
        for fact in group_facts:
            city_bucket = breakdown.setdefault(
                fact.city,
                {"value": 0.0, "fact_count": 0, "initiative_ids": set()},
            )
            city_bucket["value"] += fact.value_number or 0.0
            city_bucket["fact_count"] += 1
            city_bucket["initiative_ids"].add(fact.initiative_record_id)

        rollups.append(
            {
                "target_type": target_type,
                "target_id": target_id,
                "target_path": target_path,
                "metric_key": metric_key,
                "normalized_unit": normalized_unit,
                "aggregation_method": "sum",
                "value": sum(fact.value_number or 0.0 for fact in group_facts),
                "fact_count": len(group_facts),
                "initiative_count": len({fact.initiative_record_id for fact in group_facts}),
                "city_count": len({fact.city for fact in group_facts}),
                "breakdown_by_city": {
                    city: {
                        "value": payload["value"],
                        "fact_count": payload["fact_count"],
                        "initiative_count": len(payload["initiative_ids"]),
                    }
                    for city, payload in sorted(breakdown.items())
                },
                "included_fact_ids": sorted(fact.fact_id for fact in group_facts),
                "needs_review": any(fact.needs_review for fact in group_facts),
                "review_reasons": sorted(
                    {
                        reason
                        for fact in group_facts
                        for reason in fact.review_reasons
                        if reason != "excluded_from_default_rollup"
                    }
                ),
            }
        )

    return {
        "run_id": run_id,
        "extraction_run_id": extraction_run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "rollup_policy": {
            "default_include_only_primary_mappings": True,
            "exclude_needs_review_from_totals": False,
            "include_needs_review_with_flags": True,
            "do_not_sum_units_across_different_normalized_units": True,
        },
        "rollups": rollups,
        "unaggregated_facts": [
            {"fact_id": fact.fact_id, "reason": fact.review_reasons[0]}
            for fact in facts
            if not fact.include_in_default_rollup
        ],
    }


def write_numeric_rollup_artifacts(
    *,
    run_dir: Path,
    run_id: str,
    extraction_run_id: str | None,
    initiative_records: list[InitiativeExtractionRecord],
    final_mappings: list[TefFinalMappingRecord],
    unit_classifier: NumericUnitClassifier | None = None,
) -> dict[str, int]:
    """Write numeric fact, TEF group, and metric rollup artifacts."""
    facts = build_numeric_facts(
        run_id=run_id,
        extraction_run_id=extraction_run_id,
        initiative_records=initiative_records,
        final_mappings=final_mappings,
        unit_classifier=unit_classifier,
    )
    groups = build_grouped_initiatives(final_mappings, facts)
    rollups = build_metric_rollups(
        run_id=run_id,
        extraction_run_id=extraction_run_id,
        facts=facts,
    )

    _write_jsonl(run_dir / "07_numeric_facts" / "initiative_numeric_facts.jsonl", facts)
    _write_jsonl(run_dir / "08_tef_groups" / "tef_grouped_initiatives.jsonl", groups)
    write_json(run_dir / "08_tef_groups" / "tef_metric_rollups.json", rollups, ensure_ascii=False)
    return {
        "initiative_records_count": len(initiative_records),
        "final_mappings_count": len(final_mappings),
        "numeric_facts_count": len(facts),
        "tef_group_count": len(groups),
        "metric_rollup_count": len(rollups["rollups"]),
    }


def rollup_existing_tef_run(
    *,
    tef_run_dir: Path,
    extraction_run_dir: Path | None = None,
    initiative_records_jsonl: Path | None = None,
    unit_classifier: NumericUnitClassifier | None = None,
) -> dict[str, int]:
    """Generate numeric rollup artifacts for an existing TEF mapping run."""
    manifest = read_json_object(tef_run_dir / "00_source" / "source_manifest.json") or {}
    run_id = str(manifest.get("run_id") or tef_run_dir.name)
    extraction_run_id = manifest.get("extraction_run_id")
    records_path = resolve_initiative_records_path(
        tef_run_dir=tef_run_dir,
        extraction_run_dir=extraction_run_dir,
        initiative_records_jsonl=initiative_records_jsonl,
    )
    initiative_records = load_initiative_records(records_path)
    final_mappings = load_final_mappings(tef_run_dir / "05_final_mappings" / "final_mappings.jsonl")
    return write_numeric_rollup_artifacts(
        run_dir=tef_run_dir,
        run_id=run_id,
        extraction_run_id=str(extraction_run_id) if extraction_run_id else None,
        initiative_records=initiative_records,
        final_mappings=final_mappings,
        unit_classifier=unit_classifier,
    )
