from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.modules.initiative_extractor.models import InitiativeExtractionRecord, JsonValue
from backend.modules.tef_mapper.models import TefFinalMappingRecord, TefTargetType
from backend.utils.json_io import read_json_object, write_json

NumericFactBucket = Literal["current", "planned"]


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
    metric_type: str
    unit_raw: str | None = None
    normalized_unit: str | None = None
    aggregation_method: Literal["sum", "none"]
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


def _unit_for_key(number_key: str) -> tuple[str | None, str | None, str]:
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


def _is_additive(metric_type: str, normalized_unit: str | None) -> bool:
    """Return whether a metric can be summed in default rollups."""
    return metric_type in {"capacity", "cost", "emissions", "energy", "count"} and normalized_unit is not None


def _numeric_quality_flags(number_key: str, value_number: float | None) -> list[str]:
    """Return simple quality flags for one extracted numeric field."""
    flags: list[str] = []
    key = number_key.lower()
    if "approx" in key or "estimated" in key or "estimate" in key:
        flags.append("approximate_value")
    if value_number is None:
        flags.append("non_numeric_value")
    return flags


def _review_reasons(
    *,
    mapping: TefFinalMappingRecord,
    value_number: float | None,
    aggregation_method: str,
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
    if aggregation_method == "none":
        reasons.append("not_additive")
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
) -> list[TefNumericFactRecord]:
    """Join final mappings to clean v1 initiative numbers and emit fact rows."""
    records_by_id = {record.record_id: record for record in initiative_records}
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
                metric_type, normalized_unit, unit_raw = _unit_for_key(number_key)
                aggregation_method = "sum" if _is_additive(metric_type, normalized_unit) else "none"
                include = (
                    mapping.is_primary
                    and value_number is not None
                    and aggregation_method == "sum"
                )
                quality_flags = _numeric_quality_flags(number_key, value_number)
                review_reasons = _review_reasons(
                    mapping=mapping,
                    value_number=value_number,
                    aggregation_method=aggregation_method,
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
                        metric_type=metric_type,
                        unit_raw=unit_raw,
                        normalized_unit=normalized_unit,
                        aggregation_method=aggregation_method,
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
) -> dict[str, int]:
    """Write numeric fact, TEF group, and metric rollup artifacts."""
    facts = build_numeric_facts(
        run_id=run_id,
        extraction_run_id=extraction_run_id,
        initiative_records=initiative_records,
        final_mappings=final_mappings,
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
    )
