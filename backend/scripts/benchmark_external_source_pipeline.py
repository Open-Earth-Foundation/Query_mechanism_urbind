"""
Brief: Run the governed external-source pipeline benchmark for a city scenario.

Inputs:
- CLI args:
  - `--benchmark-file`: JSON benchmark fixture with `scenario` and `cases` fields.
    Defaults to `backend/benchmarks/external_sources/krakow_external_source_benchmark.json`.
  - `--config`: Path to `llm_config.yaml`.
  - `--output-dir`: Directory where benchmark artifacts are written.
  - `--run-id`: Optional stable run ID for artifact names.
  - `--skip-writer`: Run only external-source extraction and skip writer validation.
- Files/paths: expects `documents/source_library/sources.yaml` and Markdown files whose
  filename stems match `source_id` metadata entries.
- Env vars: `OPENROUTER_API_KEY` is required for live LLM extraction and writer calls.

Outputs:
- `benchmark_summary.json`: case-level extraction/resolution scores and tool-call counts.
- `context_bundle.json`: writer-ready enrichment context assembled from external evidence.
- `writer_answer.md`: generated writer answer when `--skip-writer` is not set.
- `external_sources/external_evidence.json`: per-run controlled tool audit artifact.

Usage (from project root):
- python -m backend.scripts.benchmark_external_source_pipeline
- python -m backend.scripts.benchmark_external_source_pipeline --skip-writer
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.modules.web_researcher.context_merger import merge_enrichment_into_context
from backend.modules.web_researcher.external_agent import run_external_source_enrichment
from backend.modules.web_researcher.models import (
    CityGap,
    ExternalEvidenceClaim,
    FieldClassification,
    GapManifest,
)
from backend.modules.writer.agent import write_markdown
from backend.utils.config import load_config, resolve_openrouter_api_key
from backend.utils.json_io import write_json
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_FILE = Path(
    "backend/benchmarks/external_sources/krakow_external_source_benchmark.json"
)
DEFAULT_OUTPUT_DIR = Path("output/external_source_benchmarks/krakow")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the governed external-source benchmark and optional writer scenario."
    )
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=DEFAULT_BENCHMARK_FILE,
        help="JSON fixture describing the city scenario and expected facts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("llm_config.yaml"),
        help="LLM config YAML path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for benchmark artifacts.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run ID. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--skip-writer",
        action="store_true",
        help="Skip writer validation and run only external-source extraction.",
    )
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("external_%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config.enrichment = config.enrichment.model_copy(
        update={"enabled": True, "external_source_search_enabled": True}
    )
    api_key = resolve_openrouter_api_key()
    benchmark = _load_benchmark(args.benchmark_file)
    scenario = benchmark["scenario"]
    cases = benchmark["cases"]
    city = str(scenario["city"])
    question = str(scenario["question"])

    logger.info(
        "Starting external-source benchmark run_id=%s city=%s cases=%d model=%s",
        run_id,
        city,
        len(cases),
        config.enrichment.model,
    )

    context_bundle = _build_context_bundle(city, question)
    gap_manifest = _build_gap_manifest(city, cases)
    claims, resolutions, no_evidence, tool_calls = run_external_source_enrichment(
        question=question,
        context_bundle=context_bundle,
        gap_manifest=gap_manifest,
        base_dir=output_dir,
        config=config,
        api_key=api_key,
        run_id=run_id,
    )
    enriched_context = merge_enrichment_into_context(
        context_bundle=context_bundle,
        gap_manifest=gap_manifest,
        web_findings=[],
        freshness_results=[],
        assumptions=[],
        non_estimable=[],
        saturation_warning=None,
        config_model=config.enrichment.model,
        assumptions_model=config.enrichment.assumptions_estimator_model or config.enrichment.model,
        elapsed_seconds=0.0,
        external_evidence=claims,
        external_resolutions=resolutions,
        external_no_evidence=no_evidence,
    )

    writer_output = None
    if not args.skip_writer:
        writer_question = str(scenario.get("writer_question", question))
        writer_output = write_markdown(
            question=writer_question,
            context_bundle=enriched_context,
            config=config,
            api_key=api_key,
            run_id=run_id,
        )
        (output_dir / "writer_answer.md").write_text(
            writer_output.content,
            encoding="utf-8",
        )

    summary = _build_summary(
        run_id=run_id,
        benchmark_file=args.benchmark_file,
        model=config.enrichment.model,
        cases=cases,
        claims=claims,
        no_evidence=no_evidence,
        tool_calls=tool_calls,
        writer_ran=writer_output is not None,
    )
    write_json(output_dir / "benchmark_summary.json", summary, ensure_ascii=False)
    write_json(output_dir / "context_bundle.json", enriched_context, ensure_ascii=False)

    logger.info(
        "External-source benchmark completed run_id=%s passed=%s output_dir=%s",
        run_id,
        summary["passed"],
        output_dir,
    )
    if not summary["passed"]:
        raise SystemExit(1)


def _load_benchmark(path: Path) -> dict[str, Any]:
    """Load and validate a benchmark fixture."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("scenario"), dict):
        raise ValueError("Benchmark fixture must contain a scenario object.")
    if not isinstance(data.get("cases"), list):
        raise ValueError("Benchmark fixture must contain a cases list.")
    return data


def _build_context_bundle(city: str, question: str) -> dict[str, Any]:
    """Build a minimal writer-ready context bundle for the benchmark scenario."""
    return {
        "research_question": question,
        "analysis_mode": "city_by_city",
        "selected_cities": [city],
        "markdown": {
            "status": "success",
            "analysis_mode": "city_by_city",
            "excerpt_count": 0,
            "excerpts": [],
            "selected_city_names": [city],
            "inspected_city_names": [city],
            "selected_cities": [city.lower()],
            "inspected_cities": [city.lower()],
        },
    }


def _build_gap_manifest(city: str, cases: list[dict[str, Any]]) -> GapManifest:
    """Convert benchmark cases into the gap manifest consumed by the pipeline."""
    blank_fields: list[str] = []
    stale_fields: list[str] = []
    classifications: list[FieldClassification] = []
    for case in cases:
        field = str(case["field"])
        field_status = str(case.get("field_status", "blank"))
        if field_status == "stale":
            stale_fields.append(field)
        else:
            blank_fields.append(field)
        classifications.append(
            FieldClassification(
                field=field,
                classification="estimable_numerical",
                searchable=True,
                rationale="Benchmark field for governed external-source extraction.",
            )
        )
    return GapManifest(
        query_fields=classifications,
        city_gaps=[
            CityGap(
                city=city,
                blank_fields=blank_fields,
                stale_flags=stale_fields,
                search_priority="high",
            )
        ],
        non_estimable_fields=[],
    )


def _build_summary(
    *,
    run_id: str,
    benchmark_file: Path,
    model: str,
    cases: list[dict[str, Any]],
    claims: list[ExternalEvidenceClaim],
    no_evidence: list[Any],
    tool_calls: list[dict[str, object]],
    writer_ran: bool,
) -> dict[str, Any]:
    """Build case-level benchmark results from extracted claims."""
    case_results = [_score_case(case, claims, no_evidence) for case in cases]
    return {
        "run_id": run_id,
        "benchmark_file": str(benchmark_file),
        "model": model,
        "passed": all(result["passed"] for result in case_results),
        "case_results": case_results,
        "claim_count": len(claims),
        "no_evidence_count": len(no_evidence),
        "tool_call_count": len(tool_calls),
        "writer_ran": writer_ran,
    }


def _score_case(
    case: dict[str, Any],
    claims: list[ExternalEvidenceClaim],
    no_evidence: list[Any],
) -> dict[str, Any]:
    """Score one benchmark case against fixture expectations."""
    field = str(case["field"])
    expected = case.get("expected", {})
    field_claims = [claim for claim in claims if claim.field == field]
    expected_must_find = bool(expected.get("must_find", True))
    if not expected_must_find:
        no_evidence_found = any(record.field == field for record in no_evidence)
        return {
            "field": field,
            "passed": not field_claims and no_evidence_found,
            "expected": "no_evidence",
            "claim_count": len(field_claims),
            "no_evidence_recorded": no_evidence_found,
        }

    best_claim = field_claims[0] if field_claims else None
    source_ok = best_claim is not None and best_claim.source_id == expected.get("source_id")
    value_ok = best_claim is not None and _terms_present(
        expected.get("value_terms", []),
        f"{best_claim.value} {best_claim.unit or ''}",
    )
    quote_ok = best_claim is not None and _terms_present(
        expected.get("quote_terms", []),
        best_claim.quote,
    )
    return {
        "field": field,
        "passed": bool(source_ok and value_ok and quote_ok),
        "claim_count": len(field_claims),
        "source_ok": source_ok,
        "value_ok": value_ok,
        "quote_ok": quote_ok,
        "best_claim": best_claim.model_dump(mode="json") if best_claim else None,
    }


def _terms_present(terms: list[str], text: str) -> bool:
    """Return True when all expected terms appear case-insensitively."""
    normalized = text.casefold()
    return all(str(term).casefold() in normalized for term in terms)


if __name__ == "__main__":
    setup_logger()
    main()
