"""Phase 1 fan-out: gather local data before per-city gap detection.

Sits between Phase 0 (field decomposition) and Phase 2 (per-city gap
detection).  Runs cheap, deterministic local lookups + similarity
retrieval against pre-built indexes — no live web access, no LLM calls
for orchestration.

Two sub-flows run concurrently:
1. Structured lookups (e.g. Bundesnetzagentur) for fields the manifest
   says they cover.
2. Benchmark vector retrieval against the local benchmark Chroma
   collection (when present).

Output is merged into ``context_bundle["phase1"]`` so Phase 2's gap
detector sees the values without needing another LLM round-trip.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from backend.modules.sources.manifest import Manifest, load_manifest
from backend.modules.web_researcher.data_lookups import (
    find_matching_structured_lookups,
)
from backend.modules.web_researcher.models import (
    BenchmarkExcerptRecord,
    FieldDecomposition,
    Phase1Artefacts,
    StructuredLookupResult,
)

logger = logging.getLogger(__name__)


def _extract_cities_from_bundle(context_bundle: dict[str, Any]) -> list[str]:
    """Pull city names out of the context bundle, preferring user-selected ones."""
    markdown = context_bundle.get("markdown")
    if not isinstance(markdown, dict):
        return []
    for key in ("selected_city_names", "selected_cities", "inspected_city_names", "inspected_cities"):
        value = markdown.get(key)
        if isinstance(value, list) and value:
            return [str(v) for v in value if isinstance(v, str) and v.strip()]
    return []


def _benchmark_queries(decomposition: FieldDecomposition) -> list[str]:
    """Build similarity queries from the decomposed fields."""
    out: list[str] = []
    for field in decomposition.query_fields:
        if field.classification == "non_estimable":
            continue
        # The field name plus the rationale gives the embedder some context.
        text = field.field.replace("_", " ")
        if field.rationale:
            text = f"{text} — {field.rationale}"
        out.append(text)
    return out


def _safe_load_manifest(path: Path | None) -> Manifest | None:
    try:
        return load_manifest(path) if path else load_manifest()
    except FileNotFoundError:
        logger.info("phase1_fanout: no sources manifest at %s; skipping local lookups", path)
        return None
    except Exception:  # noqa: BLE001
        logger.warning("phase1_fanout: failed to load sources manifest", exc_info=True)
        return None


def _run_structured_lookups(
    decomposition: FieldDecomposition,
    cities: list[str],
    manifest: Manifest | None,
) -> list[StructuredLookupResult]:
    if manifest is None:
        return []
    try:
        return find_matching_structured_lookups(decomposition, cities, manifest)
    except Exception:
        logger.warning("phase1_fanout: structured lookups raised", exc_info=True)
        return []


def _run_benchmark_retrieval(
    decomposition: FieldDecomposition,
    *,
    benchmark_retrieve: Callable | None,
    k: int,
    persist_path: Path | None,
) -> list[BenchmarkExcerptRecord]:
    if benchmark_retrieve is None:
        # Lazy default — keeps embeddings out of the import path.
        from backend.modules.vector_store.benchmark_retriever import (
            DEFAULT_BENCHMARK_PERSIST_PATH,
            retrieve_benchmark_excerpts,
        )
        benchmark_retrieve = retrieve_benchmark_excerpts
        if persist_path is None:
            persist_path = DEFAULT_BENCHMARK_PERSIST_PATH

    if persist_path is None or not persist_path.exists():
        logger.info(
            "phase1_fanout: benchmark Chroma path missing (%s); skipping benchmark retrieval",
            persist_path,
        )
        return []

    queries = _benchmark_queries(decomposition)
    if not queries:
        return []

    try:
        excerpts = benchmark_retrieve(queries, k=k, persist_path=persist_path)
    except Exception:
        logger.warning("phase1_fanout: benchmark retrieval raised", exc_info=True)
        return []

    out: list[BenchmarkExcerptRecord] = []
    for excerpt in excerpts:
        out.append(
            BenchmarkExcerptRecord(
                chunk_id=excerpt.chunk_id,
                source_id=excerpt.source_id,
                ingestion_id=excerpt.ingestion_id,
                source_path=excerpt.source_path,
                tier=excerpt.tier,
                doc_slug=excerpt.doc_slug,
                heading_path=excerpt.heading_path,
                block_type=excerpt.block_type,
                raw_text=excerpt.raw_text,
                distance=excerpt.distance,
                chunk_index=excerpt.chunk_index,
            )
        )
    return out


def run_phase1_fanout(
    decomposition: FieldDecomposition,
    context_bundle: dict[str, Any],
    *,
    manifest: Manifest | None = None,
    manifest_path: Path | None = None,
    cities_override: list[str] | None = None,
    benchmark_k: int = 10,
    benchmark_persist_path: Path | None = None,
    benchmark_retrieve: Callable | None = None,
) -> Phase1Artefacts:
    """Run the Phase 1 fan-out and return the artefacts.

    The artefacts are also a convenient input for ``merge_phase1_into_bundle``.
    """
    start = time.monotonic()
    cities = cities_override or _extract_cities_from_bundle(context_bundle)
    field_names = [f.field for f in decomposition.query_fields]

    if manifest is None:
        manifest = _safe_load_manifest(manifest_path)

    structured: list[StructuredLookupResult] = []
    benchmarks: list[BenchmarkExcerptRecord] = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        struct_future = pool.submit(
            _run_structured_lookups, decomposition, cities, manifest
        )
        bench_future = pool.submit(
            _run_benchmark_retrieval,
            decomposition,
            benchmark_retrieve=benchmark_retrieve,
            k=benchmark_k,
            persist_path=benchmark_persist_path,
        )
        structured = struct_future.result()
        benchmarks = bench_future.result()

    elapsed = time.monotonic() - start
    artefacts = Phase1Artefacts(
        structured_lookups=structured,
        benchmark_excerpts=benchmarks,
        queried_cities=cities,
        queried_fields=field_names,
        elapsed_seconds=elapsed,
    )
    logger.info(
        "phase1_fanout: cities=%d fields=%d structured=%d benchmarks=%d elapsed=%.2fs",
        len(cities),
        len(field_names),
        len(structured),
        len(benchmarks),
        elapsed,
    )
    return artefacts


def merge_phase1_into_bundle(
    context_bundle: dict[str, Any],
    artefacts: Phase1Artefacts,
) -> dict[str, Any]:
    """Return a shallow copy of the bundle with phase1 artefacts attached.

    The original bundle is left untouched.  Existing phase1 keys are
    preserved when this function is called repeatedly.
    """
    new_bundle = dict(context_bundle)
    payload = artefacts.model_dump()
    new_bundle["phase1"] = payload
    return new_bundle


__all__ = [
    "merge_phase1_into_bundle",
    "run_phase1_fanout",
]
