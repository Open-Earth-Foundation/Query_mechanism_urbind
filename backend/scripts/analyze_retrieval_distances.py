"""
Brief: Summarize Chroma retrieval distance distributions from previous runs.

Inputs:
- CLI args:
  - --runs-dir: Directory containing run folders with `markdown/retrieval.json` (default: output).
  - --city: Optional city filter; repeatable.
  - --limit-runs: Optional max number of runs to scan (newest-first).
  - --thresholds: Optional comma-separated list of distance thresholds to tabulate (e.g. "0.5,1.0,2.0").
  - --show-per-run: If set, prints one-line per-run distance summary.
- Files:
  - `<runs-dir>/<run_id>/markdown/retrieval.json` (produced when VECTOR_STORE_ENABLED=true).

Outputs:
- Logs overall distance percentiles (min/p50/p90/p95/p99/max) and per-city summaries.
- Optionally logs a threshold table: how many chunks fall under each distance cutoff.

Notes:
- `retrieval.json` stores `distance` directly per retrieved chunk.

Usage (from project root):
- python -m backend.scripts.analyze_retrieval_distances --runs-dir output
- python -m backend.scripts.analyze_retrieval_distances --city Munich --city Leipzig --thresholds "0.5,1.0,2.0"
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistanceStats:
    count: int
    minimum: float
    p50: float
    p90: float
    p95: float
    p99: float
    maximum: float


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Summarize vector retrieval distance distributions from run artifacts."
    )
    parser.add_argument(
        "--runs-dir",
        default="output",
        help="Directory containing run folders (default: output).",
    )
    parser.add_argument(
        "--city",
        action="append",
        help="Optional city filter (repeatable).",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        help="Optional maximum number of runs to scan (newest-first).",
    )
    parser.add_argument(
        "--thresholds",
        help='Optional comma-separated distance thresholds (e.g. "0.5,1.0,2.0").',
    )
    parser.add_argument(
        "--show-per-run",
        action="store_true",
        help="Print one-line distance summary per run.",
    )
    return parser.parse_args()


def _parse_thresholds(raw: str | None) -> list[float]:
    if not raw:
        return []
    thresholds: list[float] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        thresholds.append(float(value))
    return sorted(set(thresholds))


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile for already-sorted data."""
    if not sorted_values:
        return 0.0
    if fraction <= 0:
        return sorted_values[0]
    if fraction >= 1:
        return sorted_values[-1]
    idx = int(round((len(sorted_values) - 1) * fraction))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def _compute_stats(values: list[float]) -> DistanceStats | None:
    if not values:
        return None
    ordered = sorted(values)
    return DistanceStats(
        count=len(ordered),
        minimum=ordered[0],
        p50=_percentile(ordered, 0.50),
        p90=_percentile(ordered, 0.90),
        p95=_percentile(ordered, 0.95),
        p99=_percentile(ordered, 0.99),
        maximum=ordered[-1],
    )


def _read_distance(value: object) -> float | None:
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return None
    if distance < 0:
        return None
    return distance


def _iter_retrieval_paths(runs_dir: Path) -> list[Path]:
    """Find retrieval.json files, handling both parent dir and specific run dir."""
    # Check if runs_dir itself contains markdown/retrieval.json (specific run folder)
    direct_path = runs_dir / "markdown" / "retrieval.json"
    if direct_path.exists():
        return [direct_path]
    
    # Otherwise, search for */markdown/retrieval.json (parent directory)
    candidates = list(runs_dir.glob("*/markdown/retrieval.json"))
    # newest-first by run folder mtime
    candidates.sort(key=lambda p: p.parent.parent.stat().st_mtime, reverse=True)
    return candidates


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _normalize_city_filters(cities: list[str] | None) -> set[str]:
    if not cities:
        return set()
    return {city.strip().casefold() for city in cities if city.strip()}


def _format_stats(stats: DistanceStats) -> str:
    return (
        f"count={stats.count} "
        f"min={stats.minimum:.4f} p50={stats.p50:.4f} p90={stats.p90:.4f} "
        f"p95={stats.p95:.4f} p99={stats.p99:.4f} max={stats.maximum:.4f}"
    )


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    city_filter = _normalize_city_filters(args.city)
    thresholds = _parse_thresholds(args.thresholds)

    retrieval_paths = _iter_retrieval_paths(runs_dir)
    if args.limit_runs and args.limit_runs > 0:
        retrieval_paths = retrieval_paths[: args.limit_runs]

    overall_distances: list[float] = []
    per_city_distances: dict[str, list[float]] = {}

    scanned = 0
    for path in retrieval_paths:
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        chunks = payload.get("chunks", [])
        if not isinstance(chunks, list):
            continue

        run_id = path.parent.parent.name
        run_distances: list[float] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            city = str(chunk.get("city_name", "")).strip()
            if city_filter and city.casefold() not in city_filter:
                continue
            distance = _read_distance(chunk.get("distance"))
            if distance is None:
                continue
            run_distances.append(distance)
            overall_distances.append(distance)
            if city:
                per_city_distances.setdefault(city, []).append(distance)

        scanned += 1
        if args.show_per_run and run_distances:
            stats = _compute_stats(run_distances)
            if stats:
                logger.info("run=%s %s", run_id, _format_stats(stats))

    if scanned == 0 or not overall_distances:
        logger.info("No retrieval distances found under runs_dir=%s", runs_dir)
        return

    overall_stats = _compute_stats(overall_distances)
    if overall_stats:
        logger.info("Overall distances (%d runs scanned): %s", scanned, _format_stats(overall_stats))

    for city_name in sorted(per_city_distances.keys()):
        stats = _compute_stats(per_city_distances[city_name])
        if not stats:
            continue
        logger.info("City=%s %s", city_name, _format_stats(stats))

    if thresholds:
        logger.info("Threshold table (distance <= cutoff):")
        ordered = sorted(overall_distances)
        total = len(ordered)
        for cutoff in thresholds:
            under = len([d for d in ordered if d <= cutoff])
            rate = (under / total) if total else 0.0
            logger.info("cutoff=%.4f kept=%d/%d (%.1f%%)", cutoff, under, total, rate * 100.0)


if __name__ == "__main__":
    main()

