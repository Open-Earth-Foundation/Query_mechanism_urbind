"""Placeholder external-source stage for the enrichment pipeline.

The concrete external sources library is intentionally not implemented in
this PR cleanup. This module keeps the pipeline boundary explicit so a future
implementation can run between field decomposition and city-gap detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from backend.modules.web_researcher.models import FieldDecomposition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExternalSourceStageResult:
    """Result contract for the future external-source stage."""

    findings: list[dict[str, object]] = field(default_factory=list)


def run_external_source_stage(
    decomposition: FieldDecomposition,
    _context_bundle: dict[str, Any],
) -> ExternalSourceStageResult:
    """Return an empty external-source result without changing context."""
    logger.info(
        "External sources library is not implemented yet; "
        "continuing gap detection without external-source results. fields=%d",
        len(decomposition.query_fields),
    )
    return ExternalSourceStageResult()


__all__ = ["ExternalSourceStageResult", "run_external_source_stage"]
