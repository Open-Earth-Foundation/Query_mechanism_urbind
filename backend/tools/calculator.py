from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def sum_numbers(numbers: list[float], source: str = "unknown") -> float:
    """Return arithmetic sum for numeric inputs with structured diagnostics."""
    try:
        normalized_inputs = [float(value) for value in numbers]
    except (TypeError, ValueError) as exc:
        logger.exception(
            "CALCULATOR_SUM_ERROR %s",
            json.dumps(
                {
                    "source": source,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "raw_inputs": numbers,
                },
                ensure_ascii=False,
            ),
        )
        raise
    total = 0.0
    for value in normalized_inputs:
        total += value
    logger.info(
        "CALCULATOR_SUM %s",
        json.dumps(
            {"source": source, "inputs": normalized_inputs, "output": total},
            ensure_ascii=False,
        ),
    )
    return total


__all__ = ["sum_numbers"]
