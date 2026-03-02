from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def _run_operation(
    operation: str,
    *,
    numbers: list[float] | None = None,
    left_operand: float | None = None,
    right_operand: float | None = None,
) -> float:
    """Run a supported arithmetic operation on normalized numeric inputs."""
    if operation == "sum":
        if numbers is None:
            raise ValueError("Missing numbers for sum operation.")
        total = 0.0
        for value in numbers:
            total += value
        return total

    if operation == "multiply":
        if numbers is None:
            raise ValueError("Missing numbers for multiply operation.")
        total = 1.0
        for value in numbers:
            total *= value
        return total

    if operation == "subtract":
        if left_operand is None or right_operand is None:
            raise ValueError("Missing operands for subtract operation.")
        return left_operand - right_operand

    if operation == "divide":
        if left_operand is None or right_operand is None:
            raise ValueError("Missing operands for divide operation.")
        if right_operand == 0.0:
            raise ValueError("Cannot divide by zero.")
        return left_operand / right_operand

    raise ValueError(f"Unsupported calculator operation: {operation}")


def _log_calculation_error(
    *,
    event_name: str,
    source: str,
    raw_inputs: dict[str, object],
    error: Exception,
) -> None:
    """Log a structured calculator error payload."""
    logger.exception(
        "%s %s",
        event_name,
        json.dumps(
            {
                "source": source,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "raw_inputs": raw_inputs,
            },
            ensure_ascii=False,
        ),
    )


def _log_calculation_success(
    *,
    event_name: str,
    source: str,
    input_payload: dict[str, object],
    output: float,
) -> None:
    """Log a structured calculator success payload."""
    logger.info(
        "%s %s",
        event_name,
        json.dumps(
            {"source": source, **input_payload, "output": output},
            ensure_ascii=False,
        ),
    )


def sum_numbers(numbers: list[float], source: str = "unknown") -> float:
    """Return arithmetic sum for numeric inputs with structured diagnostics."""
    raw_inputs = {"numbers": numbers}
    try:
        normalized_inputs = [float(value) for value in numbers]
        total = _run_operation("sum", numbers=normalized_inputs)
    except (TypeError, ValueError) as exc:
        _log_calculation_error(
            event_name="CALCULATOR_SUM_ERROR",
            source=source,
            raw_inputs=raw_inputs,
            error=exc,
        )
        raise
    _log_calculation_success(
        event_name="CALCULATOR_SUM",
        source=source,
        input_payload={"inputs": normalized_inputs},
        output=total,
    )
    return total


def subtract_numbers(
    minuend: float,
    subtrahend: float,
    source: str = "unknown",
) -> float:
    """Return arithmetic subtraction result for two numeric operands."""
    raw_inputs = {"minuend": minuend, "subtrahend": subtrahend}
    try:
        normalized_minuend = float(minuend)
        normalized_subtrahend = float(subtrahend)
        result = _run_operation(
            "subtract",
            left_operand=normalized_minuend,
            right_operand=normalized_subtrahend,
        )
    except (TypeError, ValueError) as exc:
        _log_calculation_error(
            event_name="CALCULATOR_SUBTRACT_ERROR",
            source=source,
            raw_inputs=raw_inputs,
            error=exc,
        )
        raise
    _log_calculation_success(
        event_name="CALCULATOR_SUBTRACT",
        source=source,
        input_payload={
            "minuend": normalized_minuend,
            "subtrahend": normalized_subtrahend,
        },
        output=result,
    )
    return result


def multiply_numbers(numbers: list[float], source: str = "unknown") -> float:
    """Return arithmetic product for numeric inputs with structured diagnostics."""
    raw_inputs = {"numbers": numbers}
    try:
        normalized_inputs = [float(value) for value in numbers]
        product = _run_operation("multiply", numbers=normalized_inputs)
    except (TypeError, ValueError) as exc:
        _log_calculation_error(
            event_name="CALCULATOR_MULTIPLY_ERROR",
            source=source,
            raw_inputs=raw_inputs,
            error=exc,
        )
        raise
    _log_calculation_success(
        event_name="CALCULATOR_MULTIPLY",
        source=source,
        input_payload={"inputs": normalized_inputs},
        output=product,
    )
    return product


def divide_numbers(
    dividend: float,
    divisor: float,
    source: str = "unknown",
) -> float:
    """Return arithmetic division result for two numeric operands."""
    raw_inputs = {"dividend": dividend, "divisor": divisor}
    try:
        normalized_dividend = float(dividend)
        normalized_divisor = float(divisor)
        result = _run_operation(
            "divide",
            left_operand=normalized_dividend,
            right_operand=normalized_divisor,
        )
    except (TypeError, ValueError) as exc:
        _log_calculation_error(
            event_name="CALCULATOR_DIVIDE_ERROR",
            source=source,
            raw_inputs=raw_inputs,
            error=exc,
        )
        raise
    _log_calculation_success(
        event_name="CALCULATOR_DIVIDE",
        source=source,
        input_payload={"dividend": normalized_dividend, "divisor": normalized_divisor},
        output=result,
    )
    return result


__all__ = [
    "sum_numbers",
    "subtract_numbers",
    "multiply_numbers",
    "divide_numbers",
]
