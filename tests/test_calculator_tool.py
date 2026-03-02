import json
import logging

import pytest

from backend.tools import calculator


def test_sum_numbers_returns_value_and_logs_inputs_and_output(
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger=calculator.__name__)

    result = calculator.sum_numbers([1, 2.5, -0.5])

    assert result == 3.0
    records = [record.message for record in caplog.records if "CALCULATOR_SUM " in record.message]
    assert records
    payload = json.loads(records[-1].split("CALCULATOR_SUM ", 1)[1])
    assert payload == {"source": "unknown", "inputs": [1.0, 2.5, -0.5], "output": 3.0}


def test_subtract_numbers_returns_value_and_logs_inputs_and_output(caplog) -> None:
    caplog.set_level(logging.INFO, logger=calculator.__name__)

    result = calculator.subtract_numbers(10, 3)

    assert result == 7.0
    records = [record.message for record in caplog.records if "CALCULATOR_SUBTRACT " in record.message]
    assert records
    payload = json.loads(records[-1].split("CALCULATOR_SUBTRACT ", 1)[1])
    assert payload == {
        "source": "unknown",
        "minuend": 10.0,
        "subtrahend": 3.0,
        "output": 7.0,
    }


def test_multiply_numbers_returns_value_and_logs_inputs_and_output(caplog) -> None:
    caplog.set_level(logging.INFO, logger=calculator.__name__)

    result = calculator.multiply_numbers([2, 3, 4])

    assert result == 24.0
    records = [record.message for record in caplog.records if "CALCULATOR_MULTIPLY " in record.message]
    assert records
    payload = json.loads(records[-1].split("CALCULATOR_MULTIPLY ", 1)[1])
    assert payload == {"source": "unknown", "inputs": [2.0, 3.0, 4.0], "output": 24.0}


def test_divide_numbers_returns_value_and_logs_inputs_and_output(caplog) -> None:
    caplog.set_level(logging.INFO, logger=calculator.__name__)

    result = calculator.divide_numbers(12, 3)

    assert result == 4.0
    records = [record.message for record in caplog.records if "CALCULATOR_DIVIDE " in record.message]
    assert records
    payload = json.loads(records[-1].split("CALCULATOR_DIVIDE ", 1)[1])
    assert payload == {"source": "unknown", "dividend": 12.0, "divisor": 3.0, "output": 4.0}


def test_divide_numbers_raises_on_zero_divisor_and_logs_error(caplog) -> None:
    caplog.set_level(logging.ERROR, logger=calculator.__name__)

    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calculator.divide_numbers(12, 0)

    records = [record.message for record in caplog.records if "CALCULATOR_DIVIDE_ERROR " in record.message]
    assert records
    payload = json.loads(records[-1].split("CALCULATOR_DIVIDE_ERROR ", 1)[1])
    assert payload["source"] == "unknown"
    assert payload["error_type"] == "ValueError"
    assert payload["raw_inputs"] == {"dividend": 12, "divisor": 0}


def test_multiply_numbers_raises_for_invalid_input_and_logs_error(caplog) -> None:
    caplog.set_level(logging.ERROR, logger=calculator.__name__)

    with pytest.raises(ValueError):
        calculator.multiply_numbers([2, "abc"])

    records = [record.message for record in caplog.records if "CALCULATOR_MULTIPLY_ERROR " in record.message]
    assert records
    payload = json.loads(records[-1].split("CALCULATOR_MULTIPLY_ERROR ", 1)[1])
    assert payload["source"] == "unknown"
    assert payload["error_type"] == "ValueError"
    assert payload["raw_inputs"] == {"numbers": [2, "abc"]}
