import json
import logging

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
