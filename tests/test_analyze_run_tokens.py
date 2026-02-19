import json
from pathlib import Path

from backend.scripts.analyze_run_tokens import _parse_llm_usage_lines


def test_parse_llm_usage_lines_ignores_non_usage_entries(tmp_path: Path) -> None:
    run_log = tmp_path / "run.log"
    usage_payload_1 = {
        "event": "llm_usage",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 7},
            "total_tokens": 120,
        },
    }
    usage_payload_2 = {
        "event": "llm_usage",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total": 15,
        },
    }
    llm_end_payload = {
        "event": "llm_end",
        "response": {
            "usage": {
                "input_tokens": 9999,
                "output_tokens": 8888,
                "output_tokens_details": {"reasoning_tokens": 7777},
            }
        },
    }
    summary_payload = {
        "calls": 1,
        "totals": {"input_tokens": 9999, "output_tokens": 8888},
    }

    lines = [
        f'2026-02-09 11:00:00 agents.py:298 - INFO - LLM_USAGE {json.dumps(usage_payload_1)}',
        f'2026-02-09 11:00:01 agents.py:312 - INFO - {json.dumps(llm_end_payload)}',
        f'2026-02-09 11:00:02 run_logger.py:293 - INFO - LLM_USAGE_SUMMARY {json.dumps(summary_payload)}',
        f'2026-02-09 11:00:03 agents.py:298 - INFO - LLM_USAGE {json.dumps(usage_payload_2)}',
    ]
    run_log.write_text("\n".join(lines), encoding="utf-8")

    calls, total_input, total_output, total_reasoning = _parse_llm_usage_lines(run_log)

    assert calls == 2
    assert total_input == 110
    assert total_output == 25
    assert total_reasoning == 7
