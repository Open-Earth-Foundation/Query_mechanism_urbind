import asyncio
from types import SimpleNamespace

import pytest

from backend.services import agents as agents_service


def test_run_agent_sync_runs_from_nested_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_result = SimpleNamespace(final_output={"status": "ok"})

    async def _fake_runner_run(
        agent: object,
        input_data: str,
        max_turns: int = 10,
        hooks: object | None = None,
    ) -> object:
        _ = agent, input_data, max_turns, hooks
        return expected_result

    monkeypatch.setattr(agents_service.Runner, "run", _fake_runner_run)
    dummy_agent = SimpleNamespace(name="Dummy Agent")

    async def _invoke() -> object:
        return agents_service.run_agent_sync(
            dummy_agent,
            '{"input":"value"}',
            max_turns=3,
        )

    result = asyncio.run(_invoke())
    assert result is expected_result


def test_run_agent_sync_nested_event_loop_propagates_worker_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_runner_run(
        agent: object,
        input_data: str,
        max_turns: int = 10,
        hooks: object | None = None,
    ) -> object:
        _ = agent, input_data, max_turns, hooks
        raise RuntimeError("nested run failure")

    monkeypatch.setattr(agents_service.Runner, "run", _fake_runner_run)
    dummy_agent = SimpleNamespace(name="Dummy Agent")

    async def _invoke() -> object:
        return agents_service.run_agent_sync(
            dummy_agent,
            '{"input":"value"}',
            max_turns=3,
        )

    with pytest.raises(RuntimeError, match="nested run failure"):
        asyncio.run(_invoke())
