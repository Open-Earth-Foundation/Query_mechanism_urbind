import asyncio

import pytest

from backend.services.agents import MaxToolCallsExceeded, ToolCallLimitHooks


def test_tool_call_limit_hook_raises_when_limit_exceeded() -> None:
    hook = ToolCallLimitHooks(max_tool_calls=2)

    asyncio.run(hook.on_tool_start(None, None, None))  # type: ignore[arg-type]
    asyncio.run(hook.on_tool_start(None, None, None))  # type: ignore[arg-type]

    with pytest.raises(MaxToolCallsExceeded):
        asyncio.run(hook.on_tool_start(None, None, None))  # type: ignore[arg-type]
