"""Tests for the generic tool-calling loop.

These tests wire a ``FakeBackend`` up to a couple of simple in-memory
tools and assert that:

* the loop executes tool calls in order,
* tool results flow back into the conversation,
* unknown tool names are surfaced as structured errors,
* tool handler exceptions do not crash the loop,
* max_iterations is enforced.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aortacfd_agent.backends.base import Message, ToolCall, ToolSpec
from aortacfd_agent.backends.fake import FakeBackend, ScriptedStep
from aortacfd_agent.loop import AgentLoop


def _echo_tool() -> ToolSpec:
    def _handler(args):
        return {"echoed": args.get("value")}

    return ToolSpec(
        name="echo",
        description="Return the provided value.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        handler=_handler,
    )


def _boom_tool() -> ToolSpec:
    def _handler(_args):
        raise RuntimeError("kaboom")

    return ToolSpec(
        name="boom",
        description="Always raises.",
        parameters={"type": "object", "properties": {}},
        handler=_handler,
    )


def test_loop_executes_tool_then_finalises():
    backend = FakeBackend(
        script=[
            ScriptedStep(
                tool_calls=[ToolCall(id="a", name="echo", arguments={"value": "hi"})]
            ),
            ScriptedStep(text="all done"),
        ]
    )
    loop = AgentLoop(backend=backend, tools=[_echo_tool()], system_prompt="sys")
    result = loop.run("please echo hi")

    assert result.final_text == "all done"
    assert result.iterations == 2
    assert result.stopped_reason == "end_turn"

    tool_entries = [t for t in result.trace if t.kind == "tool"]
    assert len(tool_entries) == 1
    assert tool_entries[0].name == "echo"
    result_payload = json.loads(tool_entries[0].payload["result"])
    assert result_payload == {"echoed": "hi"}


def test_loop_handles_unknown_tool_as_error_message():
    backend = FakeBackend(
        script=[
            ScriptedStep(
                tool_calls=[ToolCall(id="a", name="ghost", arguments={})]
            ),
            ScriptedStep(text="recovered"),
        ]
    )
    loop = AgentLoop(backend=backend, tools=[_echo_tool()])
    result = loop.run("call ghost")

    tool_entries = [t for t in result.trace if t.kind == "tool"]
    assert len(tool_entries) == 1
    err = json.loads(tool_entries[0].payload["result"])
    assert "unknown tool" in err["error"]
    assert result.final_text == "recovered"


def test_loop_catches_handler_exception():
    backend = FakeBackend(
        script=[
            ScriptedStep(
                tool_calls=[ToolCall(id="a", name="boom", arguments={})]
            ),
            ScriptedStep(text="ok"),
        ]
    )
    loop = AgentLoop(backend=backend, tools=[_boom_tool()])
    result = loop.run("call boom")

    tool_entries = [t for t in result.trace if t.kind == "tool"]
    err = json.loads(tool_entries[0].payload["result"])
    assert "RuntimeError" in err["error"]
    assert result.final_text == "ok"


def test_loop_enforces_max_iterations():
    # Model keeps calling the same tool forever. Loop must give up.
    infinite = [
        ScriptedStep(
            tool_calls=[ToolCall(id=str(i), name="echo", arguments={"value": "x"})]
        )
        for i in range(10)
    ]
    backend = FakeBackend(script=infinite)
    loop = AgentLoop(backend=backend, tools=[_echo_tool()], max_iterations=3)
    result = loop.run("loop forever")
    assert result.iterations == 3
    assert result.stopped_reason == "max_iterations"
