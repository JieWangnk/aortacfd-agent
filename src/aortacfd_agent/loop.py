"""Generic tool-calling loop.

Runs a ReAct-style conversation: the loop hands the model the tools, calls
the backend, executes any tool calls, feeds the results back, and repeats
until the model stops calling tools (or ``max_iterations`` is reached).

This is the one piece of agent code that does not care which provider or
which task is in play — it works identically for the config agent, a
future repair agent, or a narrative agent. That isolation is why swapping
backends or adding new tools is a one-line change elsewhere.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .backends.base import (
    AgentResponse,
    LLMBackend,
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
)

logger = logging.getLogger(__name__)


@dataclass
class TraceEntry:
    """One recorded step in an agent run, for auditability."""

    kind: str  # "model" | "tool"
    name: str = ""
    payload: Any = None
    duration_s: float = 0.0


@dataclass
class AgentRunResult:
    """Final result of an ``AgentLoop.run`` call."""

    final_text: str
    messages: List[Message]
    trace: List[TraceEntry] = field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = "end_turn"  # "end_turn" | "max_iterations" | "error"

    def tool_calls(self) -> List[TraceEntry]:
        return [t for t in self.trace if t.kind == "tool"]


class AgentLoop:
    """A provider-agnostic tool-use loop.

    Usage::

        loop = AgentLoop(backend=backend, tools=tool_specs,
                         system_prompt="You are a CFD config assistant.")
        result = loop.run("Set up a simulation for the BPM120 patient.")
        print(result.final_text)
    """

    def __init__(
        self,
        backend: LLMBackend,
        tools: List[ToolSpec],
        system_prompt: Optional[str] = None,
        max_iterations: int = 8,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        self.backend = backend
        self.tools = tools
        self._tools_by_name: Dict[str, ToolSpec] = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.temperature = temperature

    # -- public API ----------------------------------------------------------

    def run(
        self,
        user_message: str,
        initial_messages: Optional[List[Message]] = None,
    ) -> AgentRunResult:
        """Execute the loop until the model stops calling tools."""
        messages: List[Message] = list(initial_messages or [])
        messages.append(Message(role="user", content=user_message))

        trace: List[TraceEntry] = []
        stopped = "end_turn"
        final_text = ""
        iterations = 0

        for iteration in range(self.max_iterations):
            iterations = iteration + 1

            t0 = time.perf_counter()
            try:
                response: AgentResponse = self.backend.chat(
                    messages=messages,
                    tools=self.tools,
                    system=self.system_prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            except Exception as exc:  # noqa: BLE001 — logged for trace
                trace.append(
                    TraceEntry(
                        kind="model",
                        name=self.backend.name,
                        payload={"error": str(exc)},
                        duration_s=time.perf_counter() - t0,
                    )
                )
                stopped = "error"
                break

            trace.append(
                TraceEntry(
                    kind="model",
                    name=self.backend.name,
                    payload={
                        "text": response.text,
                        "tool_calls": [
                            {"name": tc.name, "arguments": tc.arguments}
                            for tc in response.tool_calls
                        ],
                        "stop_reason": response.stop_reason,
                    },
                    duration_s=time.perf_counter() - t0,
                )
            )

            # Record the assistant turn in the conversation so the next
            # model call sees its own tool requests (required by both
            # Anthropic and OpenAI-style APIs).
            messages.append(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=list(response.tool_calls),
                )
            )

            if not response.tool_calls:
                final_text = response.text
                stopped = response.stop_reason or "end_turn"
                break

            # Execute every tool the model asked for, in order.
            for call in response.tool_calls:
                result = self._execute_tool(call)
                trace.append(
                    TraceEntry(
                        kind="tool",
                        name=call.name,
                        payload={
                            "arguments": call.arguments,
                            "result": result.content,
                            "is_error": result.is_error,
                        },
                    )
                )
                messages.append(
                    Message(
                        role="tool",
                        content=result.content,
                        tool_call_id=result.tool_call_id,
                    )
                )
        else:
            stopped = "max_iterations"

        return AgentRunResult(
            final_text=final_text,
            messages=messages,
            trace=trace,
            iterations=iterations,
            stopped_reason=stopped,
        )

    # -- internals -----------------------------------------------------------

    def _execute_tool(self, call: ToolCall) -> ToolResult:
        spec = self._tools_by_name.get(call.name)
        if spec is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=json.dumps(
                    {"error": f"unknown tool '{call.name}'"}
                ),
                is_error=True,
            )
        try:
            result = spec.handler(call.arguments or {})
        except Exception as exc:  # noqa: BLE001 - surface to model
            logger.warning("tool %s raised: %s", call.name, exc)
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                is_error=True,
            )

        if not isinstance(result, str):
            try:
                result_str = json.dumps(result, default=str)
            except TypeError:
                result_str = str(result)
        else:
            result_str = result

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=result_str,
            is_error=False,
        )
