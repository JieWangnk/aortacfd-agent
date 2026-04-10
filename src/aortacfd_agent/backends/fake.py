"""Offline scripted backend used by unit tests and ``--dry-run`` mode.

The ``FakeBackend`` replays a pre-recorded list of ``ScriptedStep`` objects.
Each step is one turn: either "call these tools" or "emit this final text".
Tests can therefore exercise the full tool-calling loop — including the
config agent and the schema validator — without ever touching the network.

Typical use::

    from agent.backends.fake import FakeBackend, ScriptedStep
    from agent.backends.base import ToolCall

    backend = FakeBackend(script=[
        ScriptedStep(tool_calls=[ToolCall("1", "inspect_geometry", {"case_dir": "..."})]),
        ScriptedStep(text=json.dumps({"physics": {"model": "laminar"}, ...})),
    ])

The loop will call ``inspect_geometry``, feed the result back, then receive
the JSON config on the second turn. Because the script is fixed, tests
remain deterministic even if the agent loop changes internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .base import AgentResponse, LLMBackend, Message, ToolCall, ToolSpec


@dataclass
class ScriptedStep:
    """One pre-recorded turn of the fake conversation."""

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"


class FakeBackend:
    """Deterministic LLM backend for tests and offline demos.

    Parameters
    ----------
    script
        Ordered list of responses to emit, one per ``chat`` call. The
        backend raises ``IndexError`` if the script is exhausted — tests
        can rely on that to catch unexpected extra turns.
    """

    name = "fake"

    def __init__(self, script: List[ScriptedStep]):
        self.script = list(script)
        self._cursor = 0
        self.calls: List[List[Message]] = []  # recorded for assertions

    def chat(
        self,
        messages: List[Message],
        tools: List[ToolSpec],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AgentResponse:
        if self._cursor >= len(self.script):
            raise IndexError(
                f"FakeBackend script exhausted after {self._cursor} turns; "
                f"tests should supply one ScriptedStep per expected model turn."
            )

        # Record the incoming history so tests can assert what the agent sent.
        self.calls.append(list(messages))

        step = self.script[self._cursor]
        self._cursor += 1
        return AgentResponse(
            text=step.text,
            tool_calls=list(step.tool_calls),
            stop_reason=step.stop_reason,
            raw={"fake_step": self._cursor - 1},
        )

    # Convenience for tests --------------------------------------------------

    def remaining(self) -> int:
        return len(self.script) - self._cursor
