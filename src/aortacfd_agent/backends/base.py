"""Provider-agnostic LLM backend interface.

Every concrete backend (Anthropic, OpenAI-compatible, Ollama, Fake) implements
``LLMBackend.chat`` with the same input/output types. The rest of the agent
package — the tool-calling loop, the config agent, the CLI — depends only on
these types. Swapping providers is a one-line change in the backend factory.

Design notes
------------
* We use plain dataclasses instead of pydantic for the wire types so this
  module has zero hard runtime dependencies. Every concrete backend brings
  its own dependency (``anthropic``, ``openai``, ``requests`` for Ollama).
* Tool schemas are expressed as JSON Schema dicts because all three real
  providers (Claude, OpenAI, Ollama) accept JSON Schema for tool parameters.
  That keeps the translation layer in each adapter trivial.
* ``AgentResponse`` is either a final text answer or a list of tool calls
  the model wants to make next. The loop in ``agent.loop.AgentLoop`` keeps
  calling ``backend.chat`` until it sees a response with no tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single message in the chat history.

    ``role`` is one of ``"system"``, ``"user"``, ``"assistant"``, ``"tool"``.
    ``tool_call_id`` is only set on tool-result messages, so the backend can
    match the result to the call the model made. ``tool_calls`` is only set
    on assistant messages that requested tool execution.
    """

    role: str
    content: str = ""
    tool_call_id: Optional[str] = None
    tool_calls: List["ToolCall"] = field(default_factory=list)


@dataclass
class ToolSpec:
    """Declarative description of a tool the model can call.

    ``parameters`` must be a JSON Schema object. ``handler`` is the Python
    callable that the agent loop will invoke when the model picks this tool.
    The handler receives the parsed argument dict and must return something
    JSON-serialisable (a dict, list, str, or number).
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Any]


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """The result of executing a tool call, echoed back to the model."""

    tool_call_id: str
    name: str
    content: str  # JSON string or plain text
    is_error: bool = False


@dataclass
class AgentResponse:
    """One turn of output from the backend.

    Exactly one of ``text`` or ``tool_calls`` is meaningful per turn:
    * If ``tool_calls`` is non-empty, the loop must execute them and send
      the results back in the next ``chat`` call.
    * Otherwise, ``text`` is the model's final answer for this turn.

    ``raw`` holds the provider-specific response object for debugging or
    logging — the loop itself never looks at it.
    """

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw: Any = None


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol every backend must implement.

    A backend is stateless: the full message history is passed on every call.
    That means tests can record a sequence of responses and replay them
    deterministically (see ``FakeBackend``), and production backends can
    be swapped mid-run without carrying any hidden state.
    """

    name: str

    def chat(
        self,
        messages: List[Message],
        tools: List[ToolSpec],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AgentResponse:
        """Send the conversation to the model and return the next turn."""
        ...


# ---------------------------------------------------------------------------
# JSON helpers shared by multiple backends
# ---------------------------------------------------------------------------


def tool_specs_to_json_schema(tools: List[ToolSpec]) -> List[Dict[str, Any]]:
    """Serialize ``ToolSpec`` list to the JSON-Schema-ish form used by
    OpenAI and Ollama (Anthropic has a slightly different shape, handled
    in its adapter). Exposed here so tests can introspect the wire format.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]
