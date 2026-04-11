"""Anthropic Claude backend.

Wraps the official ``anthropic`` Python SDK. Activated when the agent
factory is given ``provider: anthropic`` in its config. The SDK is an
optional dependency — this module only imports it inside ``__init__``
so the rest of the agent package still works without it installed.

Mapping between our wire types and Claude's native types:

* ``Message`` rows become ``{"role": ..., "content": ...}`` entries.
* Assistant turns with tool calls become a ``content`` list of
  ``tool_use`` blocks. Tool results become a ``user`` message with
  ``tool_result`` blocks whose ``tool_use_id`` matches the call id.
* ``ToolSpec`` is translated to Claude's ``{"name", "description",
  "input_schema"}`` format.
* A ``tool_use`` stop reason yields an ``AgentResponse.tool_calls``;
  any other stop reason yields a final text response.

Claude is the recommended backend for production runs because it handles
multi-step tool planning noticeably better than the 7-8B local models.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Type

from .base import AgentResponse, Message, ToolCall, ToolSpec

logger = logging.getLogger(__name__)


def _transient_exception_types() -> Tuple[Type[BaseException], ...]:
    """Return Anthropic SDK exception types that are worth retrying.

    Imported lazily so the module stays importable without the SDK. The
    list deliberately excludes authentication / bad request / permission
    errors — those indicate a caller bug, not a transient outage.
    """
    try:
        import anthropic  # type: ignore
    except ImportError:
        return tuple()

    retryable: List[Type[BaseException]] = []
    for name in (
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "RateLimitError",
        "InternalServerError",
    ):
        cls = getattr(anthropic, name, None)
        if cls is not None and isinstance(cls, type):
            retryable.append(cls)
    return tuple(retryable)


class AnthropicBackend:
    """LLM backend that calls Claude via the ``anthropic`` Python SDK."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "The Anthropic backend requires the 'anthropic' package. "
                "Install it with: pip install anthropic"
            ) from exc

        kwargs: Dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**kwargs)
        self.model = model

    # -- translation helpers -------------------------------------------------

    @staticmethod
    def _tools_to_anthropic(tools: List[ToolSpec]) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    @staticmethod
    def _messages_to_anthropic(messages: List[Message]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        i = 0
        while i < len(messages):
            m = messages[i]

            if m.role == "assistant":
                # An assistant message may carry both text and tool_use blocks.
                blocks: List[Dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
                i += 1
                continue

            if m.role == "tool":
                # Consecutive tool-result messages collapse into one user turn.
                tool_blocks: List[Dict[str, Any]] = []
                while i < len(messages) and messages[i].role == "tool":
                    tm = messages[i]
                    tool_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tm.tool_call_id or "",
                            "content": tm.content,
                        }
                    )
                    i += 1
                out.append({"role": "user", "content": tool_blocks})
                continue

            # Plain user message
            out.append({"role": "user", "content": m.content})
            i += 1
        return out

    # -- retry helper --------------------------------------------------------

    def _call_with_retry(
        self,
        payload: Dict[str, Any],
        *,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> Any:
        """Call ``messages.create`` with exponential backoff on transient errors.

        Retries on the SDK's network / rate-limit / 5xx exception classes.
        Auth errors, bad-request errors, and permission errors are *not*
        retried — those are caller bugs, not transient outages.
        """
        transient = _transient_exception_types()
        last_exc: Optional[BaseException] = None
        for attempt in range(max_attempts):
            try:
                return self._client.messages.create(**payload)
            except transient as exc:  # type: ignore[misc]
                last_exc = exc
                if attempt == max_attempts - 1:
                    break
                delay = min(max_delay, base_delay * (2 ** attempt))
                delay += random.uniform(0, 1)
                logger.warning(
                    "anthropic transient error (%s): %s — retry %d/%d in %.1fs",
                    type(exc).__name__,
                    exc,
                    attempt + 1,
                    max_attempts - 1,
                    delay,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    # -- main entry point ----------------------------------------------------

    def chat(
        self,
        messages: List[Message],
        tools: List[ToolSpec],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AgentResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": self._messages_to_anthropic(messages),
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self._tools_to_anthropic(tools)

        resp = self._call_with_retry(payload)

        # Extract text and tool calls from the returned content blocks.
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for block in getattr(resp, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                raw_input = getattr(block, "input", {}) or {}
                if isinstance(raw_input, str):
                    try:
                        raw_input = json.loads(raw_input)
                    except json.JSONDecodeError:
                        raw_input = {"_raw": raw_input}
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=raw_input,
                    )
                )

        return AgentResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=getattr(resp, "stop_reason", "end_turn") or "end_turn",
            raw=resp,
        )
