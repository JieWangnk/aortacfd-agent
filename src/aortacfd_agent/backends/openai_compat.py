"""OpenAI-compatible backend.

One adapter for every provider that speaks the OpenAI Chat Completions
protocol: OpenAI itself, Azure OpenAI, Ollama (``/v1/chat/completions``),
vLLM, LM Studio, Together, DeepSeek, Groq, Mistral, llama.cpp's server.
The only thing that changes between them is ``base_url`` and ``model``.

This single adapter is why the project can claim "provider-agnostic": it
means **any** of those backends can be used for both the quick local test
and the HPC run, with no code change — only a YAML/CLI flag.

Default settings for common local backends:

* Ollama  ->  base_url=http://localhost:11434/v1, api_key="ollama"
* vLLM    ->  base_url=http://localhost:8000/v1,  api_key="EMPTY"
* OpenAI  ->  base_url omitted, api_key from OPENAI_API_KEY

The adapter handles the small shape differences between "regular" OpenAI
tool calls and Ollama's (Ollama wraps ``arguments`` as a real dict rather
than a JSON string, and may omit the ``id`` field — both are normalised
here).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List, Optional

from .base import AgentResponse, Message, ToolCall, ToolSpec, tool_specs_to_json_schema


class OpenAICompatBackend:
    """LLM backend for any OpenAI-Chat-Completions-compatible server."""

    name = "openai_compat"

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        provider_hint: str = "generic",
    ):
        """
        Parameters
        ----------
        model
            Model name as the server expects it, e.g. ``"gpt-4o-mini"``,
            ``"qwen2.5:7b-instruct"``, ``"meta-llama/Llama-3.1-8B-Instruct"``.
        base_url
            Full URL of the ``/v1`` endpoint. If omitted, the SDK default
            (OpenAI cloud) is used.
        api_key
            API key string. For local Ollama/vLLM servers any non-empty
            value works — those servers ignore it. Falls back to the
            ``OPENAI_API_KEY`` env var if not passed.
        provider_hint
            Free-form tag stored for logging (``"ollama"``, ``"vllm"``,
            ``"openai"``, ...). Does not affect behaviour.
        """
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "The OpenAI-compatible backend requires the 'openai' package. "
                "Install it with: pip install openai"
            ) from exc

        kwargs: Dict[str, Any] = {}
        if base_url is not None:
            kwargs["base_url"] = base_url
        if api_key is not None:
            kwargs["api_key"] = api_key
        elif os.environ.get("OPENAI_API_KEY") is None:
            # Local servers still demand a non-empty key in the SDK.
            kwargs["api_key"] = "local"
        self._client = OpenAI(**kwargs)
        self.model = model
        self.provider_hint = provider_hint

    # -- translation helpers -------------------------------------------------

    @staticmethod
    def _messages_to_openai(
        messages: List[Message], system: Optional[str]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})

        for m in messages:
            if m.role == "assistant" and m.tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": m.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in m.tool_calls
                        ],
                    }
                )
            elif m.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id or "",
                        "content": m.content,
                    }
                )
            else:
                out.append({"role": m.role, "content": m.content})
        return out

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
            "messages": self._messages_to_openai(messages, system),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tool_specs_to_json_schema(tools)
            payload["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**payload)
        choice = resp.choices[0]
        msg = choice.message

        text = msg.content or ""
        tool_calls: List[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            args_raw = tc.function.arguments
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError:
                    args = {"_raw": args_raw}
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}
            tool_calls.append(
                ToolCall(
                    id=getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}",
                    name=tc.function.name,
                    arguments=args,
                )
            )

        return AgentResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "stop",
            raw=resp,
        )
