"""Backend factory.

Reads an ``AgentBackendConfig`` dictionary and returns a concrete
``LLMBackend``. Exists so CLIs, tests, and config files can switch
providers with one field::

    backend:
      provider: ollama
      model: qwen2.5:7b-instruct
      base_url: http://localhost:11434/v1

Supported providers:

* ``fake``            — offline scripted backend (tests, dry-runs)
* ``anthropic``       — Claude via the ``anthropic`` SDK
* ``openai``          — OpenAI cloud via the ``openai`` SDK
* ``ollama``          — local Ollama server, OpenAI-compatible API
* ``vllm``            — local vLLM server, OpenAI-compatible API
* ``openai_compat``   — any OpenAI-compatible endpoint (custom base_url)

The SDK for the chosen provider is only imported when ``resolve_backend``
is actually called with that provider, so ``pip install anthropic`` is not
needed unless you intend to use Claude.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base import LLMBackend


# Common local-server defaults — users can still override per call.
_PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "qwen2.5:7b-instruct",
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "openai": {
        "base_url": None,  # SDK default
        "api_key": None,   # taken from OPENAI_API_KEY
        "model": "gpt-4o-mini",
    },
    "anthropic": {
        "base_url": None,
        "api_key": None,   # taken from ANTHROPIC_API_KEY
        "model": "claude-sonnet-4-5",
    },
}


@dataclass
class AgentBackendConfig:
    """Serializable backend selection.

    ``provider`` is the only required field. Everything else is filled
    from sensible defaults for that provider, then overridden by anything
    the user set explicitly in the config.
    """

    provider: str = "fake"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentBackendConfig":
        known = {"provider", "model", "base_url", "api_key"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            provider=data.get("provider", "fake"),
            model=data.get("model"),
            base_url=data.get("base_url"),
            api_key=data.get("api_key"),
            extra=extra,
        )


def resolve_backend(config: AgentBackendConfig) -> LLMBackend:
    """Instantiate a concrete backend from a config.

    Raises
    ------
    ValueError
        If the provider name is unknown.
    ImportError
        If the provider's Python SDK is not installed.
    """
    provider = config.provider.lower()

    if provider == "fake":
        # The fake backend needs an explicit script, so it is never
        # constructable from config alone — tests build it directly.
        # We still allow resolving to a no-op fake here for CLIs that
        # want to advertise the provider but don't need a live model.
        from .fake import FakeBackend, ScriptedStep

        return FakeBackend(script=[ScriptedStep(text="(fake backend — no script)")])

    defaults = _PROVIDER_DEFAULTS.get(provider, {})
    model = config.model or defaults.get("model")
    base_url = config.base_url or defaults.get("base_url")
    api_key = config.api_key or defaults.get("api_key")

    if provider == "anthropic":
        from .anthropic_backend import AnthropicBackend

        # Anthropic SDK reads ANTHROPIC_API_KEY automatically if api_key is None.
        return AnthropicBackend(
            model=model or "claude-sonnet-4-5",
            api_key=api_key,
            base_url=base_url,
        )

    if provider in ("openai", "openai_compat", "ollama", "vllm"):
        from .openai_compat import OpenAICompatBackend

        if provider == "openai" and api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if model is None:
            raise ValueError(
                f"Provider '{provider}' requires a model name; "
                f"set backend.model in your config."
            )
        return OpenAICompatBackend(
            model=model,
            base_url=base_url,
            api_key=api_key,
            provider_hint=provider,
        )

    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        f"Supported: fake, anthropic, openai, ollama, vllm, openai_compat"
    )
