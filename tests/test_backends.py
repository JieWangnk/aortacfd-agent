"""Tests for the provider-agnostic backend layer.

These tests never hit the network: they use the ``FakeBackend`` and
exercise the shared ``LLMBackend`` protocol, the backend factory, and the
message-translation helpers. They are designed to run in milliseconds so
``pytest tests/test_agent_backends.py`` is a fast smoke check on any
laptop before committing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aortacfd_agent.backends.base import (
    AgentResponse,
    LLMBackend,
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
    tool_specs_to_json_schema,
)
from aortacfd_agent.backends.factory import AgentBackendConfig, resolve_backend
from aortacfd_agent.backends.fake import FakeBackend, ScriptedStep


# ---------------------------------------------------------------------------
# FakeBackend fundamentals
# ---------------------------------------------------------------------------


def test_fake_backend_satisfies_protocol():
    backend = FakeBackend(script=[ScriptedStep(text="hi")])
    assert isinstance(backend, LLMBackend)
    assert backend.name == "fake"


def test_fake_backend_emits_scripted_text():
    backend = FakeBackend(script=[ScriptedStep(text="hello world")])
    resp = backend.chat(
        messages=[Message(role="user", content="ping")],
        tools=[],
    )
    assert isinstance(resp, AgentResponse)
    assert resp.text == "hello world"
    assert resp.tool_calls == []


def test_fake_backend_emits_tool_call_then_text():
    tc = ToolCall(id="1", name="do_thing", arguments={"x": 1})
    backend = FakeBackend(
        script=[
            ScriptedStep(tool_calls=[tc]),
            ScriptedStep(text="done"),
        ]
    )
    first = backend.chat(
        messages=[Message(role="user", content="go")], tools=[]
    )
    assert first.tool_calls == [tc]
    second = backend.chat(
        messages=[
            Message(role="user", content="go"),
            Message(role="tool", tool_call_id="1", content="result"),
        ],
        tools=[],
    )
    assert second.text == "done"


def test_fake_backend_raises_when_exhausted():
    backend = FakeBackend(script=[ScriptedStep(text="only one")])
    backend.chat([Message(role="user", content="x")], tools=[])
    with pytest.raises(IndexError):
        backend.chat([Message(role="user", content="x")], tools=[])


def test_fake_backend_records_calls_for_assertions():
    backend = FakeBackend(
        script=[
            ScriptedStep(text="a"),
            ScriptedStep(text="b"),
        ]
    )
    backend.chat([Message(role="user", content="first")], tools=[])
    backend.chat([Message(role="user", content="second")], tools=[])
    assert len(backend.calls) == 2
    assert backend.calls[0][-1].content == "first"
    assert backend.calls[1][-1].content == "second"


# ---------------------------------------------------------------------------
# tool_specs_to_json_schema
# ---------------------------------------------------------------------------


def test_tool_specs_to_json_schema_shape():
    def _handler(_args):
        return {"ok": True}

    specs = [
        ToolSpec(
            name="echo",
            description="Echo input",
            parameters={
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            },
            handler=_handler,
        )
    ]
    schema = tool_specs_to_json_schema(specs)
    assert schema == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo input",
                "parameters": {
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                    "required": ["msg"],
                },
            },
        }
    ]


# ---------------------------------------------------------------------------
# Factory behaviour
# ---------------------------------------------------------------------------


def test_resolve_backend_fake():
    cfg = AgentBackendConfig(provider="fake")
    backend = resolve_backend(cfg)
    assert backend.name == "fake"


def test_resolve_backend_unknown_provider_raises():
    cfg = AgentBackendConfig(provider="nonsense")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        resolve_backend(cfg)


def test_resolve_backend_config_from_dict_preserves_extras():
    cfg = AgentBackendConfig.from_dict(
        {
            "provider": "ollama",
            "model": "qwen2.5:7b-instruct",
            "base_url": "http://localhost:11434/v1",
            "temperature_hint": 0.2,
        }
    )
    assert cfg.provider == "ollama"
    assert cfg.model == "qwen2.5:7b-instruct"
    assert cfg.base_url == "http://localhost:11434/v1"
    assert cfg.extra == {"temperature_hint": 0.2}


# ---------------------------------------------------------------------------
# AnthropicBackend retry behaviour
# ---------------------------------------------------------------------------


def _anthropic_sdk_available() -> bool:
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _anthropic_sdk_available(),
    reason="anthropic SDK not installed",
)
def test_anthropic_backend_retries_transient_errors(monkeypatch):
    """A flaky client that raises twice then succeeds should still return."""
    import anthropic

    from aortacfd_agent.backends.anthropic_backend import AnthropicBackend

    # Dummy block matching the shape AnthropicBackend extracts.
    class _Block:
        type = "text"
        text = "ok"

    class _Resp:
        content = [_Block()]
        stop_reason = "end_turn"

    calls = {"n": 0}

    class _FlakyMessages:
        def create(self, **payload):
            calls["n"] += 1
            if calls["n"] < 3:
                raise anthropic.APIConnectionError(request=None)  # type: ignore[arg-type]
            return _Resp()

    class _FlakyClient:
        messages = _FlakyMessages()

    backend = AnthropicBackend.__new__(AnthropicBackend)
    backend._client = _FlakyClient()
    backend.model = "claude-haiku-4-5"

    # Zero out sleeps so the test runs instantly.
    monkeypatch.setattr(
        "aortacfd_agent.backends.anthropic_backend.time.sleep",
        lambda _s: None,
    )

    resp = backend.chat(
        messages=[Message(role="user", content="ping")],
        tools=[],
    )
    assert resp.text == "ok"
    assert calls["n"] == 3


@pytest.mark.skipif(
    not _anthropic_sdk_available(),
    reason="anthropic SDK not installed",
)
def test_anthropic_backend_gives_up_after_max_attempts(monkeypatch):
    """Persistent transient errors eventually re-raise the last one."""
    import anthropic

    from aortacfd_agent.backends.anthropic_backend import AnthropicBackend

    calls = {"n": 0}

    class _AlwaysFailsMessages:
        def create(self, **payload):
            calls["n"] += 1
            raise anthropic.APIConnectionError(request=None)  # type: ignore[arg-type]

    class _AlwaysFailsClient:
        messages = _AlwaysFailsMessages()

    backend = AnthropicBackend.__new__(AnthropicBackend)
    backend._client = _AlwaysFailsClient()
    backend.model = "claude-haiku-4-5"

    monkeypatch.setattr(
        "aortacfd_agent.backends.anthropic_backend.time.sleep",
        lambda _s: None,
    )

    with pytest.raises(anthropic.APIConnectionError):
        backend.chat(messages=[Message(role="user", content="ping")], tools=[])
    assert calls["n"] == 5  # default max_attempts


def test_resolve_backend_openai_requires_openai_sdk_or_model(monkeypatch):
    # Missing model AND missing OPENAI_API_KEY: either the openai SDK is
    # not installed (ImportError) or we get a ValueError about the model.
    cfg = AgentBackendConfig(provider="openai", model=None)
    try:
        import openai  # noqa: F401
        sdk_available = True
    except ImportError:
        sdk_available = False

    if sdk_available:
        # With SDK present but no model, we expect ValueError.
        with pytest.raises(ValueError, match="requires a model"):
            resolve_backend(AgentBackendConfig(provider="openai", model=None, api_key="x"))
    else:
        # With SDK missing, resolving raises ImportError on any openai-family provider.
        with pytest.raises(ImportError):
            resolve_backend(AgentBackendConfig(provider="openai", model="gpt-4o", api_key="x"))
