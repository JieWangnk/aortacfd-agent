"""IntakeAgent — clinical text → structured ``ClinicalProfile`` JSON.

This is the first specialist in the five-agent supervisor. It does not
loop — it makes exactly one model call with a single tool whose input
schema is the ``ClinicalProfile`` JSON schema. The model's tool-use
response is the structured output.

The IntakeAgent intentionally does *not* fill in defaults. If the
referral omits a field, it is null and gets added to ``missing_fields``.
Downstream agents (Literature, Config) handle defaults with citations.

Design notes
------------

* The system prompt lives in ``prompts/intake.md`` and is loaded at
  runtime. This makes prompt iteration easy without code changes.
* The schema lives in ``schemas/clinical_profile.json`` and is loaded
  the same way. The agent emits JSON matching that schema, and we
  re-validate the output with ``jsonschema`` before returning it.
* Cost model: one ``chat`` call per referral. Sonnet or Haiku both work;
  Haiku is the cheap default.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..backends.base import (
    AgentResponse,
    LLMBackend,
    Message,
    ToolCall,
    ToolSpec,
)

logger = logging.getLogger(__name__)


# Path to the prompt and schema within the installed package.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_PATH = _PACKAGE_ROOT / "prompts" / "intake.md"
_SCHEMA_PATH = _PACKAGE_ROOT / "schemas" / "clinical_profile.json"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _load_schema() -> Dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _schema_as_tool_parameters(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Strip JSON Schema metadata so the result is accepted by provider tool APIs.

    Anthropic, OpenAI, and Ollama all expect a ``parameters`` object that
    looks like ``{"type": "object", "properties": ..., "required": ...}``.
    They reject top-level ``$schema``, ``$id``, ``title``, ``description``
    at the object level. We strip those and keep the structural parts.
    """
    cleaned: Dict[str, Any] = {"type": "object"}
    for key in ("properties", "required", "additionalProperties"):
        if key in schema:
            cleaned[key] = schema[key]
    return cleaned


@dataclass
class IntakeResult:
    """What the IntakeAgent returns."""

    profile: Dict[str, Any]
    raw_response: AgentResponse

    @property
    def confidence(self) -> str:
        return self.profile.get("confidence", "low")

    @property
    def missing_fields(self) -> List[str]:
        return list(self.profile.get("missing_fields") or [])


class IntakeAgent:
    """Extract a ``ClinicalProfile`` from a free-text clinical referral.

    Parameters
    ----------
    backend
        Any ``LLMBackend`` implementation — real or fake.
    system_prompt
        Optional override for testing. Defaults to the content of
        ``prompts/intake.md``.
    strict_validation
        If True (default), the returned profile is re-validated with
        ``jsonschema`` and a ``ValueError`` is raised on failure. Set
        False only in tests that want to inspect malformed output.
    """

    def __init__(
        self,
        backend: LLMBackend,
        system_prompt: Optional[str] = None,
        strict_validation: bool = True,
    ):
        self.backend = backend
        self.system_prompt = system_prompt if system_prompt is not None else _load_prompt()
        self.schema = _load_schema()
        self.strict_validation = strict_validation
        self._tool_spec = self._build_tool_spec()

    # -- public API ----------------------------------------------------------

    def extract(self, referral_text: str) -> IntakeResult:
        """Run the IntakeAgent on one referral.

        Raises
        ------
        ValueError
            If the model did not call the ``emit_clinical_profile`` tool,
            or if the returned JSON fails schema validation (when
            ``strict_validation`` is True).
        """
        user_message = (
            "Clinical referral to extract into a ClinicalProfile:\n\n"
            "```\n"
            f"{referral_text.strip()}\n"
            "```\n\n"
            "Call emit_clinical_profile exactly once with the structured profile."
        )

        response = self.backend.chat(
            messages=[Message(role="user", content=user_message)],
            tools=[self._tool_spec],
            system=self.system_prompt,
            temperature=0.0,
        )

        profile = self._extract_profile_from_response(response)
        if self.strict_validation:
            self._validate(profile)

        return IntakeResult(profile=profile, raw_response=response)

    # -- internals -----------------------------------------------------------

    def _build_tool_spec(self) -> ToolSpec:
        """The single tool the agent calls to emit its structured output.

        The handler is a no-op: the loop layer never calls it because
        IntakeAgent is single-turn. We still need a handler to satisfy
        the ``ToolSpec`` dataclass.
        """
        parameters = _schema_as_tool_parameters(self.schema)

        def _noop(_args: Dict[str, Any]) -> Dict[str, Any]:
            return {"ok": True}

        return ToolSpec(
            name="emit_clinical_profile",
            description=(
                "Emit a structured ClinicalProfile matching the "
                "AortaCFD clinical_profile schema. Call this exactly once "
                "after reading the referral, then stop."
            ),
            parameters=parameters,
            handler=_noop,
        )

    def _extract_profile_from_response(self, response: AgentResponse) -> Dict[str, Any]:
        if not response.tool_calls:
            raise ValueError(
                "IntakeAgent: model returned no tool call. Response text: "
                f"{response.text!r}"
            )
        call: ToolCall = response.tool_calls[0]
        if call.name != "emit_clinical_profile":
            raise ValueError(
                f"IntakeAgent: expected emit_clinical_profile, got {call.name!r}"
            )
        return dict(call.arguments or {})

    def _validate(self, profile: Dict[str, Any]) -> None:
        """Re-validate the profile against the JSON schema.

        ``jsonschema`` is in the base dependencies (see ``pyproject.toml``),
        so this is always available.
        """
        try:
            from jsonschema import Draft202012Validator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "jsonschema is required for IntakeAgent validation"
            ) from exc

        validator = Draft202012Validator(self.schema)
        errors = sorted(validator.iter_errors(profile), key=lambda e: e.path)
        if errors:
            details = [
                f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
                for e in errors
            ]
            raise ValueError(
                "IntakeAgent output failed schema validation:\n  - "
                + "\n  - ".join(details)
            )
