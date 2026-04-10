"""LiteratureAgent — ClinicalProfile → ParameterJustification via RAG.

The second specialist in the supervisor. Given a structured
``ClinicalProfile`` (produced by the IntakeAgent) and a
:class:`CorpusStore`, it searches the corpus for each parameter
decision and emits a ``ParameterJustification`` with explicit
citations.

Unlike the IntakeAgent, this agent is multi-turn: it issues several
``search_corpus`` calls before finally emitting the justification via a
second tool. The generic :class:`AgentLoop` handles the tool-call loop
identically to the way it drives the original ConfigAgent in the
landed prototype.

Design notes
------------

* The corpus store is bound at construction time, so the ``search_corpus``
  ToolSpec handler closes over it. No global state.
* The emit tool (``emit_parameter_justification``) is a sentinel: its
  handler is a no-op, but when the loop sees it called, the agent can
  inspect the arguments and finish.
* Schema validation runs after emission. If the model produced a
  malformed justification, we raise ``ValueError`` so tests can catch
  regressions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..backends.base import LLMBackend, Message, ToolSpec
from ..corpus.store import CorpusStore
from ..loop import AgentLoop, AgentRunResult
from ..tools.literature import make_search_corpus_tool

logger = logging.getLogger(__name__)


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_PATH = _PACKAGE_ROOT / "prompts" / "literature.md"
_SCHEMA_PATH = _PACKAGE_ROOT / "schemas" / "parameter_justification.json"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _load_schema() -> Dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _schema_as_tool_parameters(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Strip JSON Schema metadata so providers accept the tool input schema."""
    cleaned: Dict[str, Any] = {"type": "object"}
    for key in ("properties", "required", "additionalProperties"):
        if key in schema:
            cleaned[key] = schema[key]
    return cleaned


@dataclass
class LiteratureResult:
    """What the LiteratureAgent returns."""

    justification: Dict[str, Any]
    run: AgentRunResult
    search_queries: List[str] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        return self.justification.get("confidence", "low")

    @property
    def unresolved_decisions(self) -> List[str]:
        return list(self.justification.get("unresolved_decisions") or [])


class LiteratureAgent:
    """Search the RAG corpus and produce a cited parameter justification.

    Parameters
    ----------
    backend
        Any ``LLMBackend`` implementation. Tests use :class:`FakeBackend`
        with a scripted search-then-emit conversation.
    corpus
        A :class:`CorpusStore` — production uses :class:`ChromaCorpusStore`,
        tests use :class:`FakeCorpusStore`.
    max_iterations
        Hard cap on tool-use turns. 8 is enough for the documented
        workflow (one search per parameter decision plus the final emit).
    strict_validation
        If True (default), the returned justification is validated with
        ``jsonschema`` and malformed output raises ``ValueError``.
    """

    def __init__(
        self,
        backend: LLMBackend,
        corpus: CorpusStore,
        max_iterations: int = 10,
        system_prompt: Optional[str] = None,
        strict_validation: bool = True,
    ):
        self.backend = backend
        self.corpus = corpus
        self.system_prompt = system_prompt if system_prompt is not None else _load_prompt()
        self.schema = _load_schema()
        self.strict_validation = strict_validation

        self._emit_tool = self._build_emit_tool()
        self._search_tool = make_search_corpus_tool(corpus)

        self.loop = AgentLoop(
            backend=backend,
            tools=[self._search_tool, self._emit_tool],
            system_prompt=self.system_prompt,
            max_iterations=max_iterations,
            temperature=0.0,
        )

    # -- public API ----------------------------------------------------------

    def justify(self, clinical_profile: Dict[str, Any]) -> LiteratureResult:
        """Run the agent on one clinical profile.

        Raises
        ------
        ValueError
            If the agent never called ``emit_parameter_justification`` or
            if the emitted JSON fails schema validation.
        """
        user_message = (
            "ClinicalProfile for this patient:\n\n"
            "```json\n"
            f"{json.dumps(clinical_profile, indent=2)}\n"
            "```\n\n"
            "Follow the documented workflow: identify each parameter "
            "decision you need to justify, call search_corpus for each, "
            "then emit_parameter_justification with the full structured "
            "output."
        )

        run = self.loop.run(user_message)

        justification = self._extract_justification_from_run(run)

        # Collect the search queries the agent actually issued — useful for
        # the audit trace even if the model forgot to list them itself.
        search_queries: List[str] = []
        for entry in run.tool_calls():
            if entry.name == "search_corpus":
                args = (entry.payload or {}).get("arguments") or {}
                q = args.get("query")
                if q:
                    search_queries.append(str(q))

        # Fill in search_queries_used if the model omitted or shortened it.
        if not justification.get("search_queries_used"):
            justification["search_queries_used"] = search_queries

        if self.strict_validation:
            self._validate(justification)

        return LiteratureResult(
            justification=justification,
            run=run,
            search_queries=search_queries,
        )

    # -- internals -----------------------------------------------------------

    def _build_emit_tool(self) -> ToolSpec:
        parameters = _schema_as_tool_parameters(self.schema)

        def _noop(_args: Dict[str, Any]) -> Dict[str, Any]:
            # The loop executes tool handlers for every call. This handler
            # is a sentinel: it returns ok: true so the loop can finish on
            # the next turn, and the agent's system prompt tells the model
            # to reply with a confirmation sentence and stop.
            return {"ok": True}

        return ToolSpec(
            name="emit_parameter_justification",
            description=(
                "Emit the final ParameterJustification JSON object. Call "
                "this exactly once, after you have issued all the "
                "search_corpus calls you need."
            ),
            parameters=parameters,
            handler=_noop,
        )

    def _extract_justification_from_run(
        self, run: AgentRunResult
    ) -> Dict[str, Any]:
        """Pull the last emit_parameter_justification call from the trace."""
        for entry in reversed(run.tool_calls()):
            if entry.name == "emit_parameter_justification":
                args = (entry.payload or {}).get("arguments")
                if isinstance(args, dict):
                    return dict(args)
        raise ValueError(
            "LiteratureAgent: emit_parameter_justification was never called. "
            f"Stopped reason: {run.stopped_reason!r}, iterations: {run.iterations}."
        )

    def _validate(self, justification: Dict[str, Any]) -> None:
        try:
            from jsonschema import Draft202012Validator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "jsonschema is required for LiteratureAgent validation"
            ) from exc

        validator = Draft202012Validator(self.schema)
        errors = sorted(validator.iter_errors(justification), key=lambda e: e.path)
        if errors:
            details = [
                f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
                for e in errors
            ]
            raise ValueError(
                "LiteratureAgent output failed schema validation:\n  - "
                + "\n  - ".join(details)
            )
