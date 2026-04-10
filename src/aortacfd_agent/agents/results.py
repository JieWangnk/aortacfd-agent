"""ResultsAgent — natural-language summary + Q&A over a finished CFD run.

Fifth and final specialist in the supervisor. Takes the path to a
finished CFD run directory (with ``qoi_summary.json``,
``hemodynamics_report.txt``, ``merged_config.json``, and the OpenFOAM
``postProcessing/`` tree) and produces:

1. A one-paragraph clinical summary that cites specific numbers from
   the run's output files.
2. A standalone answer to a natural-language question about the run
   ("what is peak WSS in the arch?").
3. An interactive REPL mode for ad-hoc Q&A.

The grounding contract — every numerical value in the answer must come
from a tool call made during the current conversation — is enforced by
the system prompt, not by code. A post-hoc checker could scan the
final text for numbers not present in any tool result, but that is
left for a future iteration.

Design notes
------------

* Multi-turn agent. Uses the generic :class:`AgentLoop` just like the
  LiteratureAgent. The only differences are the tool set and the system
  prompt.
* Two entry points: :meth:`summarise` and :meth:`ask`. Both eventually
  call :meth:`_run` internally with different user messages.
* The REPL is a convenience wrapper in :meth:`repl` — it loops on
  :func:`input`, calls :meth:`ask` for each question, and prints the
  answer. Not used in tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..backends.base import LLMBackend
from ..loop import AgentLoop, AgentRunResult
from ..tools.results_io import build_results_toolset

logger = logging.getLogger(__name__)


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_PATH = _PACKAGE_ROOT / "prompts" / "results.md"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


@dataclass
class ResultsResponse:
    """What the ResultsAgent returns for one question."""

    answer: str
    run: AgentRunResult

    @property
    def tool_calls_made(self) -> List[str]:
        return [e.name for e in self.run.tool_calls()]

    @property
    def iterations(self) -> int:
        return self.run.iterations


class ResultsAgent:
    """Answer natural-language questions about a finished CFD run.

    Parameters
    ----------
    backend
        Any ``LLMBackend`` implementation. Tests script a FakeBackend.
    max_iterations
        Hard cap on tool-use turns. Eight is enough for "call every tool
        once and then write an answer"; raise if you add more tools.
    system_prompt
        Optional override for testing. Defaults to ``prompts/results.md``.
    """

    def __init__(
        self,
        backend: LLMBackend,
        max_iterations: int = 8,
        system_prompt: Optional[str] = None,
    ):
        self.backend = backend
        self.system_prompt = system_prompt if system_prompt is not None else _load_prompt()
        self.tools = build_results_toolset()
        self.loop = AgentLoop(
            backend=backend,
            tools=self.tools,
            system_prompt=self.system_prompt,
            max_iterations=max_iterations,
            temperature=0.0,
        )

    # -- public API ----------------------------------------------------------

    def summarise(self, run_dir: Path) -> ResultsResponse:
        """Produce a short clinical summary of the run."""
        run_dir = Path(run_dir).resolve()
        user_message = (
            "Produce a short clinical summary (3–6 sentences) of the "
            "CFD simulation results in this run directory. Cite specific "
            "numbers from the tool outputs. Finish with your summary text "
            "— do not ask follow-up questions.\n\n"
            f"run_dir: {run_dir}"
        )
        result = self.loop.run(user_message)
        return ResultsResponse(answer=result.final_text, run=result)

    def ask(self, run_dir: Path, question: str) -> ResultsResponse:
        """Answer one natural-language question about the run."""
        run_dir = Path(run_dir).resolve()
        user_message = (
            "The run directory for the finished simulation is below. "
            "Answer the clinician's question, grounding every numerical "
            "value in a tool call you make in this conversation.\n\n"
            f"run_dir: {run_dir}\n\n"
            f"Question: {question.strip()}"
        )
        result = self.loop.run(user_message)
        return ResultsResponse(answer=result.final_text, run=result)

    def repl(self, run_dir: Path) -> None:  # pragma: no cover — interactive
        """Interactive Q&A loop. Not covered by unit tests."""
        print(f"ResultsAgent REPL — run_dir: {run_dir}")
        print("Type a question, or 'quit' to exit.\n")
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in {"quit", "exit", "q"}:
                break
            try:
                response = self.ask(run_dir=run_dir, question=question)
            except Exception as exc:  # noqa: BLE001
                print(f"(error: {exc})")
                continue
            print(response.answer)
            print()


__all__ = ["ResultsAgent", "ResultsResponse"]
