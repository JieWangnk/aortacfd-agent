"""Top-level supervisor that chains the five specialist agents.

This is the public orchestration layer. It takes the inputs a clinician
or a script would provide:

* a free-text clinical referral,
* a path to a patient case directory with STL files,
* optionally a run-name and an output directory,

and drives the full pipeline:

1. :class:`~aortacfd_agent.agents.intake.IntakeAgent` →
   ``ClinicalProfile`` JSON.
2. :class:`~aortacfd_agent.agents.literature.LiteratureAgent` →
   ``ParameterJustification`` JSON (with citations).
3. :class:`~aortacfd_agent.agents.config.ConfigAgent` →
   ``agent_config.json`` + ``agent_rationale.md`` (deterministic).
4. :class:`~aortacfd_agent.agents.execution.ExecutionAgent` →
   subprocess-driven CFD run via ``run_patient.py`` (dry-run by default).
5. :class:`~aortacfd_agent.agents.results.ResultsAgent` →
   natural-language clinical summary.

Each stage writes a record to an :class:`AgentTraceLogger` so the final
``agent_trace.jsonl`` is a complete audit of what happened.

The coordinator owns no LLM backend of its own — it takes backends as
parameters so the caller can mix providers (for example, Haiku for
Intake, Sonnet for Literature, Haiku for Results). When a single backend
is provided, every stage uses it.

Skipping stages
---------------

Each stage can be skipped via the ``skip_*`` flags. The most common
need is ``skip_execution=True`` for fast text-in/text-out demos that
do not actually run OpenFOAM. When a stage is skipped, downstream
stages that depend on its output are also skipped automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agents.config import ConfigAgent, ConfigAgentError, ConfigAgentResult
from .agents.execution import ExecutionAgent, ExecutionAgentError, ExecutionResult
from .agents.intake import IntakeAgent, IntakeResult
from .agents.literature import LiteratureAgent, LiteratureResult
from .agents.results import ResultsAgent, ResultsResponse
from .backends.base import LLMBackend
from .corpus.store import CorpusStore
from .trace.logger import AgentTraceLogger

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorResult:
    """Everything one pipeline run produces.

    Every field is Optional because any stage may be skipped or may fail
    before producing its output. Callers should check ``success`` for
    the top-line verdict.
    """

    run_dir: Path
    success: bool
    intake: Optional[IntakeResult] = None
    literature: Optional[LiteratureResult] = None
    config: Optional[ConfigAgentResult] = None
    execution: Optional[ExecutionResult] = None
    summary: Optional[ResultsResponse] = None
    error: Optional[str] = None
    stages_run: List[str] = field(default_factory=list)
    stages_skipped: List[str] = field(default_factory=list)

    def brief(self) -> str:
        status = "OK" if self.success else f"FAILED: {self.error}"
        stages = ", ".join(self.stages_run) or "(none)"
        return f"Coordinator {status} — stages: {stages} — run_dir: {self.run_dir}"


class Coordinator:
    """Chain the five specialist agents into one pipeline.

    Parameters
    ----------
    intake_backend, literature_backend, results_backend
        LLM backends for the three agents that actually call an LLM.
        All can be the same object. ``literature_backend`` may be None
        if ``skip_literature=True`` is always passed to :meth:`run`.
    corpus
        The :class:`CorpusStore` the LiteratureAgent will search. Fake
        stores (for tests/demos) and Chroma (for production) both work.
    """

    def __init__(
        self,
        intake_backend: LLMBackend,
        literature_backend: Optional[LLMBackend] = None,
        results_backend: Optional[LLMBackend] = None,
        corpus: Optional[CorpusStore] = None,
    ):
        self.intake_backend = intake_backend
        self.literature_backend = literature_backend or intake_backend
        self.results_backend = results_backend or intake_backend
        self.corpus = corpus

    # -- public API ----------------------------------------------------------

    def run(
        self,
        clinical_text: str,
        case_dir: Path,
        output_dir: Optional[Path] = None,
        *,
        patient_id: Optional[str] = None,
        skip_intake: bool = False,
        skip_literature: bool = False,
        skip_config: bool = False,
        skip_execution: bool = True,
        skip_summary: bool = False,
        execution_dry_run: bool = True,
        run_name: Optional[str] = None,
    ) -> CoordinatorResult:
        """Run the full pipeline for one patient.

        By default ``skip_execution=True`` and ``execution_dry_run=True``
        so the pipeline produces artefacts without touching the CFD
        solver — that is the mode the demo and most tests use. Set
        ``skip_execution=False, execution_dry_run=False`` for a real
        end-to-end run.
        """
        case_dir = Path(case_dir).resolve()
        if output_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path.cwd() / "output" / "agent_runs" / f"run_{stamp}"
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        trace = AgentTraceLogger(output_dir / "agent_trace.jsonl")

        result = CoordinatorResult(run_dir=output_dir, success=False)

        # ------------------------------------------------------------------
        # Stage 1: Intake
        # ------------------------------------------------------------------
        profile: Optional[Dict[str, Any]] = None
        if skip_intake:
            result.stages_skipped.append("intake")
        else:
            try:
                with trace.start("intake") as t:
                    agent = IntakeAgent(backend=self.intake_backend)
                    intake_result = agent.extract(clinical_text)
                    profile = intake_result.profile
                    result.intake = intake_result
                    result.stages_run.append("intake")
                    t.update(
                        {
                            "confidence": intake_result.confidence,
                            "missing_fields": intake_result.missing_fields,
                            "patient_id": profile.get("patient_id"),
                            "diagnosis": profile.get("diagnosis"),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                result.error = f"intake failed: {exc}"
                return result

        # ------------------------------------------------------------------
        # Stage 2: Literature
        # ------------------------------------------------------------------
        justification: Optional[Dict[str, Any]] = None
        if skip_literature or profile is None:
            if profile is None and not skip_literature:
                trace.record(
                    "literature",
                    payload={"note": "skipped because intake produced no profile"},
                    status="warning",
                )
            result.stages_skipped.append("literature")
        elif self.corpus is None:
            trace.record(
                "literature",
                payload={"note": "skipped: no corpus store configured"},
                status="warning",
            )
            result.stages_skipped.append("literature")
        else:
            try:
                with trace.start("literature") as t:
                    agent = LiteratureAgent(
                        backend=self.literature_backend,
                        corpus=self.corpus,
                    )
                    lit_result = agent.justify(profile)
                    justification = lit_result.justification
                    result.literature = lit_result
                    result.stages_run.append("literature")
                    t.update(
                        {
                            "confidence": lit_result.confidence,
                            "unresolved": lit_result.unresolved_decisions,
                            "num_decisions": len(justification.get("decisions") or []),
                            "search_queries": lit_result.search_queries,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                result.error = f"literature failed: {exc}"
                return result

        # ------------------------------------------------------------------
        # Stage 3: Config
        # ------------------------------------------------------------------
        config_path: Optional[Path] = None
        if skip_config or profile is None or justification is None:
            result.stages_skipped.append("config")
        else:
            try:
                with trace.start("config") as t:
                    agent = ConfigAgent()
                    cfg_result = agent.generate(
                        clinical_profile=profile,
                        parameter_justification=justification,
                        output_dir=output_dir,
                        save=True,
                    )
                    result.config = cfg_result
                    result.stages_run.append("config")
                    if cfg_result.saved:
                        config_path = Path(cfg_result.config_path)
                    t.update(
                        {
                            "patches_applied": cfg_result.patches_applied,
                            "warnings": cfg_result.warnings,
                            "config_path": cfg_result.config_path,
                        }
                    )
            except ConfigAgentError as exc:
                result.error = f"config failed: {exc}"
                return result

        # ------------------------------------------------------------------
        # Stage 4: Execution
        # ------------------------------------------------------------------
        if skip_execution or config_path is None or patient_id is None:
            reason = None
            if patient_id is None and not skip_execution:
                reason = "no patient_id provided"
            elif config_path is None and not skip_execution:
                reason = "no config produced upstream"
            if reason is not None:
                trace.record("execution", payload={"note": f"skipped: {reason}"}, status="warning")
            result.stages_skipped.append("execution")
        else:
            try:
                with trace.start("execution") as t:
                    agent = ExecutionAgent()
                    exec_result = agent.run(
                        patient_id=patient_id,
                        config_path=config_path,
                        dry_run=execution_dry_run,
                        run_name=run_name,
                    )
                    result.execution = exec_result
                    result.stages_run.append("execution")
                    t.update(
                        {
                            "returncode": exec_result.returncode,
                            "duration_s": exec_result.duration_s,
                            "run_dir": exec_result.run_dir,
                            "steps": exec_result.steps,
                            "dry_run": exec_result.dry_run,
                        }
                    )
            except ExecutionAgentError as exc:
                result.error = f"execution failed: {exc}"
                return result

        # ------------------------------------------------------------------
        # Stage 5: Results summary (only if execution actually finished
        # and produced a run_dir to summarise)
        # ------------------------------------------------------------------
        if skip_summary:
            result.stages_skipped.append("summary")
        elif result.execution is None or not result.execution.run_dir:
            trace.record(
                "summary",
                payload={"note": "skipped: execution did not report a run_dir"},
                status="warning",
            )
            result.stages_skipped.append("summary")
        else:
            try:
                with trace.start("summary") as t:
                    agent = ResultsAgent(backend=self.results_backend)
                    response = agent.summarise(Path(result.execution.run_dir))
                    result.summary = response
                    result.stages_run.append("summary")
                    t.update(
                        {
                            "answer_preview": response.answer[:300],
                            "tool_calls_made": response.tool_calls_made,
                            "iterations": response.iterations,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                result.error = f"summary failed: {exc}"
                return result

        # Final bookkeeping
        result.success = True
        return result


__all__ = ["Coordinator", "CoordinatorResult"]
