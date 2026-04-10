"""Command-line interface for the aortacfd-agent package.

Exposed via the ``aortacfd-agent`` console script declared in
``pyproject.toml``. The CLI wraps the :class:`Coordinator` and adds
argument parsing, backend selection, and file-system plumbing.

Three subcommands:

* ``run``  — full pipeline (intake → literature → config → execution → summary)
* ``intake`` — only the IntakeAgent (text in, structured profile out)
* ``version`` — print the package version and exit

All subcommands default to the offline :class:`FakeBackend` and
:class:`FakeCorpusStore` so a new user can type one command and see the
pipeline produce artefacts without installing any SDKs or fetching
PDFs. Production use switches backends with ``--backend`` and feeds a
real corpus path with ``--corpus``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .backends.base import LLMBackend, ToolCall
from .backends.fake import FakeBackend, ScriptedStep
from .corpus.store import CorpusStore, FakeCorpusStore


# ---------------------------------------------------------------------------
# Backend / corpus resolution
# ---------------------------------------------------------------------------


def _build_backend(provider: str, model: Optional[str]) -> LLMBackend:
    """Instantiate the requested backend.

    The real providers (anthropic, openai, ollama, vllm) are imported
    lazily inside the factory so a CLI invocation that picks ``fake``
    does not require any SDK at all.
    """
    if provider == "fake":
        return _build_demo_fake_backend()

    from .backends.factory import AgentBackendConfig, resolve_backend

    overrides: Dict[str, Any] = {"provider": provider}
    if model:
        overrides["model"] = model
    return resolve_backend(AgentBackendConfig.from_dict(overrides))


def _build_demo_fake_backend() -> FakeBackend:
    """Scripted backend for the dry-run demo path.

    The script is deliberately generic: it drives the IntakeAgent through
    one tool call with a placeholder profile, then the LiteratureAgent
    through two search_corpus calls followed by an emit, then the
    ResultsAgent through a one-tool read + a canned narrative. It is
    not intended to be intelligent — it just proves the plumbing works
    without hitting any network.
    """
    placeholder_profile = {
        "patient_id": "DEMO",
        "age_years": 12,
        "diagnosis": "demo case — no LLM was consulted",
        "heart_rate_bpm": 78,
        "systolic_bp_mmhg": 118,
        "diastolic_bp_mmhg": 72,
        "cardiac_output_l_min": 4.8,
        "imaging_modality": ["CT_angiography"],
        "flow_waveform_source": "doppler_csv",
        "missing_fields": [],
        "confidence": "low",
        "notes": "scripted FakeBackend output; replace with --backend for real reasoning",
    }
    placeholder_justification = {
        "decisions": [
            {
                "parameter": "physics_model",
                "value": "laminar",
                "reasoning": "demo default (no real LLM)",
                "citations": [],
            },
            {
                "parameter": "mesh_goal",
                "value": "routine_hemodynamics",
                "reasoning": "demo default",
                "citations": [],
            },
            {
                "parameter": "wk_flow_allocation_method",
                "value": "murray",
                "reasoning": "demo default",
                "citations": [],
            },
            {
                "parameter": "numerics_profile",
                "value": "standard",
                "reasoning": "demo default",
                "citations": [],
            },
        ],
        "search_queries_used": ["demo query"],
        "unresolved_decisions": [],
        "confidence": "low",
        "notes": "scripted FakeBackend output — no real literature was retrieved",
    }
    narrative = (
        "This is a demo summary produced by the scripted FakeBackend. "
        "No real LLM was consulted, no CFD run was executed, and no "
        "literature was retrieved. Use --backend ollama or --backend "
        "anthropic for real reasoning."
    )

    script = [
        # Intake: one emit call
        ScriptedStep(
            text="",
            tool_calls=[
                ToolCall(id="i1", name="emit_clinical_profile", arguments=placeholder_profile)
            ],
        ),
        # Literature: one search + one emit (2 turns)
        ScriptedStep(
            text="",
            tool_calls=[
                ToolCall(id="l1", name="search_corpus", arguments={"query": "demo query"})
            ],
        ),
        ScriptedStep(
            text="",
            tool_calls=[
                ToolCall(
                    id="l2",
                    name="emit_parameter_justification",
                    arguments=placeholder_justification,
                )
            ],
        ),
        ScriptedStep(text="demo ok", stop_reason="end_turn"),
        # Results (if called): one read + narrative
        ScriptedStep(
            text="",
            tool_calls=[
                ToolCall(id="r1", name="read_qoi_summary", arguments={"run_dir": "demo"})
            ],
        ),
        ScriptedStep(text=narrative, stop_reason="end_turn"),
    ]
    return FakeBackend(script=script)


def _build_corpus(corpus_path: Optional[Path]) -> CorpusStore:
    """Load a CorpusStore from disk, or fall back to an empty fake store."""
    if corpus_path is None:
        # Tiny built-in seed so the demo path still finds something.
        return FakeCorpusStore(chunks=[])
    p = Path(corpus_path)
    if p.is_file() and p.suffix == ".json":
        return FakeCorpusStore.from_json(p)
    if p.is_dir():
        # Future: open a Chroma persistence directory here.
        try:
            from .corpus.store import ChromaCorpusStore

            return ChromaCorpusStore(persist_directory=p)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"error: corpus directory {p} cannot be opened as a Chroma "
                f"store: {exc}. Install with `pip install -e '.[rag]'`."
            )
    raise SystemExit(f"error: corpus path {p} is neither a .json fixture nor a directory")


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    from .coordinator import Coordinator

    clinical_text = _read_clinical_text(args)
    case_dir = Path(args.case).resolve()
    if not case_dir.exists():
        print(f"error: case dir does not exist: {case_dir}", file=sys.stderr)
        return 2

    output_dir = Path(args.output).resolve() if args.output else None
    backend = _build_backend(args.backend, args.model)
    corpus = _build_corpus(args.corpus)

    coordinator = Coordinator(
        intake_backend=backend,
        literature_backend=backend,
        results_backend=backend,
        corpus=corpus,
    )

    result = coordinator.run(
        clinical_text=clinical_text,
        case_dir=case_dir,
        output_dir=output_dir,
        patient_id=args.patient_id,
        skip_literature=args.skip_literature,
        skip_config=args.skip_config,
        skip_execution=not args.execute,
        execution_dry_run=not args.full,
        skip_summary=args.skip_summary,
        run_name=args.run_name,
    )

    print(result.brief())
    if result.success:
        print(f"  trace: {result.run_dir / 'agent_trace.jsonl'}")
        if result.config and result.config.config_path:
            print(f"  config: {result.config.config_path}")
        if result.summary:
            print("  summary:")
            for line in result.summary.answer.splitlines():
                print(f"    {line}")
        return 0

    print(f"error: {result.error}", file=sys.stderr)
    return 1


def _read_clinical_text(args: argparse.Namespace) -> str:
    if args.clinical_text:
        return args.clinical_text
    if args.referral:
        return Path(args.referral).read_text(encoding="utf-8")
    raise SystemExit("error: provide --clinical-text or --referral")


# ---------------------------------------------------------------------------
# Subcommand: intake
# ---------------------------------------------------------------------------


def _cmd_intake(args: argparse.Namespace) -> int:
    from .agents.intake import IntakeAgent

    clinical_text = _read_clinical_text(args)
    backend = _build_backend(args.backend, args.model)

    agent = IntakeAgent(backend=backend)
    result = agent.extract(clinical_text)
    print(json.dumps(result.profile, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: version
# ---------------------------------------------------------------------------


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"aortacfd-agent {__version__}")
    return 0


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aortacfd-agent",
        description=(
            "LLM-agent layer over AortaCFD: turn a free-text clinical referral "
            "into a reproducible, literature-grounded patient-specific CFD run "
            "with a natural-language clinical summary."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="enable INFO-level logging")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- run --------------------------------------------------------------
    p_run = sub.add_parser("run", help="Run the full agent pipeline end-to-end.")
    _add_input_args(p_run)
    _add_backend_args(p_run)
    p_run.add_argument(
        "--case",
        required=True,
        help="Path to the patient case directory containing STL files.",
    )
    p_run.add_argument("--output", help="Output directory for the agent run.")
    p_run.add_argument("--patient-id", help="Patient identifier passed to run_patient.py.")
    p_run.add_argument("--run-name", help="Optional --run-name forwarded to the CFD CLI.")
    p_run.add_argument(
        "--execute",
        action="store_true",
        help="Actually invoke run_patient.py after config generation (default: skip).",
    )
    p_run.add_argument(
        "--full",
        action="store_true",
        help="Run the complete CFD pipeline including the solver (default: dry-run only).",
    )
    p_run.add_argument("--skip-literature", action="store_true", help="Skip the LiteratureAgent stage.")
    p_run.add_argument("--skip-config", action="store_true", help="Skip the ConfigAgent stage.")
    p_run.add_argument("--skip-summary", action="store_true", help="Skip the ResultsAgent summary stage.")
    p_run.set_defaults(func=_cmd_run)

    # --- intake -----------------------------------------------------------
    p_intake = sub.add_parser("intake", help="Run only the IntakeAgent and print the profile JSON.")
    _add_input_args(p_intake)
    _add_backend_args(p_intake)
    p_intake.set_defaults(func=_cmd_intake)

    # --- version ----------------------------------------------------------
    p_version = sub.add_parser("version", help="Print the package version and exit.")
    p_version.set_defaults(func=_cmd_version)

    return parser


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--clinical-text",
        help="Free-text clinical referral (quoted). Mutually exclusive with --referral.",
    )
    group.add_argument(
        "--referral",
        help="Path to a file containing the clinical referral text.",
    )


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        default="fake",
        choices=["fake", "anthropic", "openai", "openai_compat", "ollama", "vllm"],
        help="LLM backend. Default 'fake' is offline and scripted.",
    )
    parser.add_argument("--model", help="Model id for the chosen backend.")
    parser.add_argument(
        "--corpus",
        help=(
            "Path to a corpus source. .json is loaded into a FakeCorpusStore; "
            "a directory is loaded as a Chroma persistent store."
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)-5s %(name)s: %(message)s",
    )

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
