"""Unit tests for :class:`aortacfd_agent.coordinator.Coordinator` and
:mod:`aortacfd_agent.cli`.

These tests exercise the full agent chain end-to-end with ``FakeBackend``
replacements for every LLM stage and a ``FakeCorpusStore`` for the
literature layer. Execution is always skipped (the real CFD pipeline is
not invoked), so every test runs in well under a second.

Test strategy
-------------

* Two shared fixtures (scripted intake/literature/results backend and
  fake corpus) feed the happy path.
* Each test calls ``Coordinator.run`` with different skip flags to
  verify the stage gating works.
* The trace logger is exercised indirectly by checking the JSONL file
  the coordinator writes.
* A couple of CLI tests check that ``aortacfd-agent intake`` and
  ``aortacfd-agent run`` parse their arguments and return the expected
  exit codes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from aortacfd_agent.backends.base import ToolCall
from aortacfd_agent.backends.fake import FakeBackend, ScriptedStep
from aortacfd_agent.cli import build_parser, main as cli_main
from aortacfd_agent.coordinator import Coordinator, CoordinatorResult
from aortacfd_agent.corpus.store import Chunk, FakeCorpusStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _valid_profile() -> Dict[str, Any]:
    return {
        "patient_id": "BPM120",
        "age_years": 12,
        "sex": "male",
        "diagnosis": "aortic coarctation",
        "heart_rate_bpm": 78,
        "systolic_bp_mmhg": 118,
        "diastolic_bp_mmhg": 72,
        "cardiac_output_l_min": 4.8,
        "imaging_modality": ["CT_angiography"],
        "flow_waveform_source": "doppler_csv",
        "missing_fields": [],
        "confidence": "high",
        "notes": None,
    }


def _valid_justification() -> Dict[str, Any]:
    return {
        "decisions": [
            {
                "parameter": "physics_model",
                "value": "rans",
                "reasoning": "Re > 2000",
                "citations": [
                    {"paper": "Wang2025", "page": 3, "quote": "RANS preferred"}
                ],
            },
            {
                "parameter": "mesh_goal",
                "value": "wall_sensitive",
                "reasoning": "WSS primary endpoint",
                "citations": [
                    {"paper": "ValenSendstad2018", "page": 5, "quote": "refine near wall"}
                ],
            },
            {
                "parameter": "wk_flow_allocation_method",
                "value": "user_specified",
                "reasoning": "Murray invalid for coarctation",
                "citations": [
                    {"paper": "Wang2025", "page": 4, "quote": "Murray misallocates"}
                ],
            },
            {
                "parameter": "numerics_profile",
                "value": "standard",
                "reasoning": "default 2nd-order",
                "citations": [],
            },
        ],
        "search_queries_used": ["coarctation flow split"],
        "unresolved_decisions": [],
        "confidence": "high",
    }


def _build_scripted_backend() -> FakeBackend:
    """Single backend that scripts responses for intake + literature + results."""
    return FakeBackend(
        script=[
            # --- Intake: one emit call ---
            ScriptedStep(
                text="",
                tool_calls=[
                    ToolCall(
                        id="i1",
                        name="emit_clinical_profile",
                        arguments=_valid_profile(),
                    )
                ],
            ),
            # --- Literature: one search + one emit ---
            ScriptedStep(
                text="",
                tool_calls=[
                    ToolCall(
                        id="l1",
                        name="search_corpus",
                        arguments={"query": "coarctation Murray flow split"},
                    )
                ],
            ),
            ScriptedStep(
                text="",
                tool_calls=[
                    ToolCall(
                        id="l2",
                        name="emit_parameter_justification",
                        arguments=_valid_justification(),
                    )
                ],
            ),
            ScriptedStep(text="done", stop_reason="end_turn"),
        ]
    )


def _corpus() -> FakeCorpusStore:
    return FakeCorpusStore(
        chunks=[
            Chunk(
                text="Coarctation invalidates Murray's law; use user-specified split.",
                paper="Wang2025",
                page=4,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Coordinator happy path (no execution)
# ---------------------------------------------------------------------------


class TestCoordinatorHappyPath:
    def test_intake_plus_literature_plus_config_no_execution(
        self, tmp_path: Path
    ):
        coord = Coordinator(
            intake_backend=_build_scripted_backend(),
            corpus=_corpus(),
        )
        result = coord.run(
            clinical_text="12 year old male with aortic coarctation, HR 78, BP 118/72.",
            case_dir=tmp_path,  # case dir only needs to exist for the coordinator
            output_dir=tmp_path / "out",
            skip_execution=True,
        )
        assert isinstance(result, CoordinatorResult)
        assert result.success, result.error
        assert result.stages_run == ["intake", "literature", "config"]
        assert "execution" in result.stages_skipped
        assert "summary" in result.stages_skipped
        assert result.intake is not None
        assert result.literature is not None
        assert result.config is not None
        assert result.config.saved
        # Files actually landed on disk
        assert (tmp_path / "out" / "agent_config.json").exists()
        assert (tmp_path / "out" / "agent_rationale.md").exists()
        assert (tmp_path / "out" / "agent_trace.jsonl").exists()

    def test_trace_file_contains_expected_stages(self, tmp_path: Path):
        coord = Coordinator(
            intake_backend=_build_scripted_backend(),
            corpus=_corpus(),
        )
        coord.run(
            clinical_text="demo patient",
            case_dir=tmp_path,
            output_dir=tmp_path / "out",
            skip_execution=True,
        )
        trace_path = tmp_path / "out" / "agent_trace.jsonl"
        lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
        stages = [json.loads(line)["stage"] for line in lines]
        # We expect intake, literature, config entries at minimum
        assert "intake" in stages
        assert "literature" in stages
        assert "config" in stages

    def test_config_contains_agent_patches(self, tmp_path: Path):
        coord = Coordinator(
            intake_backend=_build_scripted_backend(),
            corpus=_corpus(),
        )
        result = coord.run(
            clinical_text="demo",
            case_dir=tmp_path,
            output_dir=tmp_path / "out",
            skip_execution=True,
        )
        cfg_path = Path(result.config.config_path)
        cfg = json.loads(cfg_path.read_text())
        # Physics model was 'rans' in the justification
        assert cfg["physics"]["model"] == "rans"
        # Mesh goal was 'wall_sensitive'
        assert cfg["mesh"]["goal"] == "wall_sensitive"
        # WK methodology patched to user_flow_split
        assert (
            cfg["boundary_conditions"]["outlets"]["windkessel_settings"]["methodology"]
            == "user_flow_split"
        )
        # HR-derived cardiac cycle
        assert cfg["cardiac_cycle"] == pytest.approx(60 / 78, abs=0.01)


# ---------------------------------------------------------------------------
# Skip-flag behaviour
# ---------------------------------------------------------------------------


class TestCoordinatorSkipFlags:
    def test_skip_literature_still_produces_trace(self, tmp_path: Path):
        backend = FakeBackend(
            script=[
                ScriptedStep(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="i1",
                            name="emit_clinical_profile",
                            arguments=_valid_profile(),
                        )
                    ],
                ),
            ]
        )
        coord = Coordinator(intake_backend=backend, corpus=_corpus())
        result = coord.run(
            clinical_text="demo",
            case_dir=tmp_path,
            output_dir=tmp_path / "out",
            skip_literature=True,
            skip_execution=True,
        )
        assert result.success
        assert "literature" in result.stages_skipped
        assert "config" in result.stages_skipped  # depends on literature
        assert result.intake is not None
        assert result.literature is None
        assert result.config is None

    def test_no_corpus_skips_literature_gracefully(self, tmp_path: Path):
        backend = FakeBackend(
            script=[
                ScriptedStep(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="i1",
                            name="emit_clinical_profile",
                            arguments=_valid_profile(),
                        )
                    ],
                ),
            ]
        )
        coord = Coordinator(intake_backend=backend, corpus=None)
        result = coord.run(
            clinical_text="demo",
            case_dir=tmp_path,
            output_dir=tmp_path / "out",
            skip_execution=True,
        )
        assert result.success
        assert "literature" in result.stages_skipped
        assert result.literature is None


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


class TestCoordinatorErrors:
    def test_intake_failure_reports_error(self, tmp_path: Path):
        # Script the intake to return no tool call — triggers ValueError.
        backend = FakeBackend(
            script=[ScriptedStep(text="sorry I cannot help", stop_reason="end_turn")]
        )
        coord = Coordinator(intake_backend=backend, corpus=_corpus())
        result = coord.run(
            clinical_text="demo",
            case_dir=tmp_path,
            output_dir=tmp_path / "out",
            skip_execution=True,
        )
        assert not result.success
        assert result.error is not None
        assert "intake failed" in result.error


# ---------------------------------------------------------------------------
# CLI parser + commands (no subprocess, no real LLM)
# ---------------------------------------------------------------------------


class TestCli:
    def test_build_parser_has_expected_subcommands(self):
        parser = build_parser()
        help_text = parser.format_help()
        assert "run" in help_text
        assert "intake" in help_text
        assert "version" in help_text

    def test_version_subcommand(self, capsys: pytest.CaptureFixture[str]):
        rc = cli_main(["version"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "aortacfd-agent" in captured.out

    def test_intake_subcommand_prints_profile(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        referral = tmp_path / "r.txt"
        referral.write_text("demo referral", encoding="utf-8")
        rc = cli_main([
            "intake",
            "--backend",
            "fake",
            "--referral",
            str(referral),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert '"patient_id"' in out
        assert '"diagnosis"' in out
        # The demo FakeBackend always emits DEMO
        assert '"DEMO"' in out

    def test_run_subcommand_dry_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        referral = tmp_path / "r.txt"
        referral.write_text("demo referral text", encoding="utf-8")
        out_dir = tmp_path / "agent_out"
        rc = cli_main([
            "run",
            "--backend",
            "fake",
            "--case",
            str(tmp_path),
            "--output",
            str(out_dir),
            "--referral",
            str(referral),
        ])
        captured = capsys.readouterr()
        # Accept either success or a friendly skip — the demo FakeBackend is
        # intentionally minimal so some stages may legitimately be skipped.
        assert rc == 0, f"CLI exited {rc}: stdout={captured.out!r} stderr={captured.err!r}"
        assert "Coordinator" in captured.out
        assert out_dir.exists()
        assert (out_dir / "agent_trace.jsonl").exists()
