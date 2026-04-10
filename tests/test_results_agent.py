"""Unit tests for :class:`aortacfd_agent.agents.results.ResultsAgent` and
its read-side tool layer in :mod:`aortacfd_agent.tools.results_io`.

The tests build a synthetic "finished run" directory in a pytest
``tmp_path`` fixture, drop realistic files into it
(``qoi_summary.json``, ``hemodynamics_report.txt``, ``merged_config.json``,
and a ``postProcessing/outlet4Pressure/0/surfaceFieldValue.dat``), and
then exercise both the raw tool handlers and the full ResultsAgent loop
with a scripted ``FakeBackend``.

No real CFD output is needed — everything is cheap hand-written fixtures.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Dict

import pytest

from aortacfd_agent.agents.results import ResultsAgent, ResultsResponse
from aortacfd_agent.backends.base import ToolCall
from aortacfd_agent.backends.fake import FakeBackend, ScriptedStep
from aortacfd_agent.tools.results_io import (
    _parse_surface_field_value_dat,
    build_results_toolset,
    read_hemodynamics_report,
    read_merged_config,
    read_pressure_timeseries,
    read_qoi_summary_full,
)


# ---------------------------------------------------------------------------
# Synthetic finished-run fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def finished_run(tmp_path: Path) -> Path:
    """Build a minimal but realistic finished-run directory tree."""
    run = tmp_path / "output" / "BPM120" / "run_20260410_171530"
    (run / "results").mkdir(parents=True)
    (run / "reports").mkdir(parents=True)
    (run / "postProcessing" / "outlet4Pressure" / "0").mkdir(parents=True)

    # Primary QoI summary
    qoi = {
        "patient_id": "BPM120",
        "peak_systole_time_s": 0.12,
        "pressure_drop_mmHg_cycle_avg": 11.26,
        "pressure_drop_mmHg_peak_systole": 32.4,
        "wss_p99_Pa": 14.12,
        "wss_mean_Pa": 3.21,
        "osi_mean": 0.058,
        "tawss_p99_Pa": 14.8,
        "stroke_volume_ml": 58.2,
        "outlet_flow_split": {
            "outlet1": 0.15,
            "outlet2": 0.08,
            "outlet3": 0.09,
            "outlet4": 0.68,
        },
    }
    (run / "results" / "qoi_summary.json").write_text(json.dumps(qoi, indent=2))

    # Plain-text hemodynamics report
    report = textwrap.dedent(
        """\
        ===== Hemodynamic summary — BPM120 =====
        Peak systolic pressure drop: 32.4 mmHg (inlet → outlet4)
        Cycle-averaged pressure drop: 11.26 mmHg
        Wall shear stress p99: 14.12 Pa (elevated in descending aorta)
        Oscillatory shear index mean: 0.058

        Interpretation:
        The peak systolic gradient is consistent with the coarctation
        severity described in the referral. The WSS p99 is elevated in
        the arch and descending regions, consistent with the jet
        impinging on the outer wall.
        """
    )
    (run / "results" / "hemodynamics_report.txt").write_text(report)

    # Merged config
    merged = {
        "case_info": {"patient_id": "BPM120", "description": "paediatric coarctation"},
        "cardiac_cycle": 0.77,
        "physics": {"model": "laminar"},
        "numerics": {"profile": "standard"},
        "boundary_conditions": {
            "inlet": {"type": "TIMEVARYING"},
            "outlets": {
                "type": "3EWINDKESSEL",
                "windkessel_settings": {
                    "systolic_pressure": 118,
                    "diastolic_pressure": 72,
                    "tau": 1.5,
                    "betaT": 0.3,
                },
            },
        },
        "simulation_control": {"number_of_cycles": 3},
    }
    (run / "reports" / "merged_config.json").write_text(json.dumps(merged, indent=2))

    # Fake surfaceFieldValue.dat for outlet4 pressure
    dat_lines = [
        "# Time          areaAverage(p)",
        "# ---------------------------",
        "0.00    10340.0",
        "0.04    10520.0",
        "0.08    14210.0",
        "0.12    16880.0  # peak",
        "0.16    14950.0",
        "0.20    12100.0",
        "0.25    10680.0",
        "0.30    10020.0",
        "0.40    10310.0",
        "0.50    10390.0",
    ]
    (
        run
        / "postProcessing"
        / "outlet4Pressure"
        / "0"
        / "surfaceFieldValue.dat"
    ).write_text("\n".join(dat_lines) + "\n")

    return run


# ---------------------------------------------------------------------------
# read_qoi_summary_full
# ---------------------------------------------------------------------------


class TestReadQoiSummary:
    def test_happy_path(self, finished_run: Path):
        out = read_qoi_summary_full({"run_dir": str(finished_run)})
        assert "error" not in out
        assert out["run_dir"] == str(finished_run)
        assert "qoi_summary.json" in out["path"]
        assert out["qoi"]["patient_id"] == "BPM120"
        assert out["qoi"]["wss_p99_Pa"] == 14.12

    def test_missing_run_dir(self, tmp_path: Path):
        out = read_qoi_summary_full({"run_dir": str(tmp_path / "nope")})
        assert "error" in out

    def test_missing_qoi_file(self, tmp_path: Path):
        out = read_qoi_summary_full({"run_dir": str(tmp_path)})
        assert "error" in out
        assert "qoi_summary.json" in out["error"]

    def test_malformed_json(self, finished_run: Path):
        target = finished_run / "results" / "qoi_summary.json"
        target.write_text("{not valid json")
        out = read_qoi_summary_full({"run_dir": str(finished_run)})
        assert "error" in out


# ---------------------------------------------------------------------------
# read_hemodynamics_report
# ---------------------------------------------------------------------------


class TestReadHemodynamicsReport:
    def test_happy_path(self, finished_run: Path):
        out = read_hemodynamics_report({"run_dir": str(finished_run)})
        assert "error" not in out
        assert "Hemodynamic summary" in out["text"]
        assert "32.4 mmHg" in out["text"]

    def test_missing_report(self, tmp_path: Path):
        out = read_hemodynamics_report({"run_dir": str(tmp_path)})
        assert "error" in out


# ---------------------------------------------------------------------------
# read_merged_config
# ---------------------------------------------------------------------------


class TestReadMergedConfig:
    def test_happy_path(self, finished_run: Path):
        out = read_merged_config({"run_dir": str(finished_run)})
        assert "error" not in out
        cfg = out["config"]
        assert cfg["case_info"]["patient_id"] == "BPM120"
        assert cfg["physics"]["model"] == "laminar"
        assert cfg["boundary_conditions"]["outlets"]["windkessel_settings"]["tau"] == 1.5

    def test_missing_config(self, tmp_path: Path):
        out = read_merged_config({"run_dir": str(tmp_path)})
        assert "error" in out


# ---------------------------------------------------------------------------
# read_pressure_timeseries + dat parser
# ---------------------------------------------------------------------------


class TestReadPressureTimeseries:
    def test_happy_path(self, finished_run: Path):
        out = read_pressure_timeseries(
            {"run_dir": str(finished_run), "patch": "outlet4"}
        )
        assert "error" not in out
        assert out["patch"] == "outlet4"
        summary = out["summary"]
        assert summary["num_samples"] == 10
        assert summary["min"] < summary["max"]
        # Peak from the fixture file is 16880 at t=0.12
        assert summary["max"] == pytest.approx(16880.0)
        # Series is downsampled to <= 200 rows (all 10 here)
        assert len(out["series"]) == 10

    def test_missing_patch_arg(self, finished_run: Path):
        out = read_pressure_timeseries({"run_dir": str(finished_run)})
        assert "error" in out
        assert "patch is required" in out["error"]

    def test_unknown_patch(self, finished_run: Path):
        out = read_pressure_timeseries(
            {"run_dir": str(finished_run), "patch": "bogus_patch"}
        )
        assert "error" in out

    def test_missing_postprocessing(self, tmp_path: Path):
        out = read_pressure_timeseries(
            {"run_dir": str(tmp_path), "patch": "outlet4"}
        )
        assert "error" in out

    def test_dat_parser_handles_comments(self):
        text = "# header\n# another\n0.0 100.0\n0.1 200.0\nbad line\n0.2 300.0\n"
        rows = _parse_surface_field_value_dat(text)
        assert len(rows) == 3
        assert rows[-1]["value"] == 300.0


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_build_results_toolset_shape(self):
        tools = build_results_toolset()
        names = {t.name for t in tools}
        assert names == {
            "read_qoi_summary",
            "read_hemodynamics_report",
            "read_merged_config",
            "read_pressure_timeseries",
        }
        for tool in tools:
            assert tool.parameters["type"] == "object"
            assert "run_dir" in tool.parameters["properties"]


# ---------------------------------------------------------------------------
# ResultsAgent end-to-end (scripted)
# ---------------------------------------------------------------------------


def _scripted_summary_run(finished_run: Path) -> ResultsAgent:
    """Build a FakeBackend that reads the QoI summary and emits a narrative."""
    narrative = (
        "BPM120 shows a cycle-averaged pressure drop of 11.26 mmHg with a "
        "peak systolic spike of 32.4 mmHg across the coarctation, consistent "
        "with the referral's severity hint. Wall shear stress reaches 14.12 Pa "
        "(99th percentile) with OSI of 0.058 — elevated in the descending aorta "
        "as expected for this anatomy. Stroke volume was preserved at 58.2 mL "
        "and 68% of cardiac output is delivered to the descending aorta, which "
        "matches the user-specified flow split used in this run."
    )
    return ResultsAgent(
        backend=FakeBackend(
            script=[
                ScriptedStep(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            name="read_qoi_summary",
                            arguments={"run_dir": str(finished_run)},
                        )
                    ],
                ),
                ScriptedStep(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="t2",
                            name="read_hemodynamics_report",
                            arguments={"run_dir": str(finished_run)},
                        )
                    ],
                ),
                ScriptedStep(text=narrative, stop_reason="end_turn"),
            ]
        )
    )


def _scripted_qa_run(finished_run: Path) -> ResultsAgent:
    """Scripted Q&A: read the pressure waveform then answer."""
    answer = (
        "Peak inlet-to-outlet4 pressure at t=0.12s is 16880 Pa (≈126.6 mmHg), "
        "the maximum across the cycle. Minimum is 10020 Pa at t=0.30s. The "
        "spike at 0.12s coincides with peak systole reported in the QoI summary."
    )
    return ResultsAgent(
        backend=FakeBackend(
            script=[
                ScriptedStep(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            name="read_pressure_timeseries",
                            arguments={
                                "run_dir": str(finished_run),
                                "patch": "outlet4",
                            },
                        )
                    ],
                ),
                ScriptedStep(text=answer, stop_reason="end_turn"),
            ]
        )
    )


class TestResultsAgentEndToEnd:
    def test_summarise_happy_path(self, finished_run: Path):
        agent = _scripted_summary_run(finished_run)
        response = agent.summarise(finished_run)
        assert isinstance(response, ResultsResponse)
        assert "11.26 mmHg" in response.answer
        assert "14.12 Pa" in response.answer
        assert response.tool_calls_made == [
            "read_qoi_summary",
            "read_hemodynamics_report",
        ]
        # 2 tool-use turns + 1 final text = 3 iterations
        assert response.iterations == 3

    def test_ask_happy_path(self, finished_run: Path):
        agent = _scripted_qa_run(finished_run)
        response = agent.ask(
            finished_run,
            "What is the peak outlet4 pressure during this cycle?",
        )
        assert "16880" in response.answer or "126.6" in response.answer
        assert response.tool_calls_made == ["read_pressure_timeseries"]

    def test_ask_passes_run_dir_to_prompt(self, finished_run: Path):
        # Verify the agent's user message embeds the run_dir so the
        # model can see it — we inspect the first recorded call.
        agent = _scripted_qa_run(finished_run)
        response = agent.ask(finished_run, "anything")
        backend = agent.backend
        assert backend.calls  # at least one call was recorded
        first_call = backend.calls[0]
        # Concatenate all user messages and search for the run_dir path
        user_messages = [m.content for m in first_call if m.role == "user"]
        assert any(str(finished_run) in msg for msg in user_messages)

    def test_agent_system_prompt_loaded(self):
        backend = FakeBackend(script=[ScriptedStep(text="")])
        agent = ResultsAgent(backend=backend)
        assert "Results Agent" in agent.system_prompt
        assert "read_qoi_summary" in agent.system_prompt
