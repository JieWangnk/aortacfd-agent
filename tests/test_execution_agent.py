"""Unit tests for :class:`aortacfd_agent.agents.execution.ExecutionAgent`.

We never actually run ``run_patient.py`` in these tests — the CFD
pipeline needs OpenFOAM installed and takes minutes to hours. Instead,
the :class:`ExecutionAgent` accepts an ``executor`` keyword on
:meth:`run`, which the tests use to inject a fake that records the
command and returns a scripted ``CompletedProcess``. That gives us
deterministic coverage of:

* command construction (flags, step lists, dry-run vs full)
* stdout parsing for the run directory
* error paths (config missing, subprocess non-zero, timeout)
* the ``ExecutionResult`` dataclass fields and summary string

There is also one integration-style test that creates a real temporary
config, then drives the agent with a recording executor to prove the
full happy path works end-to-end without touching the solver.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from aortacfd_agent.agents.execution import (
    ExecutionAgent,
    ExecutionAgentError,
    ExecutionResult,
    _parse_run_dir,
)


# ---------------------------------------------------------------------------
# Executor fake: records calls + returns scripted CompletedProcess objects
# ---------------------------------------------------------------------------


class RecordingExecutor:
    """Mimics :func:`subprocess.run` for tests.

    Attributes
    ----------
    calls
        List of ``(command, kwargs)`` tuples, one per invocation.
    returncode
        Scripted return code for the next call.
    stdout / stderr
        Scripted output streams.
    raise_timeout
        If True, raise ``TimeoutExpired`` on the next call.
    raise_file_not_found
        If True, raise ``FileNotFoundError``.
    """

    def __init__(
        self,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        raise_timeout: bool = False,
        raise_file_not_found: bool = False,
    ):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.raise_timeout = raise_timeout
        self.raise_file_not_found = raise_file_not_found
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        check: bool = False,
        timeout: Optional[float] = None,
        capture_output: bool = True,
        text: bool = True,
    ) -> subprocess.CompletedProcess:
        self.calls.append(
            {
                "command": list(command),
                "cwd": cwd,
                "check": check,
                "timeout": timeout,
                "capture_output": capture_output,
                "text": text,
            }
        )
        if self.raise_timeout:
            raise subprocess.TimeoutExpired(cmd=command, timeout=timeout or 0)
        if self.raise_file_not_found:
            raise FileNotFoundError("no such python interpreter")
        return subprocess.CompletedProcess(
            args=command,
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_config(tmp_path: Path) -> Path:
    """Write a minimal placeholder config to disk so the path check passes."""
    cfg = tmp_path / "agent_config.json"
    cfg.write_text(json.dumps({"case_info": {"patient_id": "TESTPAT"}}))
    return cfg


# ---------------------------------------------------------------------------
# Constructor and submodule discovery
# ---------------------------------------------------------------------------


class TestExecutionAgentInit:
    def test_default_submodule_exists(self):
        agent = ExecutionAgent()
        assert agent.submodule_path.exists()
        assert agent.runner_path.exists()
        assert agent.runner_path.name == "run_patient.py"

    def test_missing_submodule_raises(self, tmp_path):
        with pytest.raises(ExecutionAgentError, match="submodule not found"):
            ExecutionAgent(submodule_path=tmp_path / "nope")

    def test_submodule_without_runner_raises(self, tmp_path):
        """If the submodule dir exists but run_patient.py doesn't, fail loudly."""
        fake_submodule = tmp_path / "fake_submodule"
        fake_submodule.mkdir()
        with pytest.raises(ExecutionAgentError, match="run_patient.py not found"):
            ExecutionAgent(submodule_path=fake_submodule)


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


class TestCommandConstruction:
    def test_dry_run_default_steps(self, fake_config):
        agent = ExecutionAgent()
        executor = RecordingExecutor(stdout="Run directory: output/TESTPAT/run_1\n")

        result = agent.run(
            patient_id="TESTPAT",
            config_path=fake_config,
            dry_run=True,
            executor=executor,
        )

        assert result.success
        assert len(executor.calls) == 1
        command = executor.calls[0]["command"]
        assert "run_patient.py" in command[1]
        assert "TESTPAT" in command
        assert "--config" in command
        assert str(fake_config) in command
        assert "--steps" in command
        steps_arg = command[command.index("--steps") + 1]
        assert steps_arg == "case,mesh,boundary"

    def test_full_run_steps(self, fake_config):
        agent = ExecutionAgent()
        executor = RecordingExecutor()

        agent.run(
            patient_id="FULL",
            config_path=fake_config,
            dry_run=False,
            executor=executor,
        )

        steps_arg = executor.calls[0]["command"][
            executor.calls[0]["command"].index("--steps") + 1
        ]
        assert "solver" in steps_arg
        assert "postprocess" in steps_arg

    def test_custom_steps_override_defaults(self, fake_config):
        agent = ExecutionAgent()
        executor = RecordingExecutor()
        agent.run(
            patient_id="X",
            config_path=fake_config,
            steps=["case", "mesh"],
            executor=executor,
        )
        steps_arg = executor.calls[0]["command"][
            executor.calls[0]["command"].index("--steps") + 1
        ]
        assert steps_arg == "case,mesh"

    def test_run_name_passed(self, fake_config):
        agent = ExecutionAgent()
        executor = RecordingExecutor()
        agent.run(
            patient_id="X",
            config_path=fake_config,
            run_name="agent_demo_20260410",
            executor=executor,
        )
        cmd = executor.calls[0]["command"]
        assert "--run-name" in cmd
        assert "agent_demo_20260410" in cmd

    def test_extra_args_appended(self, fake_config):
        agent = ExecutionAgent()
        executor = RecordingExecutor()
        agent.run(
            patient_id="X",
            config_path=fake_config,
            extra_args=["--verbose", "--quick"],
            executor=executor,
        )
        cmd = executor.calls[0]["command"]
        assert "--verbose" in cmd
        assert "--quick" in cmd

    def test_cwd_is_submodule_root(self, fake_config):
        agent = ExecutionAgent()
        executor = RecordingExecutor()
        agent.run(patient_id="X", config_path=fake_config, executor=executor)
        cwd = executor.calls[0]["cwd"]
        assert cwd and "aortacfd-app" in cwd

    def test_timeout_passed_through(self, fake_config):
        agent = ExecutionAgent()
        executor = RecordingExecutor()
        agent.run(
            patient_id="X",
            config_path=fake_config,
            timeout_s=300.0,
            executor=executor,
        )
        assert executor.calls[0]["timeout"] == 300.0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_missing_config_raises(self, tmp_path):
        agent = ExecutionAgent()
        with pytest.raises(ExecutionAgentError, match="config file does not exist"):
            agent.run(
                patient_id="X",
                config_path=tmp_path / "nope.json",
                executor=RecordingExecutor(),
            )

    def test_non_zero_exit_raises(self, fake_config):
        agent = ExecutionAgent()
        executor = RecordingExecutor(
            returncode=42,
            stdout="",
            stderr="snappyHexMesh failed: non-orthogonality too high\n",
        )
        with pytest.raises(ExecutionAgentError, match="exited with code 42"):
            agent.run(patient_id="X", config_path=fake_config, executor=executor)

    def test_timeout_raises(self, fake_config):
        agent = ExecutionAgent()
        executor = RecordingExecutor(raise_timeout=True)
        with pytest.raises(ExecutionAgentError, match="timed out"):
            agent.run(
                patient_id="X",
                config_path=fake_config,
                timeout_s=1.0,
                executor=executor,
            )

    def test_file_not_found_raises(self, fake_config):
        agent = ExecutionAgent(python_executable="/no/such/python")
        executor = RecordingExecutor(raise_file_not_found=True)
        with pytest.raises(ExecutionAgentError, match="failed to launch python"):
            agent.run(patient_id="X", config_path=fake_config, executor=executor)


# ---------------------------------------------------------------------------
# Result dataclass + stdout parsing
# ---------------------------------------------------------------------------


class TestExecutionResult:
    def test_summary_string_success(self, fake_config):
        agent = ExecutionAgent()
        executor = RecordingExecutor(stdout="Run directory: output/X/run_1\n")
        result = agent.run(
            patient_id="X",
            config_path=fake_config,
            dry_run=True,
            executor=executor,
        )
        assert result.success is True
        assert result.dry_run is True
        summary = result.summary()
        assert "OK" in summary
        assert "dry-run" in summary
        assert "X" in summary

    def test_run_dir_parsed_from_stdout(self, fake_config):
        agent = ExecutionAgent()
        executor = RecordingExecutor(
            stdout="something\nRun directory: output/BPM120/run_20260410\nmore\n",
        )
        result = agent.run(
            patient_id="BPM120", config_path=fake_config, executor=executor
        )
        assert result.run_dir == "output/BPM120/run_20260410"

    def test_run_dir_none_when_not_reported(self, fake_config):
        agent = ExecutionAgent()
        executor = RecordingExecutor(stdout="no markers here\n")
        result = agent.run(
            patient_id="X", config_path=fake_config, executor=executor
        )
        assert result.run_dir is None

    def test_parse_run_dir_multiple_markers(self):
        assert (
            _parse_run_dir("Output dir: foo/bar/run_1\n")
            == "foo/bar/run_1"
        )
        assert (
            _parse_run_dir("Run dir: ./x/y\n")
            == "./x/y"
        )
        assert _parse_run_dir("") is None
        assert _parse_run_dir("just a normal line") is None
