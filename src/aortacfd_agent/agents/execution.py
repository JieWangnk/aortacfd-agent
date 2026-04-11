"""ExecutionAgent — run the AortaCFD pipeline on a validated config.

Fourth specialist in the supervisor. Takes a validated ``agent_config.json``
(produced by :class:`~aortacfd_agent.agents.config.ConfigAgent`) plus a
patient_id and executes the CFD pipeline via ``subprocess`` — we never
import the submodule's solver code into this process.

Using a subprocess has three important properties:

1. **Isolation** — the solver writes a lot of files and can crash or
   deadlock. Running it as a child process means failures cannot take
   down the agent loop.
2. **Remoting ready** — the same interface works locally, on HPC (via a
   batch script), or inside a container. The agent doesn't know or care.
3. **Zero import coupling** — the agent layer's Python environment is
   independent of whatever the CFD pipeline needs (OpenFOAM, numpy-stl,
   Jinja, pyvista, …).

There is no LLM inside ExecutionAgent either, for the same reason as
ConfigAgent: the decisions were already made upstream. The class keeps
an ``agent``-style interface (``ExecutionAgent.run(...)``) so the
coordinator chains it uniformly with the others.

Dry-run mode stops after the case-setup, mesh, and boundary steps — the
pipeline still produces dictionaries and a mesh, but the solver is not
started. This is the default for demos and for most of the test suite.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class ExecutionAgentError(RuntimeError):
    """Raised when the CFD pipeline subprocess fails or cannot be located."""


@dataclass
class ExecutionResult:
    """What :meth:`ExecutionAgent.run` returns."""

    patient_id: str
    steps: List[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    command: List[str] = field(default_factory=list)
    cwd: str = ""
    duration_s: float = 0.0
    run_dir: Optional[str] = None
    dry_run: bool = False

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def summary(self) -> str:
        status = "OK" if self.success else f"FAILED (exit {self.returncode})"
        mode = " [dry-run]" if self.dry_run else ""
        return f"ExecutionAgent{mode}: {self.patient_id} {status} in {self.duration_s:.1f}s"


# ---------------------------------------------------------------------------
# ExecutionAgent
# ---------------------------------------------------------------------------


# Default AortaCFD-app submodule location inside this repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_SUBMODULE = _REPO_ROOT / "external" / "aortacfd-app"

# Pydantic mask for the subprocess environment. See the full justification
# in ``src/aortacfd_agent/_subproc_shims/pydantic_mask/pydantic.py``. Short
# version: the pinned submodule has a latent validator bug under pydantic v2
# that only triggers when pydantic is importable; prepending this dir to
# PYTHONPATH makes the subprocess think pydantic is not installed and fall
# back to the permissive dict-path. Our agent process is unaffected because
# we only set this in the child environment.
_PYDANTIC_MASK_DIR = (
    Path(__file__).resolve().parent.parent
    / "_subproc_shims"
    / "pydantic_mask"
)

# Canonical step groups the CFD CLI recognises. These match the values
# accepted by ``run_patient.py --steps``. Dry-run stops at 'boundary';
# full runs include the solver and post-processing.
_DRY_RUN_STEPS = ("case", "mesh", "boundary")
_FULL_STEPS = ("case", "mesh", "boundary", "solver", "reconstruct", "postprocess")


class ExecutionAgent:
    """Drive the ``run_patient.py`` CFD CLI on a validated config.

    Parameters
    ----------
    submodule_path
        Path to the AortaCFD-app git submodule root. Defaults to
        ``external/aortacfd-app`` relative to this repo.
    python_executable
        Which Python to use for the child process. Defaults to
        ``sys.executable`` (the same interpreter the agent is running
        under), which is what you want 95% of the time.
    """

    def __init__(
        self,
        submodule_path: Optional[Path] = None,
        python_executable: Optional[str] = None,
    ):
        self.submodule_path = Path(submodule_path) if submodule_path else _DEFAULT_SUBMODULE
        self.python_executable = python_executable or sys.executable

        if not self.submodule_path.exists():
            raise ExecutionAgentError(
                f"AortaCFD-app submodule not found at {self.submodule_path}. "
                "Run `git submodule update --init --recursive`."
            )
        runner = self.submodule_path / "run_patient.py"
        if not runner.exists():
            raise ExecutionAgentError(
                f"run_patient.py not found under {self.submodule_path}. "
                "Is the submodule pointing at a compatible AortaCFD-app commit?"
            )
        self.runner_path = runner

    # -- public API ----------------------------------------------------------

    def run(
        self,
        patient_id: str,
        config_path: Path,
        *,
        dry_run: bool = True,
        steps: Optional[Sequence[str]] = None,
        run_name: Optional[str] = None,
        extra_args: Optional[Sequence[str]] = None,
        timeout_s: Optional[float] = None,
        capture_output: bool = True,
        executor: Optional[Any] = None,
    ) -> ExecutionResult:
        """Execute the pipeline for one patient case.

        Parameters
        ----------
        patient_id
            Patient identifier the CFD CLI will use (e.g. ``"BPM120"``).
        config_path
            Absolute path to an agent-generated config.json. Passed to the
            CLI via ``--config``.
        dry_run
            If True (default), runs only ``case,mesh,boundary`` so no
            solver is started. If False, runs the full pipeline including
            solver + post-processing.
        steps
            Explicit list of steps to override the dry-run / full defaults.
        run_name
            Optional ``--run-name`` passed to the CLI. Helpful for placing
            output under a predictable directory.
        extra_args
            Extra CLI flags appended verbatim (e.g. ``["--verbose"]``).
        timeout_s
            Subprocess wall-clock timeout. ``None`` (default) waits
            indefinitely; tests pass a small value.
        capture_output
            If True (default) the stdout/stderr are captured and returned
            on the result. Set False for interactive runs where you want
            the CLI's own output streamed to the terminal.
        executor
            Test seam: if provided, must be a callable with the same
            signature as :func:`subprocess.run`. Injected by unit tests so
            they can assert on the command without actually running it.

        Raises
        ------
        ExecutionAgentError
            If the CLI exits with a non-zero return code, times out, or
            the ``config_path`` does not exist.
        """
        config_path = Path(config_path).resolve()
        if not config_path.exists():
            raise ExecutionAgentError(f"config file does not exist: {config_path}")

        effective_steps = list(steps) if steps else list(
            _DRY_RUN_STEPS if dry_run else _FULL_STEPS
        )

        command: List[str] = [
            self.python_executable,
            str(self.runner_path),
            patient_id,
            "--config",
            str(config_path),
            "--steps",
            ",".join(effective_steps),
        ]
        if run_name:
            command.extend(["--run-name", run_name])
        if extra_args:
            command.extend(list(extra_args))

        run = executor if executor is not None else subprocess.run
        cwd = str(self.submodule_path)

        # Build the subprocess environment with the pydantic mask prepended
        # to PYTHONPATH so the submodule's config validator falls back to
        # its dict path (see _PYDANTIC_MASK_DIR comment above).
        child_env = os.environ.copy()
        existing_pp = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = (
            f"{_PYDANTIC_MASK_DIR}{os.pathsep}{existing_pp}"
            if existing_pp
            else str(_PYDANTIC_MASK_DIR)
        )

        logger.info("ExecutionAgent: invoking %s", " ".join(command))
        t0 = time.perf_counter()
        try:
            completed = run(
                command,
                cwd=cwd,
                check=False,
                timeout=timeout_s,
                capture_output=capture_output,
                text=True,
                env=child_env,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionAgentError(
                f"run_patient.py timed out after {timeout_s}s: {exc}"
            ) from exc
        except FileNotFoundError as exc:
            raise ExecutionAgentError(
                f"failed to launch python interpreter {self.python_executable!r}: {exc}"
            ) from exc
        duration = time.perf_counter() - t0

        stdout = getattr(completed, "stdout", "") or ""
        stderr = getattr(completed, "stderr", "") or ""
        returncode = int(getattr(completed, "returncode", 0))
        run_dir = _parse_run_dir(stdout)

        result = ExecutionResult(
            patient_id=patient_id,
            steps=effective_steps,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            command=command,
            cwd=cwd,
            duration_s=duration,
            run_dir=run_dir,
            dry_run=dry_run,
        )

        if not result.success:
            logger.warning(
                "ExecutionAgent: %s exited non-zero (%s). stderr tail: %s",
                patient_id,
                returncode,
                stderr[-400:] if stderr else "(empty)",
            )
            raise ExecutionAgentError(
                f"run_patient.py exited with code {returncode} for {patient_id}. "
                f"stderr tail: {stderr[-400:] if stderr else '(empty)'}"
            )

        return result


# ---------------------------------------------------------------------------
# Stdout parsing helpers
# ---------------------------------------------------------------------------


_RUN_DIR_MARKERS = (
    "Run directory:",
    "Run dir:",
    "Output dir:",
    "Run output:",
)


def _parse_run_dir(stdout: str) -> Optional[str]:
    """Best-effort extraction of the run directory from CLI stdout.

    The AortaCFD CLI writes its output directory to stdout in a line like::

        Run directory: output/BPM120/run_20260410_183022

    If the format changes or the output is empty, we return None and the
    caller can fall back to scanning ``output/<patient_id>/`` itself.
    """
    if not stdout:
        return None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        for marker in _RUN_DIR_MARKERS:
            if marker in line:
                tail = line.split(marker, 1)[1].strip()
                if tail:
                    return tail
    return None


__all__ = [
    "ExecutionAgent",
    "ExecutionAgentError",
    "ExecutionResult",
]
