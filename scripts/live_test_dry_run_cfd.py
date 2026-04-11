#!/usr/bin/env python3
"""Choice 2 — local dry-run CFD on BPM120 using the agent-generated config.

This script takes the config produced by ``live_test_coordinator.py``
(which you must run first) and hands it to the ExecutionAgent with
``dry_run=True``. That invokes the AortaCFD-app submodule's
``run_patient.py`` CLI with only the ``case,mesh,boundary`` steps:

1. ``case``     — generates every OpenFOAM dictionary from the config
2. ``mesh``     — runs ``blockMesh`` + ``snappyHexMesh`` to build the mesh
3. ``boundary`` — generates ``boundaryData`` for the inlet and writes
                  the Windkessel coefficients into ``0/p``

The solver (``foamRun``) is NOT started. Expected wall time for the
``wall_sensitive`` mesh goal on BPM120's geometry:

* ``case``     — seconds
* ``mesh``     — 10–30 minutes on 8 cores locally (this is where most
                  of the time goes; snappyHexMesh builds ~2–3M cells)
* ``boundary`` — seconds

Total ≈ 10–30 minutes of wall-clock time on a laptop.

Usage::

    cd ~/GitHub/aortacfd-agent
    source venv/bin/activate
    source /opt/openfoam12/etc/bashrc        # make blockMesh/snappy visible
    python scripts/live_test_coordinator.py  # generate the config first
    python scripts/live_test_dry_run_cfd.py  # then run this
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from aortacfd_agent.agents.execution import ExecutionAgent, ExecutionAgentError  # noqa: E402


_AGENT_RUN_DIR = _REPO / "examples" / "output" / "live_BPM120"
_AGENT_CONFIG = _AGENT_RUN_DIR / "agent_config.json"
_CASE_DIR = _REPO / "external" / "aortacfd-app" / "cases_input" / "BPM120"
_PATIENT_ID = "BPM120"


def _pretty(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:6.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes} TB"


def _find_run_dir(patient_id: str) -> Path | None:
    """Return the most recently modified run_* directory for this patient."""
    base = _REPO / "external" / "aortacfd-app" / "output" / patient_id
    if not base.exists():
        return None
    candidates = sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _tail(path: Path, n: int = 20) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return f"(not found: {path})"
    return "\n".join(f"  {line}" for line in lines[-n:])


def main() -> int:
    print("=" * 72)
    print("AortaCFD agent — local dry-run CFD on BPM120 (Choice 2)")
    print("=" * 72)

    # -- Preflight checks ----------------------------------------------------
    if not _AGENT_CONFIG.exists():
        print(
            f"error: {_AGENT_CONFIG} does not exist.\n"
            "       Run `python scripts/live_test_coordinator.py` first to "
            "generate it."
        )
        return 2

    if not _CASE_DIR.exists():
        print(f"error: BPM120 case directory missing: {_CASE_DIR}")
        return 3

    # Warn early if OpenFOAM env doesn't look sourced — the subprocess will
    # fail loudly later anyway, but the error message is much friendlier now.
    if not os.environ.get("FOAM_APP") and not shutil.which("blockMesh"):
        print(
            "warning: OpenFOAM environment does not appear to be sourced.\n"
            "         Before running the CFD pipeline, do:\n"
            "             source /opt/openfoam12/etc/bashrc\n"
            "         Then re-run this script."
        )
        return 4

    print(f"Agent config:  {_AGENT_CONFIG.relative_to(_REPO)}")
    print(f"Case dir:      {_CASE_DIR.relative_to(_REPO)}")
    print(f"Patient ID:    {_PATIENT_ID}")
    print(f"OpenFOAM:      {os.environ.get('WM_PROJECT_VERSION', '(env not detected)')}")
    print()

    # -- Inspect the generated config we're about to run --------------------
    cfg = json.loads(_AGENT_CONFIG.read_text(encoding="utf-8"))
    wk = cfg["boundary_conditions"]["outlets"]["windkessel_settings"]
    print("--- config to be executed ---")
    print(f"  cardiac_cycle     : {cfg.get('cardiac_cycle')} s")
    print(f"  physics.model     : {cfg['physics']['model']}")
    print(f"  numerics.profile  : {cfg['numerics']['profile']}")
    print(f"  mesh.goal         : {cfg.get('mesh', {}).get('goal')}")
    print(f"  inlet.type        : {cfg['boundary_conditions']['inlet']['type']}")
    print(f"  inlet.csv_file    : {cfg['boundary_conditions']['inlet'].get('csv_file')}")
    print(f"  wk.methodology    : {wk.get('methodology')}")
    print(f"  wk.flow_split     : {wk.get('flow_split')}")
    print(f"  wk.tau / betaT    : {wk.get('tau')} / {wk.get('betaT')}")
    print(f"  number_of_cycles  : {cfg.get('simulation_control', {}).get('number_of_cycles')}")
    print()

    # -- Run the ExecutionAgent in dry_run mode ------------------------------
    agent = ExecutionAgent()
    print(f"Invoking run_patient.py via ExecutionAgent (dry_run=True)...")
    print("This builds case + mesh + boundary data. No solver is started.")
    print("Expected wall time: 10-30 minutes depending on mesh size.")
    print()

    run_name = f"agent_{int(time.time())}"
    t0 = time.perf_counter()
    try:
        result = agent.run(
            patient_id=_PATIENT_ID,
            config_path=_AGENT_CONFIG,
            dry_run=True,
            run_name=run_name,
            capture_output=False,  # stream CFD output straight to the terminal
            timeout_s=3600,        # 60-minute safety limit
        )
    except ExecutionAgentError as exc:
        duration = time.perf_counter() - t0
        print()
        print(f"ExecutionAgent FAILED after {duration:.1f}s")
        print(f"  {exc}")
        # Try to surface the most recent run's logs if they exist
        run_dir = _find_run_dir(_PATIENT_ID)
        if run_dir is not None:
            print()
            print(f"Most recent run dir: {run_dir}")
            for log in ("log.blockMesh", "log.snappyHexMesh", "log.checkMesh"):
                p = run_dir / "openfoam" / "logs" / log
                if p.exists():
                    print(f"\n--- tail of {p.relative_to(run_dir.parent)} ---")
                    print(_tail(p, n=20))
        return 1
    duration = time.perf_counter() - t0

    # -- Success summary -----------------------------------------------------
    print()
    print(result.summary())
    print()
    print(f"Wall time: {duration:.1f}s")
    print(f"Command:   {' '.join(result.command)}")
    print(f"CWD:       {result.cwd}")
    print()

    run_dir = _find_run_dir(_PATIENT_ID)
    if run_dir is None:
        print("warning: could not locate the CFD run directory under output/BPM120/")
        return 0
    print(f"CFD run directory: {run_dir}")

    # Mesh quality report, if available.
    check_mesh = run_dir / "openfoam" / "logs" / "log.checkMesh"
    if check_mesh.exists():
        print()
        print(f"--- tail of {check_mesh.relative_to(run_dir.parent)} ---")
        print(_tail(check_mesh, n=25))

    # Count OpenFOAM time directories in the case (should be just '0/' for a
    # dry run since the solver never advances).
    openfoam_case = run_dir / "openfoam"
    if openfoam_case.exists():
        zero_dir = openfoam_case / "0"
        constant_dir = openfoam_case / "constant"
        print()
        print("--- case tree summary ---")
        if zero_dir.exists():
            print(f"  0/           files: {sorted(p.name for p in zero_dir.iterdir() if p.is_file())[:10]}")
        if constant_dir.exists():
            poly = constant_dir / "polyMesh"
            if poly.exists():
                sizes = {p.name: p.stat().st_size for p in poly.iterdir() if p.is_file()}
                print(f"  constant/polyMesh/ files: {len(sizes)}")
                for name in ("points", "cells", "faces", "boundary", "owner", "neighbour"):
                    if name in sizes:
                        print(f"    {name:12} {_pretty(sizes[name])}")

    # Try to pull the cell count from the reconstructed or single-process mesh.
    checkmesh_text = check_mesh.read_text(encoding="utf-8") if check_mesh.exists() else ""
    for line in checkmesh_text.splitlines():
        if "cells:" in line and line.strip().startswith("cells:"):
            print()
            print(f"Mesh size: {line.strip()}")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
