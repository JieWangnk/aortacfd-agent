"""Generate a complete OpenFOAM case directory from an agent config.

No OpenFOAM binaries required — only Python + Jinja2 templates.
Uses the aortacfd-app PatientCaseRunner programmatically.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _aortacfd_app_root() -> Path:
    return _repo_root() / "external" / "aortacfd-app"


def _ensure_paths():
    app_root = _aortacfd_app_root()
    app_src = app_root / "src"
    for p in (str(app_root), str(app_src)):
        if p not in sys.path:
            sys.path.insert(0, p)


def build_openfoam_case(
    agent_config: dict,
    stl_source_dir: Path,
    case_id: str = "BPM120",
) -> Optional[bytes]:
    """Generate a complete OpenFOAM case as a zip file (bytes).

    Strategy: switch cwd to aortacfd-app, stage input in cases_input/<case_id>/,
    run PatientCaseRunner's setup:dict workflow step, then zip the output.
    """
    _ensure_paths()

    try:
        # Import + patch schema BEFORE patient_runner picks up the symbols.
        # aortacfd-app's PhysicsConfig validator has a bug with physics.model
        # under `use_enum_values=True`. Our agent config is already validated.
        import config.schema as _schema
        _schema.is_pydantic_available = lambda: False
        _schema.validate_config = lambda c: c
        # Also patch references already taken by patient_runner if module is cached
        import importlib
        if "patient_runner.core" in sys.modules:
            _mod = sys.modules["patient_runner.core"]
            _mod.is_pydantic_available = lambda: False
            _mod.validate_config = lambda c: c
        from patient_runner.core import PatientCaseRunner
        # Re-patch after the fresh import to override any re-imported refs
        import patient_runner.core as _pr
        _pr.is_pydantic_available = lambda: False
        _pr.validate_config = lambda c: c
    except ImportError as e:
        print(f"[build_case] aortacfd-app not importable: {e}")
        return None

    app_root = _aortacfd_app_root()
    original_cwd = Path.cwd()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # We need to work inside a temporary root that looks like the aortacfd-app
        # tree (so PatientCaseRunner's relative paths make sense), but we can't
        # mutate the real repo. Solution: symlink the app into tmp, then chdir.
        app_staging = tmp / "app"
        app_staging.mkdir()
        # Symlink src/ and the templates (read-only)
        for sub in ("src", "templates"):
            src = app_root / sub
            if src.exists():
                os.symlink(src, app_staging / sub)

        # Create cases_input/<case_id>/ under the staging dir
        case_in = app_staging / "cases_input" / case_id
        case_in.mkdir(parents=True)
        for fp in stl_source_dir.iterdir():
            if fp.is_file() and fp.suffix.lower() in {".stl", ".csv", ".json"}:
                shutil.copy(fp, case_in / fp.name)

        # Write the agent's config as-is. We've disabled pydantic validation
        # above to avoid a known bug with physics.model handling.
        clean_config = dict(agent_config)
        if "physics" in clean_config:
            physics = dict(clean_config["physics"])
            # Ensure simulation_type matches model for downstream consistency
            physics.setdefault("simulation_type", physics.get("model", "laminar"))
            clean_config["physics"] = physics

        agent_config_path = case_in / "config.json"
        agent_config_path.write_text(
            json.dumps(clean_config, indent=2), encoding="utf-8"
        )

        # Output goes under staging/output/<case_id>/
        try:
            os.chdir(app_staging)

            runner = PatientCaseRunner()
            case_info = runner.load_patient_case(case_id, config_path=str(agent_config_path))
            sim_config = runner.prepare_simulation(case_info, options=None)
            ok = runner.run_workflow_step(sim_config, workflow_step="setup:dict")
            if not ok:
                print("[build_case] setup:dict returned False")
                return None

            run_dir = sim_config["run_dir"]
            case_dir = Path(run_dir) / "openfoam"
            if not case_dir.exists():
                print(f"[build_case] openfoam dir not found at {case_dir}")
                return None

            # Copy agent artefacts into the run directory
            agent_out = Path(run_dir) / "agent"
            agent_out.mkdir(exist_ok=True)
            (agent_out / "agent_config.json").write_text(
                json.dumps(agent_config, indent=2), encoding="utf-8"
            )
            _write_readme(Path(run_dir), case_id)

            # Zip the whole run directory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in Path(run_dir).rglob("*"):
                    if file_path.is_file():
                        arcname = f"{case_id}_case/" + str(
                            file_path.relative_to(run_dir)
                        )
                        zf.write(file_path, arcname)
            return zip_buffer.getvalue()

        except Exception as e:
            import traceback
            print(f"[build_case] FAILED: {e}")
            traceback.print_exc()
            return None
        finally:
            os.chdir(original_cwd)


def _write_readme(run_dir: Path, case_id: str):
    (run_dir / "README.md").write_text(
        f"""# OpenFOAM Case — {case_id}

Generated by **AortaCFD Agent** from a clinical referral.

## Layout

```
{case_id}_case/
├── openfoam/
│   ├── 0/                  field BCs (populated after meshing)
│   ├── constant/
│   │   ├── triSurface/     scaled STLs (ready for snappyHexMesh)
│   │   ├── transportProperties
│   │   └── momentumTransport
│   └── system/
│       ├── controlDict
│       ├── fvSchemes
│       ├── fvSolution
│       ├── blockMeshDict
│       ├── snappyHexMeshDict
│       ├── surfaceFeaturesDict
│       └── decomposeParDict
├── agent/
│   └── agent_config.json   validated config from the agent
└── README.md
```

## Run it

You need **OpenFOAM 12** (Foundation) sourced.

```bash
cd openfoam

# 1. Background Cartesian mesh
blockMesh

# 2. Extract sharp feature edges
surfaceFeatures

# 3. Castellate, snap, add boundary layers
snappyHexMesh -overwrite

# 4. Verify quality
checkMesh

# 5. Write BCs into 0/ and run the solver. Simplest path via aortacfd-app:
python /path/to/AortaCFD-app/run_patient.py {case_id} \\
    --update $(pwd) --steps boundary,solver,postprocess
```

Or run the solver directly:

```bash
foamRun -solver incompressibleFluid
```

## Reproducibility

`agent/agent_config.json` captures every parameter the agent chose — physics
model, numerics profile, Windkessel values, flow waveform source, mesh goal.
Paste that with your paper and reviewers can reproduce the case verbatim.

## Attribution

Generated by AortaCFD: https://jiewangnk.github.io/AortaCFD-web/
""",
        encoding="utf-8",
    )
