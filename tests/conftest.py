"""Shared pytest fixtures for the aortacfd-agent test suite.

The important pieces are:

* ``submodule_path`` — absolute path to ``external/aortacfd-app``. Every test
  that touches real STL files or config templates uses this.
* ``sys_path_for_submodule`` — inserts the submodule's ``src/`` directory into
  ``sys.path`` so tests can directly import from the AortaCFD library (e.g.
  ``from aortacfd_lib.physics_advisor import recommend_physics_model``).
* ``case_bpm120`` — path to the BPM120 case directory inside the submodule,
  used as the canonical "real patient with STLs" fixture.
* ``fake_backend`` — a shared FakeBackend instance seeded with no responses,
  for tests that need to inject their own scripted responses.

The goal is that every test in this repo runs offline with zero network
calls, in under a second per test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMODULE = REPO_ROOT / "external" / "aortacfd-app"


@pytest.fixture(scope="session")
def submodule_path() -> Path:
    """Absolute path to the AortaCFD-app git submodule."""
    if not SUBMODULE.exists():
        pytest.skip(
            "external/aortacfd-app submodule not initialised; "
            "run `git submodule update --init` first"
        )
    return SUBMODULE


@pytest.fixture(scope="session", autouse=True)
def sys_path_for_submodule():
    """Make ``aortacfd_lib`` and friends importable from the submodule.

    This runs once per session and is opt-out only — tests that want to
    prove a module works without the submodule should use a subprocess.
    """
    src = SUBMODULE / "src"
    if src.exists():
        sys.path.insert(0, str(src))
    yield
    try:
        sys.path.remove(str(src))
    except ValueError:
        pass


@pytest.fixture
def case_bpm120(submodule_path: Path) -> Path:
    """Canonical BPM120 patient case directory with STLs and flow data."""
    case_dir = submodule_path / "cases_input" / "BPM120"
    if not case_dir.exists():
        pytest.skip(f"BPM120 case not found at {case_dir}")
    return case_dir
