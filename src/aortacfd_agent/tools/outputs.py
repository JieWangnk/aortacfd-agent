"""Read tools for the ResultsAgent.

These read finished-run artefacts from ``output/<case>/<run_id>/``.
The ResultsAgent (built in Phase D5) uses them to answer clinician
questions in natural language with numerical backing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..backends.base import ToolSpec


def read_qoi_summary(args: Dict[str, Any]) -> Dict[str, Any]:
    """Read ``results/qoi_summary.json`` (or top-level ``qoi_summary.json``) from a finished run."""
    run_dir = Path(args.get("run_dir", ""))
    candidates = [
        run_dir / "results" / "qoi_summary.json",
        run_dir / "qoi_summary.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                return {"run_dir": str(run_dir), "qoi": json.loads(c.read_text())}
            except json.JSONDecodeError as exc:
                return {"error": f"failed to parse {c}: {exc}"}
    return {"error": f"no qoi_summary.json under {run_dir}"}


def read_qoi_summary_spec() -> ToolSpec:
    return ToolSpec(
        name="read_qoi_summary",
        description=(
            "Read QoI summary JSON from a finished run directory. "
            "Used by the ResultsAgent after a simulation."
        ),
        parameters={
            "type": "object",
            "properties": {"run_dir": {"type": "string"}},
            "required": ["run_dir"],
        },
        handler=read_qoi_summary,
    )
