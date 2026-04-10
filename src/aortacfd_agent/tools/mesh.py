"""Mesh goal preset lookup — a pure policy lookup, no CFD computation.

The AortaCFD mesh config schema supports three named goals with different
span targets and layer policies. This tool lets the agent pick one by
describing its intent (``"pressure_fast"``, ``"routine_hemodynamics"``,
``"wall_sensitive"``) rather than setting span targets numerically.
"""

from __future__ import annotations

from typing import Any, Dict

from ..backends.base import ToolSpec


_MESH_GOALS: Dict[str, Dict[str, Any]] = {
    "pressure_fast": {
        "span_target": 10,
        "layers": "off",
        "surfaceRefinementLevels": [0, 1],
        "use_case": "Quick screening, pressure gradient only.",
    },
    "routine_hemodynamics": {
        "span_target": 16,
        "layers": "standard",
        "surfaceRefinementLevels": [2, 2],
        "use_case": "Production patient-specific runs (recommended default).",
    },
    "wall_sensitive": {
        "span_target": 22,
        "layers": "standard",
        "surfaceRefinementLevels": [2, 2],
        "use_case": "WSS, OSI, near-wall indices.",
    },
}


def suggest_mesh_profile(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return span target and layer settings for a given mesh goal.

    Arguments
    ---------
    goal : str
        One of ``pressure_fast``, ``routine_hemodynamics``, ``wall_sensitive``.
    """
    goal = args.get("goal", "routine_hemodynamics")
    if goal not in _MESH_GOALS:
        return {
            "error": f"unknown goal '{goal}'",
            "available_goals": sorted(_MESH_GOALS.keys()),
        }
    return {"goal": goal, **_MESH_GOALS[goal]}


def suggest_mesh_profile_spec() -> ToolSpec:
    return ToolSpec(
        name="suggest_mesh_profile",
        description=(
            "Look up the span target and layer settings associated with "
            "a mesh goal preset."
        ),
        parameters={
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "enum": sorted(_MESH_GOALS.keys()),
                }
            },
            "required": ["goal"],
        },
        handler=suggest_mesh_profile,
    )
