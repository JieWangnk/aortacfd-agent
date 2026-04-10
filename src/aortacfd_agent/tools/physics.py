"""Physics advisor tools — delegate to the AortaCFD physics_advisor module.

Two functions are exposed:

* :func:`estimate_reynolds` — peak Reynolds number from a partial config
* :func:`recommend_physics` — laminar / RANS / LES recommendation with reasoning

Both import ``aortacfd_lib.physics_advisor`` lazily, which means the
submodule's ``src/`` directory must be on ``sys.path`` at call time. In
tests, :mod:`tests.conftest` takes care of that automatically.
"""

from __future__ import annotations

from typing import Any, Dict

from ..backends.base import ToolSpec


def estimate_reynolds(args: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate peak Reynolds number for a tentative config.

    Arguments
    ---------
    config : dict
        A partial AortaCFD config. Must include ``physics`` and
        ``boundary_conditions.inlet``.
    inlet_radius_m : float, optional
        Inlet radius in metres, typically derived from
        :func:`~aortacfd_agent.tools.geometry.inspect_geometry`.
    """
    try:
        from aortacfd_lib.physics_advisor import estimate_reynolds_number
    except ImportError as exc:
        return {"error": f"physics_advisor unavailable: {exc}"}

    config = args.get("config") or {}
    inlet_radius_m = args.get("inlet_radius_m")
    try:
        re = estimate_reynolds_number(config, inlet_radius_m=inlet_radius_m)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    if re is None:
        return {
            "reynolds": None,
            "note": "Insufficient data (no cardiac_output, flowrate, or velocity in config).",
        }

    regime = "laminar" if re < 4000 else ("transitional" if re < 5000 else "turbulent")
    return {"reynolds": re, "regime_hint": regime}


def recommend_physics(args: Dict[str, Any]) -> Dict[str, Any]:
    """Delegate to ``physics_advisor.recommend_physics_model``."""
    try:
        from aortacfd_lib.physics_advisor import recommend_physics_model
    except ImportError as exc:
        return {"error": f"physics_advisor unavailable: {exc}"}

    config = args.get("config") or {}
    inlet_radius_m = args.get("inlet_radius_m")
    try:
        advice = recommend_physics_model(config, inlet_radius_m=inlet_radius_m)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "recommended_model": advice.recommended_model,
        "estimated_reynolds": advice.estimated_re,
        "reasoning": list(advice.reasoning),
        "warnings": list(advice.warnings),
        "mesh_compatible": advice.mesh_compatible,
    }


def estimate_reynolds_spec() -> ToolSpec:
    return ToolSpec(
        name="estimate_reynolds",
        description=(
            "Estimate peak Reynolds number from a partial config "
            "(uses cardiac_output/flowrate/velocity and inlet area)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "description": "Partial AortaCFD config dict.",
                },
                "inlet_radius_m": {
                    "type": "number",
                    "description": "Inlet radius in metres, if known.",
                },
            },
            "required": ["config"],
        },
        handler=estimate_reynolds,
    )


def recommend_physics_spec() -> ToolSpec:
    return ToolSpec(
        name="recommend_physics",
        description=(
            "Ask the built-in physics advisor which model "
            "(laminar/RANS/LES) suits the config and get its reasoning."
        ),
        parameters={
            "type": "object",
            "properties": {
                "config": {"type": "object"},
                "inlet_radius_m": {"type": "number"},
            },
            "required": ["config"],
        },
        handler=recommend_physics,
    )
