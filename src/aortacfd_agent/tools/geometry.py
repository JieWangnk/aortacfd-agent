"""Geometry inspection tool — list STL patches in a case directory.

The AortaCFD convention is one STL per patch: ``inlet.stl``,
``outlet1.stl``, ..., ``wall_aorta.stl``. The inlet equivalent radius this
function returns is used downstream by the physics advisor to estimate
peak Reynolds number.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

from ..backends.base import ToolSpec


def _stl_bbox_and_area(path: Path) -> Dict[str, Any]:
    """Return bounding box + approximate area of an STL file.

    Uses ``numpy-stl`` (pinned in ``pyproject.toml`` dependencies). If the
    read fails we return an ``{"error": ...}`` dict so the agent loop can
    surface the error to the model rather than crashing.
    """
    try:
        from stl import mesh as _stl_mesh  # numpy-stl
    except ImportError:  # pragma: no cover
        return {"error": f"numpy-stl not available to read {path.name}"}

    try:
        m = _stl_mesh.Mesh.from_file(str(path))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"failed to read {path.name}: {exc}"}

    import numpy as np  # numpy-stl already brings numpy

    pts = m.vectors.reshape(-1, 3)
    mn = pts.min(axis=0).tolist()
    mx = pts.max(axis=0).tolist()
    span = [mx[i] - mn[i] for i in range(3)]

    v0 = m.vectors[:, 0, :]
    v1 = m.vectors[:, 1, :]
    v2 = m.vectors[:, 2, :]
    tri_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total_area = float(tri_area.sum())
    equivalent_radius = math.sqrt(total_area / math.pi) if total_area > 0 else 0.0

    return {
        "filename": path.name,
        "num_triangles": int(m.vectors.shape[0]),
        "bbox_min": mn,
        "bbox_max": mx,
        "span": span,
        "total_area_units2": total_area,
        "equivalent_radius_units": equivalent_radius,
        "units_note": "values are in STL native units (typically mm for AortaCFD)",
    }


def inspect_geometry(args: Dict[str, Any]) -> Dict[str, Any]:
    """List every STL file in a case directory and classify patches.

    Arguments
    ---------
    case_dir : str
        Path to a directory containing patient STL files.

    Returns a dict with one entry per STL plus a role hint
    (``inlet`` / ``outlet`` / ``wall``) derived from the filename.
    """
    case_dir = Path(args.get("case_dir", ""))
    if not case_dir.exists() or not case_dir.is_dir():
        return {"error": f"case_dir does not exist: {case_dir}"}

    stls = sorted(case_dir.glob("*.stl"))
    if not stls:
        return {"error": f"no STL files found in {case_dir}"}

    def classify(name: str) -> str:
        lower = name.lower()
        if "inlet" in lower:
            return "inlet"
        if "outlet" in lower:
            return "outlet"
        if "wall" in lower or "aorta" in lower:
            return "wall"
        return "unknown"

    patches: List[Dict[str, Any]] = []
    for stl in stls:
        info = _stl_bbox_and_area(stl)
        info["patch_role_hint"] = classify(stl.name)
        patches.append(info)

    counts: Dict[str, int] = {}
    for p in patches:
        role = p.get("patch_role_hint", "unknown")
        counts[role] = counts.get(role, 0) + 1

    return {
        "case_dir": str(case_dir),
        "num_stls": len(patches),
        "patch_counts": counts,
        "patches": patches,
    }


def inspect_geometry_spec() -> ToolSpec:
    """Tool spec the agent loop registers."""
    return ToolSpec(
        name="inspect_geometry",
        description=(
            "List STL patches in a patient case directory and return "
            "per-patch bounding box, area, and a role hint "
            "(inlet/outlet/wall). Call this first to discover geometry."
        ),
        parameters={
            "type": "object",
            "properties": {
                "case_dir": {
                    "type": "string",
                    "description": "Path to the case directory containing STL files.",
                }
            },
            "required": ["case_dir"],
        },
        handler=inspect_geometry,
    )
