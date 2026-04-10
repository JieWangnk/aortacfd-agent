"""Tool implementations that agents can call through the LLM tool-use API.

Each submodule exposes one or more pure-ish Python functions plus a
``*_spec()`` factory that produces a ``ToolSpec`` for the agent loop to
register. Tools never raise — they return ``{"error": ...}`` dicts so the
model can see failures and retry.

The :func:`build_default_toolset` function is the canonical tool bundle
used by the config agent. Future agents (literature, execution, results)
will assemble their own bundles from the same modules, plus their own
agent-specific tools.
"""

from __future__ import annotations

from typing import List

from ..backends.base import ToolSpec

# Re-export the handler functions directly so callers can
# ``from aortacfd_agent.tools import inspect_geometry`` etc.
from .config_io import save_config, validate_config  # noqa: F401
from .geometry import inspect_geometry  # noqa: F401
from .mesh import suggest_mesh_profile  # noqa: F401
from .outputs import read_qoi_summary  # noqa: F401
from .physics import estimate_reynolds, recommend_physics  # noqa: F401


def build_default_toolset() -> List[ToolSpec]:
    """Return the tool specs exposed to the config agent.

    Add new tools here when you add new agent-facing modules. Keep the
    JSON schemas compact — every field becomes part of the model's
    context window on every turn.
    """
    from .config_io import save_config_spec, validate_config_spec
    from .geometry import inspect_geometry_spec
    from .mesh import suggest_mesh_profile_spec
    from .outputs import read_qoi_summary_spec
    from .physics import estimate_reynolds_spec, recommend_physics_spec

    return [
        inspect_geometry_spec(),
        estimate_reynolds_spec(),
        recommend_physics_spec(),
        suggest_mesh_profile_spec(),
        validate_config_spec(),
        save_config_spec(),
        read_qoi_summary_spec(),
    ]


__all__ = [
    "inspect_geometry",
    "estimate_reynolds",
    "recommend_physics",
    "suggest_mesh_profile",
    "validate_config",
    "save_config",
    "read_qoi_summary",
    "build_default_toolset",
]
