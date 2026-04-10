"""Config validation + save tools.

:func:`validate_config` delegates to the pydantic schema in
``config.schema`` (from the AortaCFD-app submodule). Note that the pydantic
schema is intentionally permissive — it catches type errors and structural
mistakes but does not enforce every enum. For stricter physiological
bounds the agent should also call
:func:`~aortacfd_agent.tools.bounds.validate_wk_bounds`.

:func:`save_config` is the only tool in the whole agent layer that writes
files. The agent is instructed to call it exactly once, at the end of the
workflow, after validation has passed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..backends.base import ToolSpec


def validate_config(args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a candidate config against the AortaCFD pydantic schema.

    Arguments
    ---------
    config : dict
        The candidate config to validate.

    Returns ``{"valid": true}`` on success, otherwise a structured error
    dict. The agent loop uses this as its stopping signal.
    """
    try:
        from config.schema import validate_config as _validate
    except ImportError as exc:
        return {"error": f"config.schema unavailable: {exc}"}

    config = args.get("config") or {}
    try:
        _validate(config)
    except Exception as exc:  # pydantic ValidationError or plain ValueError
        errors: List[Dict[str, Any]] = []
        inner = getattr(exc, "errors", None)
        if callable(inner):
            try:
                for e in inner():
                    errors.append(
                        {
                            "loc": list(e.get("loc", [])),
                            "msg": e.get("msg", ""),
                            "type": e.get("type", ""),
                        }
                    )
            except Exception:  # noqa: BLE001
                pass
        return {
            "valid": False,
            "error_summary": f"{type(exc).__name__}: {exc}",
            "errors": errors,
        }

    return {"valid": True}


def save_config(args: Dict[str, Any]) -> Dict[str, Any]:
    """Write a validated config plus rationale to disk.

    Arguments
    ---------
    config : dict
        The config to save. Re-validated at save time as a safety net.
    rationale : str
        Human-readable markdown explanation of non-default choices.
    output_dir : str
        Directory to write into (created if missing).
    config_filename : str, default ``agent_config.json``
    rationale_filename : str, default ``agent_rationale.md``
    """
    config = args.get("config") or {}
    rationale = args.get("rationale", "")
    output_dir = Path(args.get("output_dir", ""))
    if not str(output_dir):
        return {"error": "output_dir is required"}

    # Re-validate before writing.
    check = validate_config({"config": config})
    if not check.get("valid"):
        return {
            "error": "config failed schema validation at save time",
            "details": check,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / args.get("config_filename", "agent_config.json")
    rationale_path = output_dir / args.get("rationale_filename", "agent_rationale.md")

    config_path.write_text(json.dumps(config, indent=2, sort_keys=False))
    rationale_path.write_text(rationale or "(no rationale provided)\n")

    return {
        "saved": True,
        "config_path": str(config_path),
        "rationale_path": str(rationale_path),
    }


def validate_config_spec() -> ToolSpec:
    return ToolSpec(
        name="validate_config",
        description=(
            "Validate a candidate full AortaCFD config against the "
            "pydantic schema. Returns valid=true or structured errors."
        ),
        parameters={
            "type": "object",
            "properties": {"config": {"type": "object"}},
            "required": ["config"],
        },
        handler=validate_config,
    )


def save_config_spec() -> ToolSpec:
    return ToolSpec(
        name="save_config",
        description=(
            "Write the final validated config plus a markdown rationale "
            "to disk. Call this exactly once when the config is ready."
        ),
        parameters={
            "type": "object",
            "properties": {
                "config": {"type": "object"},
                "rationale": {"type": "string"},
                "output_dir": {"type": "string"},
                "config_filename": {"type": "string"},
                "rationale_filename": {"type": "string"},
            },
            "required": ["config", "rationale", "output_dir"],
        },
        handler=save_config,
    )
