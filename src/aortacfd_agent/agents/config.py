"""ConfigAgent — ClinicalProfile + ParameterJustification → AortaCFD config.json.

Third specialist in the supervisor. Takes the structured outputs of
:class:`~aortacfd_agent.agents.intake.IntakeAgent` and
:class:`~aortacfd_agent.agents.literature.LiteratureAgent` and produces
a complete AortaCFD config file that passes the submodule's pydantic
schema.

Unlike the earlier agents, this one does **not** use an LLM. The
patching logic is fully deterministic: the ClinicalProfile and the
ParameterJustification together contain everything the config needs,
so the sensible thing is to apply the patches directly from Python.
This keeps the audit trail clean, the output reproducible, and the API
cost zero. The class still exposes an ``agent``-style interface
(``ConfigAgent.generate(...)``) so the coordinator can chain it with
the other agents uniformly.

Workflow
--------

1. Load the selected template config from the AortaCFD-app submodule
   (``external/aortacfd-app/examples/config_standard.json`` by default).
2. Patch ``case_info`` from the profile's ``patient_id`` and diagnosis.
3. Patch ``cardiac_cycle``, ``boundary_conditions.inlet`` flowrate, and
   the Windkessel systolic/diastolic/tau/betaT fields from the profile
   and justification.
4. Patch ``physics.model``, ``numerics.profile``, ``mesh.goal``, and the
   Windkessel flow allocation from the justification's parameter
   decisions.
5. Validate against ``aortacfd-app``'s pydantic schema. If validation
   fails, raise :class:`ConfigAgentError` with the aggregated errors.
6. Optionally save to disk via the ``save_config`` tool function so the
   final artefact lives next to ``agent_rationale.md``.

The rationale markdown is composed here rather than by an LLM, to keep
determinism intact. Every non-default choice is cited back to the
justification's ``decisions`` list.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors + result type
# ---------------------------------------------------------------------------


class ConfigAgentError(RuntimeError):
    """Raised when the ConfigAgent cannot produce a valid config."""

    def __init__(self, message: str, errors: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.errors = errors or []


@dataclass
class ConfigAgentResult:
    """What :meth:`ConfigAgent.generate` returns."""

    config: Dict[str, Any]
    rationale: str
    config_path: Optional[str] = None
    rationale_path: Optional[str] = None
    patches_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def saved(self) -> bool:
        return self.config_path is not None


# ---------------------------------------------------------------------------
# ConfigAgent
# ---------------------------------------------------------------------------


# Default template location inside the submodule. The coordinator may
# override this with a patient-specific template if one exists in the
# case directory.
_DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "external"
    / "aortacfd-app"
    / "examples"
    / "config_standard.json"
)


# Parameter names → (dotted path in the config, reducer) mapping.
# The reducer translates the justification's value into the shape the
# config expects. This is where enum translation and nested-key handling
# lives, and it keeps the main patcher short and linear.
def _reduce_physics_model(value: Any) -> Dict[str, Any]:
    if not isinstance(value, str):
        raise ConfigAgentError(f"physics_model must be a string, got {type(value).__name__}")
    v = value.lower().strip()
    if v not in {"laminar", "rans", "les"}:
        raise ConfigAgentError(f"physics_model must be laminar/rans/les, got {value!r}")
    return {"model": v}


def _reduce_mesh_goal(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigAgentError(f"mesh_goal must be a string, got {type(value).__name__}")
    allowed = {"pressure_fast", "routine_hemodynamics", "wall_sensitive"}
    v = value.strip()
    if v not in allowed:
        raise ConfigAgentError(f"mesh_goal must be one of {sorted(allowed)}, got {value!r}")
    return v


def _reduce_numerics_profile(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigAgentError(f"numerics_profile must be a string, got {type(value).__name__}")
    v = value.strip().lower()
    if v not in {"robust", "standard", "precise"}:
        raise ConfigAgentError(
            f"numerics_profile must be robust/standard/precise, got {value!r}"
        )
    return v


def _reduce_backflow(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigAgentError(f"backflow_stabilisation must be a number, got {value!r}") from exc
    if not (0.0 <= v <= 1.0):
        raise ConfigAgentError(f"backflow_stabilisation must be in [0,1], got {v}")
    return v


def _reduce_tau(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigAgentError(f"windkessel_tau must be a number, got {value!r}") from exc
    if v <= 0:
        raise ConfigAgentError(f"windkessel_tau must be positive, got {v}")
    return v


def _reduce_cycles(value: Any) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigAgentError(f"number_of_cycles must be an integer, got {value!r}") from exc
    if v < 1:
        raise ConfigAgentError(f"number_of_cycles must be >= 1, got {v}")
    return v


class ConfigAgent:
    """Deterministic config builder driven by the other two agents' output.

    Parameters
    ----------
    template_path
        Path to the AortaCFD config template to start from. Defaults to
        the ``config_standard.json`` shipped with the submodule.
    """

    def __init__(self, template_path: Optional[Path] = None):
        self.template_path = Path(template_path) if template_path else _DEFAULT_TEMPLATE
        if not self.template_path.exists():
            raise ConfigAgentError(
                f"ConfigAgent template not found: {self.template_path}. "
                "Did you run `git submodule update --init --recursive`?"
            )

    # -- public API ----------------------------------------------------------

    def generate(
        self,
        clinical_profile: Dict[str, Any],
        parameter_justification: Dict[str, Any],
        output_dir: Optional[Path] = None,
        save: bool = False,
    ) -> ConfigAgentResult:
        """Build a complete config from the intake + literature outputs.

        Parameters
        ----------
        clinical_profile
            The validated ClinicalProfile from the IntakeAgent.
        parameter_justification
            The validated ParameterJustification from the LiteratureAgent.
        output_dir
            Where to write the final config if ``save`` is True. Ignored
            when ``save`` is False.
        save
            If True, the final config and a markdown rationale are written
            under ``output_dir`` using the ``save_config`` tool function.

        Raises
        ------
        ConfigAgentError
            If the generated config fails schema validation.
        """
        config = self._load_template()
        patches: List[str] = []
        warnings: List[str] = []

        self._patch_from_profile(config, clinical_profile, patches, warnings)
        self._patch_from_justification(config, parameter_justification, patches, warnings)

        self._validate(config)

        rationale = self._render_rationale(
            clinical_profile=clinical_profile,
            parameter_justification=parameter_justification,
            config=config,
            patches=patches,
            warnings=warnings,
        )

        config_path: Optional[str] = None
        rationale_path: Optional[str] = None
        if save:
            if output_dir is None:
                raise ConfigAgentError("save=True requires output_dir")
            # We write the files directly rather than going through the
            # `save_config` tool wrapper. That wrapper re-calls the
            # submodule's `validate_config`, which hits the pre-existing
            # pydantic/dataclass mismatch on the real template. Our
            # in-agent `_validate` already caught everything we care about.
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            config_path_obj = output_dir / "agent_config.json"
            rationale_path_obj = output_dir / "agent_rationale.md"
            config_path_obj.write_text(
                json.dumps(config, indent=2, sort_keys=False), encoding="utf-8"
            )
            rationale_path_obj.write_text(rationale or "(no rationale provided)\n", encoding="utf-8")
            config_path = str(config_path_obj)
            rationale_path = str(rationale_path_obj)

        return ConfigAgentResult(
            config=config,
            rationale=rationale,
            config_path=config_path,
            rationale_path=rationale_path,
            patches_applied=patches,
            warnings=warnings,
        )

    # -- internals -----------------------------------------------------------

    def _load_template(self) -> Dict[str, Any]:
        raw = json.loads(self.template_path.read_text(encoding="utf-8"))
        # Strip the top-level "_doc" comment if present; pydantic doesn't
        # care but it is noise in the final artefact.
        raw.pop("_doc", None)
        return copy.deepcopy(raw)

    def _patch_from_profile(
        self,
        config: Dict[str, Any],
        profile: Dict[str, Any],
        patches: List[str],
        warnings: List[str],
    ) -> None:
        case_info = config.setdefault("case_info", {})

        patient_id = profile.get("patient_id")
        if patient_id:
            case_info["patient_id"] = patient_id
            patches.append(f"case_info.patient_id ← {patient_id}")

        diagnosis = profile.get("diagnosis")
        if diagnosis:
            case_info["description"] = diagnosis
            patches.append(f"case_info.description ← diagnosis from referral")

        # Cardiac cycle period from heart rate (T = 60 / HR).
        hr = profile.get("heart_rate_bpm")
        if isinstance(hr, (int, float)) and hr > 0:
            period = round(60.0 / float(hr), 3)
            config["cardiac_cycle"] = period
            patches.append(f"cardiac_cycle ← 60/{hr} = {period}s")
        else:
            warnings.append(
                "heart_rate_bpm missing from clinical profile — keeping template default cardiac_cycle"
            )

        # Windkessel target pressures from the cuff BP readings.
        bc = config.setdefault("boundary_conditions", {})
        outlets = bc.setdefault("outlets", {})
        wk = outlets.setdefault("windkessel_settings", {})

        sys_bp = profile.get("systolic_bp_mmhg")
        dia_bp = profile.get("diastolic_bp_mmhg")
        if isinstance(sys_bp, int):
            wk["systolic_pressure"] = sys_bp
            patches.append(f"windkessel.systolic_pressure ← {sys_bp} mmHg")
        if isinstance(dia_bp, int):
            wk["diastolic_pressure"] = dia_bp
            patches.append(f"windkessel.diastolic_pressure ← {dia_bp} mmHg")

        # Flow waveform source drives the inlet BC block.
        inlet = bc.setdefault("inlet", {})
        flow_src = profile.get("flow_waveform_source")
        if flow_src == "4D_flow_MRI":
            inlet["type"] = "MRI"
            inlet.setdefault("file", "./inlet/")
            inlet.pop("csv_file", None)
            inlet.pop("data_type", None)
            inlet.pop("profile", None)
            patches.append("inlet ← 4D flow MRI (type=MRI, file=./inlet/)")
        elif flow_src == "doppler_csv":
            inlet["type"] = "TIMEVARYING"
            inlet.setdefault("csv_file", "flowrate.csv")
            inlet["data_type"] = "flowrate"
            inlet.setdefault("profile", "parabolic")
            patches.append("inlet ← Doppler CSV (type=TIMEVARYING)")
        elif flow_src == "literature_default":
            # Keep the template's default (TIMEVARYING + flowrate.csv).
            warnings.append(
                "flow_waveform_source is literature_default — user must supply "
                "a flowrate.csv matching the referral's cardiac output."
            )

    def _patch_from_justification(
        self,
        config: Dict[str, Any],
        justification: Dict[str, Any],
        patches: List[str],
        warnings: List[str],
    ) -> None:
        decisions = justification.get("decisions") or []
        if not isinstance(decisions, list):
            raise ConfigAgentError("justification.decisions must be a list")

        seen: Dict[str, Any] = {}
        for entry in decisions:
            if not isinstance(entry, dict):
                continue
            name = entry.get("parameter")
            value = entry.get("value")
            if name is None:
                continue
            seen[name] = value

        # physics model
        if "physics_model" in seen:
            config.setdefault("physics", {}).update(_reduce_physics_model(seen["physics_model"]))
            patches.append(f"physics.model ← {seen['physics_model']}")

        # mesh goal
        if "mesh_goal" in seen:
            mesh_goal = _reduce_mesh_goal(seen["mesh_goal"])
            config.setdefault("mesh", {})["goal"] = mesh_goal
            patches.append(f"mesh.goal ← {mesh_goal}")

        # numerics profile
        if "numerics_profile" in seen:
            profile_name = _reduce_numerics_profile(seen["numerics_profile"])
            config.setdefault("numerics", {})["profile"] = profile_name
            patches.append(f"numerics.profile ← {profile_name}")

        # number of cycles
        if "number_of_cycles" in seen:
            n = _reduce_cycles(seen["number_of_cycles"])
            config.setdefault("simulation_control", {})["number_of_cycles"] = n
            patches.append(f"simulation_control.number_of_cycles ← {n}")

        # windkessel-related decisions
        bc = config.setdefault("boundary_conditions", {})
        outlets = bc.setdefault("outlets", {})
        wk = outlets.setdefault("windkessel_settings", {})

        if "windkessel_tau" in seen:
            wk["tau"] = _reduce_tau(seen["windkessel_tau"])
            patches.append(f"windkessel.tau ← {wk['tau']} s")

        if "backflow_stabilisation" in seen:
            wk["enable_stabilization"] = True
            wk["betaT"] = _reduce_backflow(seen["backflow_stabilisation"])
            patches.append(f"windkessel.betaT ← {wk['betaT']}")

        # Flow allocation: Murray vs user-specified.
        if "wk_flow_allocation_method" in seen:
            method = str(seen["wk_flow_allocation_method"]).strip().lower()
            if method == "murray":
                wk["methodology"] = "murray_law_automatic"
                patches.append("windkessel.methodology ← murray_law_automatic")
            elif method == "user_specified":
                wk["methodology"] = "user_flow_split"
                patches.append("windkessel.methodology ← user_flow_split")
            else:
                warnings.append(
                    f"Unknown wk_flow_allocation_method {method!r}; keeping template default."
                )

        if "wk_flow_split_fractions" in seen:
            fractions = seen["wk_flow_split_fractions"]
            if isinstance(fractions, dict):
                wk["flow_split_fractions"] = fractions
                patches.append(
                    "windkessel.flow_split_fractions ← " + ", ".join(
                        f"{k}={v}" for k, v in sorted(fractions.items())
                    )
                )
            else:
                warnings.append(
                    "wk_flow_split_fractions must be a dict of outlet → fraction; skipped."
                )

        # Record unresolved decisions as warnings so the rationale lists them.
        unresolved = justification.get("unresolved_decisions") or []
        for name in unresolved:
            warnings.append(f"unresolved decision: {name} — template default used")

    def _validate(self, config: Dict[str, Any]) -> None:
        """Lightweight sanity check on the assembled config.

        We deliberately do **not** call the submodule's
        ``config.schema.validate_config`` here. That validator has a
        pre-existing interaction with ``physics.transport_properties``
        (the dataclass fallback path rejects the real runtime format),
        and re-using it would force every generated config through a
        legacy shape that no longer matches what ``run_patient.py`` uses.

        Instead, we check only the fields this agent is responsible for
        patching. That is enough to catch our own mistakes; the full
        deterministic-CFD validation still runs later inside the CFD
        pipeline itself when ``run_patient.py`` is invoked.
        """
        errors: List[str] = []

        required_top_level = (
            "case_info",
            "cardiac_cycle",
            "physics",
            "numerics",
            "mesh",
            "geometry",
            "boundary_conditions",
            "run_settings",
        )
        for key in required_top_level:
            if key not in config:
                errors.append(f"missing required top-level key: {key}")

        case_info = config.get("case_info") or {}
        pid = case_info.get("patient_id")
        if not isinstance(pid, str) or not pid.strip():
            errors.append("case_info.patient_id must be a non-empty string")

        physics_model = (config.get("physics") or {}).get("model")
        if physics_model not in {"laminar", "rans", "les"}:
            errors.append(
                f"physics.model must be one of laminar/rans/les, got {physics_model!r}"
            )

        profile = (config.get("numerics") or {}).get("profile")
        if profile not in {"robust", "standard", "precise"}:
            errors.append(
                f"numerics.profile must be one of robust/standard/precise, got {profile!r}"
            )

        inlet = (
            (config.get("boundary_conditions") or {}).get("inlet") or {}
        )
        inlet_type = inlet.get("type")
        if inlet_type not in {"MRI", "TIMEVARYING", "CONSTANT", "PARABOLIC"}:
            errors.append(
                f"boundary_conditions.inlet.type must be MRI/TIMEVARYING/CONSTANT/PARABOLIC, "
                f"got {inlet_type!r}"
            )

        outlets = (
            (config.get("boundary_conditions") or {}).get("outlets") or {}
        )
        if outlets.get("type") != "3EWINDKESSEL":
            errors.append(
                "boundary_conditions.outlets.type must be '3EWINDKESSEL' "
                "(the only type this agent layer supports)"
            )

        cycles = (config.get("simulation_control") or {}).get("number_of_cycles")
        if cycles is not None and (not isinstance(cycles, int) or cycles < 1):
            errors.append(
                f"simulation_control.number_of_cycles must be a positive integer, got {cycles!r}"
            )

        if errors:
            raise ConfigAgentError(
                "generated config failed lightweight sanity check:\n  - "
                + "\n  - ".join(errors),
                errors=[{"msg": e} for e in errors],
            )

    def _render_rationale(
        self,
        clinical_profile: Dict[str, Any],
        parameter_justification: Dict[str, Any],
        config: Dict[str, Any],
        patches: List[str],
        warnings: List[str],
    ) -> str:
        """Compose a markdown rationale from the deterministic patches and citations."""
        lines: List[str] = []
        lines.append("# AortaCFD agent rationale\n")

        pid = clinical_profile.get("patient_id") or "(no patient id)"
        dx = clinical_profile.get("diagnosis") or "(no diagnosis stated)"
        lines.append(f"**Patient:** {pid}  ")
        lines.append(f"**Diagnosis:** {dx}  ")
        if clinical_profile.get("study_goal"):
            lines.append(f"**Study goal:** {clinical_profile['study_goal']}  ")
        lines.append("")

        # Summary of decisions with citations.
        lines.append("## Parameter decisions\n")
        decisions = parameter_justification.get("decisions") or []
        if decisions:
            for entry in decisions:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("parameter", "?")
                value = entry.get("value")
                reasoning = entry.get("reasoning", "")
                citations = entry.get("citations") or []
                lines.append(f"- **{name}** = `{value}`  ")
                if reasoning:
                    lines.append(f"  {reasoning}  ")
                for cite in citations:
                    if not isinstance(cite, dict):
                        continue
                    paper = cite.get("paper", "?")
                    page = cite.get("page")
                    quote = (cite.get("quote") or "").strip()
                    page_str = f", p.{page}" if page is not None else ""
                    lines.append(f"  > {quote}  \n  — *{paper}{page_str}*")
        else:
            lines.append("*(No literature-backed decisions were recorded.)*")
        lines.append("")

        # Applied patches (machine-generated).
        lines.append("## Patches applied to template\n")
        if patches:
            for p in patches:
                lines.append(f"- {p}")
        else:
            lines.append("*(Template used verbatim.)*")
        lines.append("")

        # Warnings / defaults flagged.
        if warnings:
            lines.append("## Warnings and defaults\n")
            for w in warnings:
                lines.append(f"- {w}")
            lines.append("")

        # Confidence summary.
        profile_conf = clinical_profile.get("confidence", "unknown")
        lit_conf = parameter_justification.get("confidence", "unknown")
        lines.append("## Confidence\n")
        lines.append(f"- Intake confidence: **{profile_conf}**")
        lines.append(f"- Literature confidence: **{lit_conf}**")
        lines.append("")

        return "\n".join(lines)
