"""Unit tests for :class:`aortacfd_agent.agents.config.ConfigAgent`.

Covers:

* Template loading and deep-copy safety.
* Patching from a clinical profile (patient id, cardiac cycle, BP, inlet BC).
* Patching from a parameter justification (physics, mesh goal, numerics,
  windkessel tau, backflow, flow allocation).
* Rationale markdown composition.
* Save-to-disk path and validation failure paths.
* Reducer helpers that reject bad values.

All tests run offline with no LLM backend — ConfigAgent is deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from aortacfd_agent.agents.config import (
    ConfigAgent,
    ConfigAgentError,
    ConfigAgentResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bpm120_profile() -> Dict[str, Any]:
    return {
        "patient_id": "BPM120",
        "age_years": 12,
        "sex": "male",
        "diagnosis": "aortic coarctation post-balloon angioplasty",
        "diagnosis_severity_hint": None,
        "heart_rate_bpm": 78,
        "systolic_bp_mmhg": 118,
        "diastolic_bp_mmhg": 72,
        "cardiac_output_l_min": 4.8,
        "imaging_modality": ["CT_angiography"],
        "flow_waveform_source": "doppler_csv",
        "study_goal": "WSS around coarctation and pressure drop.",
        "constraints": ["≤32 cores"],
        "missing_fields": [],
        "confidence": "high",
        "notes": None,
    }


def _vol04_profile() -> Dict[str, Any]:
    return {
        "patient_id": "VOL04",
        "age_years": 34,
        "sex": "female",
        "diagnosis": "healthy volunteer",
        "heart_rate_bpm": 68,
        "systolic_bp_mmhg": 122,
        "diastolic_bp_mmhg": 76,
        "cardiac_output_l_min": None,
        "imaging_modality": ["4D_flow_MRI"],
        "flow_waveform_source": "4D_flow_MRI",
        "study_goal": "Bulk hemodynamics validation.",
        "constraints": [],
        "missing_fields": ["cardiac_output_l_min"],
        "confidence": "medium",
    }


def _valid_justification() -> Dict[str, Any]:
    return {
        "decisions": [
            {
                "parameter": "physics_model",
                "value": "rans",
                "reasoning": "Re > 2000 at ascending aorta.",
                "citations": [
                    {"paper": "Wang2025", "page": 3, "quote": "RANS is preferred"}
                ],
            },
            {
                "parameter": "mesh_goal",
                "value": "wall_sensitive",
                "reasoning": "WSS primary endpoint.",
                "citations": [
                    {"paper": "ValenSendstad2018", "page": 5, "quote": "WSS refinement"}
                ],
            },
            {
                "parameter": "wk_flow_allocation_method",
                "value": "user_specified",
                "reasoning": "Murray invalid for coarctation.",
                "citations": [
                    {"paper": "Wang2025", "page": 4, "quote": "Murray misallocates"}
                ],
            },
            {
                "parameter": "wk_flow_split_fractions",
                "value": {"descending": 0.70, "bca": 0.10, "lcca": 0.10, "lsa": 0.10},
                "reasoning": "Paediatric coarctation convention.",
                "citations": [
                    {"paper": "Wang2025", "page": 4, "quote": "seventy percent to descending"}
                ],
            },
            {
                "parameter": "windkessel_tau",
                "value": 1.5,
                "reasoning": "Default systemic tau.",
                "citations": [
                    {"paper": "Stergiopulos1999", "page": 2, "quote": "one point five seconds"}
                ],
            },
            {
                "parameter": "backflow_stabilisation",
                "value": 0.3,
                "reasoning": "Published default.",
                "citations": [
                    {"paper": "Esmaily2011", "page": 7, "quote": "beta_T zero point three"}
                ],
            },
            {
                "parameter": "numerics_profile",
                "value": "standard",
                "reasoning": "Default 2nd-order for production.",
                "citations": [],
            },
            {
                "parameter": "number_of_cycles",
                "value": 3,
                "reasoning": "Periodicity within 3 cycles with MAP init.",
                "citations": [
                    {"paper": "Pfaller2021", "page": 8, "quote": "three cardiac cycles"}
                ],
            },
        ],
        "search_queries_used": [],
        "unresolved_decisions": [],
        "confidence": "high",
    }


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


class TestConfigAgentInit:
    def test_default_template_is_from_submodule(self):
        agent = ConfigAgent()
        assert agent.template_path.exists()
        assert "aortacfd-app" in str(agent.template_path)
        assert agent.template_path.name == "config_standard.json"

    def test_missing_template_raises(self, tmp_path):
        with pytest.raises(ConfigAgentError, match="template not found"):
            ConfigAgent(template_path=tmp_path / "nope.json")

    def test_deep_copy_template_is_independent(self):
        agent = ConfigAgent()
        a = agent._load_template()
        b = agent._load_template()
        a["case_info"]["patient_id"] = "MUTATED"
        assert b["case_info"]["patient_id"] != "MUTATED"


# ---------------------------------------------------------------------------
# Happy path: full patching from both inputs
# ---------------------------------------------------------------------------


class TestConfigAgentGenerate:
    def test_bpm120_full_patching(self):
        agent = ConfigAgent()
        result = agent.generate(_bpm120_profile(), _valid_justification())

        assert isinstance(result, ConfigAgentResult)
        cfg = result.config

        # Case info
        assert cfg["case_info"]["patient_id"] == "BPM120"
        assert "coarctation" in cfg["case_info"]["description"]

        # Cardiac cycle from HR=78 → 60/78 ≈ 0.769
        assert abs(cfg["cardiac_cycle"] - 0.769) < 0.01

        # Physics model patched
        assert cfg["physics"]["model"] == "rans"

        # Mesh goal patched
        assert cfg["mesh"]["goal"] == "wall_sensitive"

        # Numerics profile
        assert cfg["numerics"]["profile"] == "standard"

        # Number of cycles
        assert cfg["simulation_control"]["number_of_cycles"] == 3

        # Windkessel settings
        wk = cfg["boundary_conditions"]["outlets"]["windkessel_settings"]
        assert wk["systolic_pressure"] == 118
        assert wk["diastolic_pressure"] == 72
        assert wk["tau"] == 1.5
        assert wk["betaT"] == 0.3
        assert wk["enable_stabilization"] is True
        assert wk["methodology"] == "user_flow_split"
        assert wk["flow_split_fractions"]["descending"] == 0.70

        # Inlet BC type from flow_waveform_source = doppler_csv
        inlet = cfg["boundary_conditions"]["inlet"]
        assert inlet["type"] == "TIMEVARYING"

        # Patches list populated
        assert len(result.patches_applied) > 5
        assert any("patient_id ← BPM120" in p for p in result.patches_applied)
        assert any("physics.model ← rans" in p for p in result.patches_applied)

        # No warnings (every field was set)
        assert result.warnings == []

        # Rationale contains the decisions
        assert "BPM120" in result.rationale
        assert "physics_model" in result.rationale
        assert "Wang2025" in result.rationale

    def test_vol04_uses_mri_inlet(self):
        agent = ConfigAgent()
        result = agent.generate(_vol04_profile(), _valid_justification())
        inlet = result.config["boundary_conditions"]["inlet"]
        assert inlet["type"] == "MRI"
        assert "file" in inlet
        assert "csv_file" not in inlet

    def test_missing_hr_produces_warning_not_error(self):
        profile = _bpm120_profile()
        profile["heart_rate_bpm"] = None
        agent = ConfigAgent()
        result = agent.generate(profile, _valid_justification())
        assert any("heart_rate_bpm missing" in w for w in result.warnings)
        # Template default cardiac_cycle should still be present
        assert "cardiac_cycle" in result.config

    def test_unresolved_decisions_become_warnings(self):
        justification = _valid_justification()
        justification["unresolved_decisions"] = ["windkessel_z_fraction"]
        agent = ConfigAgent()
        result = agent.generate(_bpm120_profile(), justification)
        assert any("windkessel_z_fraction" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Save-to-disk path
# ---------------------------------------------------------------------------


class TestConfigAgentSave:
    def test_save_writes_config_and_rationale(self, tmp_path):
        agent = ConfigAgent()
        out = tmp_path / "agent_out"
        result = agent.generate(
            _bpm120_profile(),
            _valid_justification(),
            output_dir=out,
            save=True,
        )
        assert result.saved
        cfg_path = Path(result.config_path)
        rat_path = Path(result.rationale_path)
        assert cfg_path.exists()
        assert rat_path.exists()
        loaded = json.loads(cfg_path.read_text())
        assert loaded["case_info"]["patient_id"] == "BPM120"
        rationale = rat_path.read_text()
        assert "AortaCFD agent rationale" in rationale

    def test_save_without_output_dir_raises(self):
        agent = ConfigAgent()
        with pytest.raises(ConfigAgentError, match="output_dir"):
            agent.generate(
                _bpm120_profile(),
                _valid_justification(),
                save=True,
            )


# ---------------------------------------------------------------------------
# Reducer validation — bad values must raise, not silently pass through
# ---------------------------------------------------------------------------


class TestReducerValidation:
    def test_bad_physics_model_raises(self):
        j = _valid_justification()
        j["decisions"][0]["value"] = "turbulent_magic"
        agent = ConfigAgent()
        with pytest.raises(ConfigAgentError, match="physics_model"):
            agent.generate(_bpm120_profile(), j)

    def test_bad_mesh_goal_raises(self):
        j = _valid_justification()
        j["decisions"][1]["value"] = "ultra_fast"
        agent = ConfigAgent()
        with pytest.raises(ConfigAgentError, match="mesh_goal"):
            agent.generate(_bpm120_profile(), j)

    def test_negative_tau_raises(self):
        j = _valid_justification()
        j["decisions"][4]["value"] = -1.0
        agent = ConfigAgent()
        with pytest.raises(ConfigAgentError, match="windkessel_tau"):
            agent.generate(_bpm120_profile(), j)

    def test_out_of_range_betaT_raises(self):
        j = _valid_justification()
        j["decisions"][5]["value"] = 1.5
        agent = ConfigAgent()
        with pytest.raises(ConfigAgentError, match="backflow_stabilisation"):
            agent.generate(_bpm120_profile(), j)

    def test_nonpositive_cycles_raises(self):
        j = _valid_justification()
        j["decisions"][7]["value"] = 0
        agent = ConfigAgent()
        with pytest.raises(ConfigAgentError, match="number_of_cycles"):
            agent.generate(_bpm120_profile(), j)
