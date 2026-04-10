"""Unit tests for :class:`aortacfd_agent.agents.intake.IntakeAgent`.

These tests exercise the extraction path end-to-end with a ``FakeBackend``
scripted to emit the structured profile the agent is meant to produce. No
real LLM calls happen; the test is really checking:

1. the schema is loaded and exposed to the tool layer correctly
2. the agent correctly unwraps the model's tool call and returns the profile
3. jsonschema validation catches malformed output when ``strict_validation``
   is on
4. the agent raises ``ValueError`` when the model fails to call the tool

The five sample referrals in ``tests/fixtures/sample_reports/`` each get
one test. The FakeBackend responses are deliberately hand-written so the
tests document the exact extraction behaviour we expect from a real
model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from aortacfd_agent.agents.intake import IntakeAgent, IntakeResult
from aortacfd_agent.backends.base import ToolCall
from aortacfd_agent.backends.fake import FakeBackend, ScriptedStep


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_reports"


def _load_referral(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def _scripted_backend_for_profile(profile: Dict[str, Any]) -> FakeBackend:
    """Build a FakeBackend that will emit one tool call with the given profile."""
    return FakeBackend(
        script=[
            ScriptedStep(
                text="",
                tool_calls=[
                    ToolCall(
                        id="intake-1",
                        name="emit_clinical_profile",
                        arguments=profile,
                    )
                ],
                stop_reason="tool_use",
            )
        ]
    )


# ---------------------------------------------------------------------------
# Happy-path extractions for each sample referral
# ---------------------------------------------------------------------------


def test_paediatric_coarctation_extraction():
    profile = {
        "patient_id": "BPM120",
        "age_years": 12,
        "sex": "male",
        "weight_kg": None,
        "height_cm": None,
        "bsa_m2": 1.32,
        "diagnosis": "aortic coarctation post-balloon angioplasty",
        "diagnosis_severity_hint": None,
        "heart_rate_bpm": 78,
        "systolic_bp_mmhg": 118,
        "diastolic_bp_mmhg": 72,
        "cardiac_output_l_min": 4.8,
        "imaging_modality": ["CT_angiography"],
        "flow_waveform_source": "literature_default",
        "study_goal": (
            "Wall shear stress around the coarctation and pressure drop "
            "from ascending to descending aorta, for surgical planning."
        ),
        "constraints": [
            "Must finish overnight on ≤32 cores",
            "Use 3-element Windkessel default for outlets",
        ],
        "missing_fields": ["weight_kg", "height_cm", "diagnosis_severity_hint"],
        "confidence": "high",
        "notes": (
            "Legs BP 104/65 mmHg noted separately — used as arm BP for "
            "the config's MAP target."
        ),
    }
    backend = _scripted_backend_for_profile(profile)
    agent = IntakeAgent(backend=backend)

    result = agent.extract(_load_referral("coarctation_paediatric.txt"))

    assert isinstance(result, IntakeResult)
    assert result.profile["patient_id"] == "BPM120"
    assert result.profile["age_years"] == 12
    assert result.profile["heart_rate_bpm"] == 78
    assert result.profile["cardiac_output_l_min"] == 4.8
    assert "CT_angiography" in result.profile["imaging_modality"]
    assert result.confidence == "high"


def test_healthy_adult_extraction_with_4d_mri():
    profile = {
        "patient_id": "VOL04",
        "age_years": 34,
        "sex": "female",
        "weight_kg": None,
        "height_cm": None,
        "bsa_m2": None,
        "diagnosis": "healthy volunteer, no cardiovascular history",
        "diagnosis_severity_hint": None,
        "heart_rate_bpm": 68,
        "systolic_bp_mmhg": 122,
        "diastolic_bp_mmhg": 76,
        "cardiac_output_l_min": None,
        "imaging_modality": ["4D_flow_MRI"],
        "flow_waveform_source": "4D_flow_MRI",
        "study_goal": (
            "Bulk hemodynamic reference against a wall-resolved LES from "
            "the literature — validation case."
        ),
        "constraints": [],
        "missing_fields": [
            "weight_kg",
            "height_cm",
            "bsa_m2",
            "cardiac_output_l_min",
            "diagnosis_severity_hint",
        ],
        "confidence": "medium",
        "notes": None,
    }
    backend = _scripted_backend_for_profile(profile)
    agent = IntakeAgent(backend=backend)

    result = agent.extract(_load_referral("healthy_adult.txt"))

    assert result.profile["flow_waveform_source"] == "4D_flow_MRI"
    assert "4D_flow_MRI" in result.profile["imaging_modality"]
    assert result.profile["cardiac_output_l_min"] is None
    assert "cardiac_output_l_min" in result.missing_fields
    assert result.confidence == "medium"


def test_marfan_dilated_root_with_prescribed_rcz():
    profile = {
        "patient_id": "0023",
        "age_years": 15,
        "sex": "male",
        "weight_kg": None,
        "height_cm": None,
        "bsa_m2": None,
        "diagnosis": "Marfan syndrome with dilated aortic root and mild aortic regurgitation",
        "diagnosis_severity_hint": "aortic root 45 mm at sinuses of Valsalva; vena contracta 0.4 cm",
        "heart_rate_bpm": 72,
        "systolic_bp_mmhg": 118,
        "diastolic_bp_mmhg": 68,
        "cardiac_output_l_min": 5.2,
        "imaging_modality": ["CT_angiography", "echocardiography", "Doppler"],
        "flow_waveform_source": "doppler_csv",
        "study_goal": "Cross-solver comparison against SimVascular with identical boundary conditions.",
        "constraints": [
            "Use prescribed RCZ values from the VMR case metadata (override Murray's law)",
            "Standard numerical profile, laminar physics, no RANS",
        ],
        "missing_fields": ["weight_kg", "height_cm", "bsa_m2"],
        "confidence": "high",
        "notes": "Prescribed RCR is a hard constraint — ConfigAgent must use the VMR values verbatim.",
    }
    backend = _scripted_backend_for_profile(profile)
    agent = IntakeAgent(backend=backend)

    result = agent.extract(_load_referral("marfan_dilated_root.txt"))

    assert "Marfan" in result.profile["diagnosis"]
    assert "45 mm" in result.profile["diagnosis_severity_hint"]
    # Prescribed-RCZ hard constraint must land in the constraints list verbatim-ish
    assert any("prescribed RCZ" in c or "Murray" in c for c in result.profile["constraints"])


def test_post_surgical_repair_extraction():
    profile = {
        "patient_id": "post_repair",
        "age_years": 23,
        "sex": "female",
        "weight_kg": None,
        "height_cm": None,
        "bsa_m2": 1.58,
        "diagnosis": "post end-to-end coarctation repair (5 years post-op)",
        "diagnosis_severity_hint": "minimal upper-lower limb BP gradient — good repair",
        "heart_rate_bpm": 66,
        "systolic_bp_mmhg": 115,
        "diastolic_bp_mmhg": 70,
        "cardiac_output_l_min": 4.4,
        "imaging_modality": ["CT_angiography"],
        "flow_waveform_source": "literature_default",
        "study_goal": "Baseline hemodynamics for longitudinal comparison; pressure drop primary endpoint.",
        "constraints": ["WSS reported comparatively, not as absolute magnitudes"],
        "missing_fields": ["weight_kg", "height_cm"],
        "confidence": "high",
        "notes": None,
    }
    backend = _scripted_backend_for_profile(profile)
    agent = IntakeAgent(backend=backend)

    result = agent.extract(_load_referral("post_surgical_repair.txt"))

    assert result.profile["bsa_m2"] == 1.58
    assert result.profile["cardiac_output_l_min"] == 4.4
    assert result.confidence == "high"


def test_infant_complex_with_user_flow_split():
    profile = {
        "patient_id": "PAT003",
        "age_years": 1,
        "sex": "male",
        "weight_kg": 9.2,
        "height_cm": None,
        "bsa_m2": None,
        "diagnosis": "complex paediatric aortic arch anatomy, prior balloon dilatation",
        "diagnosis_severity_hint": "peak instantaneous gradient 48 mmHg in descending aorta",
        "heart_rate_bpm": 120,
        "systolic_bp_mmhg": 95,
        "diastolic_bp_mmhg": 55,
        "cardiac_output_l_min": 1.1,
        "imaging_modality": ["CT_angiography", "echocardiography", "Doppler"],
        "flow_waveform_source": "doppler_csv",
        "study_goal": "Pre-operative planning for a complex paediatric aortic arch repair.",
        "constraints": [
            "Use user-specified flow split: 70% descending aorta, 10% each arch branch",
            "Robust numerical profile",
            "beta_T = 0.5 for backflow stability",
        ],
        "missing_fields": ["height_cm", "bsa_m2"],
        "confidence": "high",
        "notes": (
            "Referral explicitly rejects Murray's law for this case due to "
            "extreme small-vessel diameters. BP estimate is cuff-approximate "
            "(agitated child)."
        ),
    }
    backend = _scripted_backend_for_profile(profile)
    agent = IntakeAgent(backend=backend)

    result = agent.extract(_load_referral("infant_complex.txt"))

    assert result.profile["age_years"] == 1
    assert result.profile["weight_kg"] == 9.2
    # Must capture the user-specified flow split directive as a constraint
    assert any("flow split" in c.lower() for c in result.profile["constraints"])
    assert any("0.5" in c or "β_T" in c or "beta_T" in c for c in result.profile["constraints"])
    assert result.confidence == "high"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_missing_tool_call_raises():
    """If the model replies with text instead of a tool call, we must fail fast."""
    backend = FakeBackend(
        script=[ScriptedStep(text="Sorry, I don't understand.", tool_calls=[])]
    )
    agent = IntakeAgent(backend=backend)
    with pytest.raises(ValueError, match="no tool call"):
        agent.extract("some referral")


def test_schema_validation_rejects_malformed_output():
    """Missing required fields must be caught by jsonschema."""
    malformed = {
        "patient_id": "X",
        "diagnosis": "coarctation",
        # Intentionally missing 'confidence', which is required.
    }
    backend = _scripted_backend_for_profile(malformed)
    agent = IntakeAgent(backend=backend)
    with pytest.raises(ValueError, match="schema validation"):
        agent.extract("some referral")


def test_schema_validation_rejects_out_of_range_values():
    """Fields with numeric bounds should reject values outside those bounds."""
    bad = {
        "patient_id": "X",
        "diagnosis": "test",
        "heart_rate_bpm": 500,  # out of 30..220 range
        "confidence": "high",
    }
    backend = _scripted_backend_for_profile(bad)
    agent = IntakeAgent(backend=backend)
    with pytest.raises(ValueError, match="schema validation"):
        agent.extract("some referral")


# ---------------------------------------------------------------------------
# Schema / wiring checks
# ---------------------------------------------------------------------------


def test_intake_agent_tool_spec_is_object_schema():
    """The tool schema exposed to the model must be a valid JSON Schema object."""
    agent = IntakeAgent(backend=FakeBackend(script=[ScriptedStep(text="")]))
    spec = agent._tool_spec  # internal, but fine to assert on in tests
    assert spec.name == "emit_clinical_profile"
    assert spec.parameters["type"] == "object"
    assert "properties" in spec.parameters
    assert "diagnosis" in spec.parameters["properties"]
    assert "confidence" in spec.parameters["properties"]
    # JSON Schema metadata keys must NOT leak into tool parameters
    assert "$schema" not in spec.parameters
    assert "$id" not in spec.parameters


def test_intake_agent_system_prompt_loaded_from_package():
    """The system prompt is loaded from prompts/intake.md on construction."""
    agent = IntakeAgent(backend=FakeBackend(script=[ScriptedStep(text="")]))
    assert "Intake Agent" in agent.system_prompt
    assert "emit_clinical_profile" in agent.system_prompt
