#!/usr/bin/env python3
"""One-command end-to-end demo of the AortaCFD agent pipeline.

This script runs the full five-agent supervisor on a synthetic clinical
referral for the BPM120 paediatric coarctation case that lives in the
``external/aortacfd-app`` submodule. It uses the offline FakeBackend and
a small hand-curated FakeCorpusStore so anyone who has cloned the repo
can run::

    python examples/end_to_end_demo.py

and see the pipeline produce:

* an extracted ``ClinicalProfile``
* a literature-backed ``ParameterJustification``
* a validated ``agent_config.json`` ready for the CFD pipeline
* a markdown ``agent_rationale.md`` with citations for every decision
* an ``agent_trace.jsonl`` audit trail

No LLM credentials, no network, no PDFs, no OpenFOAM — the whole thing
runs in about half a second and writes its output under
``examples/output/demo_BPM120/``.

For a *real* run, replace the FakeBackend with an Anthropic or
Ollama-backed provider and point the coordinator at a ChromaCorpusStore
built from real open-access PDFs. See ``README.md`` for instructions.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from textwrap import dedent

# Make 'aortacfd_agent' importable when the script is run from the repo
# root without installing the package. pip install -e '.' also works.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from aortacfd_agent.backends.base import ToolCall  # noqa: E402
from aortacfd_agent.backends.fake import FakeBackend, ScriptedStep  # noqa: E402
from aortacfd_agent.coordinator import Coordinator  # noqa: E402
from aortacfd_agent.corpus.store import Chunk, FakeCorpusStore  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic referral — same BPM120 scenario as tests/fixtures/sample_reports/
# ---------------------------------------------------------------------------


REFERRAL = dedent(
    """\
    Patient: BPM120

    18-month-old male, neonatal diagnosis of aortic coarctation repaired by
    balloon angioplasty in early infancy. Referred for hemodynamic assessment
    of possible residual/recurrent gradient before elective re-intervention.

    Vitals:
      Heart rate 120 bpm, sinus rhythm
      Right arm BP 100/55 mmHg, leg BP 82/50 mmHg (arm-leg gradient present)
      Echo cardiac output 1.8 L/min
      Weight 11 kg, height 82 cm, BSA 0.50 m²

    Imaging:
      Gated CT angiography segmented into inlet (ascending aorta),
      three supra-aortic outlets (BCA, LCC, LSA), descending aorta outlet,
      and a wall patch. STLs in mm.

    Clinical question:
      Assess wall shear stress around the coarctation and pressure drop from
      ascending to descending aorta, for surgical planning. Must finish
      overnight on ≤32 cores. Use the 3-element Windkessel default; for this
      pathological geometry the literature suggests a user-specified flow split
      rather than Murray's law.
    """
)


# ---------------------------------------------------------------------------
# Minimal literature corpus — same 4 chunks the coordinator tests use
# ---------------------------------------------------------------------------


def build_demo_corpus() -> FakeCorpusStore:
    return FakeCorpusStore(
        chunks=[
            Chunk(
                text=(
                    "In coarctation cases Murray's law systematically misallocates flow "
                    "because the stenosis dominates the local pressure distribution. "
                    "Published paediatric coarctation studies use a user-specified "
                    "flow split with roughly seventy percent of cardiac output to the "
                    "descending aorta."
                ),
                paper="Wang2025",
                page=4,
            ),
            Chunk(
                text=(
                    "Diastolic backflow at Windkessel outlets can trigger instabilities. "
                    "A directional stabilisation with beta_T equal to zero point three "
                    "damps tangential velocity during backflow while preserving normal "
                    "pressure response, eliminating divergence with less than one "
                    "percent hemodynamic bias."
                ),
                paper="Esmaily2011",
                page=7,
            ),
            Chunk(
                text=(
                    "For studies where WSS, OSI and near-wall indices are primary "
                    "endpoints we recommend mesh refinement around the wall region "
                    "and explicit reporting of the numerical profile used."
                ),
                paper="ValenSendstad2018",
                page=5,
            ),
            Chunk(
                text=(
                    "A default tau of one point five seconds is appropriate when "
                    "patient-specific calibration is unavailable."
                ),
                paper="Stergiopulos1999",
                page=2,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Scripted FakeBackend — responses for intake and literature stages only
# (config and execution stages do not use the LLM)
# ---------------------------------------------------------------------------


def build_demo_backend() -> FakeBackend:
    profile = {
        "patient_id": "BPM120",
        "age_years": 1.5,  # 18 months
        "sex": "male",
        "weight_kg": 11.0,
        "height_cm": 82,
        "bsa_m2": 0.50,
        "diagnosis": "aortic coarctation post-balloon angioplasty",
        "diagnosis_severity_hint": None,
        "heart_rate_bpm": 120,
        "systolic_bp_mmhg": 100,
        "diastolic_bp_mmhg": 55,
        "cardiac_output_l_min": 1.8,
        "imaging_modality": ["CT_angiography"],
        "flow_waveform_source": "doppler_csv",
        "study_goal": (
            "WSS around the coarctation and pressure drop from ascending to "
            "descending aorta, for surgical planning."
        ),
        "constraints": [
            "Must finish overnight on ≤32 cores",
            "Use 3-element Windkessel with user-specified flow split",
        ],
        "missing_fields": ["weight_kg", "height_cm", "diagnosis_severity_hint"],
        "confidence": "high",
        "notes": None,
    }

    justification = {
        "decisions": [
            {
                "parameter": "physics_model",
                "value": "laminar",
                "reasoning": (
                    "Re is near 2000 at peak systole, at the edge of the laminar "
                    "regime. The referral explicitly asks for pressure drop and "
                    "WSS, and the submodule default profile is laminar."
                ),
                "citations": [
                    {
                        "paper": "ValenSendstad2018",
                        "page": 5,
                        "quote": (
                            "For studies where WSS, OSI and near-wall indices are "
                            "primary endpoints we recommend mesh refinement"
                        ),
                    }
                ],
            },
            {
                "parameter": "mesh_goal",
                "value": "wall_sensitive",
                "reasoning": "WSS around the coarctation is a primary endpoint.",
                "citations": [
                    {
                        "paper": "ValenSendstad2018",
                        "page": 5,
                        "quote": "mesh refinement around the wall region",
                    }
                ],
            },
            {
                "parameter": "wk_flow_allocation_method",
                "value": "user_specified",
                "reasoning": (
                    "Murray's law is invalid for coarctation because the stenosis "
                    "dominates the local pressure distribution."
                ),
                "citations": [
                    {
                        "paper": "Wang2025",
                        "page": 4,
                        "quote": (
                            "Murray's law systematically misallocates flow because "
                            "the stenosis dominates the local pressure distribution"
                        ),
                    }
                ],
                "alternative_considered": "murray (rejected: invalid for coarctation)",
            },
            {
                "parameter": "wk_flow_split_fractions",
                "value": {
                    "descending": 0.70,
                    "brachiocephalic": 0.10,
                    "lcca": 0.10,
                    "lsa": 0.10,
                },
                "reasoning": "Paediatric coarctation convention of ~70% to descending aorta.",
                "citations": [
                    {
                        "paper": "Wang2025",
                        "page": 4,
                        "quote": (
                            "roughly seventy percent of cardiac output to the "
                            "descending aorta"
                        ),
                    }
                ],
            },
            {
                "parameter": "windkessel_tau",
                "value": 1.5,
                "reasoning": "Default systemic tau without patient-specific calibration.",
                "citations": [
                    {
                        "paper": "Stergiopulos1999",
                        "page": 2,
                        "quote": "A default tau of one point five seconds is appropriate",
                    }
                ],
            },
            {
                "parameter": "backflow_stabilisation",
                "value": 0.3,
                "reasoning": "Published default eliminates divergence with <1% bias.",
                "citations": [
                    {
                        "paper": "Esmaily2011",
                        "page": 7,
                        "quote": (
                            "beta_T equal to zero point three damps tangential "
                            "velocity during backflow"
                        ),
                    }
                ],
            },
            {
                "parameter": "numerics_profile",
                "value": "standard",
                "reasoning": "Default second-order profile for production runs.",
                "citations": [],
            },
            {
                "parameter": "number_of_cycles",
                "value": 3,
                "reasoning": "Template default; MAP initialisation reaches periodic state in 3 cycles.",
                "citations": [],
            },
        ],
        "search_queries_used": [
            "coarctation Murray flow split",
            "backflow stabilisation beta_T diastolic divergence",
            "WSS mesh refinement near wall",
            "Windkessel tau default",
        ],
        "unresolved_decisions": [],
        "confidence": "high",
        "notes": None,
    }

    return FakeBackend(
        script=[
            # IntakeAgent — single emit call
            ScriptedStep(
                text="",
                tool_calls=[
                    ToolCall(id="i1", name="emit_clinical_profile", arguments=profile)
                ],
            ),
            # LiteratureAgent — two searches then emit
            ScriptedStep(
                text="",
                tool_calls=[
                    ToolCall(
                        id="l1",
                        name="search_corpus",
                        arguments={"query": "coarctation Murray flow split", "top_k": 3},
                    )
                ],
            ),
            ScriptedStep(
                text="",
                tool_calls=[
                    ToolCall(
                        id="l2",
                        name="search_corpus",
                        arguments={
                            "query": "backflow stabilisation beta_T",
                            "top_k": 3,
                        },
                    )
                ],
            ),
            ScriptedStep(
                text="",
                tool_calls=[
                    ToolCall(
                        id="l3",
                        name="emit_parameter_justification",
                        arguments=justification,
                    )
                ],
            ),
            ScriptedStep(text="literature ok", stop_reason="end_turn"),
        ]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    output_dir = _REPO / "examples" / "output" / "demo_BPM120"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    submodule_case = _REPO / "external" / "aortacfd-app" / "cases_input" / "BPM120"
    if not submodule_case.exists():
        print(
            f"error: submodule case not found at {submodule_case}\n"
            "run `git submodule update --init --recursive` and retry."
        )
        return 2

    print("=" * 72)
    print("AortaCFD agent — end-to-end demo (offline, no LLM, no CFD solve)")
    print("=" * 72)
    print(f"Submodule case: {submodule_case}")
    print(f"Output:         {output_dir}")
    print()

    coord = Coordinator(
        intake_backend=build_demo_backend(),
        corpus=build_demo_corpus(),
    )

    result = coord.run(
        clinical_text=REFERRAL,
        case_dir=submodule_case,
        output_dir=output_dir,
        skip_execution=True,  # no CFD solve in the demo
    )

    print(result.brief())
    print()

    if not result.success:
        print(f"FAILED: {result.error}")
        return 1

    # Show the rationale preview
    rationale_path = Path(result.config.rationale_path)
    print(f"--- {rationale_path.name} (first 30 lines) ---")
    lines = rationale_path.read_text(encoding="utf-8").splitlines()
    for line in lines[:30]:
        print(line)
    print()

    # Show a snippet of the generated config
    config_path = Path(result.config.config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"--- {config_path.name} (key patched fields) ---")
    print(f"  case_info.patient_id      = {config['case_info']['patient_id']}")
    print(f"  cardiac_cycle             = {config['cardiac_cycle']} s")
    print(f"  physics.model             = {config['physics']['model']}")
    print(f"  numerics.profile          = {config['numerics']['profile']}")
    print(f"  mesh.goal                 = {config['mesh'].get('goal')}")
    wk = config["boundary_conditions"]["outlets"]["windkessel_settings"]
    print(f"  windkessel.methodology    = {wk.get('methodology')}")
    print(f"  windkessel.tau            = {wk.get('tau')}")
    print(f"  windkessel.betaT          = {wk.get('betaT')}")
    if "flow_split_fractions" in wk:
        print(f"  windkessel.flow_split     = {wk['flow_split_fractions']}")
    print()

    trace_path = output_dir / "agent_trace.jsonl"
    print(f"Audit trace: {trace_path} ({trace_path.stat().st_size} bytes)")
    print()
    print("Next step:")
    print(
        "  To actually run the CFD pipeline on this config, use the "
        "submodule's run_patient.py:"
    )
    print(
        "      cd external/aortacfd-app && python run_patient.py BPM120 "
        f"--config {config_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
