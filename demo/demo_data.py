"""Load pre-computed demo data or run the live pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class DemoData:
    referral_text: str = ""
    clinical_profile: dict = field(default_factory=dict)
    parameter_justification: dict = field(default_factory=dict)
    agent_config: dict = field(default_factory=dict)
    base_config: dict = field(default_factory=dict)
    rationale_md: str = ""
    trace_records: list[dict] = field(default_factory=list)
    hemodynamics_report: str = ""
    flow_distribution_png: Optional[bytes] = None
    stl_path: Optional[Path] = None
    case_stls_dir: Optional[Path] = None  # dir with all STLs + CSV for case generation
    case_id: str = "BPM120"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _submodule_root() -> Path:
    return _repo_root() / "external" / "aortacfd-app"


def _sibling_app_root() -> Optional[Path]:
    candidate = _repo_root().parent / "AortaCFD-app"
    return candidate if candidate.exists() else None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""


def _read_bytes(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes()
    except (FileNotFoundError, OSError):
        return None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Pre-computed demo data (matches examples/end_to_end_demo.py output)
# ---------------------------------------------------------------------------

DEMO_PROFILE = {
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
    "flow_waveform_source": "doppler_csv",
    "study_goal": (
        "WSS around the coarctation and pressure drop from ascending to "
        "descending aorta, for surgical planning."
    ),
    "constraints": [
        "Must finish overnight on \u226432 cores",
        "Use 3-element Windkessel with user-specified flow split",
    ],
    "missing_fields": ["weight_kg", "height_cm", "diagnosis_severity_hint"],
    "confidence": "high",
    "notes": None,
}

DEMO_JUSTIFICATION = {
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
                    "paper": "valensendstad2018",
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
                    "paper": "valensendstad2018",
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
                    "paper": "wang2025hr",
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
            "value": {"descending": 0.70, "BCA": 0.10, "LCCA": 0.10, "LSA": 0.10},
            "reasoning": "Paediatric coarctation convention of ~70% to descending aorta.",
            "citations": [
                {
                    "paper": "wang2025hr",
                    "page": 4,
                    "quote": "roughly seventy percent of cardiac output to the descending aorta",
                }
            ],
        },
        {
            "parameter": "windkessel_tau",
            "value": 1.5,
            "reasoning": "Default systemic tau without patient-specific calibration.",
            "citations": [
                {
                    "paper": "stergiopulos1999",
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
                    "paper": "esmaily2011",
                    "page": 7,
                    "quote": "beta_T equal to zero point three damps tangential velocity during backflow",
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


def _assets_dir() -> Path:
    """Bundled demo assets (work on Streamlit Cloud where submodule/sibling repos don't exist)."""
    return Path(__file__).resolve().parent / "assets"


def load_demo_data(referral_text: str = "") -> DemoData:
    """Load all pre-computed demo data from disk."""
    root = _repo_root()
    demo_dir = root / "examples" / "output" / "demo_BPM120"
    sub = _submodule_root()
    sibling = _sibling_app_root()
    assets = _assets_dir()

    # Agent outputs — bundled assets first, then examples/output
    agent_config = _read_json(assets / "agent_config.json")
    if not agent_config:
        agent_config = _read_json(demo_dir / "agent_config.json")

    rationale_md = _read_text(assets / "agent_rationale.md")
    if not rationale_md:
        rationale_md = _read_text(demo_dir / "agent_rationale.md")

    # Trace — bundled assets first
    trace_records = []
    trace_path = assets / "agent_trace.jsonl"
    if not trace_path.exists():
        trace_path = demo_dir / "agent_trace.jsonl"
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                trace_records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # Base config (for diff) — bundled assets first, then submodule
    base_config = _read_json(assets / "base_config.json")
    if not base_config:
        base_config = _read_json(sub / "cases_input" / "BPM120" / "config.json")

    # Hemodynamics report — bundled assets first, then submodule/sibling
    hemo_report = _read_text(assets / "BPM120_hemodynamics_report.txt")
    if not hemo_report:
        for base in [sub, sibling]:
            if base:
                hemo_report = _read_text(base / "docs" / "tutorial" / "precomputed_results" / "BPM120_hemodynamics_report.txt")
                if hemo_report:
                    break

    # Flow distribution plot — bundled assets first, then sibling output dirs
    flow_png = _read_bytes(assets / "flow_distribution.png")
    if not flow_png and sibling:
        for candidate in [
            sibling / "output" / "BPM120" / "tutorial_300k" / "reports" / "flow_distribution.png",
            sibling / "output" / "BPM120" / "L3_span20_layers" / "reports" / "flow_distribution.png",
        ]:
            flow_png = _read_bytes(candidate)
            if flow_png:
                break

    # STL geometry — bundled assets first, then submodule
    stl_path = assets / "wall_aorta.stl"
    if not stl_path.exists():
        stl_path = sub / "cases_input" / "BPM120" / "wall_aorta.stl"
    if not stl_path.exists():
        stl_path = None

    # Full STL dir for case generation (needs all STLs + CSV)
    case_stls_dir = sub / "cases_input" / "BPM120"
    if not case_stls_dir.exists() or not any(case_stls_dir.glob("*.stl")):
        # Fallback: bundled assets/bpm120/
        case_stls_dir = assets / "bpm120"
        if not case_stls_dir.exists() or not any(case_stls_dir.glob("*.stl")):
            case_stls_dir = None

    from sample_referrals import REFERRALS
    default_referral = list(REFERRALS.values())[0]

    return DemoData(
        referral_text=referral_text or default_referral,
        clinical_profile=DEMO_PROFILE,
        parameter_justification=DEMO_JUSTIFICATION,
        agent_config=agent_config,
        base_config=base_config,
        rationale_md=rationale_md,
        trace_records=trace_records,
        hemodynamics_report=hemo_report,
        flow_distribution_png=flow_png,
        stl_path=stl_path,
        case_stls_dir=case_stls_dir,
        case_id="BPM120",
    )


def run_live_pipeline(clinical_text: str, api_key: str, model: str) -> DemoData:
    """Run the real agent pipeline with Claude and return DemoData."""
    import tempfile

    from aortacfd_agent.backends.factory import AgentBackendConfig, resolve_backend
    from aortacfd_agent.coordinator import Coordinator
    from aortacfd_agent.corpus.store import FakeCorpusStore, Chunk

    root = _repo_root()
    sub = _submodule_root()

    # Build backend
    backend_config = AgentBackendConfig(provider="anthropic", model=model, api_key=api_key)
    backend = resolve_backend(backend_config)

    # Real literature corpus: 108 papers from references.bib with OpenAlex
    # abstracts, BM25-indexed. Falls back to the 4-chunk FakeCorpusStore if
    # the corpus JSON isn't bundled.
    try:
        from aortacfd_agent.corpus.bib_store import load_default as _load_real_corpus
        corpus = _load_real_corpus()
    except (FileNotFoundError, ImportError):
        corpus = FakeCorpusStore(chunks=[
            Chunk(text="In coarctation cases Murray's law systematically misallocates flow.", paper="Wang2025", page=4),
            Chunk(text="Diastolic backflow at Windkessel outlets can trigger instabilities. beta_T=0.3 recommended.", paper="Esmaily2011", page=7),
            Chunk(text="For WSS and OSI studies, recommend mesh refinement around the wall region.", paper="ValenSendstad2018", page=5),
            Chunk(text="A default tau of 1.5 seconds is appropriate without patient-specific calibration.", paper="Stergiopulos1999", page=2),
        ])

    case_dir = sub / "cases_input" / "BPM120"
    output_dir = Path(tempfile.mkdtemp(prefix="aortacfd_demo_"))

    coord = Coordinator(intake_backend=backend, corpus=corpus)
    result = coord.run(
        clinical_text=clinical_text,
        case_dir=case_dir,
        output_dir=output_dir,
        skip_execution=True,
    )

    # Convert CoordinatorResult to DemoData
    profile = result.intake.profile if result.intake else DEMO_PROFILE
    justification = result.literature.justification if result.literature else DEMO_JUSTIFICATION
    config = result.config.config if result.config else {}
    rationale = result.config.rationale if result.config else ""

    trace = []
    trace_path = output_dir / "agent_trace.jsonl"
    if trace_path.exists():
        for line in trace_path.read_text().strip().splitlines():
            try:
                trace.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # Load static assets (same as demo mode)
    demo_static = load_demo_data(clinical_text)

    return DemoData(
        referral_text=clinical_text,
        clinical_profile=profile,
        parameter_justification=justification,
        agent_config=config,
        base_config=demo_static.base_config,
        rationale_md=rationale,
        trace_records=trace,
        hemodynamics_report=demo_static.hemodynamics_report,
        flow_distribution_png=demo_static.flow_distribution_png,
        stl_path=demo_static.stl_path,
    )
