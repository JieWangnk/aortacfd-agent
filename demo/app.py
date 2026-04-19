"""
AortaCFD Agent — Interactive Pipeline Demo

Launch:
    streamlit run demo/app.py

Modes:
    Demo (instant):  Pre-computed results, no API key needed
    Live (Claude):   Real LLM inference via Anthropic API (~15-30s)
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure the repo's src/ is importable
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "demo"))

from demo_data import load_demo_data, run_live_pipeline
from components import (
    render_pipeline_status,
    render_intake_stage,
    render_literature_stage,
    render_config_stage,
    render_case_download_stage,
    render_execution_stage,
    render_results_stage,
    render_audit_trace,
)
from sample_referrals import REFERRALS
from styles import CUSTOM_CSS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AortaCFD Agent",
    page_icon="\U0001fac0",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("AortaCFD Agent")
    st.caption("Clinical text to patient-specific CFD")

    st.markdown("---")

    mode = st.radio("Mode", ["Demo (instant)", "Live (Claude API)"], index=0)
    is_live = mode.startswith("Live")

    api_key = ""
    model = "claude-sonnet-4-5-20250514"
    if is_live:
        api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
        model = st.selectbox("Model", [
            "claude-sonnet-4-5-20250514",
            "claude-haiku-4-5-20251001",
        ])

    st.markdown("---")

    selected = st.selectbox("Sample Referral", list(REFERRALS.keys()))
    referral_text = st.text_area(
        "Clinical Referral",
        value=REFERRALS[selected],
        height=250,
        placeholder="Paste a clinical referral here...",
    )

    st.markdown("---")

    run_button = st.button("Run Pipeline", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "[GitHub](https://github.com/JieWangnk/aortacfd-agent) | "
        "[Paper](https://github.com/JieWangnk/AortaCFDappPaper) | "
        "v0.1.0"
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown("## AortaCFD Agent Pipeline")
st.markdown(
    "*Free-text clinical referral* &rarr; *structured profile* &rarr; "
    "*literature-grounded config* &rarr; *CFD simulation* &rarr; *clinical summary*"
)

if run_button:
    if is_live and not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
    elif is_live:
        with st.status("Running agent pipeline with Claude...", expanded=True) as status:
            st.write("Calling IntakeAgent + LiteratureAgent + ConfigAgent...")
            st.write(f"Model: `{model}` | Backend: `anthropic`")
            try:
                data = run_live_pipeline(referral_text, api_key, model)
                status.update(label="Pipeline complete!", state="complete")
            except Exception as e:
                status.update(label="Pipeline failed", state="error")
                st.error(f"Error: {e}")
                data = None
        if data:
            st.session_state["pipeline_data"] = data
    else:
        data = load_demo_data(referral_text)
        st.session_state["pipeline_data"] = data

# Display results
if "pipeline_data" in st.session_state:
    data = st.session_state["pipeline_data"]

    # Pipeline status bar
    render_pipeline_status(data.trace_records)
    st.markdown("---")

    # Stage 1: Intake
    render_intake_stage(data.referral_text, data.clinical_profile)
    st.markdown("---")

    # Stage 2: Literature
    render_literature_stage(data.parameter_justification)
    st.markdown("---")

    # Stage 3: Config
    render_config_stage(data.agent_config, data.base_config, data.rationale_md)
    st.markdown("---")

    # Stage 3.5: Download OpenFOAM case
    render_case_download_stage(data.agent_config, data.case_stls_dir, data.case_id)
    st.markdown("---")

    # Stage 4: CFD Results
    render_execution_stage(data.hemodynamics_report, data.flow_distribution_png, data.stl_path)
    st.markdown("---")

    # Stage 5: Summary
    render_results_stage(data.hemodynamics_report)
    st.markdown("---")

    # Audit trail
    render_audit_trace(data.trace_records)

else:
    # Landing state
    st.markdown("---")
    st.markdown(
        """
        ### How it works

        1. **Intake Agent** — Extracts structured patient data from free-text referral (LLM)
        2. **Literature Agent** — Searches a paper corpus to justify every CFD parameter with citations (LLM + RAG)
        3. **Config Agent** — Deterministically patches a template config (no LLM, fully reproducible)
        4. **Execution Agent** — Runs OpenFOAM mesh generation and solver (subprocess)
        5. **Results Agent** — Reads simulation output and writes a grounded clinical summary (LLM)

        Select a sample referral in the sidebar and click **Run Pipeline** to see the demo.

        **Demo mode** uses pre-computed results and requires no API key.
        **Live mode** calls Claude in real-time (~15-30 seconds).
        """
    )
