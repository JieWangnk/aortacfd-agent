"""Rendering functions for each pipeline stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Pipeline status bar
# ---------------------------------------------------------------------------

def render_pipeline_status(trace_records: list[dict]):
    """Render a horizontal pipeline status bar."""
    stage_names = ["Intake", "Literature", "Config", "Execution", "Summary"]
    stage_keys = ["intake", "literature", "config", "execution", "summary"]
    stage_status = {r["stage"]: r.get("status", "ok") for r in trace_records}

    cols = st.columns(len(stage_names))
    for i, (name, key) in enumerate(zip(stage_names, stage_keys)):
        with cols[i]:
            status = stage_status.get(key)
            if status == "ok":
                st.markdown(f"**:green[{name}]**")
                st.progress(1.0)
            elif status == "warning":
                st.markdown(f"**:orange[{name}]**")
                st.progress(1.0)
            else:
                st.markdown(f"**:gray[{name}]**")
                st.progress(0.0)


# ---------------------------------------------------------------------------
# Stage 1: Clinical Intake
# ---------------------------------------------------------------------------

def render_intake_stage(referral_text: str, profile: dict):
    st.markdown("### 1. Clinical Intake")
    st.caption("The IntakeAgent extracts a structured patient profile from free-text clinical referral.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Clinical Referral**")
        st.text_area("referral_display", referral_text, height=300, disabled=True, label_visibility="collapsed")

    with col2:
        st.markdown("**Extracted Patient Profile**")

        # Confidence badge
        conf = profile.get("confidence", "low")
        badge_map = {"high": ":green[HIGH]", "medium": ":orange[MEDIUM]", "low": ":red[LOW]"}
        st.markdown(f"Confidence: {badge_map.get(conf, conf)}")

        # Key metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Age", f"{profile.get('age_years', '?')} yr")
        m2.metric("HR", f"{profile.get('heart_rate_bpm', '?')} bpm")
        m3.metric("CO", f"{profile.get('cardiac_output_l_min', '?')} L/min")
        m4.metric("BSA", f"{profile.get('bsa_m2', '?')} m\u00b2")

        # Vitals table
        vitals = {
            "Systolic BP": f"{profile.get('systolic_bp_mmhg', 'N/A')} mmHg",
            "Diastolic BP": f"{profile.get('diastolic_bp_mmhg', 'N/A')} mmHg",
            "Diagnosis": profile.get("diagnosis", "N/A"),
            "Sex": profile.get("sex", "N/A"),
            "Imaging": ", ".join(profile.get("imaging_modality", [])),
            "Flow source": profile.get("flow_waveform_source", "N/A"),
        }
        for label, val in vitals.items():
            st.markdown(f"**{label}:** {val}")

        # Study goal
        goal = profile.get("study_goal", "")
        if goal:
            st.info(f"**Study goal:** {goal}")

        # Missing fields
        missing = profile.get("missing_fields", [])
        if missing:
            st.warning(f"Missing fields: {', '.join(missing)}")

        with st.expander("Raw JSON"):
            st.json(profile)


# ---------------------------------------------------------------------------
# Stage 2: Literature Decisions
# ---------------------------------------------------------------------------

def render_literature_stage(justification: dict):
    st.markdown("### 2. Literature-Grounded Decisions")
    st.caption("The LiteratureAgent searches a corpus and cites papers to justify every CFD parameter choice.")

    decisions = justification.get("decisions", [])
    queries = justification.get("search_queries_used", [])
    conf = justification.get("confidence", "N/A")

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Parameters Justified", len(decisions))
    c2.metric("Corpus Queries", len(queries))
    badge_map = {"high": ":green[HIGH]", "medium": ":orange[MEDIUM]", "low": ":red[LOW]"}
    c3.metric("Confidence", conf.upper())

    # Search queries
    with st.expander(f"Search queries issued ({len(queries)})"):
        for i, q in enumerate(queries, 1):
            st.code(f"{i}. {q}", language=None)

    # Each decision
    for d in decisions:
        param = d["parameter"]
        value = d["value"]
        if isinstance(value, dict):
            value_str = json.dumps(value, indent=2)
        else:
            value_str = str(value)

        with st.expander(f"**{param}** = `{value_str}`"):
            st.markdown(f"**Reasoning:** {d['reasoning']}")

            if d.get("alternative_considered"):
                st.markdown(f"*Alternative considered:* {d['alternative_considered']}")

            citations = d.get("citations", [])
            if citations:
                st.markdown("**Citations:**")
                for c in citations:
                    st.markdown(
                        f"> \"{c['quote']}\"  \n"
                        f"> \u2014 *{c['paper']}*, p.{c.get('page', '?')}"
                    )
            elif not citations:
                st.caption("No citation needed (default value).")


# ---------------------------------------------------------------------------
# Stage 3: Config Generation
# ---------------------------------------------------------------------------

def render_config_stage(agent_config: dict, base_config: dict, rationale_md: str):
    st.markdown("### 3. Configuration Generation")
    st.caption("The ConfigAgent deterministically patches a template config with the intake + literature decisions. No LLM used here.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Agent Rationale**")
        st.markdown(rationale_md)

    with col2:
        st.markdown("**Generated Config**")
        tabs = st.tabs(["Key Changes", "Full Config"])

        with tabs[0]:
            if base_config and agent_config:
                changes = _compute_diff(base_config, agent_config)
                if changes:
                    for path, old, new in changes[:20]:
                        old_str = _format_val(old)
                        new_str = _format_val(new)
                        if old is None:
                            st.markdown(f"**`{path}`**: :green[{new_str}] *(added)*")
                        else:
                            st.markdown(f"**`{path}`**: {old_str} :arrow_right: :green[{new_str}]")
                    if len(changes) > 20:
                        st.caption(f"... and {len(changes) - 20} more changes")
                else:
                    st.info("No differences detected.")
            elif agent_config:
                # No base config — show agent config fields as "added"
                st.caption("Base template not available — showing agent config fields:")
                for key in sorted(agent_config.keys()):
                    if not key.startswith("_"):
                        st.markdown(f"**`{key}`**: :green[{_format_val(agent_config[key])}]")
            else:
                st.info("Config not available.")

        with tabs[1]:
            if agent_config:
                st.json(agent_config)
            else:
                st.info("Config not available.")


# ---------------------------------------------------------------------------
# Stage 3.5: OpenFOAM case download
# ---------------------------------------------------------------------------

def render_case_download_stage(agent_config: dict, case_stls_dir: Optional[Path], case_id: str):
    """
    Offer a pre-built OpenFOAM case for the BPM120 demo, or fall back to
    runtime generation if the AortaCFD-app submodule is available.
    """
    st.markdown("### 3.5. Download OpenFOAM Case")
    st.caption(
        "A complete OpenFOAM case (rendered dictionaries + scaled STLs + merged config) "
        "ready to run on any machine with OpenFOAM 12. No computation needed to "
        "**generate** the case — only to run it."
    )

    if not agent_config:
        st.info("Run the pipeline first to generate the agent config.")
        return

    # Path to the pre-built zip (bundled in the repo for Streamlit Cloud)
    assets_dir = Path(__file__).resolve().parent / "assets"
    prebuilt_zip = assets_dir / "bpm120_case.zip"

    # Show what's inside the zip
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Source patches** (scaled mm → m in the zip):")
        if case_stls_dir and case_stls_dir.exists():
            stls = sorted([p.name for p in case_stls_dir.glob("*.stl")])
            csvs = sorted([p.name for p in case_stls_dir.glob("*.csv")])
            for f in stls + csvs:
                st.markdown(f"- `{f}`")
        else:
            st.markdown("- `inlet.stl`\n- `outlet1.stl` – `outlet4.stl`\n- `wall_aorta.stl`\n- `BPM120.csv`")
    with col_b:
        st.markdown("**The zip contains:**")
        st.markdown(
            "- `openfoam/constant/` — `triSurface/` (scaled), `transportProperties`, `momentumTransport`\n"
            "- `openfoam/system/` — `controlDict`, `fvSchemes`, `fvSolution`, `snappyHexMeshDict`, `blockMeshDict`, `decomposeParDict`\n"
            "- `agent/agent_config.json` — validated config (reproducibility)\n"
            "- `reports/merged_config.json`, `simulation_setup_report.txt`\n"
            "- `README.md` — run instructions"
        )

    # --- Download ---
    if prebuilt_zip.exists():
        # Serve the pre-built zip (BPM120 demo scenario)
        zip_bytes = prebuilt_zip.read_bytes()
        st.download_button(
            label=f"Download {case_id}_case.zip",
            data=zip_bytes,
            file_name=f"{case_id}_case.zip",
            mime="application/zip",
            type="primary",
            use_container_width=False,
        )
        st.caption(
            f"Demo case · {len(zip_bytes)/1024:.1f} KB · matches the BPM120 paediatric coarctation scenario. "
            "Unzip and follow `README.md`: `blockMesh` → `snappyHexMesh` → `foamRun`."
        )
        st.info(
            "**Note.** The current download is a pre-built case for the BPM120 demo scenario. "
            "Live generation from user-specific clinical text will be enabled once the "
            "**AortaCFD-app** is published alongside the paper (currently under review). "
            "The full template engine lives in that repo."
        )
    else:
        # Fallback: try runtime generation (only works when aortacfd-app is available)
        if st.button("Generate OpenFOAM case (.zip)", type="primary"):
            with st.spinner("Rendering case dictionaries..."):
                try:
                    from build_case import build_openfoam_case
                    zip_bytes = build_openfoam_case(
                        agent_config=agent_config,
                        stl_source_dir=case_stls_dir,
                        case_id=case_id,
                    )
                except Exception as e:
                    st.error(f"Case generation failed: {e}")
                    return

            if not zip_bytes:
                st.warning(
                    "Runtime case generation is not available on this deployment. "
                    "The full case-rendering engine lives in **AortaCFD-app**, which is "
                    "private pending paper publication. A pre-built BPM120 demo case will "
                    "be bundled here soon."
                )
                return

            st.success(f"Generated · {len(zip_bytes)/1024:.1f} KB")
            st.download_button(
                label=f"Download {case_id}_case.zip",
                data=zip_bytes,
                file_name=f"{case_id}_case.zip",
                mime="application/zip",
                use_container_width=False,
            )


def _format_val(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, dict):
        return json.dumps(v)
    return str(v)


def _compute_diff(base: dict, patched: dict, path: str = "") -> list[tuple[str, Any, Any]]:
    """Return (dotted_path, old_value, new_value) for changed leaf fields."""
    changes = []
    all_keys = set(list(base.keys()) + list(patched.keys()))
    for key in sorted(all_keys):
        if str(key).startswith("_"):
            continue
        p = f"{path}.{key}" if path else str(key)
        bv = base.get(key)
        pv = patched.get(key)
        if isinstance(bv, dict) and isinstance(pv, dict):
            changes.extend(_compute_diff(bv, pv, p))
        elif bv != pv:
            changes.append((p, bv, pv))
    return changes


# ---------------------------------------------------------------------------
# Stage 4: CFD Results
# ---------------------------------------------------------------------------

def render_execution_stage(hemodynamics_report: str, flow_png: Optional[bytes], stl_path: Optional[Path]):
    st.markdown("### 4. CFD Simulation Results")
    st.caption("Pre-computed results from a BPM120 production run. In a live deployment, the ExecutionAgent would run OpenFOAM.")

    # Hemodynamic metrics parsed from report
    metrics = _parse_hemo_metrics(hemodynamics_report)

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TAWSS Mean", f"{metrics.get('tawss_mean', '?')} Pa")
        c2.metric("OSI Mean", metrics.get("osi_mean", "?"))
        c3.metric("Max Pressure Drop", f"{metrics.get('max_dp', '?')} mmHg")
        significant = float(metrics.get("max_dp", 0)) > 20
        c4.metric("Clinically Significant?", "Yes" if significant else "No",
                   delta="coarctation" if significant else None,
                   delta_color="inverse" if significant else "off")

    col1, col2 = st.columns(2)

    with col1:
        if flow_png:
            st.markdown("**Flow Distribution**")
            st.image(flow_png, caption="Outlet flow distribution over cardiac cycle", use_container_width=True)
        else:
            st.info("Flow distribution plot not available (run the CFD pipeline to generate).")

    with col2:
        st.markdown("**Hemodynamics Report**")
        with st.expander("Full report", expanded=True):
            st.code(hemodynamics_report[:2000] if hemodynamics_report else "Not available", language=None)

    # 3D geometry viewer
    if stl_path and stl_path.exists():
        with st.expander("3D Aorta Geometry (interactive)"):
            _render_stl_viewer(stl_path)


def _parse_hemo_metrics(report: str) -> dict:
    """Extract key metrics from the hemodynamics report text."""
    metrics = {}
    if not report:
        return metrics
    for line in report.splitlines():
        line = line.strip()
        if line.startswith("TAWSS Mean:"):
            metrics["tawss_mean"] = line.split(":")[1].strip().split()[0]
        elif line.startswith("TAWSS Maximum:"):
            metrics["tawss_max"] = line.split(":")[1].strip().split()[0]
        elif line.startswith("OSI Mean:"):
            metrics["osi_mean"] = line.split(":")[1].strip().split()[0]
        elif "Pressure drop:" in line:
            try:
                dp = float(line.split(":")[1].strip().split()[0])
                existing = float(metrics.get("max_dp", 0))
                if dp > existing:
                    metrics["max_dp"] = f"{dp:.2f}"
            except (ValueError, IndexError):
                pass
    return metrics


def _render_stl_viewer(stl_path: Path):
    """Render an STL file as interactive 3D Plotly mesh."""
    try:
        import numpy as np
        from stl import mesh as stl_mesh
        import plotly.graph_objects as go

        m = stl_mesh.Mesh.from_file(str(stl_path))
        vertices = m.vectors.reshape(-1, 3)

        # Downsample if too many triangles
        n_faces = len(m.vectors)
        stride = max(1, n_faces // 25000)
        if stride > 1:
            sampled = m.vectors[::stride]
            vertices = sampled.reshape(-1, 3)

        unique_verts, inverse = np.unique(vertices, axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3)

        fig = go.Figure(data=[
            go.Mesh3d(
                x=unique_verts[:, 0],
                y=unique_verts[:, 1],
                z=unique_verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="lightcoral",
                opacity=0.8,
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3),
            )
        ])
        fig.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="#f8fafc",
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Geometry: {stl_path.name} ({n_faces:,} triangles, showing every {stride}th)")
    except ImportError:
        st.warning("Install plotly for 3D rendering: `pip install plotly`")
    except Exception as e:
        st.error(f"Failed to render STL: {e}")


# ---------------------------------------------------------------------------
# Stage 5: Clinical Summary
# ---------------------------------------------------------------------------

def render_results_stage(hemodynamics_report: str):
    st.markdown("### 5. Clinical Summary")
    st.caption("The ResultsAgent reads CFD outputs and writes a grounded natural-language summary for clinicians.")

    if not hemodynamics_report:
        st.info("No results available. Run the full pipeline with `--execute --full` to generate CFD results.")
        return

    st.markdown(
        "In a full run, the ResultsAgent would call tools (`read_qoi_summary`, "
        "`read_hemodynamics_report`, `read_pressure_timeseries`) and produce a "
        "3-6 sentence clinical summary citing specific numbers from the simulation."
    )

    # Show what the ResultsAgent would summarize
    metrics = _parse_hemo_metrics(hemodynamics_report)
    if metrics:
        max_dp = float(metrics.get("max_dp", 0))
        tawss = metrics.get("tawss_mean", "?")
        osi = metrics.get("osi_mean", "?")

        summary = (
            f"The simulation predicts a peak pressure gradient of **{max_dp:.1f} mmHg** "
            f"from ascending to descending aorta, "
            f"{'exceeding the 20 mmHg clinical threshold for significant coarctation' if max_dp > 20 else 'below the 20 mmHg significance threshold'}. "
            f"Time-averaged wall shear stress is **{tawss} Pa** (mean), with oscillatory "
            f"shear index of **{osi}** suggesting {'moderate' if float(osi) > 0.15 else 'low'} "
            f"flow disturbance. These findings "
            f"{'support consideration of intervention' if max_dp > 20 else 'suggest continued monitoring'}."
        )
        st.success(summary)


# ---------------------------------------------------------------------------
# Audit Trail
# ---------------------------------------------------------------------------

def render_audit_trace(trace_records: list[dict]):
    st.markdown("### Audit Trail")
    st.caption("Every agent decision is logged to agent_trace.jsonl for reproducibility.")

    if not trace_records:
        st.info("No trace data available.")
        return

    # Timeline
    cols = st.columns(len(trace_records))
    for i, rec in enumerate(trace_records):
        with cols[i]:
            status = rec.get("status", "ok")
            stage = rec.get("stage", "?")
            duration = rec.get("duration_s", 0)

            if status == "ok":
                st.success(f"**{stage}**")
            elif status == "warning":
                st.warning(f"**{stage}**")
            else:
                st.error(f"**{stage}**")
            st.caption(f"{duration:.3f}s")

    # Detail expander
    with st.expander("Full trace (JSONL)"):
        for rec in trace_records:
            st.json(rec)
