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
    st.markdown("### 1. Intake — structured patient from free-text referral")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Clinical referral**")
        st.text_area(
            "referral_display", referral_text, height=220,
            disabled=True, label_visibility="collapsed",
        )

    with col2:
        st.markdown("**Extracted profile**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Age", f"{profile.get('age_years', '?')} yr")
        m2.metric("HR", f"{profile.get('heart_rate_bpm', '?')} bpm")
        m3.metric("CO", f"{profile.get('cardiac_output_l_min', '?')} L/min")
        bp = f"{profile.get('systolic_bp_mmhg', '?')}/{profile.get('diastolic_bp_mmhg', '?')}"
        st.caption(
            f"**BP** {bp} mmHg  ·  **Diagnosis** {profile.get('diagnosis', 'N/A')}  ·  "
            f"**Imaging** {', '.join(profile.get('imaging_modality', [])) or 'N/A'}"
        )


# ---------------------------------------------------------------------------
# Stage 2: Literature Decisions
# ---------------------------------------------------------------------------

def render_literature_stage(justification: dict):
    st.markdown("### 2. Literature — decisions with citations")

    decisions = justification.get("decisions", [])
    queries = justification.get("search_queries_used", [])

    # Count unique papers cited
    papers = set()
    for d in decisions:
        for c in d.get("citations", []):
            papers.add(c.get("paper", ""))
    papers.discard("")

    # Try to enrich citations with real paper metadata from the corpus
    paper_index = _load_paper_index()

    st.markdown(
        f"**{len(decisions)}** parameter decisions · **{len(papers)}** papers cited · "
        f"**{len(queries)}** corpus queries"
    )
    if paper_index:
        n_with_abstract = sum(1 for p in paper_index.values() if p.get("abstract"))
        st.caption(
            f"Retrieval over **{len(paper_index)} papers** from the project bibliography · "
            f"title + abstract indexing ({n_with_abstract} papers have OpenAlex abstracts) · "
            f"BM25 keyword search · no full-text PDFs."
        )

    with st.expander("Show all decisions and citations"):
        for d in decisions:
            param = d["parameter"]
            value = d["value"]
            value_str = (
                json.dumps(value) if isinstance(value, dict) else str(value)
            )
            st.markdown(f"**`{param}`** = `{value_str}`")
            st.caption(d["reasoning"])
            for c in d.get("citations", []):
                _render_citation(c, paper_index)
            st.markdown("")


def _render_citation(cite: dict, paper_index: dict):
    """Render a citation, enriched with real paper metadata if available."""
    paper_id = cite.get("paper", "")
    quote = cite.get("quote", "")
    page = cite.get("page", "?")

    # Look up real paper metadata by ID (case-insensitive key match)
    meta = paper_index.get(paper_id.lower()) if paper_id else None

    if meta:
        authors = meta.get("authors", paper_id)
        year = meta.get("year", "")
        title = meta.get("title", "")
        journal = meta.get("journal", "")
        doi = meta.get("doi", "").strip()
        doi_link = f"[doi:{doi}](https://doi.org/{doi})" if doi else ""
        st.markdown(
            f"> \"{quote}\"  \n"
            f"> **{authors} {year}** — *{title}* · {journal} {doi_link}"
        )
    else:
        st.markdown(f"> \"{quote}\" — *{paper_id}*, p.{page}")


def _load_paper_index() -> dict:
    """Lazy-load the corpus JSON and index papers by lowercased ID."""
    if hasattr(_load_paper_index, "_cache"):
        return _load_paper_index._cache
    try:
        repo_root = Path(__file__).resolve().parent.parent
        corpus_path = repo_root / "src" / "aortacfd_agent" / "corpus" / "index" / "aortacfd_corpus.json"
        if not corpus_path.exists():
            _load_paper_index._cache = {}
            return {}
        with corpus_path.open(encoding="utf-8") as f:
            data = json.load(f)
        index = {p["id"].lower(): p for p in data.get("papers", [])}
        _load_paper_index._cache = index
        return index
    except Exception:
        _load_paper_index._cache = {}
        return {}


# ---------------------------------------------------------------------------
# Stage 3: Config Generation
# ---------------------------------------------------------------------------

def render_config_stage(agent_config: dict, base_config: dict, rationale_md: str):
    st.markdown("### 3. Config — validated OpenFOAM case")

    # Key changes from base (up to 6 most important)
    if base_config and agent_config:
        changes = _compute_diff(base_config, agent_config)
        if changes:
            st.markdown("**Key parameters set from clinical input:**")
            shown = 0
            for path, old, new in changes:
                # Filter: only show changes to interesting fields
                if not any(k in path for k in [
                    "cardiac_cycle", "physics", "numerics.profile",
                    "mesh.goal", "windkessel", "flow_split", "inlet.type",
                    "systolic", "diastolic"
                ]):
                    continue
                new_str = _format_val(new)
                st.markdown(f"- `{path}` → **{new_str}**")
                shown += 1
                if shown >= 6:
                    break
            if shown == 0:
                # Fallback: show first 6 changes
                for path, old, new in changes[:6]:
                    new_str = _format_val(new)
                    st.markdown(f"- `{path}` → **{new_str}**")

    with st.expander("Show agent rationale (with citations)"):
        st.markdown(rationale_md)

    with st.expander("Show full config JSON"):
        st.json(agent_config)

    st.markdown("")


# ---------------------------------------------------------------------------
# Stage 3.5: OpenFOAM case download
# ---------------------------------------------------------------------------

def render_case_download_stage(agent_config: dict, case_stls_dir: Optional[Path], case_id: str):
    """Pre-built BPM120 case download (runtime generation needs AortaCFD-app repo, which is private until paper publishes)."""
    if not agent_config:
        return

    assets_dir = Path(__file__).resolve().parent / "assets"
    prebuilt_zip = assets_dir / "bpm120_case.zip"

    if prebuilt_zip.exists():
        zip_bytes = prebuilt_zip.read_bytes()
        st.download_button(
            label=f"⬇  Download {case_id} OpenFOAM case ({len(zip_bytes)/1024:.0f} KB)",
            data=zip_bytes,
            file_name=f"{case_id}_case.zip",
            mime="application/zip",
            type="primary",
        )
        st.caption(
            "Unzip → `blockMesh` → `snappyHexMesh` → `foamRun`. Full setup instructions in the "
            "bundled `README.md`. Custom clinical input will build patient-specific cases once "
            "the AortaCFD-app is released with the paper."
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
