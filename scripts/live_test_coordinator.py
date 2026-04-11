#!/usr/bin/env python3
"""Live end-to-end Coordinator test with real Claude.

Runs the full five-agent pipeline (intake + literature + config) against
real Claude models on the BPM120 paediatric coarctation fixture:

* IntakeAgent   → claude-haiku-4-5        (cheap, single call)
* LiteratureAgent → claude-sonnet-4-5     (quality, multi-turn RAG)
* ConfigAgent     → no LLM (deterministic)
* ExecutionAgent  → SKIPPED (no CFD solver)
* ResultsAgent    → SKIPPED (no run to summarise)

The corpus is the hand-curated 12-chunk FakeCorpusStore used by the
unit tests, so the LiteratureAgent runs against real embeddings-free
keyword retrieval — fast and deterministic, but still exercises the
full agent loop end-to-end.

Output: examples/output/live_BPM120/ with the usual artefacts
(agent_config.json, agent_rationale.md, agent_trace.jsonl).

Usage::

    cd ~/GitHub/aortacfd-agent
    source venv/bin/activate
    export ANTHROPIC_API_KEY="sk-ant-..."
    python scripts/live_test_coordinator.py

Expected cost: about $0.03 - $0.15 depending on how many tool turns
Sonnet takes. Expected runtime: ~15-30 seconds.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from aortacfd_agent.backends.anthropic_backend import AnthropicBackend  # noqa: E402
from aortacfd_agent.coordinator import Coordinator  # noqa: E402
from aortacfd_agent.corpus.store import FakeCorpusStore  # noqa: E402


_FIXTURE = _REPO / "tests" / "fixtures" / "sample_reports" / "coarctation_paediatric.txt"
_CORPUS_FIXTURE = _REPO / "tests" / "fixtures" / "corpus_chunks.json"
_CASE_DIR = _REPO / "external" / "aortacfd-app" / "cases_input" / "BPM120"
_OUTPUT_DIR = _REPO / "examples" / "output" / "live_BPM120"

_INTAKE_MODEL = "claude-haiku-4-5"
_LITERATURE_MODEL = "claude-sonnet-4-5"
_RESULTS_MODEL = "claude-haiku-4-5"  # unused (summary is skipped)


def _cost_estimate(usage_by_model: dict) -> float:
    """Rough $ estimate using published Anthropic prices (April 2026)."""
    prices = {
        # per million tokens, input / output
        "claude-haiku-4-5": (1.0, 5.0),
        "claude-sonnet-4-5": (3.0, 15.0),
        "claude-opus-4-6": (15.0, 75.0),
    }
    total = 0.0
    for model, usage in usage_by_model.items():
        if model not in prices:
            continue
        inp, out = prices[model]
        total += (usage.get("input_tokens", 0) / 1_000_000) * inp
        total += (usage.get("output_tokens", 0) / 1_000_000) * out
    return total


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set in this shell.", file=sys.stderr)
        return 2
    for p in (_FIXTURE, _CORPUS_FIXTURE, _CASE_DIR):
        if not p.exists():
            print(f"error: missing fixture: {p}", file=sys.stderr)
            return 3

    if _OUTPUT_DIR.exists():
        shutil.rmtree(_OUTPUT_DIR)
    _OUTPUT_DIR.mkdir(parents=True)

    print("=" * 72)
    print("AortaCFD agent — live end-to-end Coordinator test")
    print("=" * 72)
    print(f"Fixture:     {_FIXTURE.relative_to(_REPO)}")
    print(f"Case dir:    {_CASE_DIR.relative_to(_REPO)}")
    print(f"Corpus:      {_CORPUS_FIXTURE.relative_to(_REPO)} ({len(json.loads(_CORPUS_FIXTURE.read_text()))} chunks)")
    print(f"Output:      {_OUTPUT_DIR.relative_to(_REPO)}")
    print(f"Intake:      {_INTAKE_MODEL}")
    print(f"Literature:  {_LITERATURE_MODEL}")
    print()

    referral = _FIXTURE.read_text(encoding="utf-8")
    corpus = FakeCorpusStore.from_json(_CORPUS_FIXTURE)

    intake_backend = AnthropicBackend(model=_INTAKE_MODEL)
    literature_backend = AnthropicBackend(model=_LITERATURE_MODEL)

    # Wrap both backends so we can record per-call token usage by model.
    usage_by_model: dict = {}

    def wrap(backend: AnthropicBackend, label: str):
        real_chat = backend.chat

        def recorded_chat(*args, **kwargs):
            response = real_chat(*args, **kwargs)
            raw = getattr(response, "raw", None)
            usage = getattr(raw, "usage", None) if raw is not None else None
            if usage is not None:
                bucket = usage_by_model.setdefault(
                    getattr(backend, "model", label),
                    {"input_tokens": 0, "output_tokens": 0, "calls": 0},
                )
                bucket["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
                bucket["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
                bucket["calls"] += 1
            return response

        backend.chat = recorded_chat  # type: ignore[method-assign]
        return backend

    wrap(intake_backend, "intake")
    wrap(literature_backend, "literature")

    coord = Coordinator(
        intake_backend=intake_backend,
        literature_backend=literature_backend,
        corpus=corpus,
    )

    print("Running coordinator...")
    t0 = time.perf_counter()
    result = coord.run(
        clinical_text=referral,
        case_dir=_CASE_DIR,
        output_dir=_OUTPUT_DIR,
        skip_execution=True,  # Choice 1: no CFD
        skip_summary=True,    # nothing to summarise
    )
    duration = time.perf_counter() - t0
    print(f"Done in {duration:.1f}s")
    print()
    print(result.brief())
    print()

    if not result.success:
        print(f"FAILED: {result.error}")
        return 1

    # ---- Intake summary -----------------------------------------------------
    intake = result.intake
    if intake is not None:
        profile = intake.profile
        print("--- intake result ---")
        print(f"  patient_id:        {profile.get('patient_id')}")
        print(f"  diagnosis:         {profile.get('diagnosis')}")
        print(f"  heart_rate_bpm:    {profile.get('heart_rate_bpm')}")
        print(f"  BP:                {profile.get('systolic_bp_mmhg')}/{profile.get('diastolic_bp_mmhg')}")
        print(f"  cardiac_output:    {profile.get('cardiac_output_l_min')} L/min")
        print(f"  confidence:        {profile.get('confidence')}")
        print(f"  missing_fields:    {profile.get('missing_fields')}")
        print()

    # ---- Literature summary -------------------------------------------------
    lit = result.literature
    if lit is not None:
        j = lit.justification
        print("--- literature result ---")
        print(f"  confidence:           {j.get('confidence')}")
        print(f"  decisions made:       {len(j.get('decisions') or [])}")
        print(f"  unresolved:           {j.get('unresolved_decisions') or []}")
        print(f"  search queries run:   {len(lit.search_queries)}")
        for q in lit.search_queries:
            print(f"    - {q}")
        print()
        print("  Decisions:")
        for d in j.get("decisions") or []:
            param = d.get("parameter")
            value = d.get("value")
            cites = d.get("citations") or []
            cite_summary = (
                f"{cites[0].get('paper')} (p.{cites[0].get('page')})"
                if cites
                else "no citation"
            )
            print(f"    {param:32} = {value!r:30} — {cite_summary}")
        print()

    # ---- Config summary -----------------------------------------------------
    cfg_result = result.config
    if cfg_result is not None and cfg_result.saved:
        cfg = json.loads(Path(cfg_result.config_path).read_text())
        wk = cfg["boundary_conditions"]["outlets"]["windkessel_settings"]
        print("--- generated agent_config.json (key fields) ---")
        print(f"  patient_id         : {cfg['case_info']['patient_id']}")
        print(f"  cardiac_cycle      : {cfg.get('cardiac_cycle')} s")
        print(f"  physics.model      : {cfg['physics']['model']}")
        print(f"  numerics.profile   : {cfg['numerics']['profile']}")
        print(f"  mesh.goal          : {cfg.get('mesh', {}).get('goal')}")
        print(f"  number_of_cycles   : {cfg.get('simulation_control', {}).get('number_of_cycles')}")
        print(f"  wk.methodology     : {wk.get('methodology')}")
        print(f"  wk.systolic/dia    : {wk.get('systolic_pressure')}/{wk.get('diastolic_pressure')} mmHg")
        print(f"  wk.tau             : {wk.get('tau')}")
        print(f"  wk.betaT           : {wk.get('betaT')}")
        print(f"  wk.flow_split      : {wk.get('flow_split')}")
        print(f"  inlet.type         : {cfg['boundary_conditions']['inlet']['type']}")
        print()
        print("Patches applied by ConfigAgent:")
        for p in cfg_result.patches_applied:
            print(f"  - {p}")
        if cfg_result.warnings:
            print()
            print("Warnings:")
            for w in cfg_result.warnings:
                print(f"  - {w}")
        print()

    # ---- Token usage + cost -------------------------------------------------
    if usage_by_model:
        print("--- token usage by model ---")
        total_input = 0
        total_output = 0
        for model, bucket in usage_by_model.items():
            print(
                f"  {model:24} calls={bucket['calls']:2}  "
                f"in={bucket['input_tokens']:6}  out={bucket['output_tokens']:6}"
            )
            total_input += bucket["input_tokens"]
            total_output += bucket["output_tokens"]
        estimated = _cost_estimate(usage_by_model)
        print(f"  {'TOTAL':24} calls={sum(b['calls'] for b in usage_by_model.values()):2}  "
              f"in={total_input:6}  out={total_output:6}")
        print(f"  estimated cost: ${estimated:.4f}")
        print()

    print(f"Artefacts in: {_OUTPUT_DIR}")
    print(f"  config:    {cfg_result.config_path}")
    print(f"  rationale: {cfg_result.rationale_path}")
    print(f"  trace:     {_OUTPUT_DIR / 'agent_trace.jsonl'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
