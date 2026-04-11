#!/usr/bin/env python3
"""Live LLM smoke test for IntakeAgent.

Reads ANTHROPIC_API_KEY from the environment and runs the IntakeAgent
on the BPM120 paediatric coarctation fixture against Claude Haiku 4.5.
Writes the result and token usage to ./live_test_output.json so the
caller (and any other tool) can inspect the output without re-running.

Usage::

    cd ~/GitHub/aortacfd-agent
    source venv/bin/activate
    python scripts/live_test_intake.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Make 'aortacfd_agent' importable from the source tree.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from aortacfd_agent.agents.intake import IntakeAgent  # noqa: E402
from aortacfd_agent.backends.anthropic_backend import AnthropicBackend  # noqa: E402


_FIXTURE = _REPO / "tests" / "fixtures" / "sample_reports" / "coarctation_paediatric.txt"
_OUTPUT = _REPO / "live_test_output.json"
_MODEL = "claude-haiku-4-5"  # cheapest, fastest


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set in this shell.", file=sys.stderr)
        return 2

    if not _FIXTURE.exists():
        print(f"error: fixture missing: {_FIXTURE}", file=sys.stderr)
        return 3

    print("=" * 72)
    print("AortaCFD agent — live LLM smoke test (IntakeAgent)")
    print("=" * 72)
    print(f"Fixture:  {_FIXTURE.relative_to(_REPO)}")
    print(f"Model:    {_MODEL}")
    print(f"Backend:  AnthropicBackend (real Claude API)")
    print()

    referral = _FIXTURE.read_text(encoding="utf-8")
    print("--- referral text (first 6 lines) ---")
    for line in referral.splitlines()[:6]:
        print(f"  {line}")
    print("  ...")
    print()

    print("Calling Claude... ", end="", flush=True)
    backend = AnthropicBackend(model=_MODEL)
    agent = IntakeAgent(backend=backend)

    t0 = time.perf_counter()
    try:
        result = agent.extract(referral)
    except Exception as exc:
        print(f"FAILED")
        print(f"  {type(exc).__name__}: {exc}")
        return 1
    duration = time.perf_counter() - t0
    print(f"OK ({duration:.2f}s)")
    print()

    profile = result.profile

    # Try to extract token usage from the raw provider response, if present.
    usage = {}
    raw = result.raw_response.raw
    if raw is not None:
        for attr in ("usage",):
            obj = getattr(raw, attr, None)
            if obj is not None:
                # anthropic SDK returns a Usage dataclass with input_tokens etc.
                usage = {
                    "input_tokens": getattr(obj, "input_tokens", None),
                    "output_tokens": getattr(obj, "output_tokens", None),
                }
                break

    print("--- extracted profile ---")
    for key in [
        "patient_id",
        "age_years",
        "sex",
        "diagnosis",
        "heart_rate_bpm",
        "systolic_bp_mmhg",
        "diastolic_bp_mmhg",
        "cardiac_output_l_min",
        "imaging_modality",
        "flow_waveform_source",
        "confidence",
    ]:
        value = profile.get(key)
        print(f"  {key}: {value}")
    if profile.get("missing_fields"):
        print(f"  missing_fields: {profile['missing_fields']}")
    if profile.get("constraints"):
        print(f"  constraints: {len(profile['constraints'])} item(s)")
        for c in profile["constraints"]:
            print(f"    - {c}")
    print()

    if usage:
        in_tok = usage.get("input_tokens") or 0
        out_tok = usage.get("output_tokens") or 0
        # Haiku 4.5 is roughly $1/MTok in, $5/MTok out at time of writing.
        cost = (in_tok / 1_000_000) * 1.0 + (out_tok / 1_000_000) * 5.0
        print("--- token usage ---")
        print(f"  input tokens:  {in_tok}")
        print(f"  output tokens: {out_tok}")
        print(f"  estimated cost: ${cost:.5f}")
        print()

    payload = {
        "model": _MODEL,
        "fixture": str(_FIXTURE.relative_to(_REPO)),
        "duration_s": round(duration, 3),
        "usage": usage,
        "profile": profile,
        "iterations": 1,
    }
    _OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Full output saved to: {_OUTPUT.relative_to(_REPO)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
