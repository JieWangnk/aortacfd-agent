#!/usr/bin/env python3
"""
AWS Batch entrypoint for aortacfd-agent.

Workflow:
  1. Download case data (STL + config) from S3
  2. Run the agent pipeline (intake → literature → config → execute → results)
  3. Upload outputs back to S3

Environment variables (set via Batch job overrides):
  CASE_ID          - Patient case ID (e.g. BPM120)
  S3_BUCKET        - S3 bucket name (e.g. aortacfd-runs)
  CLINICAL_TEXT    - Free-text clinical referral (optional, has default)
  BACKEND          - LLM backend: fake | anthropic | openai | ollama (default: fake)
  MODEL            - LLM model name (default: provider default)
  ANTHROPIC_API_KEY - API key (required if backend=anthropic)
  OPENAI_API_KEY   - API key (required if backend=openai)
  FULL_RUN         - Set to "1" to run solver + postprocess (default: dry-run)
  NP               - Number of MPI processes for solver (default: 4)
"""

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
CASE_ID = os.environ.get("CASE_ID", "BPM120")
S3_BUCKET = os.environ.get("S3_BUCKET", "aortacfd-runs")
CLINICAL_TEXT = os.environ.get(
    "CLINICAL_TEXT",
    "Patient referred for aortic hemodynamic assessment",
)
BACKEND = os.environ.get("BACKEND", "fake")
MODEL = os.environ.get("MODEL", "")
FULL_RUN = os.environ.get("FULL_RUN", "0") == "1"
NP = os.environ.get("NP", "4")

WORK_DIR = Path("/app")
CASE_DIR = WORK_DIR / "cases_input" / CASE_ID
OUTPUT_DIR = WORK_DIR / "output" / CASE_ID


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, streaming output."""
    print(f">>> {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def main() -> int:
    # ------------------------------------------------------------------
    # 1. Download case data from S3
    # ------------------------------------------------------------------
    print(f"=== Downloading case {CASE_ID} from s3://{S3_BUCKET}/input/{CASE_ID}/ ===")
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    run(f"aws s3 sync s3://{S3_BUCKET}/input/{CASE_ID}/ {CASE_DIR}/")

    # Verify STL exists
    stl_files = list(CASE_DIR.glob("*.stl")) + list(CASE_DIR.glob("*.STL"))
    if not stl_files:
        print(f"ERROR: No STL files found in {CASE_DIR}", file=sys.stderr)
        return 1
    print(f"Found STL files: {[f.name for f in stl_files]}")

    # ------------------------------------------------------------------
    # 2. Run agent pipeline
    # ------------------------------------------------------------------
    print(f"=== Running aortacfd-agent ({BACKEND} backend) ===")

    cmd_parts = [
        "aortacfd-agent", "run",
        "--case", str(CASE_DIR),
        "--output", str(OUTPUT_DIR),
        "--clinical-text", f'"{CLINICAL_TEXT}"',
        "--backend", BACKEND,
    ]

    if MODEL:
        cmd_parts.extend(["--model", MODEL])

    if FULL_RUN:
        cmd_parts.append("--full")
        cmd_parts.extend(["--np", NP])

    # The entrypoint needs OpenFOAM sourced (done in Batch command override)
    result = run(" ".join(cmd_parts), check=False)

    if result.returncode != 0:
        print(f"WARNING: Agent exited with code {result.returncode}", file=sys.stderr)
        # Still upload whatever output was produced

    # ------------------------------------------------------------------
    # 3. Upload results to S3
    # ------------------------------------------------------------------
    print(f"=== Uploading results to s3://{S3_BUCKET}/output/{CASE_ID}/ ===")
    if OUTPUT_DIR.exists():
        run(f"aws s3 sync {OUTPUT_DIR}/ s3://{S3_BUCKET}/output/{CASE_ID}/")
    else:
        print(f"WARNING: Output directory {OUTPUT_DIR} does not exist", file=sys.stderr)

    print("=== Batch job complete ===")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
