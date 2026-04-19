"""Claude-backed paper classifier.

For each paper: return a relevance score (0–1), a tier letter (A–F), and a
2-sentence plain-English summary + 1-sentence 'why it matters' note. Calls
Claude Haiku — cheap, fast, good enough for abstract-level triage.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .sources import Paper

logger = logging.getLogger(__name__)


TIER_NAMES: Dict[str, str] = {
    "A": "Boundary conditions (inlet/outlet, Windkessel)",
    "B": "Turbulence modelling (RANS, LES, DNS)",
    "C": "V&V and uncertainty quantification",
    "D": "Physiology & scaling laws",
    "E": "Imaging & WSS measurement",
    "F": "AI pipelines & PINN / neural operators",
    "X": "Out of scope",  # will be dropped from the digest
}


CLASSIFIER_PROMPT = """You are triaging new papers for a weekly cardiovascular-CFD research digest.

Decide if this paper is in scope, assign it to one of six tiers, and write a terse summary for a technical audience (CFD researchers, bioengineers, cardiologists who simulate).

TIERS:
  A — Boundary conditions (inlet/outlet BCs, Windkessel, lumped-parameter, 0D-3D coupling)
  B — Turbulence modelling (RANS, LES, DES, DNS applied to cardiovascular flows)
  C — V&V and uncertainty quantification (grid convergence, sensitivity, GCI, UQ)
  D — Physiology & scaling (Murray's law, Womersley, arterial biomechanics, hemodynamic principles)
  E — Imaging & WSS (4D flow MRI, Doppler echo, WSS measurement from imaging)
  F — AI pipelines / PINN (neural operators, physics-informed ML, surrogates, digital twins, AI segmentation for CFD)
  X — Out of scope (not cardiovascular CFD; pure experiment with no CFD/modelling tie; unrelated)

RULES:
  - Choose the SINGLE best tier. If a paper fits two, pick the primary contribution.
  - Relevance 0–1: how likely a cardiovascular-CFD researcher is to care. Below 0.4 is noise.
  - Summary: 2 sentences, no hype, state what the paper does and the main finding/method.
  - Relevance note: 1 sentence explaining why a CFD practitioner should care (or would skip it).

Return ONLY a JSON object, no prose. Schema:
{{
  "tier": "A"|"B"|"C"|"D"|"E"|"F"|"X",
  "relevance": 0.0-1.0,
  "summary": "2-sentence summary",
  "why": "1 sentence on why it matters"
}}

---
PAPER
Title: {title}
Authors: {authors}
Venue: {venue} ({year})
Abstract: {abstract}
"""


@dataclass
class Classification:
    tier: str
    relevance: float
    summary: str
    why: str

    @property
    def in_scope(self) -> bool:
        return self.tier in {"A", "B", "C", "D", "E", "F"}


def _parse_json_response(text: str) -> Optional[dict]:
    """Extract the first JSON object from the model's reply."""
    # Models sometimes wrap in ```json ... ``` or add prose
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def classify_paper(
    paper: Paper,
    client: Any,
    model: str = "claude-haiku-4-5-20251001",
    max_abstract_chars: int = 2000,
) -> Optional[Classification]:
    """Classify one paper. Returns ``None`` on API failure."""
    abstract = paper.abstract or ""
    if len(abstract) > max_abstract_chars:
        abstract = abstract[:max_abstract_chars] + " ..."

    prompt = CLASSIFIER_PROMPT.format(
        title=paper.title,
        authors=", ".join(paper.authors[:4]) + (", et al." if len(paper.authors) > 4 else ""),
        venue=paper.venue or paper.source,
        year=paper.year,
        abstract=abstract,
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in resp.content
            if getattr(block, "type", None) == "text"
        )
    except Exception as e:
        logger.warning("classify_paper failed for %r: %s", paper.title[:60], e)
        return None

    parsed = _parse_json_response(text)
    if not parsed:
        logger.warning("could not parse JSON from classifier for %r", paper.title[:60])
        return None

    tier = (parsed.get("tier") or "X").strip().upper()
    if tier not in TIER_NAMES:
        tier = "X"

    try:
        relevance = float(parsed.get("relevance", 0.0))
    except (TypeError, ValueError):
        relevance = 0.0
    relevance = max(0.0, min(1.0, relevance))

    return Classification(
        tier=tier,
        relevance=relevance,
        summary=str(parsed.get("summary", "")).strip(),
        why=str(parsed.get("why", "")).strip(),
    )


def make_client(api_key: Optional[str] = None):
    """Instantiate an Anthropic client. Key falls back to ``ANTHROPIC_API_KEY``."""
    import anthropic  # lazy — not required for source fetching tests
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Set ANTHROPIC_API_KEY or pass api_key=")
    return anthropic.Anthropic(api_key=key)
