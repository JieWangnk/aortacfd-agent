"""Weekly paper digest for cardiovascular CFD + PINN research.

Pipeline: OpenAlex + arXiv → dedupe → Claude classifier (tier + summary) →
Jinja2 markdown → committed to AortaCFD-web under ``docs/paper-digest/``.
"""

from .sources import Paper, fetch_openalex, fetch_arxiv
from .classifier import classify_paper, TIER_NAMES

__all__ = [
    "Paper",
    "fetch_openalex",
    "fetch_arxiv",
    "classify_paper",
    "TIER_NAMES",
]
