"""RAG glue between the paper digest and the user's existing bibliography.

Two features:

  1. *Overlap flag* — "Already in your bibliography" when the new paper's
     DOI matches one in ``aortacfd_corpus.json``.
  2. *Related reading* — BM25-search the new paper's abstract against
     ``BibCorpusStore`` and return the top-N related passages, skipping
     self-matches (same DOI).

Kept in a separate module so ``digest.py`` stays focused on orchestration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .sources import Paper

logger = logging.getLogger(__name__)


@dataclass
class RelatedHit:
    paper_id: str
    title: str
    authors_short: str
    year: Optional[int]
    page: Optional[int]
    doi: Optional[str]
    score: float


@dataclass
class CorpusContext:
    """Bundles the DOI index and the BM25 store so callers do one load."""

    doi_to_paper: Dict[str, Dict]   # lowercase DOI → raw paper dict from corpus JSON
    store: object                   # BibCorpusStore; typed as object to keep import lazy

    def is_in_bibliography(self, paper: Paper) -> bool:
        if not paper.doi:
            return False
        return paper.doi.lower() in self.doi_to_paper

    def related(
        self,
        paper: Paper,
        top_k: int = 2,
        min_score: float = 3.0,
    ) -> List[RelatedHit]:
        """Return up to ``top_k`` corpus chunks most related to this abstract.

        - Skips the paper itself (by DOI match) so a paper doesn't recommend
          its own bibliography entry.
        - Drops hits with BM25 score below ``min_score`` (relevance floor;
          otherwise every paper gets a weak match to the most-common corpus
          entry).
        """
        query = f"{paper.title}. {paper.abstract}"
        if not query.strip():
            return []

        self_doi = (paper.doi or "").lower()
        hits = []
        # Ask for more candidates than we'll return — we filter and dedupe
        candidates = self.store.search(query, top_k=max(top_k * 3, 6))
        seen_papers: set[str] = set()
        for chunk in candidates:
            if chunk.score < min_score:
                break
            pid = chunk.paper
            if pid in seen_papers:
                continue
            meta_doi = (chunk.metadata.get("doi") or "").lower()
            if self_doi and meta_doi == self_doi:
                continue
            seen_papers.add(pid)
            hits.append(RelatedHit(
                paper_id=pid,
                title=_unlatex(chunk.metadata.get("title", pid)),
                authors_short=_authors_short(chunk.metadata.get("authors", "")),
                year=chunk.metadata.get("year"),
                page=chunk.page,
                doi=chunk.metadata.get("doi") or None,
                score=chunk.score,
            ))
            if len(hits) >= top_k:
                break
        return hits


def _unlatex(s: str) -> str:
    """Undo the most common BibTeX diacritic escapes from the corpus JSON.

    The bib store serves authors verbatim from ``references.bib``, which
    uses LaTeX syntax like ``B{\"u}chner`` and ``Sa{\\~n}a``. Render these
    as their actual Unicode characters so the digest doesn't leak markup.
    """
    if not s:
        return s
    # Accented letters: {\"u} → ü, etc. Handle the small set that actually
    # appears in the current bibliography.
    replacements = {
        '{\\"a}': "ä", '{\\"o}': "ö", '{\\"u}': "ü",
        '{\\"A}': "Ä", '{\\"O}': "Ö", '{\\"U}': "Ü",
        "{\\'a}": "á", "{\\'e}": "é", "{\\'i}": "í",
        "{\\'o}": "ó", "{\\'u}": "ú",
        "{\\^a}": "â", "{\\^e}": "ê", "{\\^i}": "î",
        "{\\^o}": "ô", "{\\^u}": "û",
        "{\\~n}": "ñ", "{\\~a}": "ã", "{\\~o}": "õ",
        "{\\c c}": "ç", "{\\c C}": "Ç",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # Strip any remaining bare braces
    return s.replace("{", "").replace("}", "")


def _authors_short(raw: str) -> str:
    """Compact author string: 'Smith, Lee, Patel' → 'Smith et al.'"""
    raw = _unlatex(raw)
    if not raw:
        return ""
    parts = [p.strip() for p in raw.replace(" and ", ",").split(",") if p.strip()]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    if len(parts) == 2:
        return f"{parts[0]} & {parts[1]}"
    return f"{parts[0]} et al."


def load_context() -> Optional[CorpusContext]:
    """Load the bundled AortaCFD corpus. Returns ``None`` if unavailable.

    We import ``aortacfd_agent.corpus.bib_store`` lazily so ``paper_digest``
    can still be imported in environments where the main package isn't
    installed (the GitHub Action does install it now; this is defensive).
    """
    try:
        from aortacfd_agent.corpus.bib_store import load_default
    except Exception as e:
        logger.warning("Corpus unavailable (aortacfd_agent not installed): %s", e)
        return None

    try:
        store = load_default()
    except FileNotFoundError as e:
        logger.warning("Corpus file missing: %s", e)
        return None

    doi_to_paper: Dict[str, Dict] = {}
    for p in store.papers:
        doi = (p.get("doi") or "").strip().lower()
        if doi:
            doi_to_paper[doi] = p

    logger.info(
        "Loaded corpus context: %d papers, %d with DOI, %d searchable units",
        len(store.papers), len(doi_to_paper), len(store._units),
    )
    return CorpusContext(doi_to_paper=doi_to_paper, store=store)
