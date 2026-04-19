"""BibTeX-backed corpus store with BM25 keyword search.

Loads a pre-ingested JSON corpus (built by ``scripts/ingest_bib_corpus.py``)
that contains paper metadata + OpenAlex abstracts. Queries are answered
with BM25 over the concatenated title+abstract text.

No embeddings, no external API calls at query time, no heavy dependencies.
Suitable for Streamlit Cloud deployment.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .store import Chunk


# ---------------------------------------------------------------------------
# Minimal BM25 (Okapi BM25) implementation — no rank-bm25 dependency
# ---------------------------------------------------------------------------


class BM25:
    """Bare-bones Okapi BM25 scorer suitable for a ~100-doc corpus."""

    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.n_docs = len(docs)
        self.doc_lens = [len(d) for d in docs]
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.n_docs)
        self.doc_freq: Dict[str, int] = {}
        for doc in docs:
            for term in set(doc):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
        # Pre-compute idf
        self.idf: Dict[str, float] = {}
        for term, df in self.doc_freq.items():
            # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            import math
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: List[str], doc_idx: int) -> float:
        doc = self.docs[doc_idx]
        dl = self.doc_lens[doc_idx]
        tf: Dict[str, int] = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1

        s = 0.0
        for term in query:
            if term not in tf:
                continue
            idf = self.idf.get(term, 0.0)
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avg_doc_len)
            s += idf * (freq * (self.k1 + 1)) / denom
        return s

    def search(self, query: List[str], top_k: int) -> List[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


_STOPWORDS = set(
    """
    a an the and or of for in on at to with by from as is are was were be been
    this that these those it its it's we our us they them their there here
    which who whom whose what when where why how can could should would will
    may might must shall not no nor so than too very s t all each any both some
    such only than if then else into about above below through during over under
    between against among before after beyond within without because while although
    """.split()
)


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, filter stopwords, keep tokens of length ≥ 2."""
    text = text.lower()
    # Keep only letters, digits, hyphens, periods (for Re, K_t, etc.)
    text = re.sub(r"[^a-z0-9\-._]", " ", text)
    tokens = [t.strip("-._") for t in text.split() if len(t) >= 2]
    return [t for t in tokens if t and t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class BibCorpusStore:
    """A paper-backed corpus that responds to natural-language queries with BM25 ranking."""

    name = "bib_corpus"

    def __init__(self, corpus_path: Path):
        self.corpus_path = Path(corpus_path)
        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus file not found: {self.corpus_path}. "
                f"Run scripts/ingest_bib_corpus.py to build it."
            )
        data = json.loads(self.corpus_path.read_text(encoding="utf-8"))
        self.papers: List[Dict[str, Any]] = data.get("papers", [])

        # Build search corpus: title + abstract for each paper
        self._texts: List[str] = []
        self._tokenized: List[List[str]] = []
        for p in self.papers:
            text = f"{p.get('title', '')}. {p.get('abstract', '')}"
            self._texts.append(text)
            self._tokenized.append(_tokenize(text))

        self._bm25 = BM25(self._tokenized)

    # ------------------------------------------------------------------
    # CorpusStore protocol
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Return the top-k papers most relevant to the query."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        hits = self._bm25.search(query_tokens, top_k=max(top_k, 1))

        chunks: List[Chunk] = []
        for idx, score in hits:
            if score <= 0:
                continue
            p = self.papers[idx]
            title = p.get("title", "")
            abstract = p.get("abstract", "")
            snippet = self._best_snippet(abstract or title, query_tokens)
            chunks.append(
                Chunk(
                    text=snippet,
                    paper=p.get("id", "unknown"),
                    page=None,
                    score=float(score),
                    metadata={
                        "authors": p.get("authors", ""),
                        "year": p.get("year"),
                        "title": title,
                        "journal": p.get("journal", ""),
                        "doi": p.get("doi", ""),
                    },
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _best_snippet(self, text: str, query_tokens: List[str], window: int = 280) -> str:
        """Return a short passage from `text` centred on the densest query match."""
        if not text:
            return ""
        if len(text) <= window:
            return text.strip()

        lower = text.lower()
        # Find positions of each query token
        positions: List[int] = []
        for tok in query_tokens:
            for m in re.finditer(re.escape(tok), lower):
                positions.append(m.start())
        if not positions:
            return text[:window].strip() + " ..."

        # Pick the position with most query tokens nearby
        positions.sort()
        best_center = positions[0]
        best_count = 0
        for pivot in positions:
            count = sum(1 for p in positions if abs(p - pivot) < window)
            if count > best_count:
                best_count = count
                best_center = pivot

        half = window // 2
        start = max(0, best_center - half)
        end = min(len(text), start + window)
        snippet = text[start:end]
        if start > 0:
            snippet = "... " + snippet
        if end < len(text):
            snippet = snippet + " ..."
        return snippet.strip()

    # Some convenience for agent-side rendering
    def get_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        for p in self.papers:
            if p.get("id") == paper_id:
                return p
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_default() -> BibCorpusStore:
    """Load the bundled aortacfd corpus from the package."""
    here = Path(__file__).resolve().parent
    return BibCorpusStore(here / "index" / "aortacfd_corpus.json")
