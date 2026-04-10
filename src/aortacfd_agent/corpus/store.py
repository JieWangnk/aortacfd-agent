"""Vector store abstraction for the LiteratureAgent RAG pipeline.

Two backends are supported:

* :class:`FakeCorpusStore` — an in-memory, dependency-free keyword matcher
  used by tests and offline demos. Deterministic, no embeddings, no disk.
* :class:`ChromaCorpusStore` — a thin wrapper around ``chromadb`` for real
  retrieval. Imported lazily so it is only required when you actually use
  it (see the ``rag`` optional-deps group in ``pyproject.toml``).

Both implement the :class:`CorpusStore` protocol: one ``search`` method
that takes a query string and returns the top-k most relevant
:class:`Chunk` records. Every chunk carries the metadata needed to cite
it in a ``ParameterJustification`` (paper id, page, verbatim text).

The dataclass and protocol have zero third-party imports, so this
module is always importable — the SDK-heavy bits only load when you
call the Chroma backend explicitly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """One retrieval unit: a verbatim passage plus metadata for citation.

    Attributes
    ----------
    text
        The verbatim chunk of paper text. Short is better for citation
        quoting — 200-500 characters works well.
    paper
        Citation key (e.g. ``"Steinman2013"``, ``"Esmaily2011"``).
    page
        Page number in the original PDF, or a section label. Optional.
    score
        Retrieval score the store returned for this chunk. Higher is
        better. Exact meaning depends on the store (keyword overlap for
        FakeCorpusStore, embedding cosine similarity for Chroma).
    metadata
        Any extra key-value pairs the store wants to ship along.
    """

    text: str
    paper: str
    page: Optional[Any] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "paper": self.paper,
            "page": self.page,
            "score": self.score,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class CorpusStore(Protocol):
    """Minimal protocol every corpus backend must implement."""

    name: str

    def search(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Return up to ``top_k`` chunks most relevant to ``query``."""
        ...


# ---------------------------------------------------------------------------
# FakeCorpusStore — in-memory keyword matcher for tests and demos
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")


def _tokenise(text: str) -> List[str]:
    """Lowercase word tokens for keyword matching."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


class FakeCorpusStore:
    """Deterministic keyword-overlap corpus store.

    Takes a list of :class:`Chunk` records at construction time and
    returns the top-k chunks whose token overlap with the query is
    largest. If multiple chunks tie, the original insertion order is
    preserved — this keeps tests deterministic even when the scoring
    degenerates.

    The score is ``matching_tokens / max(1, len(query_tokens))`` which
    puts it on a 0..1 scale and makes it easy to reason about in tests.
    """

    name = "fake_corpus"

    def __init__(self, chunks: Iterable[Chunk]):
        self.chunks: List[Chunk] = list(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Chunk]:
        if not query.strip() or not self.chunks:
            return []

        query_tokens = set(_tokenise(query))
        if not query_tokens:
            return []

        scored: List[tuple[float, int, Chunk]] = []
        for idx, chunk in enumerate(self.chunks):
            tokens = set(_tokenise(chunk.text))
            overlap = len(query_tokens & tokens)
            if overlap == 0:
                continue
            score = overlap / max(1, len(query_tokens))
            scored.append((score, idx, chunk))

        # Sort by score desc, then insertion order asc for stability.
        scored.sort(key=lambda triple: (-triple[0], triple[1]))
        top = scored[: max(0, int(top_k))]

        # Clone the chunks so the caller can mutate them freely.
        return [
            Chunk(
                text=c.text,
                paper=c.paper,
                page=c.page,
                score=round(score, 4),
                metadata=dict(c.metadata),
            )
            for score, _, c in top
        ]

    # -- convenience for tests ----------------------------------------------

    @classmethod
    def from_json(cls, path: Path) -> "FakeCorpusStore":
        """Load chunks from a JSON file with shape ``[{"text": ..., ...}, ...]``."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        chunks: List[Chunk] = []
        for entry in data:
            chunks.append(
                Chunk(
                    text=entry["text"],
                    paper=entry["paper"],
                    page=entry.get("page"),
                    metadata=entry.get("metadata", {}),
                )
            )
        return cls(chunks=chunks)


# ---------------------------------------------------------------------------
# ChromaCorpusStore — production backend, lazy import
# ---------------------------------------------------------------------------


class ChromaCorpusStore:
    """Persistent vector store backed by ``chromadb``.

    ``chromadb`` is an optional dependency — install it via::

        pip install -e '.[rag]'

    The import is deferred to :meth:`__init__` so users who stick with
    the fake store never pay the ``chromadb`` startup cost.
    """

    name = "chroma"

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "aortacfd_corpus",
        embedding_function: Optional[Any] = None,
    ):
        try:
            import chromadb  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ChromaCorpusStore requires chromadb. Install it via "
                "`pip install -e '.[rag]'`."
            ) from exc

        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )

    def search(self, query: str, top_k: int = 5) -> List[Chunk]:
        result = self._collection.query(query_texts=[query], n_results=top_k)
        chunks: List[Chunk] = []

        docs_list = (result.get("documents") or [[]])[0]
        metas_list = (result.get("metadatas") or [[]])[0]
        dists_list = (result.get("distances") or [[]])[0]

        for doc, meta, dist in zip(docs_list, metas_list, dists_list):
            # Cosine distance → similarity score in [0, 1] (rough).
            score = max(0.0, 1.0 - float(dist))
            chunks.append(
                Chunk(
                    text=doc,
                    paper=(meta or {}).get("paper", "unknown"),
                    page=(meta or {}).get("page"),
                    score=round(score, 4),
                    metadata=dict(meta or {}),
                )
            )
        return chunks
