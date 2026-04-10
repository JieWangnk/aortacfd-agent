"""PDF → chunks → Chroma vector store.

This module is only used when building the production RAG corpus. Tests
and dry-runs use :class:`FakeCorpusStore` instead and never import this
file, so the heavy optional dependencies (``pypdf``,
``sentence-transformers``, ``chromadb``) stay optional.

Usage::

    python scripts/ingest_corpus.py \\
        --papers src/aortacfd_agent/corpus/papers/ \\
        --index src/aortacfd_agent/corpus/index/

Each PDF in ``papers/`` becomes a set of ~500-token chunks. Chunk metadata
includes the source filename, page number, and a derived paper citation
key (filename without the extension, uppercased prefix).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunking — deterministic, dependency-free
# ---------------------------------------------------------------------------


@dataclass
class RawChunk:
    """One raw chunk before it lands in the vector store."""

    paper: str
    page: int
    text: str


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def chunk_page(text: str, target_tokens: int = 500) -> List[str]:
    """Split one page of text into ~target_tokens chunks by sentence.

    Uses a cheap sentence splitter; tokens are approximated as whitespace
    words. Good enough for the seed corpus. Swap for a real sentence
    tokenizer if retrieval quality matters more than setup simplicity.
    """
    if not text.strip():
        return []

    sentences = _SENT_RE.split(text.strip())
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = len(sent.split())
        if current_tokens + sent_tokens > target_tokens and current:
            chunks.append(" ".join(current).strip())
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c]


def derive_paper_key(filename: str) -> str:
    """Filename → citation key. ``Steinman2013_challenge.pdf`` → ``Steinman2013``."""
    stem = Path(filename).stem
    match = re.match(r"([A-Za-z]+\d{4})", stem)
    if match:
        return match.group(1)
    return stem


# ---------------------------------------------------------------------------
# PDF loading (optional pypdf)
# ---------------------------------------------------------------------------


def load_pdf_pages(path: Path) -> List[str]:
    """Return one string per page. Returns empty list if pypdf is missing."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pypdf is required for PDF ingestion. Install it via "
            "`pip install -e '.[rag]'`."
        ) from exc

    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to extract text from a page of %s: %s", path.name, exc)
            pages.append("")
    return pages


def ingest_directory(papers_dir: Path, target_tokens: int = 500) -> Iterable[RawChunk]:
    """Walk every PDF in ``papers_dir`` and yield :class:`RawChunk` records."""
    papers_dir = Path(papers_dir)
    pdfs = sorted(p for p in papers_dir.glob("*.pdf") if p.is_file())
    if not pdfs:
        logger.warning("no PDFs found in %s", papers_dir)
        return

    for pdf_path in pdfs:
        paper_key = derive_paper_key(pdf_path.name)
        pages = load_pdf_pages(pdf_path)
        for page_num, page_text in enumerate(pages, start=1):
            for chunk_text in chunk_page(page_text, target_tokens=target_tokens):
                yield RawChunk(paper=paper_key, page=page_num, text=chunk_text)


# ---------------------------------------------------------------------------
# Chroma persistence (optional chromadb + sentence-transformers)
# ---------------------------------------------------------------------------


def build_chroma_index(
    papers_dir: Path,
    index_dir: Path,
    collection_name: str = "aortacfd_corpus",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    target_tokens: int = 500,
) -> int:
    """Read every PDF under ``papers_dir`` and persist chunks to Chroma.

    Returns the number of chunks written. This is a one-shot script —
    re-running it wipes the existing collection and rebuilds from scratch,
    which is the simplest behaviour for a small seed corpus.
    """
    try:
        import chromadb  # type: ignore
        from chromadb.utils import embedding_functions  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "chromadb is required for indexing. Install it via "
            "`pip install -e '.[rag]'`."
        ) from exc

    client = chromadb.PersistentClient(path=str(index_dir))

    # Wipe any previous collection so the index is reproducible.
    try:
        client.delete_collection(collection_name)
    except Exception:  # noqa: BLE001
        pass

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model,
    )
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )

    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []
    for idx, chunk in enumerate(ingest_directory(papers_dir, target_tokens=target_tokens)):
        ids.append(f"{chunk.paper}_p{chunk.page}_{idx}")
        docs.append(chunk.text)
        metas.append({"paper": chunk.paper, "page": chunk.page})

    if not ids:
        logger.warning("no chunks produced; nothing written to %s", index_dir)
        return 0

    collection.add(ids=ids, documents=docs, metadatas=metas)
    logger.info("wrote %d chunks to %s / %s", len(ids), index_dir, collection_name)
    return len(ids)
