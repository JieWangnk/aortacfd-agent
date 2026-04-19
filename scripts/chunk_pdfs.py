#!/usr/bin/env python3
"""Chunk PDFs into short passages for RAG, and update the corpus JSON.

For each PDF in corpus/papers/:
  1. Extract text with pypdf
  2. Split into ~500-character passages with page numbers
  3. Keep passages that look like English prose (reject tables, references)

Updates ``aortacfd_corpus.json`` in place: adds a ``chunks`` list per paper.

Chunks are short (200-500 chars each) and serve as search results; this is
fair-use transformative indexing (like Google Scholar), not redistribution.

Usage:
    pip install pypdf
    python scripts/chunk_pdfs.py \\
        --corpus src/aortacfd_agent/corpus/index/aortacfd_corpus.json \\
        --papers src/aortacfd_agent/corpus/papers
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


def extract_pdf_text_by_page(pdf_path: Path) -> List[str]:
    """Return a list of page texts."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("error: pypdf not installed. run: pip install pypdf", file=sys.stderr)
        sys.exit(2)

    try:
        reader = PdfReader(str(pdf_path))
        return [page.extract_text() or "" for page in reader.pages]
    except Exception as e:
        print(f"    could not read {pdf_path.name}: {e}", file=sys.stderr)
        return []


# Characters that appear almost exclusively in tables / formulas / references
_NOISE_CHARS = set("#|_*<>~^@")


def _looks_like_prose(text: str) -> bool:
    """Heuristic: filter out gibberish (math, reference lists, figures)."""
    if len(text) < 80:
        return False
    # English prose has lots of spaces and letters
    letter_frac = sum(1 for c in text if c.isalpha() or c == " ") / len(text)
    if letter_frac < 0.75:
        return False
    # Bulk of noise chars → probably math/diagram
    noise = sum(1 for c in text if c in _NOISE_CHARS)
    if noise / len(text) > 0.02:
        return False
    # Must contain at least a few periods (sentence structure)
    if text.count(".") < 2:
        return False
    return True


def chunk_page(text: str, target_chars: int = 400, overlap: int = 80) -> List[str]:
    """Split a page's text into ~target_chars passages, breaking on sentences."""
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Split on sentence boundaries (simple but effective)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    chunks: List[str] = []
    buf = ""
    for sent in sentences:
        if len(buf) + len(sent) <= target_chars:
            buf = (buf + " " + sent).strip() if buf else sent
        else:
            if buf:
                chunks.append(buf)
            # Overlap: carry the last sentence of buf into the next chunk
            buf = sent
    if buf:
        chunks.append(buf)

    # Keep only prose-y chunks
    return [c for c in chunks if _looks_like_prose(c)]


def chunk_pdf(pdf_path: Path, paper_id: str, max_chunks: int = 30) -> List[Dict[str, Any]]:
    """Extract a small number of representative chunks from one PDF."""
    pages = extract_pdf_text_by_page(pdf_path)
    if not pages:
        return []

    # Skip the first page (title/abstract) since we already have the abstract.
    # Skip references / appendices (heuristic: very last ~20% of pages).
    n_pages = len(pages)
    start = 1 if n_pages > 3 else 0
    end = int(n_pages * 0.8) if n_pages > 6 else n_pages

    all_chunks: List[Dict[str, Any]] = []
    for page_idx in range(start, end):
        for chunk_text in chunk_page(pages[page_idx]):
            all_chunks.append({
                "text": chunk_text,
                "page": page_idx + 1,  # 1-indexed for humans
            })

    # Downsample: keep up to max_chunks per paper, spread across pages
    if len(all_chunks) > max_chunks:
        step = len(all_chunks) / max_chunks
        all_chunks = [all_chunks[int(i * step)] for i in range(max_chunks)]

    return all_chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--papers", type=Path, required=True)
    ap.add_argument("--max-chunks", type=int, default=25, help="Max chunks per paper (default 25)")
    ap.add_argument("--only", nargs="*", help="Only chunk these paper IDs")
    args = ap.parse_args()

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    papers: List[Dict[str, Any]] = corpus.get("papers", [])
    pdfs = {p.stem: p for p in args.papers.glob("*.pdf")}

    wanted_ids = set(args.only) if args.only else None

    total_chunks = 0
    with_fulltext = 0
    for paper in papers:
        pid = paper["id"]
        if wanted_ids and pid not in wanted_ids:
            continue
        pdf = pdfs.get(pid)
        if pdf is None:
            paper.pop("chunks", None)
            continue

        chunks = chunk_pdf(pdf, pid, max_chunks=args.max_chunks)
        if chunks:
            paper["chunks"] = chunks
            total_chunks += len(chunks)
            with_fulltext += 1
            print(f"  {pid}: {len(chunks):3d} chunks from {pdf.name}")
        else:
            paper.pop("chunks", None)

    # Save back
    corpus["n_with_fulltext"] = with_fulltext
    corpus["n_total_chunks"] = total_chunks
    args.corpus.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"Chunked {with_fulltext} papers, {total_chunks} total chunks.")
    print(f"Updated: {args.corpus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
