#!/usr/bin/env python3
"""Build the RAG corpus from PDFs under src/aortacfd_agent/corpus/papers/.

This script uses ``chromadb`` and ``sentence-transformers``, which are in
the ``rag`` optional-dependency group. Install with::

    pip install -e '.[rag]'

After installation, drop the open-access PDFs listed in
``src/aortacfd_agent/corpus/papers/README.md`` into that directory and
run::

    python scripts/ingest_corpus.py

The script walks the papers directory, chunks each page into ~500-token
passages, embeds them with a small MiniLM model, and writes the result
to ``src/aortacfd_agent/corpus/index/`` as a Chroma persistent store.
The LiteratureAgent can then be pointed at that directory via
:class:`~aortacfd_agent.corpus.store.ChromaCorpusStore`.

Re-running the script wipes the existing collection and rebuilds from
scratch, which is the simplest behaviour for a small seed corpus.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the AortaCFD agent RAG corpus from seed PDFs."
    )
    parser.add_argument(
        "--papers",
        type=Path,
        default=_repo_root() / "src" / "aortacfd_agent" / "corpus" / "papers",
        help="Directory containing open-access PDFs (default: corpus/papers/).",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=_repo_root() / "src" / "aortacfd_agent" / "corpus" / "index",
        help="Output directory for the Chroma persistent store.",
    )
    parser.add_argument(
        "--collection",
        default="aortacfd_corpus",
        help="Chroma collection name (default: aortacfd_corpus).",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model id for sentence-transformers embeddings.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=500,
        help="Target tokens per chunk (default: 500).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print INFO-level progress messages.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    # Make 'aortacfd_agent' importable without `pip install -e` by adding src/.
    repo_src = _repo_root() / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    from aortacfd_agent.corpus.ingest import build_chroma_index  # noqa: E402

    if not args.papers.exists():
        print(f"error: papers directory does not exist: {args.papers}", file=sys.stderr)
        return 2

    pdfs = sorted(p for p in args.papers.glob("*.pdf"))
    if not pdfs:
        print(
            f"error: no PDFs found in {args.papers}. See the corpus README for "
            "the list of papers to fetch.",
            file=sys.stderr,
        )
        return 3

    print(f"Found {len(pdfs)} PDFs in {args.papers}")
    args.index.mkdir(parents=True, exist_ok=True)

    count = build_chroma_index(
        papers_dir=args.papers,
        index_dir=args.index,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        target_tokens=args.chunk_tokens,
    )

    print(f"Wrote {count} chunks to {args.index} (collection: {args.collection})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
