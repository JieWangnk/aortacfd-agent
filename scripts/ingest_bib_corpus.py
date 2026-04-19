#!/usr/bin/env python3
"""Build a searchable corpus from a BibTeX bibliography.

Reads references.bib, fetches abstracts from OpenAlex (free, no API key),
and writes a JSON corpus ready for BibCorpusStore at runtime.

Usage:
    python scripts/ingest_bib_corpus.py path/to/references.bib \\
        --out src/aortacfd_agent/corpus/index/aortacfd_corpus.json

Output JSON structure:
    {
      "version": 1,
      "source": "references.bib",
      "papers": [
        {
          "id": "steinman2013",
          "authors": "Steinman, Hoi, Fahy, et al.",
          "year": 2013,
          "title": "Variability of CFD solutions for pressure and flow...",
          "journal": "Journal of Biomechanical Engineering",
          "doi": "10.1115/1.4023382",
          "abstract": "...",
        },
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# BibTeX parser (dependency-free, sufficient for our bib format)
# ---------------------------------------------------------------------------


ENTRY_RE = re.compile(r"^@(\w+)\s*\{([^,]+),\s*$", re.MULTILINE)
FIELD_RE = re.compile(r"^\s*(\w+)\s*=\s*\{(.*)\}(?:,)?\s*$", re.MULTILINE)


def parse_bib(text: str) -> List[Dict[str, Any]]:
    """Very small BibTeX parser. Handles the subset our bib file uses."""
    entries: List[Dict[str, Any]] = []

    # Split on @type{ key, and then recover each block
    # Simpler: iterate line-by-line with a state machine
    current: Optional[Dict[str, Any]] = None
    depth = 0
    buffer = ""
    key_buffer = ""
    in_field = False

    for line in text.splitlines():
        stripped = line.strip()

        if current is None:
            m = re.match(r"^@(\w+)\s*\{\s*([^,\s]+)\s*,", stripped)
            if m:
                current = {
                    "_type": m.group(1).lower(),
                    "_id": m.group(2),
                }
            continue

        # End of entry?
        if stripped == "}" and not in_field:
            entries.append(current)
            current = None
            continue

        m = re.match(r"^\s*(\w+)\s*=\s*\{(.*)$", line)
        if m:
            key_buffer = m.group(1).lower()
            rest = m.group(2)
            # Count braces
            open_count = rest.count("{") + 1
            close_count = rest.count("}")
            if open_count == close_count:
                # Complete single-line field
                current[key_buffer] = rest.rstrip("},").rstrip("}")
                key_buffer = ""
                in_field = False
            else:
                buffer = rest
                in_field = True
            continue

        if in_field:
            buffer += "\n" + line.rstrip()
            if buffer.count("{") + 1 <= buffer.count("}"):
                # Field closed
                val = buffer.rstrip("},").rstrip("}").rstrip("},")
                current[key_buffer] = val
                buffer = ""
                key_buffer = ""
                in_field = False

    return entries


def shorten_authors(author_field: str) -> str:
    """Turn 'Steinman, David A. and Hoi, ...' into a short 'Steinman, Hoi et al.'"""
    names = [n.strip() for n in author_field.split(" and ")]
    surnames = []
    for n in names:
        if "," in n:
            surnames.append(n.split(",")[0].strip())
        else:
            surnames.append(n.split()[-1])
    if len(surnames) == 0:
        return author_field
    if len(surnames) == 1:
        return surnames[0]
    if len(surnames) == 2:
        return f"{surnames[0]} & {surnames[1]}"
    return f"{surnames[0]} et al."


# ---------------------------------------------------------------------------
# OpenAlex abstract fetcher
# ---------------------------------------------------------------------------


OPENALEX_BASE = "https://api.openalex.org/works"
USER_AGENT = "aortacfd-agent/0.1 (research; jieandwang@gmail.com)"


def fetch_abstract_by_doi(doi: str, timeout: float = 12.0) -> Optional[str]:
    """Query OpenAlex for a paper by DOI and return its reconstructed abstract."""
    if not doi:
        return None
    clean_doi = doi.strip().lower()
    if not clean_doi.startswith("10."):
        return None
    url = f"{OPENALEX_BASE}/doi:{clean_doi}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"  ! OpenAlex failed for {doi}: {exc}", file=sys.stderr)
        return None

    # OpenAlex returns abstract as an inverted index
    inv = payload.get("abstract_inverted_index")
    if not inv:
        return None

    positions: List[tuple[int, str]] = []
    for word, positions_list in inv.items():
        for pos in positions_list:
            positions.append((pos, word))
    positions.sort()
    return " ".join(w for _, w in positions)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bib", type=Path, help="Path to references.bib")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/aortacfd_agent/corpus/index/aortacfd_corpus.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--skip-abstracts",
        action="store_true",
        help="Skip fetching abstracts from OpenAlex (faster; title-only corpus)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process this many entries (for testing)",
    )
    args = parser.parse_args()

    if not args.bib.exists():
        print(f"error: {args.bib} not found", file=sys.stderr)
        return 2

    print(f"Parsing {args.bib} ...")
    text = args.bib.read_text(encoding="utf-8")
    entries = parse_bib(text)
    print(f"  parsed {len(entries)} entries")

    if args.limit:
        entries = entries[: args.limit]
        print(f"  (limiting to {args.limit} for test run)")

    papers: List[Dict[str, Any]] = []
    for i, e in enumerate(entries, 1):
        paper = {
            "id": e.get("_id", f"paper_{i}"),
            "authors": shorten_authors(e.get("author", "")),
            "authors_full": e.get("author", ""),
            "year": _int_or_none(e.get("year")),
            "title": _clean_braces(e.get("title", "")),
            "journal": e.get("journal") or e.get("booktitle") or e.get("publisher") or "",
            "doi": e.get("doi", "").strip(),
            "type": e.get("_type", "article"),
        }

        if args.skip_abstracts or not paper["doi"]:
            paper["abstract"] = ""
        else:
            print(f"  [{i}/{len(entries)}] {paper['id']} ...", end=" ", flush=True)
            abstract = fetch_abstract_by_doi(paper["doi"])
            paper["abstract"] = abstract or ""
            print("ok" if abstract else "no abstract")
            time.sleep(0.15)  # be polite to OpenAlex

        papers.append(paper)

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    corpus = {
        "version": 1,
        "source": str(args.bib.name),
        "n_papers": len(papers),
        "n_with_abstracts": sum(1 for p in papers if p["abstract"]),
        "papers": papers,
    }
    args.out.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"\nWrote {len(papers)} papers "
        f"({corpus['n_with_abstracts']} with abstracts) to {args.out}"
    )
    return 0


def _int_or_none(s: Any) -> Optional[int]:
    if s is None:
        return None
    try:
        return int(str(s).strip().rstrip("}"))
    except (ValueError, AttributeError):
        return None


def _clean_braces(s: str) -> str:
    # Remove LaTeX-style protection braces from titles like {CFD}
    return re.sub(r"[{}]", "", s).strip()


if __name__ == "__main__":
    raise SystemExit(main())
