#!/usr/bin/env python3
"""Download open-access PDFs for all DOIs in the corpus.

Uses three free, legal APIs in order:

  1. Unpaywall       — finds author-uploaded / publisher OA versions
  2. OpenAlex        — sometimes has a PDF URL even when Unpaywall doesn't
  3. arXiv metadata  — direct lookup if DOI has arxiv ID hint

Outputs to ``src/aortacfd_agent/corpus/papers/<paper_id>.pdf`` and writes
a report to ``src/aortacfd_agent/corpus/papers/FETCH_REPORT.md``
listing what succeeded and what still needs manual download.

Usage:
    python scripts/fetch_open_access.py \\
        --corpus src/aortacfd_agent/corpus/index/aortacfd_corpus.json \\
        --email your.email@university.edu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


USER_AGENT = "aortacfd-agent/0.1 (research)"


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------


def unpaywall_best_pdf(doi: str, email: str, timeout: float = 12.0) -> Optional[str]:
    """Query Unpaywall. Returns a direct PDF URL, or None."""
    if not doi:
        return None
    url = f"https://api.unpaywall.org/v2/{doi.lower()}?email={urllib.parse.quote(email)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"    unpaywall: {e}", file=sys.stderr)
        return None
    best = data.get("best_oa_location") or {}
    return best.get("url_for_pdf") or best.get("url")


def openalex_pdf_url(doi: str, timeout: float = 12.0) -> Optional[str]:
    """Query OpenAlex works by DOI. Returns a PDF URL if present."""
    if not doi:
        return None
    url = f"https://api.openalex.org/works/doi:{doi.lower()}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    # Prefer OA-hosted PDF
    if data.get("open_access", {}).get("is_oa"):
        loc = data.get("best_oa_location") or {}
        if loc.get("pdf_url"):
            return loc["pdf_url"]
    for loc in data.get("locations", []) or []:
        if loc.get("pdf_url"):
            return loc["pdf_url"]
    return None


def download_pdf(url: str, out_path: Path, timeout: float = 30.0) -> Tuple[bool, str]:
    """Download a PDF. Returns (success, reason)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()
    except Exception as e:
        return False, f"download error: {e}"

    # Sanity: content should be a PDF (or at least binary-ish)
    if not data.startswith(b"%PDF-"):
        if "pdf" not in content_type.lower():
            return False, f"not a PDF (content-type: {content_type or 'unknown'})"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return True, f"{len(data)/1024:.0f} KB"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True, help="Path to aortacfd_corpus.json")
    p.add_argument("--out-dir", type=Path, default=Path("src/aortacfd_agent/corpus/papers"), help="Where to save PDFs")
    p.add_argument("--email", type=str, required=True, help="Your email (required by Unpaywall — they use it only for rate-limit tracking)")
    p.add_argument("--only", nargs="*", help="Only fetch these paper IDs (default: all)")
    p.add_argument("--skip-existing", action="store_true", help="Skip papers whose PDF already exists")
    args = p.parse_args()

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    papers: List[Dict[str, Any]] = corpus.get("papers", [])
    if args.only:
        wanted = set(args.only)
        papers = [p_ for p_ in papers if p_["id"] in wanted]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    succeeded: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for i, paper in enumerate(papers, 1):
        pid = paper["id"]
        doi = paper.get("doi", "").strip()
        title = paper.get("title", "")[:80]
        out = args.out_dir / f"{pid}.pdf"

        print(f"[{i}/{len(papers)}] {pid} — {title}")

        if args.skip_existing and out.exists():
            print("    skip (already exists)")
            succeeded.append({"id": pid, "path": str(out), "source": "cached"})
            continue

        if not doi:
            failed.append({"id": pid, "reason": "no DOI in bib entry"})
            print("    no DOI")
            continue

        # Try Unpaywall
        url = unpaywall_best_pdf(doi, args.email)
        source = "unpaywall"
        if not url:
            url = openalex_pdf_url(doi)
            source = "openalex"
        if not url:
            failed.append({"id": pid, "reason": "no OA location found", "doi": doi})
            print("    no open-access PDF found")
            time.sleep(0.2)
            continue

        ok, msg = download_pdf(url, out)
        if ok:
            succeeded.append({"id": pid, "path": str(out), "source": source, "size": msg})
            print(f"    OK  ({source}, {msg})")
        else:
            failed.append({"id": pid, "reason": msg, "doi": doi, "url": url})
            print(f"    FAIL ({msg})")

        time.sleep(0.3)  # polite rate limit

    # Write report
    report_path = args.out_dir / "FETCH_REPORT.md"
    _write_report(report_path, succeeded, failed, papers)
    print(f"\n✅ {len(succeeded)} succeeded · ❌ {len(failed)} failed")
    print(f"Report: {report_path}")

    return 0


def _write_report(path: Path, succeeded: List[Dict], failed: List[Dict], all_papers: List[Dict]):
    paper_by_id = {p["id"]: p for p in all_papers}
    lines = [
        "# Corpus PDF fetch report",
        "",
        f"Total requested: **{len(succeeded) + len(failed)}**",
        f"Succeeded (auto-downloaded): **{len(succeeded)}**",
        f"Failed (manual download needed): **{len(failed)}**",
        "",
        "## Downloaded automatically",
        "",
    ]
    for s in succeeded:
        meta = paper_by_id.get(s["id"], {})
        lines.append(f"- **{s['id']}** — {meta.get('authors', '')} {meta.get('year', '')} · {meta.get('title', '')[:80]} · *{s.get('source', '?')}*")

    lines += [
        "",
        "## Need manual download",
        "",
        "Click each DOI below (via your institutional login) and save the PDF as",
        f"`src/aortacfd_agent/corpus/papers/<id>.pdf`.",
        "",
    ]
    for f in failed:
        meta = paper_by_id.get(f["id"], {})
        doi = meta.get("doi", "")
        authors = meta.get("authors", "")
        year = meta.get("year", "")
        title = meta.get("title", "")[:80]
        doi_link = f"https://doi.org/{doi}" if doi else ""
        lines.append(f"- **{f['id']}** — {authors} {year} · {title}")
        if doi_link:
            lines.append(f"    - DOI: {doi_link}")
        lines.append(f"    - Save as: `src/aortacfd_agent/corpus/papers/{f['id']}.pdf`")
        lines.append(f"    - Reason: {f.get('reason', 'unknown')}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
