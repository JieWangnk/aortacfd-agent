"""Weekly digest generator — the CLI entry point.

Usage:
    python -m paper_digest.digest \
        --out /path/to/AortaCFD-web/docs/paper-digest \
        --since 7

Pipeline:
  1. Fetch OpenAlex + arXiv for the last N days
  2. Dedupe
  3. Classify each via Claude Haiku (tier + relevance + summary)
  4. Drop relevance < threshold and tier == X
  5. Render weekly page + refresh index
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from .classifier import Classification, TIER_NAMES, classify_paper, make_client
from .sources import Paper, dedupe, fetch_arxiv, fetch_openalex

logger = logging.getLogger("paper_digest")


TIER_ORDER = ["A", "B", "C", "D", "E", "F"]


def render_digest(
    week_start: dt.date,
    papers_with_class: List[Tuple[Paper, Classification]],
    relevance_threshold: float,
    template_dir: Path,
) -> str:
    import jinja2

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("digest.md.j2")

    # Group by tier, sort within tier by relevance desc
    tiers: Dict[str, Dict] = {
        k: {"name": TIER_NAMES[k], "papers": []} for k in TIER_ORDER
    }
    total = 0
    for paper, cls in papers_with_class:
        if cls.tier not in TIER_ORDER:
            continue
        if cls.relevance < relevance_threshold:
            continue
        tiers[cls.tier]["papers"].append({
            "title": paper.title,
            "authors_short": paper.authors_short,
            "venue": paper.venue or paper.source,
            "year": paper.year,
            "doi": paper.doi,
            "doi_url": paper.doi_url,
            "arxiv_id": paper.arxiv_id,
            "arxiv_url": paper.arxiv_url,
            "summary": cls.summary,
            "relevance": cls.why,
            "_rel_score": cls.relevance,
        })
        total += 1

    for t in tiers.values():
        t["papers"].sort(key=lambda p: p.get("_rel_score", 0), reverse=True)

    counts = {k: len(v["papers"]) for k, v in tiers.items()}

    # Pick top 3 across all tiers (by relevance) as "highlights"
    all_classified = sorted(
        [(p, c) for p, c in papers_with_class if c.relevance >= relevance_threshold and c.tier in TIER_ORDER],
        key=lambda pc: pc[1].relevance,
        reverse=True,
    )
    highlights = [
        {
            "title": p.title,
            "authors_short": p.authors_short,
            "venue_short": (p.venue or p.source)[:40],
            "url": p.doi_url or p.arxiv_url or p.url,
        }
        for p, _ in all_classified[:3]
    ]

    return tpl.render(
        week_start=week_start.isoformat(),
        total=total,
        counts=counts,
        tiers=tiers,
        highlights=highlights,
        relevance_threshold=relevance_threshold,
    )


def render_index(docs_dir: Path, template_dir: Path) -> str:
    """Rebuild the archive index from existing digest files."""
    import jinja2
    import re

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("index.md.j2")

    # Pick up existing digest files named YYYY-MM-DD.md
    digest_files = sorted(
        [f for f in docs_dir.glob("*.md") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", f.stem)],
        reverse=True,
    )
    archive = []
    latest = None
    for f in digest_files:
        text = f.read_text()
        total_m = re.search(r"— (\d+) new papers", text)
        total = int(total_m.group(1)) if total_m else 0
        # Find tier with largest count
        top_tier_key = "A"
        top_count = -1
        for tier_key in TIER_ORDER:
            m = re.search(rf"\|\s+\*\*{tier_key}\*\*\s+\|.*?\|\s+(\d+)\s+\|", text)
            if m:
                c = int(m.group(1))
                if c > top_count:
                    top_count = c
                    top_tier_key = tier_key
        entry = {
            "week_start": f.stem,
            "slug": f.stem,
            "total": total,
            "top_tier_name": TIER_NAMES[top_tier_key],
        }
        if latest is None:
            # Short teaser: pull first "Highlights" bullet if present
            hm = re.search(r"## Highlights this week\s*\n\n(- \*\*.+?\*\*[^\n]+)", text)
            entry["teaser"] = hm.group(1) if hm else ""
            latest = entry
        archive.append(entry)

    return tpl.render(latest=latest, archive=archive)


def run(
    out_dir: Path,
    since_days: int = 7,
    relevance_threshold: float = 0.55,
    max_per_source: int = 100,
    dry_run: bool = False,
    api_key: str | None = None,
) -> Path:
    today = dt.date.today()
    start = today - dt.timedelta(days=since_days)
    logger.info("Fetching papers %s → %s", start, today)

    openalex = fetch_openalex(start, today, max_results=max_per_source)
    arxiv = fetch_arxiv(start, today, max_results=max_per_source)
    all_papers = dedupe(openalex + arxiv)
    logger.info("Total unique candidates: %d (OpenAlex %d + arXiv %d)",
                len(all_papers), len(openalex), len(arxiv))

    if dry_run:
        for p in all_papers[:20]:
            print(f"  [{p.source}] {p.title[:100]}")
        return out_dir

    client = make_client(api_key=api_key)
    classified: List[Tuple[Paper, Classification]] = []
    for i, p in enumerate(all_papers, 1):
        logger.info("[%d/%d] classifying: %s", i, len(all_papers), p.title[:80])
        cls = classify_paper(p, client)
        if cls is None:
            continue
        classified.append((p, cls))

    in_scope = [(p, c) for p, c in classified if c.in_scope and c.relevance >= relevance_threshold]
    logger.info("Classifier: %d in-scope / %d classified / %d candidates",
                len(in_scope), len(classified), len(all_papers))

    template_dir = Path(__file__).parent / "templates"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write the dated digest. Use today's publish date (the Monday the cron
    # ran) as both filename and title — readers think in terms of the date
    # they can read it on, not the internal 7-day-window start.
    digest_md = render_digest(today, classified, relevance_threshold, template_dir)
    out_path = out_dir / f"{today.isoformat()}.md"
    out_path.write_text(digest_md)
    logger.info("Wrote %s", out_path)

    # Rebuild the archive index
    index_md = render_index(out_dir, template_dir)
    (out_dir / "index.md").write_text(index_md)
    logger.info("Wrote index.md")

    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True,
                    help="Destination dir (e.g. ../AortaCFD-web/docs/paper-digest)")
    ap.add_argument("--since", type=int, default=7, help="Look back N days (default 7)")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="Relevance threshold 0-1 (default 0.55)")
    ap.add_argument("--max-per-source", type=int, default=100,
                    help="Max candidates per source before classification")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch + dedupe only, no LLM calls, no writes")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run(
        out_dir=args.out,
        since_days=args.since,
        relevance_threshold=args.threshold,
        max_per_source=args.max_per_source,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
