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
import email.utils
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .classifier import Classification, TIER_NAMES, classify_paper, make_client
from .corpus import CorpusContext, load_context
from .sources import Paper, dedupe, fetch_arxiv, fetch_openalex

logger = logging.getLogger("paper_digest")


TIER_ORDER = ["A", "B", "C", "D", "E", "F"]


def _format_related_line(r) -> str:
    """Render one related-reading hit as a markdown bullet fragment."""
    year = r.year or "—"
    page = f" (p. {r.page})" if r.page else ""
    doi = f" — [DOI](https://doi.org/{r.doi})" if r.doi else ""
    return f"*{r.authors_short} ({year}) · {r.title}*{page}{doi}"


def render_digest(
    week_start: dt.date,
    papers_with_class: List[Tuple[Paper, Classification]],
    relevance_threshold: float,
    template_dir: Path,
    corpus: Optional[CorpusContext] = None,
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
        # RAG enrichment — overlap flag + related reading from the user's corpus.
        in_bibliography = corpus.is_in_bibliography(paper) if corpus else False
        related = corpus.related(paper) if corpus else []
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
            "in_bibliography": in_bibliography,
            # Pre-render each related-reading bullet as one string so the
            # Jinja template doesn't have to juggle conditional block tags
            # (trim_blocks strips newlines after them, collapsing bullets).
            "related": [_format_related_line(r) for r in related],
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


def _scan_archive(docs_dir: Path) -> List[Dict[str, Any]]:
    """Scan ``docs_dir`` for YYYY-MM-DD.md files, extract per-issue metadata.

    Returns a list sorted newest-first, where each entry contains: slug,
    date, total paper count, title line, top tier, highlight bullets
    (as a raw markdown string), and the full file text.
    """
    digest_files = sorted(
        [f for f in docs_dir.glob("*.md") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", f.stem)],
        reverse=True,
    )
    issues: List[Dict[str, Any]] = []
    for f in digest_files:
        text = f.read_text()
        title_m = re.search(r"^#\s+(.+)", text, flags=re.MULTILINE)
        total_m = re.search(r"— (\d+) new papers", text)
        total = int(total_m.group(1)) if total_m else 0

        top_tier_key = "A"
        top_count = -1
        for tier_key in TIER_ORDER:
            m = re.search(rf"\|\s+\*\*{tier_key}\*\*\s+\|.*?\|\s+(\d+)\s+\|", text)
            if m:
                c = int(m.group(1))
                if c > top_count:
                    top_count = c
                    top_tier_key = tier_key

        # "Highlights this week" block — a bullet list, useful for teasers.
        hl_m = re.search(
            r"## Highlights this week\s*\n\n(- \*\*.+?)(?=\n\n|\n---)",
            text,
            flags=re.DOTALL,
        )
        highlights_md = hl_m.group(1).strip() if hl_m else ""

        try:
            date = dt.date.fromisoformat(f.stem)
        except ValueError:
            continue

        issues.append({
            "slug": f.stem,
            "date": date,
            "total": total,
            "title": (title_m.group(1) if title_m else f.stem).strip(),
            "top_tier_key": top_tier_key,
            "top_tier_name": TIER_NAMES[top_tier_key],
            "highlights_md": highlights_md,
            "text": text,
        })
    return issues


def render_index(docs_dir: Path, template_dir: Path) -> str:
    """Rebuild the archive index from existing digest files."""
    import jinja2

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("index.md.j2")

    issues = _scan_archive(docs_dir)
    archive = []
    latest = None
    for issue in issues:
        entry = {
            "week_start": issue["slug"],
            "slug": issue["slug"],
            "total": issue["total"],
            "top_tier_name": issue["top_tier_name"],
        }
        if latest is None:
            entry["teaser"] = issue["highlights_md"].splitlines()[0] if issue["highlights_md"] else ""
            latest = entry
        archive.append(entry)

    return tpl.render(latest=latest, archive=archive)


def _highlights_to_html(md: str) -> str:
    """Convert the Markdown highlights bullets to minimal HTML for RSS.

    Handles ``**bold**`` and ``[text](url)`` — the only two constructs the
    highlight bullets currently use.
    """
    if not md:
        return ""
    items = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body)
        body = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            r'<a href="\2">\1</a>',
            body,
        )
        items.append(f"<li>{body}</li>")
    if not items:
        return ""
    return "<ul>" + "".join(items) + "</ul>"


def render_rss(
    docs_dir: Path,
    template_dir: Path,
    base_url: str = "https://jiewangnk.github.io/AortaCFD-web/",
) -> str:
    """Render an RSS 2.0 feed from the archive of digest issues."""
    import jinja2

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("rss.xml.j2")

    issues_meta = _scan_archive(docs_dir)
    # RFC-822 pub dates anchored at Monday 09:00 UTC (the cron schedule).
    publish_time = dt.time(9, 0, tzinfo=dt.timezone.utc)
    items = []
    for issue in issues_meta:
        pub_dt = dt.datetime.combine(issue["date"], publish_time)
        items.append({
            "slug": issue["slug"],
            "title": issue["title"],
            "pub_date": email.utils.format_datetime(pub_dt),
            "description_html": _highlights_to_html(issue["highlights_md"])
            or f"<p>{issue['total']} new papers, top theme: {issue['top_tier_name']}.</p>",
        })

    last_build = items[0]["pub_date"] if items else None
    return tpl.render(base_url=base_url, issues=items, last_build=last_build)


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

    # Load the bibliography corpus for RAG enrichment (overlap flag +
    # related reading). Degrades gracefully if aortacfd_agent isn't
    # installed or the JSON is missing.
    corpus_ctx = load_context()

    # Write the dated digest. Use today's publish date (the Monday the cron
    # ran) as both filename and title — readers think in terms of the date
    # they can read it on, not the internal 7-day-window start.
    digest_md = render_digest(
        today, classified, relevance_threshold, template_dir, corpus=corpus_ctx
    )
    out_path = out_dir / f"{today.isoformat()}.md"
    out_path.write_text(digest_md)
    logger.info("Wrote %s", out_path)

    # Rebuild the archive index
    index_md = render_index(out_dir, template_dir)
    (out_dir / "index.md").write_text(index_md)
    logger.info("Wrote index.md")

    # Rebuild the RSS feed
    rss_xml = render_rss(out_dir, template_dir)
    (out_dir / "rss.xml").write_text(rss_xml)
    logger.info("Wrote rss.xml")

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
