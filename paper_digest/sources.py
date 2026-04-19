"""Paper sources: OpenAlex (peer-reviewed) + arXiv (preprints).

Both APIs are free, no key required. We filter in two passes:
  1. Server-side: date window + concepts / categories
  2. Client-side: keyword match on title+abstract
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

USER_AGENT = "aortacfd-agent-digest/0.1 (https://github.com/JieWangnk/aortacfd-agent)"


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass
class Paper:
    source: str  # "openalex" or "arxiv"
    title: str
    authors: List[str]
    abstract: str
    year: int
    venue: str = ""  # journal or arXiv category
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    published: Optional[str] = None  # ISO date
    url: str = ""

    @property
    def doi_url(self) -> str:
        return f"https://doi.org/{self.doi}" if self.doi else ""

    @property
    def arxiv_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}" if self.arxiv_id else ""

    @property
    def authors_short(self) -> str:
        if not self.authors:
            return ""
        if len(self.authors) == 1:
            return self.authors[0]
        if len(self.authors) == 2:
            return f"{self.authors[0]} & {self.authors[1]}"
        return f"{self.authors[0]} et al."

    @property
    def venue_short(self) -> str:
        v = self.venue or ""
        return v if len(v) <= 40 else v[:37] + "..."

    @property
    def dedup_key(self) -> str:
        """Canonical string used to dedupe across sources."""
        if self.doi:
            return f"doi:{self.doi.lower()}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        # Fallback: normalised title
        t = re.sub(r"\s+", " ", self.title.lower().strip())
        return f"title:{t}"


# ---------------------------------------------------------------------------
# Keyword filter shared by both sources
# ---------------------------------------------------------------------------


# Two axes: a paper must have at least one *context* term (cardiovascular
# anatomy / physiology) AND at least one *method* term (CFD, turbulence,
# PINN, etc.). This knocks out clinical-only papers that happen to mention
# "hemodynamic" once and pure-engineering CFD papers with no vascular tie.
CONTEXT_TERMS = [
    "aorta", "aortic", "coronary", "carotid", "pulmonary artery",
    "cerebral artery", "intracranial", "aneurysm", "stenosis",
    "bifurcation", "dissection", "left ventric", "right ventric",
    "cardiovascular", "cardiac flow", "arterial", "vascular",
    "hemodynamic", "haemodynamic", "blood flow", "pulsatile flow",
    "wall shear stress", "wss ", "oscillatory shear", "windkessel",
    "womersley", "vessel", "valve",
]

METHOD_TERMS = [
    # CFD / numerics
    "cfd ", "cfd,", "cfd)", "cfd:", "computational fluid",
    "finite element", "finite volume", "openfoam", "simvascular",
    "les ", " rans", "large eddy", "direct numerical simulation",
    "turbulen", "reynolds-averaged", "navier-stokes",
    # ML / PINN / surrogates
    "pinn", "physics-informed", "neural operator", "fourier neural",
    "deeponet", "graph neural", "digital twin", "surrogate model",
    "machine learning", "deep learning", "neural network",
    # Boundary conditions & modelling
    "boundary condition", "lumped parameter", "0d-3d", "3d-0d",
    "multiscale", "reduced order", "fluid-structure",
    # Imaging→CFD
    "4d flow", "phase contrast", "patient-specific",
]


def _matches_keywords(title: str, abstract: str) -> bool:
    """Paper must hit at least one *context* AND one *method* keyword."""
    blob = f" {title.lower()} {abstract.lower()} "
    has_context = any(kw in blob for kw in CONTEXT_TERMS)
    has_method = any(kw in blob for kw in METHOD_TERMS)
    return has_context and has_method


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------


# Concept IDs used by OpenAlex (openalex.org/concepts). We cast a wide server-
# side net and then rely on the two-axis keyword filter (context AND method)
# below to trim clinical noise before classification.
OPENALEX_CONCEPTS = [
    "C2779022549",   # Hemodynamics
    "C126322002",    # Cardiovascular
    "C159985019",    # Computational fluid dynamics
]


def _reconstruct_abstract(inverted: Optional[dict]) -> str:
    """OpenAlex serves abstracts as inverted index. Rebuild linear text."""
    if not inverted:
        return ""
    positions: List[tuple[int, str]] = []
    for word, idxs in inverted.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


# Server-side keyword query — OpenAlex's `search` parameter does BM25 over
# title+abstract, so we narrow the 45K/week cardiovascular concept hits down
# to a few hundred that actually look CFD/modelling-adjacent.
OPENALEX_SEARCH = (
    '"hemodynamic" OR "haemodynamic" OR "wall shear stress" OR "aortic flow" '
    'OR "coronary flow" OR "blood flow simulation" OR "cardiovascular cfd" '
    'OR "physics-informed" OR "neural operator" OR "windkessel" '
    'OR "4d flow mri" OR "patient-specific cfd"'
)


def fetch_openalex(
    start_date: dt.date,
    end_date: dt.date,
    max_results: int = 200,
    timeout: float = 20.0,
) -> List[Paper]:
    """Fetch OpenAlex works created between ``start_date`` and ``end_date``."""
    params = {
        "filter": ",".join([
            f"from_publication_date:{start_date.isoformat()}",
            f"to_publication_date:{end_date.isoformat()}",
            "has_abstract:true",
        ]),
        "search": OPENALEX_SEARCH,
        "per-page": "50",
        "sort": "relevance_score:desc",
    }
    url = f"https://api.openalex.org/works?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    papers: List[Paper] = []
    cursor_url: Optional[str] = url
    seen = 0
    # OpenAlex paginates via `cursor=*` — we only pull what we need
    for _ in range(5):  # max 5 pages = 250 hits
        if not cursor_url or seen >= max_results:
            break
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning("OpenAlex fetch failed: %s", e)
            break

        for w in data.get("results", []):
            abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
            title = (w.get("title") or "").strip()
            if not abstract or not title:
                continue
            if not _matches_keywords(title, abstract):
                continue

            authors = []
            for a in (w.get("authorships") or [])[:6]:
                nm = (a.get("author") or {}).get("display_name")
                if nm:
                    authors.append(nm)

            venue = ""
            loc = w.get("primary_location") or {}
            src = loc.get("source") or {}
            if isinstance(src, dict):
                venue = src.get("display_name", "") or ""

            doi = w.get("doi") or ""
            if doi.startswith("https://doi.org/"):
                doi = doi[len("https://doi.org/"):]

            papers.append(Paper(
                source="openalex",
                title=title,
                authors=authors,
                abstract=abstract,
                year=int(w.get("publication_year") or end_date.year),
                venue=venue,
                doi=doi.lower() if doi else None,
                published=w.get("publication_date"),
                url=f"https://doi.org/{doi}" if doi else w.get("id", ""),
            ))
            seen += 1
            if seen >= max_results:
                break

        nxt = data.get("meta", {}).get("next_cursor")
        if not nxt:
            break
        cursor_url = url + f"&cursor={nxt}"
        req = urllib.request.Request(cursor_url, headers={"User-Agent": USER_AGENT})

    logger.info("OpenAlex: %d papers after keyword filter", len(papers))
    return papers


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------


# arXiv doesn't have the OpenAlex concept index — we hit it with keywords in
# the abstract and filter by date client-side. This is far more reliable than
# trying to URL-encode a submittedDate range bracket expression.
ARXIV_KEYWORD_QUERY = (
    'abs:"hemodynamic" OR abs:"haemodynamic" OR abs:"wall shear stress" '
    'OR abs:"aortic flow" OR abs:"coronary flow" OR abs:"blood flow" '
    'OR abs:"windkessel" OR abs:"cardiovascular cfd" OR abs:"patient-specific cfd"'
)


def fetch_arxiv(
    start_date: dt.date,
    end_date: dt.date,
    max_results: int = 200,
    timeout: float = 20.0,
) -> List[Paper]:
    """Fetch arXiv preprints submitted between ``start_date`` and ``end_date``."""
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={urllib.parse.quote(ARXIV_KEYWORD_QUERY)}"
        f"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            xml_text = resp.read().decode("utf-8")
    except Exception as e:
        logger.warning("arXiv fetch failed: %s", e)
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(xml_text)

    papers: List[Paper] = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        abstract = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        title = re.sub(r"\s+", " ", title)
        abstract = re.sub(r"\s+", " ", abstract)
        if not _matches_keywords(title, abstract):
            continue

        # Client-side date filter (since we dropped the submittedDate range)
        pub_raw = entry.findtext("atom:published", default="", namespaces=ns) or ""
        try:
            pub_date = dt.date.fromisoformat(pub_raw[:10])
        except ValueError:
            pub_date = end_date  # keep if date unparseable
        if not (start_date <= pub_date <= end_date):
            continue

        # arXiv id from the <id> URL
        url_id = entry.findtext("atom:id", default="", namespaces=ns) or ""
        m = re.search(r"/abs/([^/]+?)(v\d+)?$", url_id)
        arxiv_id = m.group(1) if m else None

        authors = [
            (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
            for a in entry.findall("atom:author", ns)
        ]
        authors = [a for a in authors if a][:6]

        published = entry.findtext("atom:published", default="", namespaces=ns) or ""
        year = int(published[:4]) if published else end_date.year

        # Primary category
        pcat = entry.find("arxiv:primary_category", ns)
        venue = pcat.get("term", "arXiv") if pcat is not None else "arXiv"

        # DOI (optional; some arXiv submissions include publisher DOI)
        doi = entry.findtext("arxiv:doi", default="", namespaces=ns)
        doi = doi.strip().lower() if doi else None

        papers.append(Paper(
            source="arxiv",
            title=title,
            authors=authors,
            abstract=abstract,
            year=year,
            venue=f"arXiv · {venue}",
            doi=doi,
            arxiv_id=arxiv_id,
            published=published[:10] if published else None,
            url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else url_id,
        ))

    logger.info("arXiv: %d papers after keyword filter", len(papers))
    return papers


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def dedupe(papers: Iterable[Paper]) -> List[Paper]:
    """Keep first occurrence by DOI / arxiv-id / normalised title."""
    seen: set[str] = set()
    out: List[Paper] = []
    for p in papers:
        key = p.dedup_key
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
