"""Literature search tool — thin wrapper over a :class:`CorpusStore`.

Unlike most tools in this package, ``search_corpus`` is stateful: the
handler needs a reference to a corpus store, and different agents (or
different test runs) bind different stores. So the module exposes a
``make_search_corpus_tool`` factory that returns a ``ToolSpec`` with the
store baked into the handler closure.

The LiteratureAgent builds this tool at construction time from the
store it was given.
"""

from __future__ import annotations

from typing import Any, Dict

from ..backends.base import ToolSpec
from ..corpus.store import CorpusStore


def make_search_corpus_tool(store: CorpusStore) -> ToolSpec:
    """Build a ``ToolSpec`` bound to a specific corpus store.

    The returned tool exposes one parameter — a query string — and an
    optional top-k override. Results are a list of chunk dicts matching
    :meth:`Chunk.to_dict`, truncated to ``top_k`` entries.

    The handler never raises. On any error it returns an ``{"error": ...}``
    dict so the model sees the failure and can retry with a different
    query.
    """

    def search_corpus(args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            return {"error": "query is required and must be non-empty"}
        top_k = int(args.get("top_k") or 5)
        if top_k < 1:
            top_k = 5

        try:
            chunks = store.search(query=query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

        return {
            "query": query,
            "top_k": top_k,
            "store": getattr(store, "name", "unknown"),
            "num_results": len(chunks),
            "results": [c.to_dict() for c in chunks],
        }

    return ToolSpec(
        name="search_corpus",
        description=(
            "Search the cardiovascular CFD literature corpus for passages "
            "supporting a parameter decision. Returns short verbatim "
            "quotes plus citation metadata (paper, page). Call this "
            "separately for each parameter you need to justify — do not "
            "assume one search answers every decision."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language query describing the parameter "
                        "decision, e.g. 'Windkessel flow split for aortic "
                        "coarctation' or 'backflow stabilisation beta_T "
                        "diastolic divergence'."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Max number of results to return. Default 5.",
                },
            },
            "required": ["query"],
        },
        handler=search_corpus,
    )
