"""Unit tests for the LiteratureAgent and its dependencies.

Three groups of tests:

1. :class:`FakeCorpusStore` — keyword-overlap scoring, stability, edge cases.
2. ``search_corpus`` tool factory — bound-store behaviour and error handling.
3. :class:`LiteratureAgent` — scripted end-to-end conversations that
   exercise the search-then-emit loop and the schema validation path.

All tests run offline with a hand-written 12-chunk corpus in
``tests/fixtures/corpus_chunks.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from aortacfd_agent.agents.literature import LiteratureAgent
from aortacfd_agent.backends.base import ToolCall
from aortacfd_agent.backends.fake import FakeBackend, ScriptedStep
from aortacfd_agent.corpus.store import Chunk, FakeCorpusStore
from aortacfd_agent.tools.literature import make_search_corpus_tool


FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus_chunks.json"


@pytest.fixture(scope="module")
def corpus() -> FakeCorpusStore:
    return FakeCorpusStore.from_json(FIXTURE_CORPUS)


# ---------------------------------------------------------------------------
# FakeCorpusStore behaviour
# ---------------------------------------------------------------------------


class TestFakeCorpusStore:
    def test_loads_from_json_fixture(self, corpus: FakeCorpusStore):
        assert len(corpus.chunks) == 12
        assert all(isinstance(c, Chunk) for c in corpus.chunks)
        assert any(c.paper == "Steinman2013" for c in corpus.chunks)
        assert any(c.paper == "Esmaily2011" for c in corpus.chunks)

    def test_search_returns_relevant_chunks_for_coarctation(self, corpus: FakeCorpusStore):
        results = corpus.search("coarctation user specified flow split", top_k=3)
        assert len(results) >= 1
        # The Wang2025 chunk about coarctation misallocation should rank highly
        top_papers = [r.paper for r in results]
        assert "Wang2025" in top_papers

    def test_search_for_backflow_finds_esmaily(self, corpus: FakeCorpusStore):
        results = corpus.search("backflow stabilisation beta_T diastolic", top_k=3)
        assert len(results) >= 1
        assert any(r.paper == "Esmaily2011" for r in results)

    def test_search_for_windkessel_tau_finds_stergiopulos(self, corpus: FakeCorpusStore):
        results = corpus.search("Windkessel tau time constant default", top_k=3)
        papers = [r.paper for r in results]
        assert "Stergiopulos1999" in papers

    def test_search_empty_query_returns_empty(self, corpus: FakeCorpusStore):
        assert corpus.search("", top_k=5) == []
        assert corpus.search("   ", top_k=5) == []

    def test_search_no_match_returns_empty(self, corpus: FakeCorpusStore):
        results = corpus.search("xylophone marimba concerto", top_k=5)
        assert results == []

    def test_search_respects_top_k(self, corpus: FakeCorpusStore):
        results = corpus.search("aorta", top_k=2)
        assert len(results) <= 2

    def test_search_score_is_normalised(self, corpus: FakeCorpusStore):
        results = corpus.search("pressure drop", top_k=5)
        for chunk in results:
            assert 0.0 < chunk.score <= 1.0


# ---------------------------------------------------------------------------
# search_corpus tool factory
# ---------------------------------------------------------------------------


class TestSearchCorpusTool:
    def test_tool_spec_has_expected_schema(self, corpus: FakeCorpusStore):
        tool = make_search_corpus_tool(corpus)
        assert tool.name == "search_corpus"
        assert tool.parameters["type"] == "object"
        assert "query" in tool.parameters["properties"]
        assert "top_k" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["query"]

    def test_tool_handler_returns_structured_results(self, corpus: FakeCorpusStore):
        tool = make_search_corpus_tool(corpus)
        out = tool.handler({"query": "Windkessel flow split coarctation", "top_k": 3})
        assert out["query"] == "Windkessel flow split coarctation"
        assert out["store"] == "fake_corpus"
        assert out["num_results"] >= 1
        assert len(out["results"]) == out["num_results"]
        first = out["results"][0]
        assert "text" in first
        assert "paper" in first
        assert "score" in first

    def test_tool_handler_rejects_empty_query(self, corpus: FakeCorpusStore):
        tool = make_search_corpus_tool(corpus)
        out = tool.handler({"query": "", "top_k": 3})
        assert "error" in out

    def test_tool_handler_defaults_top_k(self, corpus: FakeCorpusStore):
        tool = make_search_corpus_tool(corpus)
        out = tool.handler({"query": "aorta"})  # top_k omitted
        assert out["top_k"] == 5

    def test_tool_handler_catches_store_errors(self):
        """If the store raises, the tool must return {"error": ...}, not crash."""

        class BrokenStore:
            name = "broken"

            def search(self, query, top_k=5):
                raise RuntimeError("boom")

        tool = make_search_corpus_tool(BrokenStore())
        out = tool.handler({"query": "anything"})
        assert "error" in out
        assert "RuntimeError" in out["error"]


# ---------------------------------------------------------------------------
# LiteratureAgent end-to-end (scripted)
# ---------------------------------------------------------------------------


def _bpm120_profile() -> Dict[str, Any]:
    """Minimal clinical profile representing BPM120 (paediatric coarctation)."""
    return {
        "patient_id": "BPM120",
        "age_years": 12,
        "sex": "male",
        "diagnosis": "aortic coarctation post-balloon angioplasty",
        "heart_rate_bpm": 78,
        "systolic_bp_mmhg": 118,
        "diastolic_bp_mmhg": 72,
        "cardiac_output_l_min": 4.8,
        "imaging_modality": ["CT_angiography"],
        "study_goal": "WSS around coarctation and inlet-to-descending pressure drop.",
        "constraints": ["≤32 cores", "3-element Windkessel default"],
        "missing_fields": [],
        "confidence": "high",
    }


def _valid_justification() -> Dict[str, Any]:
    """A well-formed ParameterJustification matching the schema."""
    return {
        "decisions": [
            {
                "parameter": "physics_model",
                "value": "rans",
                "reasoning": (
                    "Peak Reynolds exceeds 2000 in the ascending aorta; laminar "
                    "over-predicts pressure drop by ~40% relative to RANS or LES "
                    "in this regime."
                ),
                "citations": [
                    {
                        "paper": "Wang2025",
                        "page": 3,
                        "quote": (
                            "For Reynolds above two thousand, laminar simulations "
                            "over-predict the cycle averaged pressure drop by "
                            "roughly forty percent"
                        ),
                    }
                ],
                "alternative_considered": "laminar (rejected: over-prediction at transitional Re)",
            },
            {
                "parameter": "mesh_goal",
                "value": "wall_sensitive",
                "reasoning": "Study goal includes WSS around the coarctation.",
                "citations": [
                    {
                        "paper": "ValenSendstad2018",
                        "page": 5,
                        "quote": (
                            "For studies where WSS, OSI and near-wall indices are "
                            "primary endpoints we recommend mesh refinement around "
                            "the wall region"
                        ),
                    }
                ],
            },
            {
                "parameter": "wk_flow_allocation_method",
                "value": "user_specified",
                "reasoning": "Coarctation invalidates Murray's law; use fixed split.",
                "citations": [
                    {
                        "paper": "Wang2025",
                        "page": 4,
                        "quote": (
                            "In coarctation cases Murray's law systematically "
                            "misallocates flow because the stenosis dominates the "
                            "local pressure distribution"
                        ),
                    }
                ],
                "alternative_considered": "murray (rejected: invalid for coarctation)",
            },
            {
                "parameter": "wk_flow_split_fractions",
                "value": {"descending": 0.70, "brachiocephalic": 0.10, "lcca": 0.10, "lsa": 0.10},
                "reasoning": "Literature convention of ~70% to descending aorta in paediatric coarctation.",
                "citations": [
                    {
                        "paper": "Wang2025",
                        "page": 4,
                        "quote": (
                            "Published paediatric coarctation studies use a "
                            "user-specified flow split with roughly seventy percent "
                            "of cardiac output to the descending aorta"
                        ),
                    }
                ],
            },
            {
                "parameter": "windkessel_tau",
                "value": 1.5,
                "reasoning": "Default systemic tau for adult/paediatric circulation without patient-specific calibration.",
                "citations": [
                    {
                        "paper": "Stergiopulos1999",
                        "page": 2,
                        "quote": (
                            "A default tau of one point five seconds is appropriate "
                            "when patient-specific calibration is unavailable"
                        ),
                    }
                ],
            },
            {
                "parameter": "backflow_stabilisation",
                "value": 0.3,
                "reasoning": "Published default that eliminates divergence with minimal hemodynamic bias.",
                "citations": [
                    {
                        "paper": "Esmaily2011",
                        "page": 7,
                        "quote": (
                            "A directional stabilisation with beta_T equal to zero "
                            "point three damps tangential velocity during backflow "
                            "while preserving normal pressure response"
                        ),
                    }
                ],
            },
            {
                "parameter": "numerics_profile",
                "value": "standard",
                "reasoning": "Default 2nd-order profile balances accuracy and stability for production runs.",
                "citations": [],
            },
            {
                "parameter": "number_of_cycles",
                "value": 3,
                "reasoning": "MAP initialisation reaches periodic steady state within three cycles.",
                "citations": [
                    {
                        "paper": "Pfaller2021",
                        "page": 8,
                        "quote": (
                            "Windkessel-coupled aortic simulations initialised at "
                            "mean arterial pressure reached periodic steady state "
                            "within three cardiac cycles"
                        ),
                    }
                ],
            },
        ],
        "search_queries_used": [
            "paediatric coarctation Reynolds physics model",
            "wall shear stress mesh refinement coarctation",
            "Windkessel flow split coarctation Murray's law",
            "Windkessel tau time constant default adult",
            "backflow stabilisation beta_T diastolic divergence",
            "cardiac cycle periodicity MAP initialisation",
        ],
        "unresolved_decisions": [],
        "confidence": "high",
        "notes": None,
    }


def _scripted_agent_run(
    corpus: FakeCorpusStore, justification: Dict[str, Any]
) -> LiteratureAgent:
    """Build a FakeBackend that issues 2 searches and then emits the justification."""
    backend = FakeBackend(
        script=[
            ScriptedStep(
                text="",
                tool_calls=[
                    ToolCall(
                        id="s1",
                        name="search_corpus",
                        arguments={
                            "query": "coarctation Murray's law flow split",
                            "top_k": 3,
                        },
                    )
                ],
            ),
            ScriptedStep(
                text="",
                tool_calls=[
                    ToolCall(
                        id="s2",
                        name="search_corpus",
                        arguments={
                            "query": "backflow stabilisation beta_T diastolic",
                            "top_k": 3,
                        },
                    )
                ],
            ),
            ScriptedStep(
                text="",
                tool_calls=[
                    ToolCall(
                        id="emit",
                        name="emit_parameter_justification",
                        arguments=justification,
                    )
                ],
            ),
            ScriptedStep(text="Parameter justification emitted.", stop_reason="end_turn"),
        ]
    )
    return LiteratureAgent(backend=backend, corpus=corpus)


class TestLiteratureAgent:
    def test_happy_path_emits_justification(self, corpus: FakeCorpusStore):
        justification = _valid_justification()
        agent = _scripted_agent_run(corpus, justification)

        result = agent.justify(_bpm120_profile())

        assert result.justification["confidence"] == "high"
        assert len(result.justification["decisions"]) == 8
        assert result.unresolved_decisions == []
        # search_queries list the agent actually issued through the loop
        assert "coarctation Murray's law flow split" in result.search_queries
        assert "backflow stabilisation beta_T diastolic" in result.search_queries

    def test_iterations_count_includes_all_turns(self, corpus: FakeCorpusStore):
        agent = _scripted_agent_run(corpus, _valid_justification())
        result = agent.justify(_bpm120_profile())
        # 3 tool-use turns plus one final text turn = 4
        assert result.run.iterations == 4

    def test_missing_emit_raises(self, corpus: FakeCorpusStore):
        # Only a search, no emit
        backend = FakeBackend(
            script=[
                ScriptedStep(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="s1",
                            name="search_corpus",
                            arguments={"query": "anything"},
                        )
                    ],
                ),
                ScriptedStep(text="I give up.", stop_reason="end_turn"),
            ]
        )
        agent = LiteratureAgent(backend=backend, corpus=corpus)
        with pytest.raises(ValueError, match="emit_parameter_justification"):
            agent.justify(_bpm120_profile())

    def test_malformed_justification_fails_validation(self, corpus: FakeCorpusStore):
        # Missing the 'decisions' required field
        bad = {
            "search_queries_used": [],
            "confidence": "low",
        }
        backend = FakeBackend(
            script=[
                ScriptedStep(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="emit",
                            name="emit_parameter_justification",
                            arguments=bad,
                        )
                    ],
                ),
                ScriptedStep(text="done", stop_reason="end_turn"),
            ]
        )
        agent = LiteratureAgent(backend=backend, corpus=corpus)
        with pytest.raises(ValueError, match="schema validation"):
            agent.justify(_bpm120_profile())

    def test_invalid_parameter_enum_fails(self, corpus: FakeCorpusStore):
        bad = {
            "decisions": [
                {
                    "parameter": "made_up_parameter",  # not in enum
                    "value": "whatever",
                    "reasoning": "nope",
                    "citations": [],
                }
            ],
            "search_queries_used": [],
            "confidence": "low",
        }
        backend = FakeBackend(
            script=[
                ScriptedStep(
                    text="",
                    tool_calls=[
                        ToolCall(id="e", name="emit_parameter_justification", arguments=bad)
                    ],
                ),
                ScriptedStep(text="done", stop_reason="end_turn"),
            ]
        )
        agent = LiteratureAgent(backend=backend, corpus=corpus)
        with pytest.raises(ValueError, match="schema validation"):
            agent.justify(_bpm120_profile())

    def test_agent_system_prompt_loaded(self, corpus: FakeCorpusStore):
        backend = FakeBackend(script=[ScriptedStep(text="")])
        agent = LiteratureAgent(backend=backend, corpus=corpus)
        assert "Literature Agent" in agent.system_prompt
        assert "search_corpus" in agent.system_prompt
        assert "emit_parameter_justification" in agent.system_prompt
