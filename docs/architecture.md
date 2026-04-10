# Architecture

This document describes how `aortacfd-agent` is structured, why each
piece exists, and where to look when you want to modify or extend it.
For a pitch-level overview see [`sophia_pitch.md`](sophia_pitch.md).
For a quickstart see the top-level [`README.md`](../README.md).

## Two-repo model

`aortacfd-agent` is deliberately a **separate repository** from the CFD
pipeline it drives:

```
AortaCFD-app        stable CFD pipeline, JSON-driven (paper / HPC / production)
aortacfd-agent      research LLM layer (this repo), depends on AortaCFD-app
   └ external/
     └ aortacfd-app/    git submodule, pinned to a known-good commit
```

This has four practical consequences:

1. **The CFD core never imports LLM code.** That makes the paper and HPC
   releases auditable and the agent layer skippable.
2. **The two repos evolve on independent release cycles.** Breaking
   experimental agent features cannot destabilise the CFD pipeline.
3. **The agent repo can be private while the CFD repo is public**, which
   matches the current state of each project.
4. **Pitching to NLP collaborators is easier** because the agent repo is
   a focused, small, well-tested codebase they can read in half an hour
   rather than a 30-KLOC CFD pipeline.

The submodule is pinned at a specific commit, updated deliberately, and
treated as a read-only dependency. The agent layer talks to it in two
ways: read-only Python imports from `external/aortacfd-app/src/` (for
the physics advisor, STL reader, and schema helpers) and `subprocess`
invocations of `run_patient.py` (for actually running a CFD case).

## Layers

```
┌───────────────────────────────────────────────────────────────┐
│ CLI: aortacfd-agent run | intake | version                    │
├───────────────────────────────────────────────────────────────┤
│ Coordinator (src/aortacfd_agent/coordinator.py)               │
│   orchestrates 5 agents, writes agent_trace.jsonl             │
├───────────────┬───────────────┬─────────────┬────────┬────────┤
│ IntakeAgent   │ LiteratureAgt │ ConfigAgent │ ExecAg │ ResAgt │
│ (LLM, 1 turn) │ (LLM, N turns)│ (no LLM)    │ (none) │ (LLM)  │
├───────────────┴───────────────┴─────────────┴────────┴────────┤
│ AgentLoop (ReAct)    +    LLMBackend protocol                 │
│                                                               │
│ Backends:  fake | anthropic | openai_compat (ollama/vllm/...) │
├───────────────────────────────────────────────────────────────┤
│ Tools:   geometry.py  physics.py  mesh.py  config_io.py       │
│          literature.py  results_io.py  outputs.py             │
├───────────────────────────────────────────────────────────────┤
│ Corpus:  store.py (FakeCorpusStore + ChromaCorpusStore)       │
│          ingest.py (PDF → chunks → Chroma, optional deps)     │
├───────────────────────────────────────────────────────────────┤
│ external/aortacfd-app (git submodule, pinned)                 │
│   src/aortacfd_lib/…  run_patient.py  cases_input/…           │
└───────────────────────────────────────────────────────────────┘
```

## The five agents in detail

Each agent has a narrow responsibility and a well-defined JSON contract
with the next agent. All five share the generic
[`AgentLoop`](../src/aortacfd_agent/loop.py) and
[`LLMBackend`](../src/aortacfd_agent/backends/base.py), so swapping
providers or adding a new agent requires touching only:

1. a new module under `agents/`,
2. a new system prompt under `prompts/`,
3. optionally a new JSON schema under `schemas/`,
4. new tool wrappers if the agent needs capabilities the existing tools
   don't cover.

### 1. IntakeAgent
`agents/intake.py` + `prompts/intake.md` + `schemas/clinical_profile.json`

Single-turn LLM call. Receives the free-text referral and the schema,
returns a `ClinicalProfile` JSON validated with `jsonschema`. Explicit
no-invention rule: missing fields are `null` and listed in
`missing_fields`. Tests cover five synthetic referrals
(paediatric coarctation, healthy adult, Marfan, post-surgical,
complex infant) plus three failure modes.

### 2. LiteratureAgent
`agents/literature.py` + `prompts/literature.md` + `schemas/parameter_justification.json`

Multi-turn LLM agent. Receives the `ClinicalProfile` and a
`CorpusStore`, iteratively issues `search_corpus` calls, then emits
`emit_parameter_justification` exactly once. Every parameter decision
must carry at least one verbatim quote from a retrieved chunk.
Handles the `FakeCorpusStore` (tests, offline demos) and
`ChromaCorpusStore` (production, optional dependency) identically.

### 3. ConfigAgent
`agents/config.py`

**Deterministic — no LLM.** Takes the `ClinicalProfile` + the
`ParameterJustification`, loads the submodule's
`examples/config_standard.json` as a template, and patches it field
by field. Reducer helpers validate every value before writing it into
the config. A lightweight in-agent `_validate` pass catches patching
mistakes; the full schema validation runs inside the CFD pipeline when
`run_patient.py` is eventually invoked.

Writes `agent_config.json` and a composed `agent_rationale.md` with
every decision cited back to its literature chunk.

### 4. ExecutionAgent
`agents/execution.py`

**Deterministic — no LLM.** Invokes `run_patient.py` as a
subprocess, with the agent-generated config. Dry-run mode
(`case,mesh,boundary`) is the default; full runs
(`…,solver,reconstruct,postprocess`) are one flag away. An `executor`
keyword on `run()` is a test seam that injects a recording fake so the
test suite never starts OpenFOAM.

### 5. ResultsAgent
`agents/results.py` + `prompts/results.md` + `tools/results_io.py`

Multi-turn LLM agent. Receives a finished run directory, reads
`qoi_summary.json`, `hemodynamics_report.txt`, `merged_config.json`,
and per-patch pressure time-series via four read-only tools. Produces
a one-paragraph clinical narrative (via `summarise()`) or answers a
direct question (via `ask()`). System prompt enforces the grounding
rule: every numerical value, unit, and interpretation must come from a
tool call in the current conversation.

## The trace file

Every `Coordinator.run()` call writes an `agent_trace.jsonl` next to its
outputs. Each line is one stage record with:

```json
{
  "stage": "literature",
  "timestamp": 1712770000.123,
  "duration_s": 0.84,
  "status": "ok",
  "payload": {
    "confidence": "high",
    "unresolved": [],
    "num_decisions": 8,
    "search_queries": ["coarctation Murray flow split", "..."]
  }
}
```

The trace is append-only, has no external dependencies, and is the
single source of truth for the "trustworthy AI" provenance story. A
reviewer can reproduce exactly what every agent did by reading the
trace alongside the input referral.

## Testing strategy

All tests run offline, with zero network calls, in about a quarter of a
second total. The approach is:

- **`FakeBackend`** replays scripted LLM responses so agent loops can be
  driven deterministically without hitting an API.
- **`FakeCorpusStore`** is an in-memory keyword matcher over hand-written
  chunks — no embeddings, no vector database.
- **`RecordingExecutor`** is a `subprocess.run`-compatible callable that
  records the command and returns a scripted `CompletedProcess` so the
  ExecutionAgent is testable without OpenFOAM.
- **Synthetic finished-run fixtures** are built in `tmp_path` with just
  enough JSON/text/`.dat` files for the ResultsAgent to find.

Two tests are marked `xfail` — they exercise the submodule's legacy
`validate_config` function, which has a pre-existing enum-enforcement
bug. When that is fixed in the CFD repo, the `xfail` markers can be
removed.

## Extending the system

Common extension points:

- **Add a new agent.** Copy `agents/intake.py` as a template, wire a new
  prompt and (optionally) a new schema, then hook it into the
  coordinator. The generic `AgentLoop` handles the tool-use plumbing.
- **Add a new tool.** Create a module under `tools/`, define
  `*_spec()` factories, and bundle them via
  `tools/__init__.py::build_default_toolset` or a per-agent
  equivalent.
- **Swap the LLM provider.** Pick one of the existing backends or add a
  new one under `backends/` — anything that implements the
  `LLMBackend` protocol works. The factory wires it into the CLI.
- **Replace the RAG backend.** Implement a new class that satisfies the
  `CorpusStore` protocol (`search(query, top_k) → List[Chunk]`). No
  other code needs to change.
