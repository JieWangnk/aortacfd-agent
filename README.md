# aortacfd-agent

> **LLM-agent layer over [AortaCFD-app](https://github.com/JieWangnk/AortaCFD-app)** —
> turn a free-text clinical referral into a reproducible, literature-grounded,
> patient-specific CFD simulation with a natural-language clinical summary.

This is the research-facing companion to `AortaCFD-app`. The core CFD pipeline
lives in that repo and is included here as a git submodule. This repo adds a
multi-agent reasoning layer on top of it — nothing in `AortaCFD-app` changes,
and the two codebases evolve independently.

## What this is

A five-agent supervisor that runs this loop:

```
clinical text ──► IntakeAgent    ──► ClinicalProfile (JSON)
                  LiteratureAgent ──► ParameterJustification (JSON, with citations)
                  ConfigAgent     ──► AortaCFD config.json (validated)
                  ExecutionAgent  ──► runs the full pipeline via run_patient.py
                  ResultsAgent    ──► natural-language summary + interactive Q&A
```

Every agent decision is logged to `output/<case>/agent_trace.jsonl` so you
can audit exactly which tools were called, which citations were used, and
which numerical values backed every clinical statement.

## Why it exists

The original `AortaCFD-app` pipeline is JSON-driven and assumes the user can
configure a CFD case. That is a reasonable assumption for research engineers
but a barrier for:

- **Clinicians** — who have free-text reports, not JSON configs
- **Independent reviewers** — who want to see why a parameter was chosen, not
  just the final value
- **Anyone reproducing a result** — who benefits from an auditable chain from
  clinical input to simulation output

The agent layer closes those gaps while leaving the CFD core untouched.

## Status

- **Phase 0** — repo scaffolded, submodule wired to `AortaCFD-app@afeffe5a`
- Phase A–B (current): skeleton + shared infrastructure
- Phase C: port backends/loop/tools from the prior single-agent prototype
- Phase D1–D5: five specialist agents
- Phase D6: coordinator + end-to-end CLI
- Phase E: demo + pitch

See `/home/mchi4jw4/.claude/plans/typed-herding-jellyfish.md` (author's
working plan) for the full rollout.

## Quick start

```bash
# One-time setup
git clone --recurse-submodules https://github.com/JieWangnk/aortacfd-agent.git
cd aortacfd-agent
python -m venv venv && source venv/bin/activate
pip install -e .

# Optional: install the Anthropic SDK if you want to use Claude
pip install -e '.[anthropic]'

# Dry-run (no LLM required — uses a scripted FakeBackend)
aortacfd-agent run --dry-run \
    --case external/aortacfd-app/cases_input/BPM120 \
    --clinical-text "5-year-old with aortic coarctation, peak echo gradient 35 mmHg, HR 110, BP 110/65" \
    --output output/BPM120_demo
```

## Architecture

```
aortacfd-agent/
├── external/aortacfd-app/     ← git submodule (pinned to a known-good commit)
├── src/aortacfd_agent/
│   ├── backends/               provider-agnostic LLM protocol
│   │   ├── base.py             LLMBackend protocol + wire types
│   │   ├── fake.py             scripted backend (tests + dry-run)
│   │   ├── anthropic_backend.py
│   │   ├── openai_compat.py    Ollama / vLLM / OpenAI / any compatible
│   │   └── factory.py
│   ├── loop.py                 generic ReAct tool-use loop
│   ├── agents/                 the five specialists
│   │   ├── intake.py
│   │   ├── literature.py
│   │   ├── config.py
│   │   ├── execution.py
│   │   └── results.py
│   ├── tools/                  tool implementations each agent can call
│   ├── corpus/                 RAG corpus + ingestion
│   ├── schemas/                JSON schemas for inter-agent contracts
│   ├── prompts/                system prompts (versioned)
│   ├── trace/                  JSONL audit logger
│   ├── coordinator.py          top-level supervisor
│   └── cli.py                  `aortacfd-agent` entry point
├── tests/
└── docs/
```

## Design principles

- **Two repos, not one.** CFD core stays stable; agent layer evolves separately.
- **Provider-agnostic.** One `LLMBackend` protocol; swap Claude for Ollama with a CLI flag.
- **Deterministic by default.** `temperature=0.0`, stateless backends, reproducible `FakeBackend` for tests and demos.
- **Trust through provenance.** Every agent decision, tool call, and citation is logged in a JSONL trace.
- **No vendor lock-in.** Zero dependencies on LangChain / LlamaIndex / frameworks. Raw SDKs only, imported lazily.
- **AortaCFD-app is never modified.** The submodule is read-only; the agent layer calls it via subprocess or read-only imports.

## License

MIT — same as `AortaCFD-app`.
