# Free-text clinical referral → reproducible patient-specific CFD

### A multi-agent LLM layer over AortaCFD for bulk-hemodynamic assessment with explicit provenance

---

**One-sentence summary.** `aortacfd-agent` turns a free-text clinical referral into a reproducible cardiovascular CFD simulation and a natural-language clinical summary, with every numerical parameter traced back to a literature citation and every agent decision recorded in an auditable JSONL trace.

**Why this exists.** The underlying CFD pipeline ([`AortaCFD-app`](https://github.com/JieWangnk/AortaCFD-app)) is reliable and well-documented, but it is JSON-driven and assumes the user can configure an OpenFOAM case. That is a reasonable assumption for CFD researchers and a barrier for everyone else — clinicians have narrative reports, reviewers want to see *why* a parameter was picked, and reproducibility requires an explicit chain from clinical input to simulation output. This project closes all three gaps without modifying the CFD core.

---

### Architecture at a glance

```
Clinical referral (free text)
        │
        ▼
┌────────────────────┐   ClinicalProfile JSON (validated)
│ 1. IntakeAgent     │── age, HR, BP, diagnosis, imaging, constraints
└──────────┬─────────┘
           ▼
┌────────────────────┐   ParameterJustification JSON with citations
│ 2. LiteratureAgent │── RAG over a curated cardiovascular-CFD corpus
└──────────┬─────────┘   (Wang 2025, Esmaily 2011, Stergiopulos 1999, …)
           ▼
┌────────────────────┐   agent_config.json  +  agent_rationale.md
│ 3. ConfigAgent     │── deterministic patcher, lightweight validator
└──────────┬─────────┘
           ▼
┌────────────────────┐   run_patient.py via subprocess (dry-run or full)
│ 4. ExecutionAgent  │── isolated, remoting-ready, HPC-compatible
└──────────┬─────────┘
           ▼
┌────────────────────┐   natural-language clinical summary + Q&A REPL
│ 5. ResultsAgent    │── every number grounded in tool-call outputs
└────────────────────┘
           ▼
   agent_trace.jsonl   (append-only audit trail of every stage)
```

All five agents share a single provider-agnostic `LLMBackend` protocol — Claude, GPT, Ollama, vLLM, or a scripted `FakeBackend` all plug in with one CLI flag. SDKs are imported lazily so a new user can run the full demo (`examples/end_to_end_demo.py`) offline in about half a second with no credentials and no network.

---

### Where this intersects NaCTeM's research programme

- **Generative AI in biomedicine (Horizon Europe projects).** The IntakeAgent is a production-grade example of structured information extraction from clinical narrative, enforced by a JSON schema and validated at every step. The LiteratureAgent is a domain-specific retrieval-augmented generation loop with explicit citation discipline.
- **Patient-oriented clinical NLP.** The input is narrative clinical text; the output is a narrative clinical summary. The pipeline is designed to preserve the clinician's wording where it is clinically meaningful and to flag every inferred value as such.
- **Knowledge-enhanced search (Kleio-style).** The `search_corpus` tool is a first-class citizen in the agent loop. Every parameter decision must be backed by a verbatim quote from the corpus, and the audit trail records the exact queries the agent issued.
- **Trustworthy AI with provenance.** Every model call, every tool call, every agent decision, and every citation is persisted to `agent_trace.jsonl`. The trace is the single source of truth for "what did the system do on behalf of this patient?". This is the same design principle NaCTeM has been arguing for in its trustworthy-AI work on mental health NLP.
- **Cardiovascular modelling as a new application domain.** The NLP group has a track record on mental health, drug safety, and biomedical literature mining. Cardiovascular CFD is a natural next target because the clinical data (echo, CT, 4D flow MRI) is well-structured, the decision points are well-documented, and the downstream simulation is deterministic and reproducible — a cleaner testbed than most clinical NLP targets.

---

### Current status (April 2026)

| Component | Status | Tests |
|---|---|---|
| Provider-agnostic backend layer | ✓ ready | 14 unit tests |
| Generic ReAct tool-use loop | ✓ ready | 4 unit tests |
| Focused tool wrappers (geometry, physics, mesh, config, results) | ✓ ready | 37 unit tests |
| IntakeAgent | ✓ ready | 10 unit tests + 5 synthetic referrals |
| LiteratureAgent + FakeCorpusStore | ✓ ready | 19 unit tests |
| ConfigAgent (deterministic patcher) | ✓ ready | 14 unit tests |
| ExecutionAgent (subprocess wrapper) | ✓ ready | 18 unit tests |
| ResultsAgent + REPL | ✓ ready | 18 unit tests |
| Coordinator + `aortacfd-agent` CLI | ✓ ready | 10 unit tests |
| RAG corpus with real PDFs | 🟡 seed list only, ingestion script ready |
| End-to-end live-LLM integration | 🟡 skeleton ready, awaiting API keys |
| Clinical validation on real referrals | ⬜ future collaboration |

**115 passing tests, 0 failing, 0.25 seconds total runtime, zero network calls.** The whole pipeline runs end-to-end offline right now.

---

### What a collaboration would look like

This is where I would welcome NaCTeM input most:

1. **Corpus curation.** A joint choice of ~20 open-access cardiovascular-CFD papers plus a biomedical-NLP pass to evaluate how well the RAG retrieval matches expert expectations.
2. **Extraction quality evaluation.** Running the IntakeAgent on anonymised real referrals (not synthetic fixtures) and measuring extraction accuracy per field against a human-annotated reference set.
3. **Grounding-violation detection.** A post-hoc checker that scans every ResultsAgent response for numerical values not present in any tool call — the same trustworthy-AI principle applied to CFD outputs. This looks like natural PhD-student work.
4. **Clinical usability study.** Comparing clinician satisfaction and confidence on CFD reports generated by the agent versus the existing JSON-driven workflow, with and without the literature-backed rationale document.

None of these require modifying the CFD core or bringing NLP researchers up to speed on OpenFOAM — the agent layer is fully self-contained.

---

### One-paragraph ask

If any of this looks interesting, I would value 30 minutes to walk through the live demo and discuss whether a lightweight collaboration (a joint meeting, a shared paper on the clinical-NLP layer, or just informal feedback on the corpus and schemas) fits into the group's programme. The code is on a private GitHub branch and can be made available immediately; the pipeline is designed to be handed off to someone who wants to work on the NLP side without needing any CFD background.

— Jie Wang,  Department of Mechanical, Aerospace and Civil Engineering
