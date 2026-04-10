# Seed RAG corpus

The LiteratureAgent retrieves from a small set of cardiovascular CFD papers
chosen to cover the default parameter-selection decisions the config agent
needs to justify. We do **not** redistribute the PDFs — fetch each one from
its open-access source and drop it into this directory. The ingest script
(`scripts/ingest_corpus.py`) will index whatever it finds.

## Papers

Fetch these 10 seed papers. Open-access versions are linked where available.

1. **Steinman DA, Hoi Y, Fahy P, et al. (2013).** Variability of computational
   fluid dynamics solutions for pressure and flow in a giant aneurysm: the
   ASME 2012 CFD Challenge. *J Biomech Eng* 135(2):021016.
   — why the field needs reproducibility

2. **Valen-Sendstad K, Bergersen AW, Shimogonya Y, et al. (2018).** Real-world
   variability in the prediction of intracranial aneurysm wall shear stress.
   *Cardiovasc Eng Technol* 9(4):544–553.
   — WSS variability across groups

3. **Pfaller MR, Pham J, Verma A, et al. (2021).** On the periodicity of
   cardiovascular fluid dynamics simulations. *Ann Biomed Eng* 49(12):3574–3592.
   — cycle convergence, Windkessel initialisation

4. **Murray CD (1926).** The physiological principle of minimum work: I. The
   vascular system and the cost of blood volume. *Proc Natl Acad Sci USA*
   12(3):207–214.
   — Murray's law, flow allocation by radius³

5. **Olufsen MS (1999).** Structured tree outflow condition for blood flow in
   larger systemic arteries. *Am J Physiol* 276(1):H257–H268.
   — characteristic impedance, PWV from vessel area

6. **Stergiopulos N, Westerhof BE, Westerhof N (1999).** Total arterial
   inertance as the fourth element of the windkessel model. *Am J Physiol*
   276(1):H81–H88.
   — tau scaling, compliance distribution

7. **Wang J et al. (2025).** Pediatric aortic coarctation CFD with
   user-specified flow splits. [Preprint / in-preparation].
   — why Murray's law fails for coarctation, flow split calibration

8. **Esmaily Moghadam M, Bazilevs Y, Hsia T-Y, et al. (2011).** A comparison of
   outlet boundary treatments for prevention of backflow divergence with
   relevance to blood flow simulations. *Comput Mech* 48(3):277–291.
   — backflow stabilisation, βT parameter

9. **Büchner A et al. (2024).** Reference LES of a healthy adult aorta with
   4D flow MRI comparison. [Journal TBD].
   — validation baseline for VOL04-class geometries

10. **Updegrove A, Wilson NM, Merkow J, et al. (2017).** SimVascular: an open
    source pipeline for cardiovascular simulation. *Ann Biomed Eng* 45(3):525–541.
    — cross-solver baseline, FEM tet meshing conventions

## After you've dropped PDFs here

```bash
# From the repo root
python scripts/ingest_corpus.py \
    --papers src/aortacfd_agent/corpus/papers/ \
    --index src/aortacfd_agent/corpus/index/
```

This chunks each PDF into ~500-token passages, embeds them, and writes a
Chroma persistence directory that the `search_corpus` tool queries.
