# Proposal: DeepChem Genomic Sequence Support

**Standalone review repo for upstream integration.**  
This document outlines the project scope, work plan, milestones, and resources for integrating genomic sequence support into DeepChem (e.g. for a GSoC or similar program).

---

## Introduction

DeepChem provides featurizers, loaders, and models for molecular and materials ML. Genomic sequence data (DNA/RNA) shares similar needs: standardized featurization, benchmark datasets with official splits, and model wrappers for foundation models (e.g. DNABERT-2). This project delivers a **complete DeepChem-style genomics pipeline** that is:

- **PR-review-ready**: Featurizers, loader, and DNABERT-2 model follow existing DeepChem patterns (MolNet loader API, `Featurizer`, `HuggingFaceModel`).
- **Benchmarked**: Results on Genomic Benchmarks (e.g. `dummy_mouse_enhancers_ensembl`, `human_nontata_promoters`) with ROC-AUC and interpretation in [RESULTS.md](RESULTS.md).
- **Designed for upstream**: Clear mapping to DeepChem paths and `__init__.py` registrations (see README).

By creating a minimal, review-friendly implementation in this repo, the work can be proposed as **Phase 1: upstream integration** and **Phase 2: extended benchmarks and polish**.

### Scope: starting work

The **starting work** for this project is:

1. **DNA BERT** — Foundation models for DNA sequence understanding. The repo currently implements **DNABERT-2** (BPE tokenization, HuggingFace); the original **DNABERT** (Ji et al., k-mer tokenization) can be supported via the same `HuggingFaceModel` pattern or a dedicated wrapper.
2. **MethylBERT** — A BERT-style model for DNA methylation prediction (e.g. N6-methyladenine and related tasks). To be added as a DeepChem `HuggingFaceModel` wrapper, reusing the same pipeline (raw sequences → tokenization in model → classification/regression).
3. **One more loader** — In addition to the existing **Genomic Benchmarks** loader (`load_genomic_benchmark`), add one additional dataset loader with the same MolNet-style API. **Recommended: DeepSEA** (chromatin prediction from 1000 bp DNA; 919 multi-label tasks; Princeton train/val/test). Alternatives: generic FASTA+labels loader, or METABRIC for clinical/expression (different modality). See [docs/SECOND_LOADER_OPTIONS.md](docs/SECOND_LOADER_OPTIONS.md).

Featurizers and the first loader are already in place; the starting scope completes the picture with DNA BERT, MethylBERT, and a second loader.

---

## Current State (Pre–Work Plan)

| Component        | Status | Notes |
|------------------|--------|--------|
| Featurizers      | Done   | `DNAOneHotFeaturizer`, `DNAKmerFeaturizer`, `DNAKmerCountFeaturizer`, `KmerFrequencyFeaturizer`; 21 tests. |
| Loader           | Done   | `load_genomic_benchmark()` (MolNet-style); official split; 8 tests. |
| DNABERT-2 model  | Done   | `DNABERT2Model` (HuggingFaceModel); 6 tests; benchmarked. |
| Pipeline / example| Done   | `run_pipeline.py`; tutorial notebook (`human_nontata_promoters`). |
| Results & plots  | Done   | [RESULTS.md](RESULTS.md), `plot_results.py` → `plots/`. |
| Integration review| Done   | [docs/DEEPCHEM_INTEGRATION_REVIEW.md](docs/DEEPCHEM_INTEGRATION_REVIEW.md). |

**Planned (not yet done):** Run on `human_enhancers_cohn`; multi-seed averaging for confidence intervals; optional NucleotideTransformer following the same pattern.

---

## Work Plan

### Phase 1: Upstream integration (DeepChem)

1. **Featurizers**
   - Move `genomics/featurizers.py` → `deepchem/feat/dna_featurizers.py`.
   - Register in `deepchem/feat/__init__.py`.
   - Move and adapt `tests/test_featurizers.py` → `deepchem/feat/tests/test_dna_featurizers.py`.

2. **Loaders** (Genomic Benchmarks + one more)
   - Add MolNet-style loader: `deepchem/molnet/load_function/load_genomic_benchmark.py` (from `genomics/loader.py`).
   - Add **one more** dataset loader with the same MolNet-style API. Recommended: **DeepSEA** (DNA → chromatin multi-label; see [docs/SECOND_LOADER_OPTIONS.md](docs/SECOND_LOADER_OPTIONS.md)). Alternatives: generic FASTA+labels, or METABRIC (expression/clinical).
   - Register both in `deepchem/molnet/__init__.py`.
   - Move and adapt `tests/test_loader.py` → `deepchem/molnet/tests/test_load_genomic_benchmark.py`; add tests for the second loader.

3. **Models: DNA BERT and MethylBERT**
   - Move `genomics/dnabert2.py` → `deepchem/models/torch_models/dnabert2.py` (DNABERT-2; DNA BERT family).
   - Add **MethylBERT** wrapper (e.g. `methylbert.py`) following the same `HuggingFaceModel` pattern for methylation prediction.
   - Register both in `deepchem/models/torch_models/__init__.py`.
   - Move and adapt `tests/test_dnabert2.py`; add tests for MethylBERT.

4. **Example and docs**
   - Add `examples/genomics_pipeline.py` (from `genomics/run_pipeline.py`).
   - Update DeepChem docs/README as needed for genomics usage and install (`.[data]` / `.[all]`).

Deliverables: All components live in DeepChem; existing tests pass in-tree; one canonical example runs.

### Phase 2: Benchmarks and robustness

1. **Additional benchmark**
   - Run pipeline on `human_enhancers_cohn` (27K sequences); add results and interpretation to RESULTS.md (or equivalent in DeepChem).

2. **Multi-seed evaluation**
   - Add optional multi-seed runs (e.g. 3 seeds) for key benchmarks; report mean ± std or confidence intervals in RESULTS.

3. **Integration test**
   - Add a minimal integration test: load a benchmark with `featurizer=None`, build `DNABERT2Model`, run one step of `fit()` (can be marked slow/GPU if needed).

4. **Tutorial and type hints**
   - Align tutorial notebook with the public featurizer set (e.g. `DNAKmerCountFeaturizer` for k-mer section).
   - Add type hints in loader and `run_pipeline` where helpful (e.g. `Tuple[Dataset, ...]`).

Deliverables: One more dataset in RESULTS; multi-seed option; one integration test; clearer tutorial and types.

### Phase 3 (Optional / stretch)

- Add **MethylBERT** wrapper (if not completed in Phase 1–2) and/or **NucleotideTransformer** using the same `HuggingFaceModel` pattern as DNABERT-2.
- Document DummyFeaturizer requirement (DeepChem version) to avoid reliance on the local stub.

---

## Future work: GENA-LM

**GENA-LM** (Fishman et al., bioRxiv 2023) is a family of open-source foundational DNA language models for **long sequences** (up to **36 thousand base pairs**), addressing a limitation of earlier models like DNABERT (512 bp). Key aspects:

- **Architecture**: Transformer with BPE tokenization; supports both full attention (≈4.5 kb, 512 tokens) and **sparse attention** (BigBird-style, up to 36 kb / 4096 tokens).
- **Pre-training**: Masked language modeling on human T2T genome (and optionally multispecies / 1000 Genomes augmented data).
- **Downstream tasks** (in the paper): promoter activity, splice sites, chromatin profiles (DeepSEA), enhancer activity (DeepSTARR), polyadenylation (APARENT). GENA-LM matches or exceeds task-specific and prior foundation models.
- **Availability**: [GitHub](https://github.com/AIRI-Institute/GENA_LM), HuggingFace [AIRI-Institute](https://huggingface.co/AIRI-Institute) with `gena-lm-` prefix.

**Proposed future work** (beyond the current proposal): Integrate GENA-LM into DeepChem as a `HuggingFaceModel` wrapper (same pattern as DNABERT-2 and MethylBERT). This would enable:

- Fine-tuning and evaluation of long-context DNA models (4.5–36 kb) within the same featurizer–loader–model pipeline.
- Fair comparison with DNA BERT, MethylBERT, and NucleotideTransformer on tasks where long context matters (e.g. promoter prediction at 2–16 kb, chromatin profiling with extended context).

Reference: Fishman, V. et al. *GENA-LM: A Family of Open-Source Foundational Models for Long DNA Sequences.* bioRxiv (2023). doi:10.1101/2023.06.12.544594.

---

## Milestones and Timeline

| Milestone | Target | Deliverables |
|-----------|--------|--------------|
| **M1: Featurizers + loaders upstream** | Week 3–4 | Featurizers and both loaders (Genomic Benchmarks + one more) in DeepChem; tests passing; `__init__.py` registrations. |
| **M2: Models (DNA BERT, MethylBERT) + example** | Week 5–6 | DNABERT-2 and MethylBERT in DeepChem; tests; `examples/genomics_pipeline.py`; basic docs. |
| **M3: Benchmarks and quality** | Week 7–8 | `human_enhancers_cohn` results; multi-seed option; integration test; tutorial/type hints. |
| **M4 (stretch)** | Week 9+ | NucleotideTransformer and/or extra docs. |

Timeline is indicative; adjust to program duration (e.g. 12-week GSoC: M1 by week 4, M2 by week 8, M3 by week 12).

---

## Testing Plan

- **Unit tests**
  - Featurizers: input/output shapes, padding/truncation, base alphabet (existing 21 tests).
  - Loaders: dataset list, invalid name raises, official split preservation, featurizer resolution, caching (existing 8 tests for Genomic Benchmarks; add tests for second loader).
  - DNABERT-2 / MethylBERT: loading, config, `_prepare_batch` (existing 6 tests for DNABERT-2; add analogous tests for MethylBERT).
- **Integration test**
  - One end-to-end test: `load_genomic_benchmark(..., featurizer=None)` → `DNABERT2Model` → one `fit()` step (optional: mark `slow` or require GPU).
- **How to run**
  - Fast (no network): `pytest tests/ -v -m "not slow"`.
  - Full: `pytest tests/ -v` (may need `.[data]` and `.[all]` for loader/model).

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| **DeepChem review bandwidth** | Split into small PRs (featurizers → loader → model → example). Link to this repo and [DEEPCHEM_INTEGRATION_REVIEW.md](docs/DEEPCHEM_INTEGRATION_REVIEW.md). |
| **GPU needed for DNABERT-2** | Document CPU vs GPU; CI can skip heavy model tests or use a tiny run; tutorial/benchmarks can note “e.g. Kaggle T4”. |
| **Dependency churn (transformers, genomic-benchmarks)** | Pin or lower-bound key deps in `setup.py` / `requirements.txt`; use extras `.[data]` and `.[all]` so minimal install still works. |
| **Scope creep** | Keep Phase 1–2 as core scope; Phase 3 as optional. |

---

## Resources Required

- **Environment**: Python 3.8+; `pip install -e .` for minimal; `pip install -e ".[data]"` for loader/datasets; `pip install -e ".[all]"` for full pipeline (torch, transformers, genomic-benchmarks).
- **Compute**: Featurizer and loader tests run on CPU. DNABERT-2 training/benchmarks benefit from a GPU (e.g. T4); full reproduction can be documented as “with GPU”.
- **Storage**: Space for Genomic Benchmarks datasets and model weights (order of a few GB depending on datasets used).
- **Contingency**: If upstream PRs are delayed, Phase 2 (benchmarks, integration test, tutorial) can proceed in this repo and be merged to DeepChem later.

---

## Summary

This proposal turns the existing **dc-genomics-review** implementation into a clear work plan: **Starting work** is DNA BERT (DNABERT-2), MethylBERT, and one more loader alongside Genomic Benchmarks. **Phase 1** upstreams featurizers and both loaders; **Phase 2** upstreams DNABERT-2 and MethylBERT with an example; **Phase 3** adds benchmarks, integration test, and optional NucleotideTransformer. **Future work** proposes GENA-LM integration for long-sequence DNA (up to 36 kb). Milestones, testing, risks, and resources are specified so the project is easy to evaluate and execute (e.g. for GSoC).
