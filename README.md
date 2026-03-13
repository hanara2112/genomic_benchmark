# dc-genomics-review

**Standalone review repo for DeepChem genomic sequence support.**

A minimal, PR-review-ready repository that demonstrates a complete DeepChem-style genomics pipeline: featurizer → loader → model → evaluation. Designed for a DeepChem maintainer to scan in one sitting.

## Quick Start

```bash
git clone <this-repo-url>
cd dc-genomics-review
pip install -e .

# Run featurizer tests (fast, no network)
pytest tests/test_featurizers.py -v

# Run all offline tests
pytest tests/ -v -m "not slow"

# Run full test suite (requires network for dataset/model downloads)
pytest tests/ -v
```

## Pipeline Overview

```
Raw DNA sequences (FASTA / CSV / GenomicsBenchmarks)
        │
        ▼
[1] Loader (load_genomic_benchmark)
        │   Downloads via genomic-benchmarks package
        │   Returns (tasks, datasets, transformers) — MolNet convention
        ▼
[2] Featurizer (DNAOneHotFeaturizer / DNAKmerFeaturizer)
        │   Converts DNA strings → numerical arrays
        │   Subclasses dc.feat.Featurizer
        ▼
[3] Dataset (NumpyDataset)
        │   Holds (X, y, w, ids)
        │   Split via dc.splits.RandomSplitter
        ▼
[4] Model (DNABERT2Model)
        │   Subclasses dc.models.torch_models.HuggingFaceModel
        │   BPE tokenization in _prepare_batch (like ChemBERTa)
        │   fit(dataset) / predict(dataset) / evaluate(dataset, metrics)
        ▼
[5] Metrics (dc.metrics.Metric)
        ROC-AUC, Accuracy, etc.
```

## Repository Structure

```
dc-genomics-review/
├── README.md                      ← You are here
├── RESULTS.md                     ← Benchmark status
├── requirements.txt
├── setup.py
├── genomics/
│   ├── __init__.py                ← Public API
│   ├── featurizers.py             ← DNAOneHotFeaturizer + DNAKmerFeaturizer
│   ├── loader.py                  ← _GenomicBenchmarkLoader + load_genomic_benchmark()
│   ├── dnabert2.py                ← DNABERT2Model (HuggingFaceModel subclass)
│   └── run_pipeline.py            ← One canonical end-to-end example
└── tests/
    ├── conftest.py                ← Shared fixtures
    ├── test_featurizers.py        ← 21 featurizer tests
    ├── test_loader.py             ← 5 loader tests
    └── test_dnabert2.py           ← 6 model tests
```

## DeepChem Mapping

| Standalone File | DeepChem Target Path | PR Scope |
|----------------|---------------------|----------|
| `genomics/featurizers.py` | `deepchem/feat/dna_featurizers.py` | New file |
| `genomics/loader.py` | `deepchem/molnet/load_function/load_genomic_benchmark.py` | New file |
| `genomics/dnabert2.py` | `deepchem/models/torch_models/dnabert2.py` | New file |
| `genomics/run_pipeline.py` | `examples/genomics_pipeline.py` | New file |
| `tests/test_featurizers.py` | `deepchem/feat/tests/test_dna_featurizers.py` | New file |
| `tests/test_loader.py` | `deepchem/molnet/tests/test_load_genomic_benchmark.py` | New file |
| `tests/test_dnabert2.py` | `deepchem/models/torch_models/tests/test_dnabert2.py` | New file |

**Registrations needed in DeepChem (`__init__.py` edits):**
- `deepchem/feat/__init__.py` — add `DNAOneHotFeaturizer`, `DNAKmerFeaturizer`
- `deepchem/models/torch_models/__init__.py` — add `DNABERT2Model`
- `deepchem/molnet/__init__.py` — add `load_genomic_benchmark`

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| BPE tokenization in model wrapper, not featurizer | Matches ChemBERTa pattern. `_prepare_batch` tokenizes raw strings on-the-fly. |
| `NumpyDataset` in loader, not `DiskDataset` | Genomic benchmark datasets fit in memory. Simpler for review. |
| Two featurizers in one file | Share constants (`BASE_TO_INDEX`), both are small. Reviewer sees all featurizer logic in one place. |
| One model (DNABERT-2) only | Once the pattern is accepted, NucleotideTransformer follows identically. Keeps review surface minimal. |
| `DummyFeaturizer` for transformer models | Raw DNA sequences stored in `X`, tokenized at train time. Standard DeepChem HuggingFace pattern. |
| `RandomSplitter` for genomics | DNA sequences lack molecule scaffolds. Chromosome-aware splitting is a future enhancement. |

## References

- Zhou, Z. et al. *DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome.* arXiv:2306.15006 (2023).
- Ji, Y. et al. *DNABERT.* Bioinformatics 37(15), 2112-2120 (2021).
- Gresova, K. et al. *Genomic Benchmarks.* BMC Genomic Data 24, 25 (2023).
