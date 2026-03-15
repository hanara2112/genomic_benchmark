# Second Loader Options (MolNet-Style API)

In addition to **Genomic Benchmarks** (`load_genomic_benchmark`), the project plans one additional dataset loader with the same MolNet-style API: `(tasks, (train, valid, test), transformers)`. Below are recommended options.

---

## 1. **DeepSEA** (recommended for same pipeline)

**What it is:** DeepSEA (Zhou & Troyanskaya, *Nature Methods* 2015) predicts chromatin effects from DNA sequence: transcription factor binding, DNase I hypersensitivity, and histone marks across cell types. Used in GENA-LM and many DNA foundation model papers.

| Aspect | Details |
|--------|---------|
| **Input** | DNA sequences, 1000 bp (200 bp target + 400 bp flanking each side; or extended 8000 bp) |
| **Output** | Multi-label: 919 chromatin features (DHS, histone marks, TF binding) |
| **Task** | Multi-label classification (binary per feature) |
| **Splits** | Official train/validation/test from Princeton |
| **Size** | ~4.4M training samples in the full bundle |
| **Download** | `deepsea_train_bundle.v0.9.tar.gz` (~3.7 GB) from [deepsea.princeton.edu](http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz) |

**Why it fits:**

- Same pipeline as Genomic Benchmarks: **raw DNA sequences → featurizer → model** (or DummyFeaturizer for DNABERT-2).
- Clear train/val/test; multi-task `y` and `w` map naturally to DeepChem’s `(X, y, w, ids)`.
- Establishes a second *source* (Princeton/DeepSEA) alongside Genomic Benchmarks, so users can compare models on both regulatory (Genomic Benchmarks) and chromatin (DeepSEA) tasks.

**Implementation sketch:**

- Add `load_deepsea()` in a new file (e.g. `load_deepsea.py` or under the same MolNet loader module).
- Download or accept path to the DeepSEA bundle; parse sequences and label matrix (format described in the bundle / [PyDeepSEA](https://github.com/PuYuQian/PyDeepSEA), [build-deepsea-training-dataset](https://github.com/jakublipinski/build-deepsea-training-dataset)).
- Build `NumpyDataset` or `DiskDataset` with `X` = list of 1000 bp sequences (or featurized via `DNAOneHotFeaturizer` / `DummyFeaturizer`), `y` = (N, 919), `w` = ones or mask for missing labels.
- Use the official split (or a fixed seed) to produce `train`, `valid`, `test`; return `(tasks, (train, valid, test), transformers)` where `tasks` is the list of 919 feature names (or a single task name like `"deepsea_chromatin"` for the full multi-label setup).
- Register in `deepchem/molnet/__init__.py` and add tests analogous to `test_load_genomic_benchmark.py`.

**Reference:** Zhou, J. & Troyanskaya, O. G. *Predicting effects of noncoding variants with deep learning-based sequence model.* Nature Methods 12, 931–934 (2015).

---

## 2. **Generic FASTA + labels loader**

**What it is:** A loader that reads (1) a FASTA (or directory of FASTA files) and (2) a CSV/TSV with sample IDs and labels, then returns a single dataset. Splits are created via a DeepChem `Splitter` (e.g. random, scaffold) so the return signature is still `(tasks, (train, valid, test), transformers)`.

| Aspect | Details |
|--------|---------|
| **Input** | FASTA file(s) + CSV/TSV (id, label_1, label_2, …) |
| **Output** | Classification or regression; task names from column headers |
| **Splits** | User-provided splitter (no “official” split unless encoded in the CSV) |
| **Size** | Arbitrary |

**Why it fits:**

- Same API as other MolNet loaders; works with any custom benchmark (e.g. user’s own FASTA + labels).
- Reuses existing featurizers (`DNAOneHotFeaturizer`, `DNAKmerCountFeaturizer`, `DummyFeaturizer`).
- Small implementation surface: parse FASTA, join with CSV by ID, build `NumpyDataset`, run splitter, return `(tasks, (train, valid, test), transformers)`.

**Implementation sketch:**

- `load_fasta_labels(fasta_path, labels_path, task_names=None, featurizer=None, splitter="random", ...)`.
- Parse FASTA to get `{seq_id: sequence}`; parse CSV to get `{id: [label_1, ...]}`; align by ID, build `X`, `y`, `w`, `ids`; create dataset and split; return MolNet tuple.

---

## 3. **METABRIC (clinical / expression — different modality)**

**What it is:** METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) is a breast cancer dataset: gene expression, copy number, clinical variables, and sometimes mutation data. Used for survival analysis, subtype classification, and biomarker discovery.

| Aspect | Details |
|--------|---------|
| **Input** | Sample × gene expression (and/or CNA, clinical); **not** raw DNA sequences |
| **Output** | Survival, subtype, or other clinical endpoints |
| **Splits** | Often random or by cohort; no standard “benchmark” split like Genomic Benchmarks |
| **Access** | cBioPortal, [synapse](https://www.synapse.org/), or processed versions (e.g. R/Python packages) |

**Why it might still fit:**

- **Same API:** A loader can return `(tasks, (train, valid, test), transformers)` with `tasks = ["survival", "subtype"]` (or similar), so it fits the “one more loader” goal from an API perspective.
- **Different pipeline:** Input is **not** raw DNA; it’s expression (or other sample-level features). So the pipeline is “expression (or clinical) → model,” not “sequence → featurizer → model.” Useful if the project wants to support “clinical genomics” in addition to “sequence-based genomics.”

**Caveats:**

- METABRIC does **not** provide long DNA sequences per sample; it’s expression/copy number/clinical. So it doesn’t exercise the same featurizers (DNAOneHot, DNAKmer, DummyFeaturizer for BERT) or the same models (DNABERT-2). It’s a complementary loader, not a drop-in replacement for a second *sequence* benchmark.
- If the GSoC scope is “DNA sequence loaders + models,” **DeepSEA or FASTA+labels** are a better fit. If the scope includes “genomics more broadly,” METABRIC is a reasonable second loader with a clear caveat in the docs.

**Implementation sketch:**

- Download or accept path to METABRIC expression (and optional clinical) matrix; define `tasks` (e.g. survival, subtype); build `X` (expression matrix), `y` (labels), `w`; split; return `(tasks, (train, valid, test), transformers)`. No DNA featurizer in the loop.

---

## Summary

| Loader | Input | Task | Same pipeline as Genomic Benchmarks? | Recommendation |
|--------|--------|------|--------------------------------------|-----------------|
| **DeepSEA** | DNA 1000 bp | Multi-label chromatin (919) | Yes (DNA → featurizer → model) | **Best for a second sequence loader** |
| **FASTA + labels** | FASTA + CSV | User-defined | Yes | Good for flexibility and custom benchmarks |
| **METABRIC** | Expression / clinical | Survival, subtype | No (different modality) | Optional “clinical genomics” loader |

**Suggested choice for the GSoC “one more loader”:** Implement **DeepSEA** as the second loader so both loaders are sequence-based and share the same featurizer/model pipeline. Optionally add a **generic FASTA+labels** loader for user-provided benchmarks. METABRIC can be documented as a possible future addition for clinical genomics with a different input type.
