# DeepChem Code Structure & Conventions vs dc-genomics-review

This document compares **dc-genomics-review** against **DeepChem’s code structure and conventions** as described in the contributor handbook (`helper-documents/GUIDE.md`). The “reference” here is the **DeepChem codebase layout, API contracts, and patterns**, not the deepchem-genomics research pipeline.

---

## 1. DeepChem structure (from GUIDE.md)

### 1.1 Pipeline (six stations)

```
Raw Data → [1] Loader → [2] Featurizer → [3] Dataset (X, y, w, ids)
                → [4] Splitter  → [5] Transformer
                → [6] Model (fit / predict / evaluate) → Metrics
```

### 1.2 Where things live

| Component   | DeepChem location | Convention |
|------------|-------------------|-------------|
| Featurizers | `deepchem/feat/` | Subclass `Featurizer`, implement `_featurize(datapoint)`; `featurize(datapoints)` from base. |
| Datasets    | `deepchem/data/datasets.py` | `NumpyDataset`, `DiskDataset` with `X`, `y`, `w`, `ids`. |
| Loaders     | `deepchem/data/data_loader.py` | Loaders use a Featurizer and create a Dataset. |
| MolNet loaders | `deepchem/molnet/load_function/` | Each loader returns `(tasks, (train, valid, test), transformers)`. |
| Torch/HF models | `deepchem/models/torch_models/` | New model = new file, subclass `TorchModel` or `HuggingFaceModel`, register in `__init__.py`. |
| Examples    | `examples/` | Notebooks/scripts showing user-facing API. |
| Tests       | Next to code or `tests/` | e.g. `deepchem/models/torch_models/tests/test_chemberta.py`. |

### 1.3 API contracts (GUIDE)

- **Dataset**: `X` (features), `y` (labels `[N, T]`), `w` (weights), `ids` (identifiers).
- **Featurizer**: “Knows nothing about models”; converts one instance → numerical feature; `_featurize(datapoint)`.
- **Model**: `fit(dataset, nb_epoch, ...)`, `predict(dataset)`, `evaluate(dataset, metrics)`, `save_checkpoint()` / `restore()`.
- **HuggingFaceModel**: Tokenization in `_prepare_batch`, not during featurization; loss from `outputs['loss']`; raw strings in `X` when using `DummyFeaturizer`.
- **MolNet**: Return `(tasks, (train, valid, test), transformers)`.

### 1.4 GUIDE’s “Quick Reference: DeepChem DNA/Biology Extension Map”

```
deepchem/
├── feat/
│   ├── bio_seq_featurizer.py   ← SAM/BAM/FASTA parsers (exists)
│   ├── sequence_featurizers/   ← extend here
│   └── dna_tokenizer.py        ← [NEW] K-mer or BPE tokenizer
├── data/
│   ├── datasets.py             ← use as-is
│   └── data_loader.py          ← [EXTEND] if new loader type
├── models/torch_models/
│   ├── hf_models.py            ← extend, don't modify
│   ├── chemberta.py            ← template to follow
│   ├── prot_bert.py            ← biological precedent
│   └── dna_bert2.py            ← [NEW] primary deliverable
├── molnet/load_function/
│   └── load_genomics_benchmarks.py ← [NEW] GenomicsBenchmarks loader
└── metrics/
    └── genomics_metrics.py     ← [NEW] optional
```

---

## 2. Your repo vs DeepChem structure and conventions

### 2.1 File layout and target paths

| Your repo (standalone) | DeepChem target (from GUIDE + your README) | Convention match |
|------------------------|--------------------------------------------|------------------|
| `genomics/featurizers.py` | `deepchem/feat/dna_featurizers.py` (GUIDE suggests `sequence_featurizers/` or new file) | **OK** — GUIDE says “no DNAFeaturizer, no KmerFeaturizer”; a single `dna_featurizers.py` or under `sequence_featurizers/` both fit. |
| `genomics/loader.py` | `deepchem/molnet/load_function/load_genomic_benchmark.py` | **OK** — MolNet loaders live in `load_function/`; GUIDE explicitly asks for “load_genomics_benchmarks” or similar. |
| `genomics/dnabert2.py` | `deepchem/models/torch_models/dnabert2.py` (GUIDE: `dna_bert2.py`) | **OK** — Same directory and naming idea; minor naming (dna_bert2 vs dnabert2) is cosmetic. |
| `genomics/run_pipeline.py` | `examples/genomics_pipeline.py` | **OK** — Examples live in `examples/`. |
| `tests/test_*.py` | `deepchem/feat/tests/`, `molnet/tests/`, `torch_models/tests/` | **OK** — DC keeps tests next to or under the module; your layout is fine for a standalone repo and maps cleanly. |

Your layout matches where DeepChem would put each piece.

---

### 2.2 Featurizers: contract and style

| Convention (GUIDE) | Your implementation | Verdict |
|--------------------|---------------------|--------|
| Subclass `Featurizer`, implement `_featurize(datapoint)` | All four featurizers extend `deepchem.feat.base_classes.Featurizer` and implement `_featurize` | **Match** |
| “Featurizer knows nothing about models” | Featurizers only convert DNA → arrays/lists | **Match** |
| Docstrings | NumPy-style docstrings with Parameters/Returns/Examples | **Match** (GUIDE: “NumPy docstring style that DeepChem uses”) |

**Gap (optional):** GUIDE suggests `deepchem/feat/dna_tokenizer.py` for a K-mer or BPE tokenizer. You don’t add a separate tokenizer module; BPE is inside the model. That matches the HuggingFaceModel pattern (tokenization in `_prepare_batch`), so **no change required** unless you want a reusable DNA tokenizer for other use cases.

---

### 2.3 Dataset and loader

| Convention (GUIDE) | Your implementation | Verdict |
|--------------------|---------------------|--------|
| Dataset has `X`, `y`, `w`, `ids` | Loader builds `DiskDataset.from_numpy(X, y, w, ids)`; `X` = featurized or raw strings | **Match** |
| Loader uses a Featurizer and creates a Dataset | `_GenomicBenchmarkLoader` uses `self.featurizer`, builds one Dataset then splits | **Match** |
| MolNet return | `(tasks, (train, valid, test), transformers)` | **Match** |
| DummyFeaturizer for raw sequences | “Use DummyFeaturizer (raw sequences in X)” (GUIDE expected API); you use `featurizer=None` / `"raw"` → DummyFeaturizer | **Match** |

So your loader and dataset usage follow DeepChem’s data contract and MolNet convention.

---

### 2.4 Model: HuggingFaceModel pattern

| Convention (GUIDE) | Your implementation | Verdict |
|--------------------|---------------------|--------|
| Subclass `HuggingFaceModel` | `DNABERT2Model` subclasses `HuggingFaceModel` (with fallback to bundled wrapper if needed) | **Match** |
| “Study ChemBERTa / ProtBERT” | Same pattern: task, tokenizer, model, `super().__init__(model=..., tokenizer=..., ...)` | **Match** |
| Tokenization in `_prepare_batch`, not in featurizer | `_prepare_batch` tokenizes raw strings from `X`; BPE inside model | **Match** |
| `trust_remote_code=True` for custom HF | Used for AutoTokenizer / AutoConfig / from_pretrained | **Match** |
| `fit(dataset, nb_epoch)`, `predict`, `evaluate` | Exposed via HuggingFaceModel base | **Match** |

Optional from GUIDE: “Add `feature_extractor` task mode” and “expose `get_last_hidden_state()`”. Your model focuses on classification; adding a `feature_extractor` mode and an embedding API would align even more with GUIDE’s “reusable templates” but is not required for the current scope.

---

### 2.5 Expected API contract (GUIDE Section 5)

GUIDE’s expected usage:

```python
# 1. Load via Loader with DummyFeaturizer (raw sequences in X)
loader = dc.data.CSVLoader(["label"], feature_field="sequence",
                           featurizer=dc.feat.DummyFeaturizer())
dataset = loader.create_dataset("genomics_data.csv")

# 2–5. Model instantiation, fit, predict, evaluate
model = DNAFoundationModel(task='classification', n_tasks=1)
model.fit(dataset, nb_epoch=5)
preds = model.predict(test_dataset)
results = model.evaluate(test_dataset, metrics=[dc.metrics.Metric(dc.metrics.roc_auc_score)])
```

Your equivalent:

- Load: `load_genomic_benchmark(..., featurizer=None)` → DummyFeaturizer, raw sequences in `X`. ✅  
- Model: `DNABERT2Model(task='classification', n_tasks=1, ...)`, `fit(train, nb_epoch=3)`, `evaluate(test, metrics)`. ✅  

So you satisfy the expected API contract; the only difference is MolNet-style loader (`load_genomic_benchmark`) instead of a generic CSV loader, which is exactly what GUIDE asks for (“Add load_genomics_benchmarks”).

---

### 2.6 Registration and discoverability (GUIDE)

| Convention | Your repo | Verdict |
|------------|-----------|--------|
| “Register model in `deepchem/models/__init__.py`” | Documented in README “Registrations needed”; not done in standalone repo | **Expected** — you’re standalone; PR would add this. |
| “Register tokenizer in `deepchem/feat/__init__.py`” | No separate tokenizer; BPE in model — nothing to register in feat | **OK** |
| “Add example notebook to `examples/`” | `run_pipeline.py` → `examples/genomics_pipeline.py`; you have examples/ | **Match** |
| “Write unit tests in `deepchem/models/torch_models/tests/`” | You have `tests/test_dnabert2.py`, etc. | **Match** |

---

### 2.7 Code quality (GUIDE Section 11)

| Convention | Your repo | Verdict |
|------------|-----------|--------|
| “Well-typed, well-documented” | Docstrings present; GUIDE suggests stricter typing in loader/run_pipeline | **Mostly match**; optional: tighten types. |
| “Docstrings using NumPy docstring style” | Featurizers and main classes use Parameters/Returns/Examples | **Match** |
| “Every class/function you add must have corresponding pytest tests” | test_featurizers, test_loader, test_dnabert2 | **Match** |

---

## 3. Gaps vs DeepChem structure/conventions (actionable)

1. **File naming**  
   GUIDE’s map uses `dna_bert2.py`; you use `dnabert2.py`. Both are fine; if you want to mirror GUIDE exactly when upstreaming, use `dna_bert2.py` in DeepChem.

2. **Featurizer location**  
   GUIDE mentions `sequence_featurizers/` or a new file. Your mapping to `deepchem/feat/dna_featurizers.py` is consistent; alternative is `deepchem/feat/sequence_featurizers/dna_featurizers.py`. Either is acceptable.

3. **Loader function name**  
   GUIDE says `load_genomics_benchmarks`; you have `load_genomic_benchmark`. Minor; DeepChem could accept either; consider aligning name with GUIDE when submitting the PR.

4. **Feature extractor / embeddings**  
   GUIDE recommends “Add `feature_extractor` task mode” and “expose `get_last_hidden_state()`” (ProtBERT precedent). Optional for first PR but would strengthen alignment.

5. **Genomics metrics**  
   GUIDE’s map has `metrics/genomics_metrics.py` as [NEW]. You use `dc.metrics` (e.g. ROC-AUC). Adding genomics-specific metrics (e.g. MCC, AUPRC) would match GUIDE’s “Current Gaps” but is optional.

6. **Type hints**  
   Stricter typing in `loader.py` and `run_pipeline.py` (e.g. `Tuple[Dataset, ...]`) would match GUIDE’s “well-typed” expectation.

---

## 4. Summary: structure and conventions

| Area | DeepChem convention (GUIDE) | dc-genomics-review | Verdict |
|------|-----------------------------|---------------------|--------|
| Pipeline | Loader → Featurizer → Dataset → Model → Metrics | Same flow | **Match** |
| Featurizers | `feat/`, `_featurize`, no model knowledge | Same | **Match** |
| Dataset | X, y, w, ids; NumpyDataset/DiskDataset | Same | **Match** |
| MolNet | `(tasks, (train, valid, test), transformers)` | Same | **Match** |
| Raw DNA in X | DummyFeaturizer | featurizer=None / "raw" → DummyFeaturizer | **Match** |
| Model location | `models/torch_models/`, new file per model | dnabert2.py → dna_bert2.py | **Match** |
| Model base | HuggingFaceModel, _prepare_batch for tokenization | Same | **Match** |
| Loader location | `molnet/load_function/` | load_genomic_benchmark | **Match** |
| Examples | `examples/` | run_pipeline → examples | **Match** |
| Tests | pytest, next to or under module | test_*.py | **Match** |
| Docstrings | NumPy style | Used in featurizers/model | **Match** |
| feature_extractor / get_embeddings | Recommended (ProtBERT) | Not yet | **Optional** |
| genomics_metrics.py | In extension map | Use dc.metrics only | **Optional** |

**Conclusion:** dc-genomics-review follows **DeepChem’s code structure and conventions** from GUIDE.md: correct placement of featurizers, loader, model, and examples; same API contracts (Dataset, Featurizer, Model, MolNet); and the same HuggingFaceModel pattern (including tokenization in `_prepare_batch` and DummyFeaturizer for raw sequences). Optional improvements: align loader/model filenames with GUIDE, add feature_extractor/embeddings and genomics metrics if you want to match the extension map fully, and tighten type hints.
