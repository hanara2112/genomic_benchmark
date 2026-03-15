# Google Summer of Code — Final Proposal
# DeepChem Genomic Sequence Support

**Project:** Standalone review repo for upstream integration of DNA/RNA sequence support into DeepChem  
**Document:** Overlay of proposal narrative, key tables, code snippets, and plot descriptions (~10–15 pages)

---

## 1. Introduction and Scope

DeepChem provides featurizers, loaders, and models for molecular and materials ML. Genomic sequence data (DNA/RNA) shares similar needs: standardized featurization, benchmark datasets with official splits, and model wrappers for foundation models (e.g. DNABERT-2). This project delivers a **complete DeepChem-style genomics pipeline** that is:

- **PR-review-ready**: Featurizers, loader, and DNABERT-2 model follow existing DeepChem patterns (MolNet loader API, `Featurizer`, `HuggingFaceModel`).
- **Benchmarked**: Results on Genomic Benchmarks with ROC-AUC and interpretation.
- **Designed for upstream**: Clear mapping to DeepChem paths and `__init__.py` registrations.

### 1.1 Starting work

| Component | Description |
|-----------|-------------|
| **DNA BERT** | DNABERT-2 (BPE) implemented; original DNABERT (k-mer) can follow the same `HuggingFaceModel` pattern. |
| **MethylBERT** | BERT-style model for DNA methylation prediction; to be added as a DeepChem `HuggingFaceModel` wrapper. |
| **One more loader** | In addition to Genomic Benchmarks, add one loader with the same MolNet-style API. **Recommended: DeepSEA** (chromatin from DNA; 919 tasks). See [SECOND_LOADER_OPTIONS.md](SECOND_LOADER_OPTIONS.md). |

---

## 2. Current State and Repository Structure

### 2.1 Component status

| Component | Status | Notes |
|-----------|--------|--------|
| Featurizers | Done | `DNAOneHotFeaturizer`, `DNAKmerFeaturizer`, `DNAKmerCountFeaturizer`, `KmerFrequencyFeaturizer`; 21 tests. |
| Loader | Done | `load_genomic_benchmark()` (MolNet-style); official split; 8 tests. |
| DNABERT-2 model | Done | `DNABERT2Model` (HuggingFaceModel); 6 tests; benchmarked. |
| Pipeline / example | Done | `run_pipeline.py`; tutorial notebook (`human_nontata_promoters`). |
| Results & plots | Done | RESULTS.md, `plot_results.py` → `plots/`. |
| Integration review | Done | docs/DEEPCHEM_INTEGRATION_REVIEW.md. |

### 2.2 Repository layout

```
dc-genomics-review/
├── README.md
├── RESULTS.md
├── PROPOSAL.md
├── plot_results.py              # Matplotlib scripts → plots/
├── plots/                        # PNG figures
├── requirements.txt
├── setup.py
├── genomics/
│   ├── __init__.py               # Public API
│   ├── featurizers.py            # DNA featurizers
│   ├── loader.py                 # load_genomic_benchmark()
│   ├── dnabert2.py               # DNABERT2Model
│   └── run_pipeline.py           # End-to-end example
└── tests/
    ├── conftest.py
    ├── test_featurizers.py       # 21 tests
    ├── test_loader.py            # 8 tests
    └── test_dnabert2.py          # 6 tests
```

---

## 3. Pipeline Overview

```
Raw DNA sequences (FASTA / CSV / GenomicsBenchmarks)
        │
        ▼
[1] Loader (load_genomic_benchmark)
        │   Downloads via genomic-benchmarks package
        │   Preserves official benchmark test split
        │   Returns (tasks, datasets, transformers) — MolNet convention
        ▼
[2] Featurizer (DNAOneHotFeaturizer / DNAKmerFeaturizer / DummyFeaturizer for BERT)
        │   Converts DNA strings → numerical arrays (or keeps raw for BPE models)
        ▼
[3] Dataset (NumpyDataset / DiskDataset)
        │   (X, y, w, ids); validation split from official train when needed
        ▼
[4] Model (DNABERT2Model)
        │   BPE tokenization in _prepare_batch; fit / predict / evaluate
        ▼
[5] Metrics (dc.metrics.Metric)
        ROC-AUC, Accuracy, F1, etc.
```

---

## 4. Key Benchmark Tables

### 4.1 dummy_mouse_enhancers_ensembl (binary classification)

**Task:** Enhancer vs. non-enhancer. **Splits:** Train 968 | Valid 121 | Test 121. Max length 256.

| Model | Paradigm | Input | Valid ROC-AUC | Valid Acc | Test ROC-AUC | Test Acc | Time |
|-------|----------|-------|---------------|-----------|--------------|----------|------|
| CNN | Convolution | One-hot | 0.794 | 0.686 | 0.751 | 0.686 | 4 s |
| BiLSTM | Recurrent | One-hot | 0.811 | 0.736 | 0.774 | 0.711 | 21 s |
| DNABERT-1 | Attention | 6-mer | 0.797 | 0.736 | 0.801 | 0.694 | 157 s |
| **DNABERT-2** | **Attention** | **BPE** | **0.882** | **0.826** | **0.816** | **0.719** | **102 s** |

*Config: CNN/BiLSTM 10 epochs, batch 32, lr 1e-3; DNABERT-1/2 3 epochs, batch 8, lr 2e-5.*

### 4.2 human_nontata_promoters (binary classification)

| Model | Paradigm | Input | Test ROC-AUC | Notes |
|-------|----------|-------|--------------|-------|
| Random Forest | — | 4-mer freq. | **0.980** | 100 trees; tutorial baseline. |
| **DNABERT-2** | **Attention** | **BPE** | **0.9663** | 1 epoch, max_seq_length 128, batch 4. |

### 4.3 Training configuration summary

| Parameter | CNN / BiLSTM | DNABERT-1 / DNABERT-2 |
|-----------|----------------|-------------------------|
| Epochs | 10 | 3 (dummy_mouse); 1 (human_nontata) |
| Batch size | 32 | 8 (dummy_mouse); 4 (human_nontata) |
| Learning rate | 1e-3 | 2e-5 |
| Max length | 256 | 256 / 128 |

---

## 5. Plots and Visualization Code

All figures are generated by `plot_results.py` (matplotlib) and saved under `plots/`.

### 5.1 Plot list and descriptions

| Figure | Description |
|--------|-------------|
| `plot_test_roc_mouse.png` | Test ROC-AUC by model for dummy_mouse_enhancers_ensembl (bar chart). |
| `plot_valid_vs_test_mouse.png` | Valid vs Test ROC-AUC (grouped bars) for mouse enhancers. |
| `plot_time_mouse.png` | Training time (seconds) by model for mouse enhancers. |
| `plot_promoters_roc.png` | Test ROC-AUC for human_nontata_promoters (RF vs DNABERT-2). |
| `plot_summary.png` | 2×2 summary: Test ROC mouse, Valid vs Test mouse, Time mouse, Promoters ROC. |

### 5.2 Code: plot_results.py (data and one plot)

```python
# --- Data from RESULTS.md ---
models_mouse = ["CNN", "BiLSTM", "DNABERT-1", "DNABERT-2"]
valid_roc_mouse = [0.794, 0.811, 0.797, 0.882]
test_roc_mouse = [0.751, 0.774, 0.801, 0.816]
time_sec_mouse = [4, 21, 157, 102]

models_prom = ["Random Forest", "DNABERT-2"]
test_roc_prom = [0.980, 0.9663]

def plot_test_roc_mouse():
    """Test ROC-AUC by model (dummy_mouse_enhancers_ensembl)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(models_mouse))
    bars = ax.bar(x, test_roc_mouse, color=["#6b8e9a", "#6b8e9a", "#7a9e8a", "#4a7c59"],
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models_mouse)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("dummy_mouse_enhancers_ensembl — Test ROC-AUC by model")
    ax.set_ylim(0, 1)
    for i, v in enumerate(test_roc_mouse):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_test_roc_mouse.png"), dpi=150, bbox_inches="tight")
    plt.close()
```

Reproduction: `python plot_results.py` (creates `plots/` and all five figures).

---

## 6. Code Snippets from the Codebase

### 6.1 Featurizer: DNAOneHotFeaturizer (featurizers.py)

One-hot encoding with vectorized lookup; subclasses `deepchem.feat.Featurizer`.

```python
from deepchem.feat.base_classes import Featurizer

_DNA_BASES = "ACGT"
BASE_TO_INDEX = {base: i for i, base in enumerate(_DNA_BASES)}

class DNAOneHotFeaturizer(Featurizer):
    """One-hot encode DNA sequences into fixed-length matrices.
    A=0, C=1, G=2, T=3. N/other -> [0,0,0,0]. Pad/truncate to max_length."""

    def __init__(self, max_length: int = 2048):
        if max_length < 0:
            raise ValueError("max_length must be non-negative.")
        self.max_length = max_length
        self._base_vectors = np.vstack([
            np.eye(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),  # ambiguous
        ])
        self._char_lookup = np.full(128, 4, dtype=np.intp)
        for base, idx in BASE_TO_INDEX.items():
            self._char_lookup[ord(base)] = idx
            self._char_lookup[ord(base.lower())] = idx

    def _featurize(self, datapoint: str, **kwargs) -> np.ndarray:
        seq = datapoint.upper()[:self.max_length]
        raw = seq.encode("ascii")
        indices = self._char_lookup[np.frombuffer(raw, dtype=np.uint8)]
        encoding = self._base_vectors[indices]
        # Pad to max_length
        pad_len = self.max_length - len(seq)
        if pad_len > 0:
            padding = np.zeros((pad_len, 4), dtype=np.float32)
            encoding = np.vstack([encoding, padding])
        return encoding.astype(np.float32)
```

### 6.2 Loader: load_genomic_benchmark (loader.py)

MolNet-style API; supports `featurizer=None` (DummyFeaturizer) for raw DNA (e.g. DNABERT-2).

```python
def load_genomic_benchmark(
    dataset_name: str = "human_nontata_promoters",
    featurizer: Any = None,
    splitter: Optional[str] = "official",
    transformers: Optional[List] = None,
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[List[str], Tuple[Dataset, ...], List]:
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")

    actual_splitter = None if splitter == "official" else splitter
    loader = _GenomicBenchmarkLoader(
        featurizer=_resolve_featurizer(featurizer),
        splitter=actual_splitter,
        transformer_generators=transformers or [],
        tasks=[dataset_name],
        data_dir=data_dir,
        save_dir=save_dir,
        dataset_name=dataset_name,
        **kwargs,
    )
    tasks, datasets, transformers = loader.load_dataset(dataset_name, reload)

    if splitter == "official" and len(datasets) == 1:
        full_dataset = datasets[0]
        metadata_path = os.path.join(loader.data_dir, dataset_name, "metadata.pkl")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        split_indices = metadata.get("split_indices", {})
        train = full_dataset.select(split_indices.get("train", []))
        valid = full_dataset.select(split_indices.get("valid", []))
        test = full_dataset.select(split_indices.get("test", []))
        datasets = tuple([d for d in [train, valid, test] if len(d) > 0])

    return tasks, datasets, transformers
```

### 6.3 Model: DNABERT2Model and _prepare_batch (dnabert2.py)

BPE tokenization in the model; raw sequences in `X` (DummyFeaturizer).

```python
class DNABERT2Model(HuggingFaceModel):
    """DNABERT-2 for DNA sequence analysis. BPE handled in _prepare_batch."""

    def __init__(self, task: str, model_path: str = "zhihan1996/DNABERT-2-117M",
                 n_tasks: int = 1, config: Optional[Dict] = None,
                 max_seq_length: int = 512, **kwargs):
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        _stub_flash_attn_if_needed()
        self.n_tasks = n_tasks
        self.max_seq_length = max_seq_length
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # ... config patching (num_labels, problem_type) ...
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, config=config, trust_remote_code=True
        )
        super().__init__(model=model, tokenizer=tokenizer, task=task, **kwargs)

    def _prepare_batch(self, batch: Tuple) -> Tuple[Dict, Any, Any]:
        """Tokenize raw DNA sequences (inputs[0]) with BPE."""
        inputs, labels, weights = batch
        sequences = [seq.upper().replace("N", "") for seq in inputs[0]]
        tokens = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        inputs_dict = {k: v.to(self.device) for k, v in tokens.items()}
        if labels is not None:
            y_tensor = torch.as_tensor(labels[0].squeeze(-1), dtype=torch.long, device=self.device)
            inputs_dict["labels"] = y_tensor
        w_tensor = torch.as_tensor(weights[0], dtype=torch.float32, device=self.device) if weights is not None else None
        return inputs_dict, y_tensor, w_tensor
```

### 6.4 End-to-end pipeline (run_pipeline.py)

```python
def main():
    # 1. Load dataset
    tasks, datasets, transformers = load_genomic_benchmark(
        dataset_name="dummy_mouse_enhancers_ensembl",
        splitter="random",
        reload=False,
    )
    if len(datasets) == 2:
        train, test = datasets; valid = train
    else:
        train, valid, test = datasets[0], datasets[1], datasets[2]

    # 2. Model
    from genomics.dnabert2 import DNABERT2Model
    model = DNABERT2Model(
        task="classification",
        n_tasks=1,
        model_dir=model_dir,
        batch_size=8,
        learning_rate=2e-5,
        max_seq_length=256,
    )

    # 3. Train
    loss = model.fit(train, nb_epoch=3)

    # 4. Evaluate
    metrics = [
        dc.metrics.Metric(dc.metrics.roc_auc_score),
        dc.metrics.Metric(dc.metrics.accuracy_score),
    ]
    results = model.evaluate(test, metrics)
```

Run: `python -m genomics.run_pipeline`.

---

## 7. DeepChem Mapping and Design Decisions

### 7.1 Target paths for upstream PRs

| Standalone File | DeepChem Target Path |
|-----------------|----------------------|
| genomics/featurizers.py | deepchem/feat/dna_featurizers.py |
| genomics/loader.py | deepchem/molnet/load_function/load_genomic_benchmark.py |
| genomics/dnabert2.py | deepchem/models/torch_models/dnabert2.py |
| genomics/run_pipeline.py | examples/genomics_pipeline.py |
| tests/test_*.py | deepchem/feat/tests/, molnet/tests/, torch_models/tests/ |

**Registrations:** `deepchem/feat/__init__.py`, `deepchem/models/torch_models/__init__.py`, `deepchem/molnet/__init__.py`.

### 7.2 Design decisions

| Decision | Rationale |
|----------|-----------|
| BPE in model, not featurizer | Matches ChemBERTa; `_prepare_batch` tokenizes raw strings on-the-fly. |
| Preserve official test split | Benchmark comparability. |
| DummyFeaturizer for BERT models | Raw DNA in `X`; tokenization at train time. |
| Split only official train when no valid | Keeps held-out test untouched; yields train/valid/test. |

---

## 8. Work Plan (Phases 1–3)

### Phase 1: Upstream integration

1. **Featurizers** — Move to `deepchem/feat/dna_featurizers.py`; register; move tests.
2. **Loaders** — Genomic Benchmarks loader + **one more** MolNet-style loader (recommended: **DeepSEA**; alternatives: FASTA+labels, METABRIC). Register both; tests.
3. **Models** — DNABERT-2 and **MethylBERT** wrappers in `deepchem/models/torch_models/`; register; tests.
4. **Example and docs** — `examples/genomics_pipeline.py`; install/docs for `.[data]` / `.[all]`.

### Phase 2: Benchmarks and robustness

1. Run on `human_enhancers_cohn`; add to RESULTS.
2. Multi-seed evaluation (e.g. 3 seeds); report mean ± std.
3. Integration test: load benchmark with `featurizer=None` → DNABERT2Model → one `fit()` step.
4. Tutorial alignment and type hints in loader/run_pipeline.

### Phase 3 (Optional)

- MethylBERT / NucleotideTransformer; DummyFeaturizer documentation.

---

## 9. Milestones and Timeline

| Milestone | Target | Deliverables |
|-----------|--------|---------------|
| M1: Featurizers + loaders | Week 3–4 | Featurizers and both loaders in DeepChem; tests; registrations. |
| M2: Models + example | Week 5–6 | DNABERT-2 and MethylBERT in DeepChem; tests; examples/genomics_pipeline.py; docs. |
| M3: Benchmarks and quality | Week 7–8 | human_enhancers_cohn results; multi-seed; integration test; tutorial/type hints. |
| M4 (stretch) | Week 9+ | NucleotideTransformer and/or extra docs. |

Timeline is indicative (e.g. 12-week GSoC: M1 by week 4, M2 by week 8, M3 by week 12).

---

## 10. Testing Plan

- **Unit:** Featurizers (21 tests), Loaders (8 + second loader), DNABERT-2 / MethylBERT (6 + analogous).
- **Integration:** One end-to-end test: `load_genomic_benchmark(..., featurizer=None)` → DNABERT2Model → one `fit()` step (optional `slow`/GPU).
- **Run:** `pytest tests/ -v -m "not slow"` (fast); `pytest tests/ -v` (full; may need `.[data]`/`.[all]`).

---

## 11. Risks and Resources

| Risk | Mitigation |
|------|------------|
| DeepChem review bandwidth | Small PRs (featurizers → loaders → models → example); link to integration review doc. |
| GPU for DNABERT-2 | Document CPU vs GPU; CI can skip heavy tests or use tiny run. |
| Dependency churn | Pin/lower-bound deps; extras `.[data]` and `.[all]`. |
| Scope creep | Phase 1–2 as core; Phase 3 optional. |

**Resources:** Python 3.8+; `pip install -e ".[data]"` or `.[all]` for full reproduction; optional GPU (e.g. T4); storage for datasets and model weights.

---

## 12. Future Work: GENA-LM

**GENA-LM** (Fishman et al., bioRxiv 2023) is a family of open-source foundational DNA language models for **long sequences** (up to **36 kb**):

- BPE tokenization; full attention (≈4.5 kb) and sparse attention (BigBird-style, up to 36 kb).
- Pre-trained on human T2T (and optionally multispecies / 1000 Genomes).
- Downstream: promoter activity, splice sites, chromatin (DeepSEA), enhancers (DeepSTARR), polyadenylation.
- GitHub: https://github.com/AIRI-Institute/GENA_LM ; HuggingFace: AIRI-Institute, `gena-lm-` prefix.

**Proposed future work:** Integrate GENA-LM as a `HuggingFaceModel` wrapper in DeepChem for long-context DNA and comparison with DNA BERT, MethylBERT, and NucleotideTransformer.

**Reference:** Fishman, V. et al. *GENA-LM: A Family of Open-Source Foundational Models for Long DNA Sequences.* bioRxiv (2023). doi:10.1101/2023.06.12.544594.

---

## 13. References

- Zhou, Z. et al. *DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome.* arXiv:2306.15006 (2023).
- Ji, Y. et al. *DNABERT.* Bioinformatics 37(15), 2112-2120 (2021).
- Gresova, K. et al. *Genomic Benchmarks.* BMC Genomic Data 24, 25 (2023).
- Fishman, V. et al. *GENA-LM: A Family of Open-Source Foundational Models for Long DNA Sequences.* bioRxiv (2023). doi:10.1101/2023.06.12.544594.

---

**Summary.** This proposal defines **starting work** as DNA BERT (DNABERT-2), MethylBERT, and one more loader; **Phases 1–2** upstream featurizers, both loaders, DNABERT-2 and MethylBERT, with benchmarks and tests; **Phase 3** is optional (NucleotideTransformer, docs). **Future work** proposes GENA-LM for long-sequence DNA. Key tables and code snippets above are taken from the dc-genomics-review repository; plots are produced by `plot_results.py` in `plots/`.
