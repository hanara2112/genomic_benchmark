# Benchmark Results

This file consolidates all benchmark results for the dc-genomics-review pipeline, with clear interpretation for each dataset and model.

---

## 1. Dataset: `dummy_mouse_enhancers_ensembl`

**Task:** Binary classification — enhancer vs. non-enhancer sequences.  
**Splits:** Train 968 | Valid 121 | Test 121. Seed 42. Max length 256.

### Results (validation and test)

| Model           | Paradigm   | Input   | Valid ROC-AUC | Valid Acc | Valid F1 | Test ROC-AUC | Test Acc | Test F1 | Time  |
|----------------|------------|---------|---------------|-----------|----------|--------------|----------|---------|-------|
| CNN            | Convolution| One-hot | 0.794         | 0.686     | 0.729    | 0.751        | 0.686    | 0.729   | 4 s  |
| BiLSTM         | Recurrent | One-hot | 0.811         | 0.736     | 0.750    | 0.774        | 0.711    | 0.720   | 21 s |
| DNABERT-1      | Attention | 6-mer   | 0.797         | 0.736     | 0.729    | 0.801        | 0.694    | 0.678   | 157 s|
| **DNABERT-2**  | **Attention** | **BPE** | **0.882** | **0.826** | **0.817** | **0.816** | **0.719** | **0.702** | **102 s** |

*Config: CNN/BiLSTM 10 epochs, batch 32, lr 1e-3; DNABERT-1/2 3 epochs, batch 8, lr 2e-5.*

### Interpretation

- **DNABERT-2 is the best performer** on this dataset: highest validation ROC-AUC (0.88), validation accuracy (0.83), and validation F1 (0.82), and best test ROC-AUC (0.82) with fewer epochs (3) than the baselines (10).
- **ROC-AUC** measures ranking quality (ability to separate enhancer vs non-enhancer); 0.82 test means the model ranks true enhancers higher than non-enhancers about 82% of the time.
- **Valid vs test:** DNABERT-2 valid (0.88) > test (0.82) indicates mild overfitting on this small dataset; more data or regularization (e.g. dropout, early stopping) would likely improve generalization.
- **Baselines:** CNN and BiLSTM are competitive given their simplicity and 10 epochs; DNABERT-1 is similar to DNABERT-2 on test ROC-AUC (0.80 vs 0.82) but DNABERT-2’s BPE tokenization and architecture yield better validation metrics and lower final loss.

---

## 2. Dataset: `human_nontata_promoters`

**Task:** Binary classification — promoter vs. non-promoter (human non-TATA promoters).  
**Splits:** Official benchmark splits (train/valid/test). Larger dataset; used in the tutorial.

### Results

| Model           | Paradigm   | Input        | Test ROC-AUC | Notes |
|----------------|------------|--------------|--------------|-------|
| Random Forest  | —          | 4-mer freq.  | **0.980**    | 100 trees, `SklearnModel`; tutorial baseline. |
| **DNABERT-2**  | **Attention** | **BPE**  | **0.9663**   | 1 epoch, max_seq_length 128, batch 4; tutorial. |

*Pipeline: `load_genomic_benchmark` with `featurizer=None` for DNABERT-2, `featurizer=DNAKmerCountFeaturizer(4)` for RF (see `examples/tutorial.ipynb`).*

### Interpretation

- **Both models perform very well** (ROC-AUC > 0.96), showing that this promoter task has strong signal and that the pipeline is correctly set up.
- **Random Forest (0.98)** slightly edges DNABERT-2 (0.9663) with simple 4-mer counts and no deep learning — consistent with “traditional” features often being sufficient when the dataset is large and the motif structure is relatively simple.
- **DNABERT-2 (0.9663)** reaches strong performance with only **1 epoch** of fine-tuning, demonstrating that the foundation model quickly adapts to the task and that the integration (loader → model → metrics) works as intended.
- **Takeaway:** For human non-TATA promoters, either a fast RF baseline or a single-epoch DNABERT-2 run is viable; DNABERT-2 is more suitable when you need sequence-level representations or plan to scale to harder or multi-species tasks.

---

## Training configuration summary

| Parameter     | CNN / BiLSTM   | DNABERT-1 / DNABERT-2 |
|---------------|----------------|-------------------------|
| Epochs        | 10             | 3 (dummy_mouse); 1 (human_nontata tutorial) |
| Batch size    | 32             | 8 (dummy_mouse); 4 (human_nontata tutorial) |
| Learning rate | 1e-3           | 2e-5                    |
| Max length    | 256            | 256 (dummy_mouse); 128 (human_nontata tutorial) |

---

## Reproduction

```bash
# DNABERT-2 only (this repo)
python -m genomics.run_pipeline

# Full model comparison on dummy_mouse_enhancers_ensembl was run via the deepchem-genomics repo;
# results were exported to JSON and summarized here.
```

Tutorial (human_nontata_promoters, RF + DNABERT-2):

```bash
# Run examples/tutorial.ipynb
jupyter notebook examples/tutorial.ipynb
```

Plot figures from these results (classic matplotlib):

```bash
python plot_results.py
# Saves figures in plots/: plot_test_roc_mouse.png, plot_valid_vs_test_mouse.png, plot_time_mouse.png, plot_promoters_roc.png, plot_summary.png
```

---

## Status

| Component              | Status |
|------------------------|--------|
| Featurizers            | ✅ 21 tests passing |
| Loader                 | ✅ 8 tests passing  |
| DNABERT-2 model       | ✅ Benchmarked (Kaggle T4 GPU; tutorial) |
| CNN / BiLSTM baselines| ℹ️ Run via deepchem-genomics (not in this repo) |

## Planned

- [x] Run on `human_nontata_promoters` (RF + DNABERT-2 in tutorial)
- [x] Consolidate all results and interpretation in this file
- [ ] Run on `human_enhancers_cohn` (27K sequences)
- [ ] Multi-seed averaging (e.g. 3 seeds) for confidence intervals
