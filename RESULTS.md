# Benchmark Results

## Dataset: `dummy_mouse_enhancers_ensembl`

Binary classification (enhancer vs. non-enhancer).
Train: 968 | Valid: 121 | Test: 121 sequences. Seed: 42.

### Results Summary

| Model | Paradigm | Input | Test ROC-AUC | Test Acc | Test F1 | Time |
|-------|----------|-------|-------------|----------|---------|------|
| CNN | Convolution | One-hot | 0.751 | 0.686 | 0.729 | 4s |
| BiLSTM | Recurrent | One-hot | 0.774 | 0.711 | 0.720 | 21s |
| DNABERT-1 | Attention | 6-mer | 0.801 | 0.694 | 0.678 | 157s |
| **DNABERT-2** | **Attention** | **BPE** | **0.816** | **0.719** | **0.702** | **102s** |
| Nucl. Transf. | Attention | 6-mer (NT) | — | — | — | ❌ import error |

### Validation Metrics

| Model | Valid ROC-AUC | Valid Acc | Valid F1 |
|-------|-------------|----------|---------|
| CNN | 0.794 | 0.686 | 0.729 |
| BiLSTM | 0.811 | 0.736 | 0.750 |
| DNABERT-1 | 0.797 | 0.736 | 0.729 |
| **DNABERT-2** | **0.882** | **0.826** | **0.817** |

### Key Observations

- **DNABERT-2 achieves highest ROC-AUC** (0.816 test, 0.882 valid) with only 3 epochs of fine-tuning.
- CNN and BiLSTM baselines are competitive on this small dataset but trained for 10 epochs vs. 3 for transformers.
- Nucleotide Transformer failed due to a `transformers` library version incompatibility (`find_pruneable_heads_and_indices`).
- Valid-to-test gap for DNABERT-2 (0.882 → 0.816) suggests mild overfitting on this small dataset; more data or regularization would help.

### Training Configuration

| Parameter | CNN / BiLSTM | DNABERT-1 / DNABERT-2 |
|-----------|-------------|----------------------|
| Epochs | 10 | 3 |
| Batch size | 32 | 8 |
| Learning rate | 1e-3 | 2e-5 |
| Max length | 256 | 256 |

### Reproduction

```bash
# This repo (DNABERT-2 only)
python -m genomics.run_pipeline

# Full comparison was run using the deepchem-genomics repo
```

## Status

| Component | Status |
|-----------|--------|
| Featurizers | ✅ 21 tests passing |
| Loader | ✅ 5 tests passing |
| DNABERT-2 Model | ✅ Benchmarked on Kaggle T4 GPU |
| CNN / BiLSTM baselines | ℹ️ Run via deepchem-genomics (not in this repo) |
| Nucleotide Transformer | ❌ Blocked by transformers version issue |

## Planned

- [ ] Run on `human_nontata_promoters` (36K sequences, larger benchmark)
- [ ] Run on `human_enhancers_cohn` (27K sequences)
- [ ] Multi-seed averaging (3 seeds) for error bars
