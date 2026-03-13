# Benchmark Results

## Status

| Component | Status | Notes |
|-----------|--------|-------|
| Featurizers | ✅ Tested | 21 tests passing, no network required |
| Loader | ✅ Tested | Integration tests with `dummy_mouse_enhancers_ensembl` |
| DNABERT-2 Model | ✅ API Tested | Fit/predict/evaluate validated on toy data |
| Full Pipeline | ⏳ Pending | End-to-end `run_pipeline.py` not yet run on full dataset |

## DNABERT-2 Classification

### Dataset: `dummy_mouse_enhancers_ensembl`

| Metric | Value | Notes |
|--------|-------|-------|
| ROC-AUC | *pending* | To be run with `python -m genomics.run_pipeline` |
| Accuracy | *pending* | |
| Epochs | 3 | |
| Batch size | 8 | |
| Learning rate | 2e-5 | |
| Max seq length | 128 | |

### Reproduction

```bash
# Run the canonical pipeline
python -m genomics.run_pipeline

# Results will be printed to stdout
```

## Planned Benchmarks

| Dataset | Classes | Sequences | Status |
|---------|---------|-----------|--------|
| `dummy_mouse_enhancers_ensembl` | 2 | ~1,200 | Pipeline ready |
| `human_nontata_promoters` | 2 | ~36,000 | Loader ready |
| `human_enhancers_cohn` | 2 | ~27,000 | Loader ready |

## Notes

- All results should be reproduced with `seed=42` for consistency.
- Transformer model tests require GPU for practical training runs.
- Featurizer and loader tests run on CPU without network access.
