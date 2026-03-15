# DeepChem Genomics Integration — Review

This document reviews the `dc-genomics-review/genomics` implementation for DeepChem integration readiness.

---

## 1. Architecture alignment

### 1.1 Overall design

The module correctly follows DeepChem’s patterns:

| Component        | DeepChem contract                         | Implementation |
|-----------------|--------------------------------------------|----------------|
| **Featurizers** | Subclass `Featurizer`, implement `_featurize` | `featurizers.py`: `DNAOneHotFeaturizer`, `DNAKmerFeaturizer`, `DNAKmerCountFeaturizer`, `KmerFrequencyFeaturizer` all extend `deepchem.feat.base_classes.Featurizer` and use `featurize()` → `_featurize()`. |
| **Data**        | `Dataset` / `NumpyDataset` / `DiskDataset`  | Loader returns `DiskDataset` (or split subsets); `X`/`y`/`w`/`ids` match DC expectations. |
| **Models**      | `HuggingFaceModel` for HF-backed models    | `DNABERT2Model` subclasses `HuggingFaceModel`, overrides `_prepare_batch`, uses `fit`/`predict`/`evaluate`. |
| **Pipelines**   | MolNet-style loader: tasks, datasets, transformers | `load_genomic_benchmark()` returns `(tasks, (train, valid, test), transformers)` and supports `splitter="official"` and other splitters. |

So the design is consistent with DeepChem’s featurizer → dataset → model flow.

### 1.2 DNABERT-2 and raw sequences

- For DNABERT-2, **raw DNA strings** are required (BPE is inside the model).
- The loader uses **`DummyFeaturizer`** when `featurizer is None` or `"raw"`/`"dummy"`, so `X` remains a 1D array of strings.
- `DNABERT2Model._prepare_batch()` expects `inputs[0]` to be those strings and tokenizes them on-the-fly. This matches ChemBERTa/ProtBERT (raw strings in `X`, tokenization in the model).

Data flow is therefore correct: **DummyFeaturizer → raw strings in `X` → DNABERT2 tokenizes in `_prepare_batch`**.

### 1.3 HuggingFaceModel import

- `dnabert2.py` uses the same two-step import as ProtBERT/ChemBERTa: `deepchem.models.torch_models` then `deepchem.models.torch_models.hf_models`, with a clear `ImportError` if both fail.
- No custom `TorchModel`-based HuggingFaceModel fallback; this avoids HF/import issues in notebooks.

---

## 2. Component-by-component notes

### 2.1 `featurizers.py`

- **DNAOneHotFeaturizer**: Fixed-length one-hot (A/C/G/T), vectorized, padding/truncation; good for CNNs.
- **DNAKmerCountFeaturizer**: Bag-of-k-mers (counts or normalized); good for traditional ML and the loader’s `"kmer"` shortcut.
- **DNAKmerFeaturizer**: Overlapping k-mer tokens (strings or vocab indices); good for DNABERT-1–style k-mer LMs.
- **KmerFrequencyFeaturizer**: Same idea as count featurizer (fixed-size frequency vector); naming clarifies “frequency” and is exported for users who want that API.

**Minor**: `DNAKmerCountFeaturizer` and `KmerFrequencyFeaturizer` are similar (both produce a vector of size 4^k). Having both is fine (count vs normalized frequency, different defaults). The loader uses `DNAKmerCountFeaturizer` for the `"kmer"` shortcut; the package now exports both so the public API matches the loader and tutorials.

### 2.2 `loader.py`

- **Genomic Benchmarks**: Uses `genomic_benchmarks.loc2seq.download_dataset` and the standard train/valid/test directory layout; builds a single `DiskDataset` and stores `split_indices` in `metadata.pkl`.
- **Official split**: When `splitter="official"`, the loader passes `actual_splitter=None` to the parent, then after `load_dataset()` replaces the single dataset with `train`/`valid`/`test` via `metadata["split_indices"]`. This preserves benchmark comparability.
- **Featurizer resolution**: `_resolve_featurizer()` supports `None`/`"raw"`/`"dummy"`, `"onehot"`, `"kmer"` (→ `DNAKmerCountFeaturizer`), and `"kmer_token"` (→ `DNAKmerFeaturizer`). Only non-`DummyFeaturizer` featurizers trigger `featurize()`; raw data is left as strings for DNABERT-2.
- **DummyFeaturizer import**: Multiple fallbacks (base_classes, feat, local stub) keep compatibility across DC versions; the local stub is a last resort.

**Robustness**: If `genomic_benchmarks` is missing, `load_genomic_benchmark` raises a clear `ImportError`; `_HAS_GENOMIC_BENCHMARKS` avoids silent failures.

### 2.3 `dnabert2.py`

- **Model loading**: Uses `transformers` only: `AutoTokenizer`/`AutoConfig`/`AutoModel*`.`from_pretrained(..., trust_remote_code=True)`. No direct `huggingface_hub` or `safetensors` usage in the main path, which avoids extra import issues.
- **Config**: Config is loaded and patched (pad/bos/eos, dropout, task-specific `num_labels`/`problem_type`) and passed into `from_pretrained` to avoid DNABERT-2’s `config_class` mismatch on the Hub.
- **Classification/regression fallback**: If `AutoModelForSequenceClassification.from_pretrained` fails (e.g. hub only has MLM), the code builds the seq-class model from config and loads backbone weights from the MLM model with `strict=False`. This keeps compatibility with the current Hub layout.
- **Flash attention**: `_stub_flash_attn_if_needed()` stubs `flash_attn_triton` before any model load so environments without `flash_attn` can still load DNABERT-2.
- **_prepare_batch**: Matches ChemBERTa-style: tokenize sequences (with uppercase and `N` stripping), move to device, attach labels/weights; single-label classification uses long labels, multi-label/regression use float.

### 2.4 `run_pipeline.py`

- End-to-end flow: load benchmark → create DNABERT-2 → fit → evaluate.
- **Change made**: When the splitter returns only two datasets (e.g. train, test), the pipeline now treats them as `(train, test)` and uses `train` as the validation set so `train, valid, test` are always defined and unpacking doesn’t fail.

### 2.5 `generate_tutorial.py`

- **Change made**: Output path is no longer hardcoded. It is now `examples/tutorial.ipynb` relative to the repo root (parent of `genomics/`), with `os.makedirs(..., exist_ok=True)` so the script works from any CWD and on different machines.

### 2.6 `__init__.py`

- **Change made**: `DNAKmerCountFeaturizer` is imported and included in `__all__` so the public API matches the loader and the tutorial (which use the `"kmer"` shortcut and `DNAKmerCountFeaturizer`).
- `DNABERT2Model` is imported in a try/except so the package can be imported even when `transformers`/DeepChem HF stack is not installed; in that case the model is simply not exposed.

---

## 3. Gaps and recommendations

### 3.1 Optional improvements

1. **Type hints**: `loader.py` and `run_pipeline.py` could use stricter typing (e.g. `Tuple[Dataset, ...]`) and docstring types where it helps.
2. **Tests**: Adding a small test that loads a benchmark with `featurizer=None`, builds a `DNABERT2Model`, and runs one step of `fit()` would lock in the integration (e.g. in a `tests/` dir or under `dc-genomics-review`).
3. **Tutorial notebook**: The in-repo `tutorial.ipynb` still uses `DNAKmerFeaturizer` in one cell; the generated tutorial uses `DNAKmerCountFeaturizer`. Align both with the same featurizer set (e.g. `DNAKmerCountFeaturizer` for the k-mer section) so behavior and docs match.
4. **DummyFeaturizer**: Relying on a local stub is a last resort. Prefer documenting that users need a DeepChem version that provides `DummyFeaturizer` (e.g. from `deepchem.feat` or `deepchem.feat.base_classes`) so the stub is rarely used.

### 3.2 Environment

- For DNABERT-2 and the tutorial: `pip install 'deepchem[torch]' transformers genomic_benchmarks` (and optionally `genomic-benchmarks` if the package name differs).
- If the notebook reports “Skipped loading modules with transformers dependency”, install `transformers` (and ensure PyTorch is available) so `HuggingFaceModel` and `DNABERT2Model` load.

### 3.3 Target locations (for upstreaming)

As in docstrings:

- `genomics/featurizers.py` → `deepchem/feat/dna_featurizers.py`
- `genomics/loader.py` → MolNet-style loader under `deepchem/molnet/load_function/` (e.g. genomic_benchmarks loader)
- `genomics/dnabert2.py` → `deepchem/models/torch_models/dnabert2.py`
- `genomics/run_pipeline.py` → `examples/genomics_pipeline.py`

---

## 4. Summary

| Area              | Status | Notes |
|-------------------|--------|--------|
| Featurizers       | OK     | Aligned with `Featurizer`; all four featurizers usable with DC datasets. |
| Loader            | OK     | MolNet-style API, official/random splits, DummyFeaturizer for raw DNA. |
| DNABERT-2         | OK     | HuggingFaceModel, transformers-only loading, flash_attn stub, correct _prepare_batch. |
| Pipeline / tutorial | OK   | run_pipeline handles 2 or 3 splits; generate_tutorial path fixed; __init__ exports DNAKmerCountFeaturizer. |
| Docs / tests      | Optional | Add minimal integration test and align tutorial notebook with generated content. |

The implementation is consistent with DeepChem’s abstractions and ready for integration; the changes above (exports, pipeline robustness, tutorial path) keep the package consistent and portable.
