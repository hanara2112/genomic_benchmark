# DeepChem Mapping

File-by-file mapping for upstream PRs.

| Standalone File | DeepChem Target Path | PR Scope |
|-----------------|----------------------|----------|
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
