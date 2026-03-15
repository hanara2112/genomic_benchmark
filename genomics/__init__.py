"""DeepChem Genomics — Standalone review repo for genomic sequence support.

This package provides DNA featurizers, a Genomic Benchmarks dataset loader,
and a DNABERT-2 model wrapper, built on top of DeepChem's standard
abstractions (``Featurizer``, ``NumpyDataset``, ``HuggingFaceModel``).
"""

from genomics.featurizers import (
    DNAKmerCountFeaturizer,
    DNAKmerFeaturizer,
    DNAOneHotFeaturizer,
)

from genomics.loader import (
    load_genomic_benchmark,
    load_human_nontata_promoters,
    AVAILABLE_DATASETS,
)

try:
    from genomics.dnabert2 import DNABERT2Model
except ImportError:
    pass

__all__ = [
    # Featurizers
    "DNAKmerCountFeaturizer",
    "DNAKmerFeaturizer",
    "DNAOneHotFeaturizer",
    # Dataset loaders
    "load_genomic_benchmark",
    "load_human_nontata_promoters",
    "AVAILABLE_DATASETS",
    # Models
    "DNABERT2Model",
]
