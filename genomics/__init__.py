"""DeepChem Genomics — Standalone review repo for genomic sequence support.

This package provides DNA featurizers, a Genomic Benchmarks dataset loader,
and a DNABERT-2 model wrapper, built on top of DeepChem's standard
abstractions (``Featurizer``, ``NumpyDataset``, ``HuggingFaceModel``).
"""

from genomics.featurizers import DNAOneHotFeaturizer, DNAKmerFeaturizer

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
    "DNAOneHotFeaturizer",
    "DNAKmerFeaturizer",
    # Dataset loaders
    "load_genomic_benchmark",
    "load_human_nontata_promoters",
    "AVAILABLE_DATASETS",
    # Models
    "DNABERT2Model",
]
