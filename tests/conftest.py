"""Shared test fixtures following DeepChem's conftest.py pattern."""

import numpy as np
import pytest
import deepchem as dc


@pytest.fixture
def toy_dna_sequences():
    """A small set of DNA sequences for testing."""
    return [
        "ACGTACGTACGTACGT",
        "TGCATGCATGCATGCA",
        "GGGGCCCCAAAATTTT",
        "AAAAAAAACCCCCCCC",
    ]


@pytest.fixture
def toy_dna_dataset(toy_dna_sequences):
    """A NumpyDataset with raw DNA sequences and binary labels."""
    X = np.array(toy_dna_sequences, dtype=object)
    y = np.array([0, 1, 0, 1], dtype=np.float32).reshape(-1, 1)
    return dc.data.NumpyDataset(X=X, y=y)
