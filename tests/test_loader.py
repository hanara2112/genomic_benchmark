"""Tests for the Genomic Benchmarks dataset loader."""

import pytest
import numpy as np


def _genomic_benchmarks_available():
    try:
        import genomic_benchmarks  # noqa: F401
        return True
    except ImportError:
        return False


class TestLoaderValidation:
    """Tests that don't require network access."""

    def test_available_datasets_list(self):
        from genomics.loader import AVAILABLE_DATASETS
        assert isinstance(AVAILABLE_DATASETS, list)
        assert "human_nontata_promoters" in AVAILABLE_DATASETS
        assert len(AVAILABLE_DATASETS) == 8

    def test_invalid_dataset_raises(self):
        from genomics.loader import load_genomic_benchmark
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_genomic_benchmark(dataset_name="nonexistent_dataset")


@pytest.mark.slow
@pytest.mark.skipif(
    not _genomic_benchmarks_available(),
    reason="genomic-benchmarks not installed",
)
class TestLoaderIntegration:
    """Integration tests requiring network access."""

    def test_load_returns_molnet_tuple(self):
        """Loader returns (tasks, datasets, transformers) tuple."""
        from genomics.loader import load_genomic_benchmark
        tasks, datasets, transformers = load_genomic_benchmark(
            dataset_name="dummy_mouse_enhancers_ensembl",
            splitter="random",
            reload=False,
        )
        assert isinstance(tasks, list)
        assert len(tasks) == 1
        assert len(datasets) == 3  # train, valid, test
        assert isinstance(transformers, list)

    def test_dataset_has_correct_attributes(self):
        """Each dataset has X, y, w, ids."""
        from genomics.loader import load_genomic_benchmark
        tasks, datasets, transformers = load_genomic_benchmark(
            dataset_name="dummy_mouse_enhancers_ensembl",
            splitter="random",
            reload=False,
        )
        train, valid, test = datasets
        for ds in [train, valid, test]:
            assert hasattr(ds, "X")
            assert hasattr(ds, "y")
            assert hasattr(ds, "w")
            assert hasattr(ds, "ids")
            assert ds.y.ndim == 2  # (N, 1)

    def test_no_split_returns_single_dataset(self):
        """With splitter=None, returns a single dataset."""
        from genomics.loader import load_genomic_benchmark
        tasks, datasets, transformers = load_genomic_benchmark(
            dataset_name="dummy_mouse_enhancers_ensembl",
            splitter=None,
            reload=False,
        )
        assert len(datasets) == 1
