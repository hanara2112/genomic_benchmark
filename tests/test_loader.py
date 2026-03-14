"""Tests for the Genomic Benchmarks dataset loader."""

import pathlib

import pytest


def _write_sequence(path: pathlib.Path, sequence: str) -> None:
    """Write a tiny FASTA-style sequence file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">{path.stem}\n{sequence}\n", encoding="utf-8")


def _build_fake_benchmark_dataset(
    root: pathlib.Path,
    include_valid: bool = False,
) -> pathlib.Path:
    """Create a local fake Genomic Benchmarks dataset layout."""
    for index in range(6):
        _write_sequence(root / "train" / "negative" / f"neg_{index:02d}.fa",
                        "ACGTACGT")
        _write_sequence(root / "train" / "positive" / f"pos_{index:02d}.fa",
                        "TTTTCCCC")

    for index in range(2):
        _write_sequence(root / "test" / "negative" / f"neg_{index:02d}.fa",
                        "GGGGAAAA")
        _write_sequence(root / "test" / "positive" / f"pos_{index:02d}.fa",
                        "CCCCGGGG")

    if include_valid:
        _write_sequence(root / "valid" / "negative" / "neg_valid.fa",
                        "AACCAACC")
        _write_sequence(root / "valid" / "positive" / "pos_valid.fa",
                        "TTGGTTGG")
    return root


def _patch_genomic_benchmark_download(
    monkeypatch: pytest.MonkeyPatch,
    dataset_root: pathlib.Path,
) -> None:
    """Patch the loader to use a local fake dataset."""
    from genomics import loader as loader_module

    monkeypatch.setattr(loader_module, "_HAS_GENOMIC_BENCHMARKS", True)
    monkeypatch.setattr(
        loader_module,
        "_gb_download",
        lambda dataset_name, version=0: str(dataset_root),
    )


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

    def test_preserves_official_test_split(self,
                                           monkeypatch: pytest.MonkeyPatch,
                                           tmp_path: pathlib.Path):
        """Random splitting should only touch the official train split."""
        dataset_root = _build_fake_benchmark_dataset(tmp_path / "benchmark")
        _patch_genomic_benchmark_download(monkeypatch, dataset_root)

        from genomics.loader import load_genomic_benchmark

        tasks, datasets, transformers = load_genomic_benchmark(
            dataset_name="dummy_mouse_enhancers_ensembl",
            splitter="random",
            reload=False,
            seed=7,
        )

        train, valid, test = datasets
        assert tasks == ["label"]
        assert isinstance(transformers, list)
        assert len(valid) > 0
        assert all(sample_id.startswith("train/") for sample_id in train.ids)
        assert all(sample_id.startswith("train/") for sample_id in valid.ids)
        assert all(sample_id.startswith("test/") for sample_id in test.ids)
        assert set(test.ids) == {
            "test/negative/neg_00.fa",
            "test/negative/neg_01.fa",
            "test/positive/pos_00.fa",
            "test/positive/pos_01.fa",
        }

    def test_uses_official_valid_split_when_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ):
        """Official validation data should be used as-is when provided."""
        dataset_root = _build_fake_benchmark_dataset(tmp_path / "benchmark",
                                                     include_valid=True)
        _patch_genomic_benchmark_download(monkeypatch, dataset_root)

        from genomics.loader import load_genomic_benchmark

        _, datasets, _ = load_genomic_benchmark(
            dataset_name="dummy_mouse_enhancers_ensembl",
            splitter="random",
            reload=False,
            seed=11,
        )

        train, valid, test = datasets
        assert all(sample_id.startswith("train/") for sample_id in train.ids)
        assert set(valid.ids) == {
            "valid/negative/neg_valid.fa",
            "valid/positive/pos_valid.fa",
        }
        assert set(test.ids) == {
            "test/negative/neg_00.fa",
            "test/negative/neg_01.fa",
            "test/positive/pos_00.fa",
            "test/positive/pos_01.fa",
        }

    def test_no_split_merges_official_splits(self,
                                             monkeypatch: pytest.MonkeyPatch,
                                             tmp_path: pathlib.Path):
        """splitter=None should merge the official splits into one dataset."""
        dataset_root = _build_fake_benchmark_dataset(tmp_path / "benchmark")
        _patch_genomic_benchmark_download(monkeypatch, dataset_root)

        from genomics.loader import load_genomic_benchmark

        _, datasets, _ = load_genomic_benchmark(
            dataset_name="dummy_mouse_enhancers_ensembl",
            splitter=None,
            reload=False,
        )

        assert len(datasets) == 1
        merged = datasets[0]
        assert len(merged) == 16
        assert any(sample_id.startswith("train/") for sample_id in merged.ids)
        assert any(sample_id.startswith("test/") for sample_id in merged.ids)

    def test_string_onehot_uses_dna_featurizer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ):
        """The string shortcut should resolve to the DNA one-hot featurizer."""
        dataset_root = _build_fake_benchmark_dataset(tmp_path / "benchmark")
        _patch_genomic_benchmark_download(monkeypatch, dataset_root)

        from genomics.loader import load_genomic_benchmark

        _, datasets, _ = load_genomic_benchmark(
            dataset_name="dummy_mouse_enhancers_ensembl",
            featurizer="onehot",
            splitter=None,
            reload=False,
        )

        merged = datasets[0]
        assert merged.X.shape == (16, 2048, 4)

    def test_unsupported_string_featurizer_raises(self):
        """Chemistry-only featurizer shortcuts should be rejected."""
        from genomics.loader import load_genomic_benchmark

        with pytest.raises(ValueError, match="Unsupported genomics featurizer"):
            load_genomic_benchmark(dataset_name="human_nontata_promoters",
                                   featurizer="graphconv")

    def test_reload_uses_cached_dataset(self,
                                        monkeypatch: pytest.MonkeyPatch,
                                        tmp_path: pathlib.Path):
        """A second reload=True call should come from the local cache."""
        dataset_root = _build_fake_benchmark_dataset(tmp_path / "benchmark")
        _patch_genomic_benchmark_download(monkeypatch, dataset_root)

        from genomics import loader as loader_module
        from genomics.loader import load_genomic_benchmark

        save_dir = tmp_path / "cache"
        _, datasets, _ = load_genomic_benchmark(
            dataset_name="dummy_mouse_enhancers_ensembl",
            splitter=None,
            reload=True,
            save_dir=str(save_dir),
        )
        assert len(datasets) == 1

        def _unexpected_download(dataset_name: str, version: int = 0) -> str:
            raise AssertionError("download should not be called when cache exists")

        monkeypatch.setattr(loader_module, "_gb_download", _unexpected_download)
        _, cached_datasets, _ = load_genomic_benchmark(
            dataset_name="dummy_mouse_enhancers_ensembl",
            splitter=None,
            reload=True,
            save_dir=str(save_dir),
        )
        assert len(cached_datasets[0]) == 16
