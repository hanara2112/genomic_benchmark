"""Genomic Benchmarks dataset loader.

Follows the MolNet loader convention: returns
``(tasks, (train, valid, test), transformers)``.

The loader downloads datasets from the `genomic-benchmarks
<https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks>`_ package
and wraps them as ``deepchem.data.NumpyDataset`` objects.

Target location in DeepChem:
``deepchem/molnet/load_function/load_genomic_benchmark.py``
"""

import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import deepchem as dc
from deepchem.data import NumpyDataset, Dataset
from deepchem.molnet.load_function.molnet_loader import (
    _MolnetLoader,
    TransformerGenerator,
)

try:
    from genomic_benchmarks.loc2seq import download_dataset as _gb_download
    _HAS_GENOMIC_BENCHMARKS = True
except ImportError:
    _gb_download = None  # type: ignore[assignment]
    _HAS_GENOMIC_BENCHMARKS = False

logger = logging.getLogger(__name__)

AVAILABLE_DATASETS = [
    "dummy_mouse_enhancers_ensembl",
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
]

_VALID_DNA_CHARS = set("ACGTNRYSWKMBDHV")


def _read_split_dir(
    split_dir: pathlib.Path,
    label_map: Dict[str, int],
) -> Tuple[List[str], List[int]]:
    """Read sequences and labels from a split directory.

    Parameters
    ----------
    split_dir : pathlib.Path
        Directory containing one subdirectory per class label.
    label_map : dict
        Mapping from class name to integer label. New class names
        encountered in ``split_dir`` are appended to this mapping.

    Returns
    -------
    sequences : list of str
        DNA sequences read from the directory.
    labels : list of int
        Integer labels corresponding to each sequence.
    """
    sequences: List[str] = []
    labels: List[int] = []

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in label_map:
            label_map[class_name] = len(label_map)
        label = label_map[class_name]

        for seq_file in class_dir.iterdir():
            if not seq_file.is_file():
                continue
            raw = seq_file.read_text().strip()
            lines = raw.split("\n")
            seq = "".join(
                line.strip() for line in lines if not line.startswith(">")
            ).upper()
            if not seq:
                continue
            bad = set(seq) - _VALID_DNA_CHARS
            if bad:
                logger.warning(
                    "File %s contains non-DNA characters: %s; skipping.",
                    seq_file,
                    bad,
                )
                continue
            sequences.append(seq)
            labels.append(label)

    return sequences, labels


class _GenomicBenchmarkLoader(_MolnetLoader):
    """Loader for Genomic Benchmarks datasets.

    Subclasses ``_MolnetLoader`` and implements ``create_dataset()``
    to download data via the ``genomic-benchmarks`` package and
    return a ``NumpyDataset``.
    """

    def __init__(
        self,
        featurizer: Union[dc.feat.Featurizer, str],
        splitter: Union[dc.splits.Splitter, str, None],
        transformer_generators: List[Union[TransformerGenerator, str]],
        tasks: List[str],
        data_dir: Optional[str],
        save_dir: Optional[str],
        dataset_name: str,
        **kwargs,
    ):
        super().__init__(
            featurizer, splitter, transformer_generators,
            tasks, data_dir, save_dir, **kwargs,
        )
        self.dataset_name = dataset_name
        self.label_map: Dict[str, int] = {}

    def create_dataset(self) -> Dataset:
        """Download and load the Genomic Benchmarks dataset.

        Returns
        -------
        Dataset
            A ``NumpyDataset`` containing DNA sequences (as object
            arrays in ``X``) and integer labels in ``y``.

        Raises
        ------
        ImportError
            If ``genomic-benchmarks`` is not installed.
        """
        if not _HAS_GENOMIC_BENCHMARKS:
            raise ImportError(
                "The 'genomic-benchmarks' package is required. "
                "Install with: pip install genomic-benchmarks"
            )

        logger.info(
            "Downloading dataset '%s' via genomic_benchmarks.",
            self.dataset_name,
        )
        downloaded_path = _gb_download(self.dataset_name, version=0)
        base_dir = pathlib.Path(downloaded_path)

        self.label_map = {}
        all_seqs: List[str] = []
        all_labels: List[int] = []

        for split_name in ("train", "test"):
            split_dir = base_dir / split_name
            if split_dir.exists():
                seqs, labels = _read_split_dir(split_dir, self.label_map)
                all_seqs.extend(seqs)
                all_labels.extend(labels)

        X = np.array(all_seqs, dtype=object)
        y = np.array(all_labels, dtype=np.float32).reshape(-1, 1)

        if self.featurizer is not None:
            X = self.featurizer.featurize(X)

        return NumpyDataset(X=X, y=y, ids=np.array(all_seqs, dtype=object))


def load_genomic_benchmark(
    dataset_name: str = "human_nontata_promoters",
    featurizer: Any = None,
    splitter: Optional[str] = "random",
    transformers: Optional[List[Union[TransformerGenerator, str]]] = None,
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load a Genomic Benchmarks dataset.

    Genomic Benchmarks [1]_ is a collection of genomic sequence
    classification datasets. This loader follows the standard MolNet
    return convention of ``(tasks, datasets, transformers)``.

    Parameters
    ----------
    dataset_name : str, default 'human_nontata_promoters'
        Name of the dataset. Must be one of ``AVAILABLE_DATASETS``.
    featurizer : Featurizer or str or None, default None
        Featurizer to apply. ``None`` stores raw DNA sequences in
        ``X`` (use ``DummyFeaturizer`` if DeepChem requires one).
    splitter : str or None, default 'random'
        ``'random'`` for train/valid/test split, or ``None`` to
        return a single unsplit dataset.
    transformers : list or None, default None
        Transformers to apply after splitting.
    reload : bool, default True
        Whether to reload from disk if previously cached.
    data_dir : str or None
        Directory for downloaded data.
    save_dir : str or None
        Directory to save the processed dataset.

    Returns
    -------
    tasks : list of str
        Task names.
    datasets : tuple of Dataset
        ``(train, valid, test)`` when splitter is not ``None``.
    transformers : list of Transformer
        Applied transformers.

    Raises
    ------
    ValueError
        If ``dataset_name`` is not in ``AVAILABLE_DATASETS``.

    Examples
    --------
    >>> tasks, datasets, transformers = load_genomic_benchmark(
    ...     dataset_name="dummy_mouse_enhancers_ensembl",
    ...     splitter="random",
    ... )
    >>> train, valid, test = datasets
    >>> train.X.shape[0] > 0
    True

    References
    ----------
    .. [1] Gresova, K. et al. Genomic Benchmarks: A Collection of
       Datasets for Genomic Sequence Classification. BMC Genomic
       Data 24, 25 (2023).
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {AVAILABLE_DATASETS}"
        )

    loader = _GenomicBenchmarkLoader(
        featurizer=featurizer,
        splitter=splitter,
        transformer_generators=transformers or [],
        tasks=[dataset_name],
        data_dir=data_dir,
        save_dir=save_dir,
        dataset_name=dataset_name,
        **kwargs,
    )
    return loader.load_dataset(dataset_name, reload)


def load_human_nontata_promoters(
    featurizer: Any = None,
    splitter: Optional[str] = "random",
    transformers: Optional[List[Union[TransformerGenerator, str]]] = None,
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load the Human Non-TATA Promoters dataset.

    Convenience wrapper around ``load_genomic_benchmark()`` with
    ``dataset_name='human_nontata_promoters'``.
    """
    return load_genomic_benchmark(
        dataset_name="human_nontata_promoters",
        featurizer=featurizer,
        splitter=splitter,
        transformers=transformers,
        reload=reload,
        data_dir=data_dir,
        save_dir=save_dir,
        **kwargs,
    )
