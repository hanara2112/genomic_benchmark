"""Genomic Benchmarks dataset loader.

The loader downloads datasets from the `genomic-benchmarks
<https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks>`_ package
and wraps them as DeepChem ``Dataset`` objects.

This implementation follows the standard MolNet pattern and is
memory-efficient by using DiskDataset and supporting featurization
during the loading process.

Target location in DeepChem:
``deepchem/molnet/load_function/load_genomic_benchmark.py``
"""

import logging
import os
import pathlib
import tempfile
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import deepchem as dc
from deepchem.data import Dataset, DiskDataset, NumpyDataset
from deepchem.molnet.load_function.molnet_loader import (
    _MolnetLoader,
    TransformerGenerator,
)
from genomics.featurizers import (
    DNAKmerCountFeaturizer,
    DNAKmerFeaturizer,
    DNAOneHotFeaturizer,
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
_RAW_FEATURIZER_NAMES = {"raw", "none", "dummy", "dummyfeaturizer"}


def _read_split_dir(
    split_dir: pathlib.Path,
    label_map: Dict[str, int],
):
    """Generator to read sequences and labels from a split directory.

    Yields
    ------
    Tuple[str, int, str]
        (sequence, label, id)
    """
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in label_map:
            label_map[class_name] = len(label_map)
        label = label_map[class_name]

        for seq_file in sorted(class_dir.iterdir()):
            if not seq_file.is_file():
                continue
            try:
                raw = seq_file.read_text(encoding="utf-8").strip()
                lines = raw.splitlines()
                seq = "".join(
                    line.strip() for line in lines
                    if not line.lstrip().startswith(">")
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
                # Use a relative path as a stable ID
                sid = seq_file.relative_to(split_dir.parent).as_posix()
                yield seq, label, sid
            except Exception as e:
                logger.error("Error reading %s: %s", seq_file, e)
                continue


def _resolve_featurizer(
    featurizer: Any,
) -> Optional[dc.feat.Featurizer]:
    """Resolve genomics-specific featurizer shortcuts.

    Unlike MoleculeNet's chemistry defaults, genomics loaders should
    not interpret strings like ``"onehot"`` as SMILES featurizers.
    """
    if featurizer is None:
        return None
    if isinstance(featurizer, str):
        key = featurizer.lower()
        if key in _RAW_FEATURIZER_NAMES:
            return None
        if key in {"onehot", "dnaonehot", "dna_onehot"}:
            return DNAOneHotFeaturizer()
        if key in {"kmer", "dnakmer", "dna_kmer", "bagofkmers"}:
            return DNAKmerCountFeaturizer()
        if key in {"kmer_token", "kmer_list"}:
            return DNAKmerFeaturizer()
        raise ValueError(
            f"Unsupported genomics featurizer '{featurizer}'. "
            "Use None/'raw', 'onehot', 'kmer', or pass a Featurizer instance."
        )
    return featurizer


class _GenomicBenchmarkLoader(_MolnetLoader):
    """Loader for Genomic Benchmarks datasets.

    This implementation follows the standard MolNet pattern and is
    memory-efficient by using DiskDataset.
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
            featurizer,
            splitter,
            transformer_generators,
            tasks,
            data_dir,
            save_dir,
            **kwargs,
        )
        self.dataset_name = dataset_name
        self.label_map: Dict[str, int] = {}

    def create_dataset(self) -> Dataset:
        """Create a DiskDataset for the Genomic Benchmark."""
        if not _HAS_GENOMIC_BENCHMARKS:
            raise ImportError(
                "The 'genomic-benchmarks' package is required. "
                "Install with: pip install genomic-benchmarks"
            )

        logger.info("Downloading dataset '%s' via genomic_benchmarks.",
                    self.dataset_name)
        downloaded_path = pathlib.Path(_gb_download(self.dataset_name,
                                                    version=0))

        self.label_map = {}
        all_seqs: List[str] = []
        all_labels: List[int] = []
        all_ids: List[str] = []
        split_indices: Dict[str, List[int]] = {"train": [], "valid": [], "test": []}

        curr_idx = 0
        for split_name in ["train", "valid", "test"]:
            split_dir = downloaded_path / split_name
            if split_dir.exists():
                for seq, label, sid in _read_split_dir(split_dir, self.label_map):
                    all_seqs.append(seq)
                    all_labels.append(label)
                    all_ids.append(sid)
                    split_indices[split_name].append(curr_idx)
                    curr_idx += 1

        if not all_seqs:
            raise ValueError(f"No data found for dataset {self.dataset_name}")

        # Update tasks to be the sorted class names
        sorted_tasks = [
            name for name, _ in sorted(self.label_map.items(), key=lambda x: x[1])
        ]
        self.tasks = sorted_tasks

        X = np.array(all_seqs, dtype=object)
        y = np.array(all_labels, dtype=np.float32).reshape(-1, 1)
        w = np.ones_like(y)

        if self.featurizer is not None and not isinstance(self.featurizer, dc.feat.DummyFeaturizer):
            logger.info("Featurizing dataset...")
            X = self.featurizer.featurize(X)

        dataset_path = os.path.join(self.data_dir or tempfile.gettempdir(),
                                    self.dataset_name)
        dataset = DiskDataset.from_numpy(X=X,
                                         y=y,
                                         w=w,
                                         ids=np.array(all_ids, dtype=object),
                                         data_dir=dataset_path)

        # Store split indices and tasks as metadata
        metadata_path = os.path.join(dataset_path, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump({"split_indices": split_indices, "tasks": sorted_tasks}, f)
            
        return dataset


def load_genomic_benchmark(
    dataset_name: str = "human_nontata_promoters",
    featurizer: Any = None,
    splitter: Optional[str] = "official",
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
        Featurizer to apply.
    splitter : str or None, default 'official'
        'official' uses the benchmark's pre-defined train/test splits.
        'random' or other DeepChem splitters will shuffle the entire dataset.
        None returns a single merged dataset.
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
        Task names (class labels).
    datasets : tuple of Dataset
        ``(train, valid, test)`` or ``(dataset,)``.
    transformers : list of Transformer
        Applied transformers.
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {AVAILABLE_DATASETS}"
        )

    actual_splitter = splitter
    if splitter == "official":
        actual_splitter = None

    loader = _GenomicBenchmarkLoader(
        featurizer=_resolve_featurizer(featurizer),
        splitter=actual_splitter,
        transformer_generators=transformers or [],
        tasks=[dataset_name], # Default task name
        data_dir=data_dir,
        save_dir=save_dir,
        dataset_name=dataset_name,
        **kwargs,
    )

    # load_dataset returns (tasks, datasets, transformers)
    tasks, datasets, transformers = loader.load_dataset(dataset_name, reload)

    # Try to recover metadata if it exists
    dataset_path = os.path.join(loader.data_dir or tempfile.gettempdir(), dataset_name)
    metadata_path = os.path.join(dataset_path, "metadata.pkl")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            if "tasks" in metadata:
                tasks = metadata["tasks"]

    if splitter == "official" and len(datasets) == 1:
        full_dataset = datasets[0]
        split_indices = metadata.get("split_indices", {})
        
        train = full_dataset.select(split_indices.get("train", []))
        valid = full_dataset.select(split_indices.get("valid", []))
        test = full_dataset.select(split_indices.get("test", []))
        
        # Filter out empty datasets
        results = []
        if len(train) > 0: results.append(train)
        if len(valid) > 0: results.append(valid)
        if len(test) > 0: results.append(test)
        datasets = tuple(results)

    return tasks, datasets, transformers


def load_human_nontata_promoters(
    featurizer: Any = None,
    splitter: Optional[str] = "official",
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
