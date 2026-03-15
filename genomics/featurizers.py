"""DNA sequence featurizers for genomic tasks.

Provides featurizers that subclass ``deepchem.feat.Featurizer``:

* :class:`DNAOneHotFeaturizer` — fixed-length one-hot encoding
* :class:`DNAKmerFeaturizer` — overlapping k-mer tokenization

Both follow DeepChem's featurizer contract: implement ``_featurize(datapoint)``
and are called via ``featurize(datapoints)``.

Target location in DeepChem: ``deepchem/feat/dna_featurizers.py``
"""

import itertools
import logging
from typing import Dict, List, Optional, Union

import numpy as np

from deepchem.feat.base_classes import Featurizer

logger = logging.getLogger(__name__)

# Standard genomics convention: A=0, C=1, G=2, T=3 (alphabetical)
_DNA_BASES = "ACGT"
BASE_TO_INDEX = {base: i for i, base in enumerate(_DNA_BASES)}

# ASCII lookup table for vectorized encoding (only 0..127 used)
_ASCII_SIZE = 128


class DNAOneHotFeaturizer(Featurizer):
    """One-hot encode DNA sequences into fixed-length matrices.

    Each nucleotide is mapped to a 4-dimensional vector using the
    alphabetical convention (A=0, C=1, G=2, T=3), which is the standard
    used by Selene, Kipoi, and most genomics papers:

    - ``A`` -> ``[1, 0, 0, 0]``
    - ``C`` -> ``[0, 1, 0, 0]``
    - ``G`` -> ``[0, 0, 1, 0]``
    - ``T`` -> ``[0, 0, 0, 1]``

    Ambiguous bases (``N`` or any non-ACGT character) are encoded as
    ``[0, 0, 0, 0]``. Sequences shorter than ``max_length`` are
    zero-padded on the right; longer sequences are truncated.

    The encoding is implemented with a single NumPy indexing pass (no
    Python loop over positions).

    Parameters
    ----------
    max_length : int, default 2048
        Fixed output sequence length. All sequences are padded or
        truncated to this length.

    Examples
    --------
    >>> featurizer = DNAOneHotFeaturizer(max_length=6)
    >>> result = featurizer.featurize(["ACGT"])
    >>> result.shape
    (1, 6, 4)
    >>> result[0, 0].tolist()
    [1.0, 0.0, 0.0, 0.0]

    Note
    ----
    This featurizer requires only ``numpy``.

    References
    ----------
    .. [1] Chen, K. et al. Selene: a PyTorch-based deep learning library
       for sequence data. Nature Methods 16, 315-318 (2019).
    """

    def __init__(self, max_length: int = 2048):
        if max_length < 0:
            raise ValueError("max_length must be non-negative.")
        self.max_length = max_length
        self._warned_truncation = False

        # Build lookup: base_vectors[char_lookup[ord(c)]] = one-hot vector
        self._base_vectors = np.vstack([
            np.eye(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),  # row 4: ambiguous / padding
        ])
        self._char_lookup = np.full(_ASCII_SIZE, 4, dtype=np.intp)
        for base, idx in BASE_TO_INDEX.items():
            self._char_lookup[ord(base)] = idx
            self._char_lookup[ord(base.lower())] = idx

    def _featurize(self, datapoint: str, **kwargs) -> np.ndarray:
        """One-hot encode a single DNA sequence.

        Parameters
        ----------
        datapoint : str
            A DNA sequence string.

        Returns
        -------
        np.ndarray
            Array of shape ``(max_length, 4)``, dtype ``float32``.
        """
        seq = datapoint.upper()
        if len(seq) > self.max_length and not self._warned_truncation:
            logger.warning(
                "Sequence of length %d truncated to max_length=%d.",
                len(seq),
                self.max_length,
            )
            self._warned_truncation = True
        seq = seq[:self.max_length]

        # Vectorized: convert chars to row indices, then index base_vectors
        raw = seq.encode("ascii")
        indices = self._char_lookup[np.frombuffer(raw, dtype=np.uint8)]
        encoding = self._base_vectors[indices]

        # Pad if shorter than max_length
        if len(indices) < self.max_length:
            pad = np.zeros(
                (self.max_length - len(indices), 4), dtype=np.float32
            )
            encoding = np.concatenate([encoding, pad], axis=0)
        return encoding


class DNAKmerCountFeaturizer(Featurizer):
    """Featurize DNA sequences into k-mer frequency vectors (Bag-of-Kmers).

    This featurizer counts the occurrences of all possible k-mers in a
    sequence and returns a normalized or raw frequency vector of size 4^k.
    For example, for k=2, the vector size is 16 (AA, AC, AG, AT, CA, ...).

    Parameters
    ----------
    k : int, default 3
        The length of k-mers to count.
    normalize : bool, default True
        If True, the counts are divided by the total number of k-mers
        in the sequence to produce frequencies that sum to 1.0.
    """

    def __init__(self, k: int = 3, normalize: bool = True):
        self.k = k
        self.normalize = normalize
        
        # Pre-generate all possible k-mers in alphabetical order
        from itertools import product
        self._kmers = ["".join(p) for p in product("ACGT", repeat=k)]
        self._kmer_to_idx = {kmer: i for i, kmer in enumerate(self._kmers)}

    def _featurize(self, datapoint: str, **kwargs) -> np.ndarray:
        """Calculate k-mer frequencies for a sequence."""
        seq = datapoint.upper()
        counts = np.zeros(4**self.k, dtype=np.float32)
        
        # Sliding window count
        valid_kmers = 0
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i : i + self.k]
            if kmer in self._kmer_to_idx:
                counts[self._kmer_to_idx[kmer]] += 1
                valid_kmers += 1
        
        if self.normalize and valid_kmers > 0:
            counts /= valid_kmers
            
        return counts


class DNAKmerFeaturizer(Featurizer):
    """Tokenize DNA sequences into overlapping k-mers.

    K-mer tokenization is a standard preprocessing step for genomic
    language models. A sliding window of width ``k`` is moved across
    the sequence one base at a time, producing overlapping substrings.
    For example, with ``k=3`` the sequence ``ACGTG`` yields
    ``["ACG", "CGT", "GTG"]``.

    This tokenization strategy was popularised by the original
    DNABERT [1]_, which used 6-mer tokens. Newer models like
    DNABERT-2 [2]_ use byte-pair encoding (BPE) instead, but k-mer
    tokenization remains widely used for interpretability and as input
    to classical ML models.

    When ``vocab`` is provided, each k-mer is mapped to its integer
    index. When ``vocab`` is ``None`` (the default), raw k-mer
    strings are returned.

    Parameters
    ----------
    k : int, default 3
        Length of each k-mer token.
    vocab : dict or None, default None
        A mapping from k-mer strings to integer indices. If ``None``,
        the featurizer returns a list of k-mer strings. If provided,
        the featurizer returns an ``int64`` array of token indices;
        unknown k-mers are mapped to index ``0`` (reserved for
        ``<UNK>``).
    strip_n : bool, default False
        If ``True``, remove ambiguous ``N`` characters from the
        sequence before k-mer extraction.

    Examples
    --------
    >>> featurizer = DNAKmerFeaturizer(k=3)
    >>> result = featurizer.featurize(["ACGTG"])
    >>> list(result[0])
    ['ACG', 'CGT', 'GTG']

    Note
    ----
    This featurizer requires only ``numpy``.

    References
    ----------
    .. [1] Ji, Y. et al. DNABERT: pre-trained Bidirectional Encoder
       Representations from Transformers model for DNA-language in
       genome. Bioinformatics 37(15), 2112-2120 (2021).
    .. [2] Zhou, Z. et al. DNABERT-2: Efficient Foundation Model and
       Benchmark for Multi-Species Genome. arXiv:2306.15006 (2023).
    """

    def __init__(
        self,
        k: int = 3,
        vocab: Optional[Dict[str, int]] = None,
        strip_n: bool = False,
    ):
        self.k = k
        self.vocab = vocab
        self.strip_n = strip_n

    def _featurize(
        self,
        datapoint: str,
        **kwargs,
    ) -> Union[List[str], np.ndarray]:
        """Extract k-mers from a single DNA sequence.

        Parameters
        ----------
        datapoint : str
            A DNA sequence string.

        Returns
        -------
        list[str] or np.ndarray
            If ``vocab`` is ``None``, returns a list of k-mer strings.
            Otherwise returns an ``int64`` array of token indices.
        """
        seq = datapoint.upper()
        if self.strip_n:
            seq = seq.replace("N", "")
        kmers = [seq[i:i + self.k] for i in range(len(seq) - self.k + 1)]

        if self.vocab is not None:
            return np.array(
                [self.vocab.get(km, 0) for km in kmers], dtype=np.int64
            )
        return kmers


