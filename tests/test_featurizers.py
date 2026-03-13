"""Tests for DNA sequence featurizers."""

import numpy as np
import pytest
from genomics.featurizers import DNAOneHotFeaturizer, DNAKmerFeaturizer


# ──────────────────────────────────────────────────────────────────
# DNAOneHotFeaturizer
# ──────────────────────────────────────────────────────────────────

class TestDNAOneHotFeaturizer:
    """Tests for DNAOneHotFeaturizer."""

    def test_basic_encoding(self):
        """ACGT should produce an identity matrix."""
        featurizer = DNAOneHotFeaturizer(max_length=4)
        result = featurizer.featurize(["ACGT"])
        assert result.shape == (1, 4, 4)
        np.testing.assert_array_equal(result[0], np.eye(4, dtype=np.float32))

    def test_encoding_matches_alphabetical_convention(self):
        """A=0, C=1, G=2, T=3 (standard genomics convention)."""
        featurizer = DNAOneHotFeaturizer(max_length=4)
        result = featurizer.featurize(["ACGT"])[0]
        assert result[0].argmax() == 0  # A
        assert result[1].argmax() == 1  # C
        assert result[2].argmax() == 2  # G
        assert result[3].argmax() == 3  # T

    def test_padding(self):
        """Short sequences are zero-padded on right."""
        featurizer = DNAOneHotFeaturizer(max_length=6)
        result = featurizer.featurize(["AC"])
        assert result.shape == (1, 6, 4)
        np.testing.assert_array_equal(result[0, 0], [1, 0, 0, 0])
        np.testing.assert_array_equal(result[0, 1], [0, 1, 0, 0])
        np.testing.assert_array_equal(result[0, 2], [0, 0, 0, 0])

    def test_truncation(self):
        """Long sequences are truncated to max_length."""
        featurizer = DNAOneHotFeaturizer(max_length=3)
        result = featurizer.featurize(["ACGTACGT"])
        assert result.shape == (1, 3, 4)
        np.testing.assert_array_equal(result[0, 0], [1, 0, 0, 0])  # A
        np.testing.assert_array_equal(result[0, 2], [0, 0, 1, 0])  # G

    def test_lowercase_handling(self):
        """Lowercase input should be handled identically."""
        featurizer = DNAOneHotFeaturizer(max_length=4)
        result = featurizer.featurize(["acgt"])
        np.testing.assert_array_equal(
            result[0], np.eye(4, dtype=np.float32)
        )

    def test_ambiguous_base_zeros(self):
        """N and non-ACGT chars should encode as all-zeros."""
        featurizer = DNAOneHotFeaturizer(max_length=3)
        result = featurizer.featurize(["ANA"])
        np.testing.assert_array_equal(result[0, 0], [1, 0, 0, 0])
        np.testing.assert_array_equal(result[0, 1], [0, 0, 0, 0])
        np.testing.assert_array_equal(result[0, 2], [1, 0, 0, 0])

    def test_empty_sequence(self):
        """Empty string produces all-zero matrix."""
        featurizer = DNAOneHotFeaturizer(max_length=4)
        result = featurizer.featurize([""])
        assert result.shape == (1, 4, 4)
        np.testing.assert_array_equal(result[0], np.zeros((4, 4)))

    def test_multiple_sequences(self):
        """Batch dimension works correctly."""
        featurizer = DNAOneHotFeaturizer(max_length=4)
        result = featurizer.featurize(["AAAA", "TTTT"])
        assert result.shape == (2, 4, 4)
        np.testing.assert_array_equal(result[0, 0], [1, 0, 0, 0])
        np.testing.assert_array_equal(result[1, 0], [0, 0, 0, 1])

    def test_output_dtype(self):
        """Output should be float32."""
        featurizer = DNAOneHotFeaturizer(max_length=4)
        result = featurizer.featurize(["ACGT"])
        assert result.dtype == np.float32

    def test_row_sums(self):
        """Each row sums to 1 (for bases) or 0 (for N/padding)."""
        featurizer = DNAOneHotFeaturizer(max_length=10)
        result = featurizer.featurize(["ACNGT"])[0]
        row_sums = result.sum(axis=1)
        for s in row_sums:
            assert s in [0.0, 1.0]

    def test_negative_max_length_raises(self):
        """Negative max_length should raise ValueError."""
        with pytest.raises(ValueError):
            DNAOneHotFeaturizer(max_length=-1)


# ──────────────────────────────────────────────────────────────────
# DNAKmerFeaturizer
# ──────────────────────────────────────────────────────────────────

class TestDNAKmerFeaturizer:
    """Tests for DNAKmerFeaturizer."""

    def test_basic_kmer_extraction(self):
        """3-mers of ACGTG should be [ACG, CGT, GTG]."""
        featurizer = DNAKmerFeaturizer(k=3)
        result = featurizer.featurize(["ACGTG"])
        assert list(result[0]) == ["ACG", "CGT", "GTG"]

    def test_kmer_count_formula(self):
        """Number of k-mers = len(seq) - k + 1."""
        for k in [1, 3, 6]:
            seq = "A" * 20
            featurizer = DNAKmerFeaturizer(k=k)
            result = featurizer.featurize([seq])[0]
            assert len(result) == len(seq) - k + 1

    def test_k1_returns_individual_bases(self):
        """k=1 returns individual characters."""
        featurizer = DNAKmerFeaturizer(k=1)
        result = featurizer.featurize(["ACGT"])
        assert list(result[0]) == ["A", "C", "G", "T"]

    def test_k_equals_sequence_length(self):
        """k = len(seq) returns the whole sequence as one k-mer."""
        featurizer = DNAKmerFeaturizer(k=4)
        result = featurizer.featurize(["ACGT"])
        assert list(result[0]) == ["ACGT"]

    def test_vocab_mapping(self):
        """With vocab, returns integer indices."""
        vocab = {"ACG": 1, "CGT": 2, "GTG": 3}
        featurizer = DNAKmerFeaturizer(k=3, vocab=vocab)
        result = featurizer.featurize(["ACGTG"])
        np.testing.assert_array_equal(result[0], [1, 2, 3])

    def test_vocab_unknown_maps_to_zero(self):
        """Unknown k-mers map to index 0 (UNK)."""
        vocab = {"ACG": 1}
        featurizer = DNAKmerFeaturizer(k=3, vocab=vocab)
        result = featurizer.featurize(["ACGTG"])
        np.testing.assert_array_equal(result[0], [1, 0, 0])

    def test_lowercase_handling(self):
        """Lowercase input should produce uppercase k-mers."""
        featurizer = DNAKmerFeaturizer(k=3)
        result = featurizer.featurize(["acgtg"])
        assert list(result[0]) == ["ACG", "CGT", "GTG"]

    def test_strip_n(self):
        """strip_n=True removes N before k-mer extraction."""
        featurizer = DNAKmerFeaturizer(k=3, strip_n=True)
        result = featurizer.featurize(["ACNGT"])
        assert list(result[0]) == ["ACG", "CGT"]

    def test_strip_n_false_preserves_n(self):
        """strip_n=False includes N in k-mers."""
        featurizer = DNAKmerFeaturizer(k=3, strip_n=False)
        result = featurizer.featurize(["ACNGT"])
        assert list(result[0]) == ["ACN", "CNG", "NGT"]

    def test_matches_dnabert_reference(self):
        """Output matches DNABERT's generate_kmer_str reference."""
        seq = "ACGTAGCATCGGATCTATCTATCGAC"
        for k in [3, 6]:
            featurizer = DNAKmerFeaturizer(k=k)
            our_str = " ".join(list(featurizer.featurize([seq])[0]))
            ref_str = " ".join(
                seq[i:i + k] for i in range(len(seq) - k + 1)
            )
            assert our_str == ref_str, f"Mismatch for k={k}"

    def test_multiple_sequences(self):
        """Multiple input sequences each get their own k-mer list."""
        featurizer = DNAKmerFeaturizer(k=2)
        result = featurizer.featurize(["ACG", "TGA"])
        assert list(result[0]) == ["AC", "CG"]
        assert list(result[1]) == ["TG", "GA"]
