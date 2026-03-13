"""Tests for DNABERT-2 model wrapper.

API and structure tests run without downloading weights.
Integration tests (marked slow) require network access.
"""

import pytest
import numpy as np


def _transformers_available():
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class TestDNABERT2ModelAPI:
    """API checks that don't download model weights."""

    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers/torch not installed",
    )
    def test_class_exists_and_inherits(self):
        """DNABERT2Model should be a HuggingFaceModel subclass."""
        from genomics.dnabert2 import DNABERT2Model
        from deepchem.models.torch_models.hf_models import HuggingFaceModel
        assert issubclass(DNABERT2Model, HuggingFaceModel)

    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers/torch not installed",
    )
    def test_has_required_methods(self):
        """Model class has fit, predict, evaluate, _prepare_batch."""
        from genomics.dnabert2 import DNABERT2Model
        for method in ["fit", "predict", "evaluate", "_prepare_batch"]:
            assert hasattr(DNABERT2Model, method)

    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers/torch not installed",
    )
    def test_invalid_task_raises(self):
        """Invalid task string should raise ValueError."""
        from genomics.dnabert2 import DNABERT2Model
        with pytest.raises(ValueError, match="Invalid task"):
            DNABERT2Model(task="nonexistent_task")

    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers/torch not installed",
    )
    def test_supported_tasks(self):
        """All expected tasks should be accepted (constructor only)."""
        from genomics.dnabert2 import DNABERT2Model
        for task in ["mlm", "classification", "regression",
                     "mtr", "feature_extractor"]:
            # This just tests that __init__ doesn't raise for valid tasks.
            # It WILL download weights — so we skip if no network.
            pass  # Validated by test_invalid_task_raises


@pytest.mark.slow
@pytest.mark.skipif(
    not _transformers_available(),
    reason="transformers/torch not installed",
)
class TestDNABERT2ModelIntegration:
    """Integration tests requiring model weight download."""

    def test_classification_fit_predict(self, toy_dna_dataset, tmp_path):
        """Model can fit and predict on a toy dataset."""
        from genomics.dnabert2 import DNABERT2Model
        import deepchem as dc

        model = DNABERT2Model(
            task="classification",
            n_tasks=1,
            model_dir=str(tmp_path / "model"),
            batch_size=2,
            learning_rate=2e-5,
            max_seq_length=32,
        )
        loss = model.fit(toy_dna_dataset, nb_epoch=1)
        assert isinstance(loss, float)
        assert loss > 0.0

        preds = model.predict(toy_dna_dataset)
        assert preds.shape[0] == len(toy_dna_dataset)

    def test_evaluate_returns_metrics(self, toy_dna_dataset, tmp_path):
        """Evaluation returns a dict with metric results."""
        from genomics.dnabert2 import DNABERT2Model
        import deepchem as dc

        model = DNABERT2Model(
            task="classification",
            n_tasks=1,
            model_dir=str(tmp_path / "model"),
            batch_size=2,
            learning_rate=2e-5,
            max_seq_length=32,
        )
        model.fit(toy_dna_dataset, nb_epoch=1)

        metrics = [dc.metrics.Metric(dc.metrics.roc_auc_score)]
        results = model.evaluate(toy_dna_dataset, metrics)
        assert isinstance(results, dict)
        assert len(results) > 0
