"""Canonical end-to-end genomics pipeline.

Demonstrates the full DeepChem-style workflow:
  1. Load a Genomic Benchmarks dataset
  2. Split into train / valid / test
  3. Train DNABERT-2 for classification
  4. Evaluate with standard metrics

Run with::

    python -m genomics.run_pipeline

Target location in DeepChem: ``examples/genomics_pipeline.py``
"""

import logging
import tempfile
import os

import numpy as np
import deepchem as dc

from genomics.loader import load_genomic_benchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run the full genomics classification pipeline."""

    # ── 1. Load dataset ──────────────────────────────────────────
    logger.info("Loading dataset...")
    tasks, datasets, transformers = load_genomic_benchmark(
        dataset_name="dummy_mouse_enhancers_ensembl",
        splitter="random",
        reload=False,
    )
    train, valid, test = datasets
    logger.info(
        "Loaded: train=%d, valid=%d, test=%d samples",
        len(train), len(valid), len(test),
    )

    # ── 2. Instantiate DNABERT-2 model ───────────────────────────
    logger.info("Instantiating DNABERT-2 model...")
    model_dir = os.path.join(tempfile.mkdtemp(), "dnabert2-pipeline")

    from genomics.dnabert2 import DNABERT2Model

    model = DNABERT2Model(
        task="classification",
        n_tasks=1,
        model_dir=model_dir,
        batch_size=8,
        learning_rate=2e-5,
        max_seq_length=256,
    )

    # ── 3. Train ─────────────────────────────────────────────────
    logger.info("Training for 3 epochs...")
    loss = model.fit(train, nb_epoch=3)
    logger.info("Training complete. Final loss: %.4f", loss)

    # ── 4. Evaluate ──────────────────────────────────────────────
    logger.info("Evaluating on test set...")
    metrics = [
        dc.metrics.Metric(dc.metrics.roc_auc_score),
        dc.metrics.Metric(dc.metrics.accuracy_score),
    ]
    results = model.evaluate(test, metrics)

    # ── 5. Print results ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("DNABERT-2 Classification Results")
    print("=" * 50)
    print(f"Dataset:  dummy_mouse_enhancers_ensembl")
    print(f"Epochs:   3")
    print(f"Loss:     {loss:.4f}")
    for metric_name, value in results.items():
        print(f"{metric_name}: {value:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
