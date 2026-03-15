import os
import argparse
import logging
import json
import time

import numpy as np
import torch
import torch.nn as nn

import deepchem as dc
from deepchem.models import SklearnModel, TorchModel
from sklearn.ensemble import RandomForestClassifier

from genomics.loader import load_genomic_benchmark
from genomics.featurizers import DNAKmerCountFeaturizer, DNAOneHotFeaturizer
from genomics.dnabert2 import DNABERT2Model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple PyTorch CNN & LSTM Models
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=4, out_channels=64, kernel_size=8, n_classes=2):
        super().__init__()
        # PyTorch Conv1d expects (batch, channels, seq_len), so we transpose later
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1) # Global average pooling
        self.fc = nn.Linear(out_channels, n_classes)

    def forward(self, x):
        # x is (batch, seq_len, 4) -> swap to (batch, 4, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        return logits


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size=4, embedding_dim=16, hidden_dim=32, n_classes=2):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        # x is (batch, seq_len, 4) from OneHot
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        # pool over sequence (mean pooling)
        pooled = torch.mean(lstm_out, dim=1)
        logits = self.fc(pooled)
        return logits

def _loss_fn(outputs, labels, weights):
    # outputs is logits
    loss = nn.CrossEntropyLoss(reduction="none")(outputs, labels[0].long())
    if weights is not None:
        loss = loss * weights[0]
    return loss.mean()

# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------
def train_rf(dataset_name, max_seq_length):
    logger.info("Loading Random Forest (4-mer) data...")
    tasks, datasets, _ = load_genomic_benchmark(
        dataset_name=dataset_name,
        featurizer=DNAKmerCountFeaturizer(k=4),
        splitter="official",
        reload=False
    )
    train, test = datasets[0], datasets[-1]
    
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model = dc.models.SklearnModel(rf)
    model.fit(train)
    duration = time.time() - start_time
    
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode="classification")
    return model.evaluate(test, [metric])['roc_auc_score'], duration


def train_cnn(dataset_name, max_seq_length):
    logger.info("Loading CNN (OneHot) data...")
    tasks, datasets, _ = load_genomic_benchmark(
        dataset_name=dataset_name,
        featurizer=DNAOneHotFeaturizer(max_length=max_seq_length),
        splitter="official",
        reload=False
    )
    train, test = datasets[0], datasets[-1]
    
    start_time = time.time()
    pytorch_model = SimpleCNN()
    model = TorchModel(
        pytorch_model,
        loss=_loss_fn,
        output_types=["prediction"],
        batch_size=32,
        learning_rate=1e-3,
    )
    
    # Needs categorical to long for cross entropy loss
    def _generator(dataset, batch_size):
        for batch in dataset.iterbatches(batch_size):
            yield (batch[0], batch[1].astype(np.int64), batch[2])

    model.fit_generator(_generator(train, 32), nb_epoch=10)
    duration = time.time() - start_time
    
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode="classification")
    
    preds = model.predict(test)
    probs = np.exp(preds) / np.sum(np.exp(preds), axis=-1, keepdims=True)
    score = metric.compute_metric(test.y, probs, test.w)
    
    return score, duration


def train_lstm(dataset_name, max_seq_length):
    logger.info("Loading BiLSTM (OneHot) data...")
    tasks, datasets, _ = load_genomic_benchmark(
        dataset_name=dataset_name,
        featurizer=DNAOneHotFeaturizer(max_length=max_seq_length),
        splitter="official",
        reload=False
    )
    train, test = datasets[0], datasets[-1]
    
    start_time = time.time()
    pytorch_model = SimpleLSTM()
    model = TorchModel(
        pytorch_model,
        loss=_loss_fn,
        output_types=["prediction"],
        batch_size=32,
        learning_rate=1e-3,
    )
    
    # Needs categorical to long for cross entropy loss
    def _generator(dataset, batch_size):
        for batch in dataset.iterbatches(batch_size):
            yield (batch[0], batch[1].astype(np.int64), batch[2])

    model.fit_generator(_generator(train, 32), nb_epoch=10)
    duration = time.time() - start_time
    
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode="classification")
    
    # Softmax the logits back to probabilities for ROC-AUC
    preds = model.predict(test)
    probs = np.exp(preds) / np.sum(np.exp(preds), axis=-1, keepdims=True)
    score = metric.compute_metric(test.y, probs, test.w)
    
    return score, duration


def train_bert(dataset_name, max_seq_length):
    logger.info("Loading DNABERT-2 (Raw) data...")
    tasks, datasets, _ = load_genomic_benchmark(
        dataset_name=dataset_name,
        featurizer="raw",
        splitter="official",
        reload=False
    )
    train, test = datasets[0], datasets[-1]
    
    start_time = time.time()
    model_dir = f"dnabert2_ckpt_{dataset_name}"
    model = DNABERT2Model(
        task="classification",
        n_tasks=1,
        model_dir=model_dir,
        batch_size=4,
        max_seq_length=max_seq_length,
        learning_rate=2e-5,
    )
    model.fit(train, nb_epoch=1)
    duration = time.time() - start_time
    
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode="classification")
    return model.evaluate(test, [metric], n_classes=2)[metric.name], duration

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Full Model Benchmark Sweep")
    parser.add_argument("--datasets", nargs="+", default=["dummy_mouse_enhancers_ensembl", "human_nontata_promoters"])
    parser.add_argument("--models", nargs="+", default=["rf", "cnn", "lstm", "dnabert2"])
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    results = {}
    
    for dataset in args.datasets:
        logger.info(f"=== Starting benchmark for {dataset} ===")
        results[dataset] = {}
        
        runs = {
            "rf": train_rf,
            "cnn": train_cnn,
            "lstm": train_lstm,
            "dnabert2": train_bert,
        }
        
        for m_name in args.models:
            if m_name not in runs:
                logger.warning(f"Unknown model skip: {m_name}")
                continue
                
            logger.info(f"--- Training {m_name.upper()} ---")
            try:
                score, duration = runs[m_name](dataset, args.max_seq_length)
                logger.info(f"Result for {m_name} on {dataset} -> ROC-AUC: {score:.4f} (Time: {duration:.1f}s)")
                results[dataset][m_name] = {
                    "roc_auc": float(score),
                    "time_seconds": float(duration)
                }
            except Exception as e:
                logger.error(f"Error training {m_name} on {dataset}: {e}")
                results[dataset][m_name] = {"error": str(e)}
                
    # Save results to cleanly formatted JSON
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"\nAll models finished. Results saved to {args.output}")

if __name__ == "__main__":
    main()
