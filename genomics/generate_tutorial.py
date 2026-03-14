import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# --- Markdown & Code Content ---

title = """\
# Tutorial: Mastering Genomics with DeepChem
### *From Raw DNA to State-of-the-Art Foundation Models*

This tutorial demonstrates how to use the `genomics` module to process, featurize, and model genomic sequences using DeepChem's standard MolNet architecture. 

We will use the **Human Non-TATA Promoters** dataset from Genomic Benchmarks to walk through a complete research workflow.
"""

setup_code = """\
import os
import deepchem as dc
import numpy as np
import matplotlib.pyplot as plt
from genomics.loader import load_genomic_benchmark
from genomics.featurizers import DNAKmerCountFeaturizer, DNAOneHotFeaturizer
from sklearn.ensemble import RandomForestClassifier

# Set random seeds for reproducibility
np.random.seed(42)
"""

section1_md = """\
## Section 1: Standardized Data Loading
DeepChem's `load_genomic_benchmark` function handles downloading, splitting, and memory-efficient storage. 

We use `splitter="official"` to ensure we use the benchmark's standardized Train/Test sets, which is critical for scientific reproducibility.
"""

section1_code = """\
# Load the dataset using the official benchmark splits
tasks, datasets, transformers = load_genomic_benchmark(
    dataset_name="human_nontata_promoters",
    featurizer=None,    # Load raw sequences first
    splitter="official",
    reload=False
)

# Some benchmarks might only have Train/Test (Valid may be empty)
train, test = datasets[0], datasets[-1]

print(f"Tasks: {tasks}")
print(f"Train samples: {len(train)}")
print(f"Test samples:  {len(test)}")
print(f"\\nSample DNA sequence: {train.ids[0][:50]}...")
"""

section2_md = """\
## Section 2: Flexible DNA Featurization
Genomic sequences need to be converted into numerical tensors. DeepChem makes this effortless.

1. **One-Hot Encoding**: Best for CNNs.
2. **K-mer Frequency**: Best for traditional ML and Transformers.
"""

section2_code = """\
# 1. One-Hot Featurization (Shortcut: "onehot")
_, oh_datasets, _ = load_genomic_benchmark(
    dataset_name="human_nontata_promoters",
    featurizer="onehot",
    splitter="official",
    reload=False
)
oh_train = oh_datasets[0]
print(f"One-Hot Shape: {oh_train.X.shape} (N, Length, 4)")

# 2. K-mer Featurization (Using 4-mers)
kmer_feat = DNAKmerCountFeaturizer(k=4)
_, kmer_datasets, _ = load_genomic_benchmark(
    dataset_name="human_nontata_promoters",
    featurizer=kmer_feat,
    splitter="official",
    reload=False
)
kmer_train = kmer_datasets[0]
print(f"K-mer Shape: {kmer_train.X.shape} (N, 256 frequencies)")
"""

section3_md = """\
## Section 3: Data Visualization
Understanding the "signal" in your genomic data. Let's visualize the 4-mer frequency distribution between promoters and non-promoters.
"""

section3_code = """\
# Compare mean K-mer frequencies across classes
pos_mask = (kmer_train.y == 1).flatten()
neg_mask = (kmer_train.y == 0).flatten()

pos_mean = kmer_train.X[pos_mask].mean(axis=0)
neg_mean = kmer_train.X[neg_mask].mean(axis=0)

plt.figure(figsize=(12, 4))
plt.plot(pos_mean, label="Promoter (Positive)", alpha=0.8)
plt.plot(neg_mean, label="Non-Promoter (Negative)", alpha=0.8)
plt.fill_between(range(256), pos_mean, neg_mean, color='gray', alpha=0.2)
plt.title("4-mer Frequency Signature: Promoters vs Non-Promoters")
plt.xlabel("K-mer Index")
plt.ylabel("Mean Frequency")
plt.legend()
plt.show()
"""

section4_md = """\
## Section 4: Baseline Modeling with Random Forest
We use DeepChem's `SklearnModel` wrapper to train a quick baseline on our K-mer features.
"""

section4_code = """\
# Initialize and train
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
model = dc.models.SklearnModel(rf)
model.fit(kmer_train)

# Evaluate using ROC-AUC
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
kmer_test = kmer_datasets[-1]
test_score = model.evaluate(kmer_test, [metric])

print(f"Baseline Test ROC-AUC: {test_score['roc_auc_score']:.4f}")
"""

section5_md = """\
## Section 5: Foundation Models (DNABERT-2)
Finally, we scale up to a Transformer-based foundation model. DNABERT-2 is pre-trained on massive genomic corpora and can be fine-tuned for specific tasks.
"""

section5_code = """\
from genomics.dnabert2 import DNABERT2Model

# Note: Using a small max_seq_length and batch_size for demonstration
bert_model = DNABERT2Model(
    task="classification",
    n_tasks=1,
    batch_size=4,
    max_seq_length=128
)

# Train for a few steps
bert_model.fit(train, nb_epoch=1)

# Evaluate
bert_score = bert_model.evaluate(test, [metric])
print(f"DNABERT-2 Test ROC-AUC: {bert_score['roc_auc_score']:.4f}")
"""

# --- Build Notebook ---

nb['cells'] = [
    nbf.v4.new_markdown_cell(title),
    nbf.v4.new_code_cell(setup_code),
    nbf.v4.new_markdown_cell(section1_md),
    nbf.v4.new_code_cell(section1_code),
    nbf.v4.new_markdown_cell(section2_md),
    nbf.v4.new_code_cell(section2_code),
    nbf.v4.new_markdown_cell(section3_md),
    nbf.v4.new_code_cell(section3_code),
    nbf.v4.new_markdown_cell(section4_md),
    nbf.v4.new_code_cell(section4_code),
    nbf.v4.new_markdown_cell(section5_md),
    nbf.v4.new_code_cell(section5_code)
]

# Use absolute path to ensure saving
save_path = '/Users/aryamanbahl/IIITH/opensource/dc-genomics-review/examples/tutorial.ipynb'
with open(save_path, 'w') as f:
    nbf.write(nb, f)

print(f"Successfully generated tutorial at {save_path}")
