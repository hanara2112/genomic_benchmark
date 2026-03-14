import nbformat as nbf

nb = nbf.v4.new_notebook()

title = """\
# Tutorial: Genomic Benchmarks in DeepChem
**Introduction to DNA Sequence Loaders, Featurizers, and Models**

This tutorial comprehensively demonstrates how to align with DeepChem's architecture to process, featurize, evaluate, and visualize genomic sequences using the newly integrated `genomics` tools.

We will focus on the **Human Non-TATA Promoters** dataset from the Genomic Benchmarks repository.
"""

setup_code = """\
# Required Imports
import os
import deepchem as dc
import numpy as np
import matplotlib.pyplot as plt

from genomics.loader import load_human_nontata_promoters
from genomics.featurizers import DNAKmerFeaturizer, DNAOneHotFeaturizer
from sklearn.ensemble import RandomForestClassifier
"""

part1_md = """\
## Part 1: Exploring Raw Sequence Datasets
When experimenting with HuggingFace language models (like DNABert or Nucleotide Transformer), the model requires pure string inputs since it uses its own subword sequences (like Byte-Pair Encoding). 

DeepChem handles string pipelines explicitly via the `DummyFeaturizer` under the hood when passing `featurizer="raw"`.
"""

part1_code = """\
# 1. Load the raw sequences using the DeepChem molnet-style loader pattern
print("Downloading and processing Genomic Benchmarks Database...")
tasks, datasets, transformers = load_human_nontata_promoters(
    featurizer="raw",
    splitter="official", # Use the benchmark's standardized Train/Test split
    reload=False
)

train_dataset, valid_dataset, test_dataset = datasets

print(f"\\nTasks: {tasks}")
print(f"Train Dataset Size: {len(train_dataset)}")
print(f"Test Dataset Size: {len(test_dataset)}")

# Notice that X is raw python strings!
print(f"\\nSample raw string sequence (X[0]):\\n{train_dataset.X[0][:80]}...")
print(f"\\nLabel (y[0]): {train_dataset.y[0]}")
"""

part2_md = """\
## Part 2: Feature Engineering (K-mers & One-Hot Encoding)
Traditional machine learning and numerical Deep Learning models cannot consume variable-length strings. They require structured numerical tensors. We can extract these tensors seamlessly using DeepChem `Featurizers`.

Let's convert our DNA strings into **4-mer frequency counts**.
"""

part2_code = """\
# We define a 4-mer Featurizer (k=4)
kmer_feat = DNAKmerFeaturizer(k=4)

# Reload the EXACT same benchmark, but instantly swapping out the `featurizer`
print("Featurizing dataset using 4-mers...")
tasks, kmer_datasets, transformers = load_human_nontata_promoters(
    featurizer=kmer_feat,
    splitter="official",
    reload=False
)

kmer_train, _, kmer_test = kmer_datasets

print(f"\\nNew Train Dataset Shape: {kmer_train.X.shape}")
print(f"(Expected (N, 256) since 4^4 possible combinations exist for 4-mers)")
"""

part3_md = """\
## Part 3: Data Visualization
Let's see what a DNA sequence looks like after being transformed into K-mer frequencies!
"""

part3_code = """\
# Let's visualize the differences in K-mer frequencies between a positive vs negative sequence
pos_idx = np.where(kmer_train.y == 1)[0][0]
neg_idx = np.where(kmer_train.y == 0)[0][0]

pos_features = kmer_train.X[pos_idx]
neg_features = kmer_train.X[neg_idx]

plt.figure(figsize=(15, 5))
plt.plot(pos_features, label="Positive class (Promoter)", alpha=0.7)
plt.plot(neg_features, label="Negative class (Non-Promoter)", alpha=0.7)
plt.title("4-mer Frequency Distributions")
plt.xlabel("K-mer Index (0-255)")
plt.ylabel("Frequency")
plt.legend()
plt.show()
"""

part4_md = """\
## Part 4: Training and Evaluation using DeepChem
DeepChem wraps standard implementations uniformly. Let's use an `SklearnModel` Random Forest baseline, train it on our K-mer dataset, and evaluate it using standard DeepChem Metrics!
"""

part4_code = """\
# Initialize Random Forest inside DeepChem wrapper
print("Training Random Forest Classifier on 4-mer dataset...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
model = dc.models.SklearnModel(rf)

# Model Training
model.fit(kmer_train)

# Evaluation using DeepChem ROC-AUC Metric
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
print("\\nEvaluating performance...")
train_scores = model.evaluate(kmer_train, [metric], transformers)
test_scores = model.evaluate(kmer_test, [metric], transformers)

print(f"Train ROC-AUC: {train_scores['roc_auc_score']:.3f}")
print(f"Test ROC-AUC:  {test_scores['roc_auc_score']:.3f}")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(title),
    nbf.v4.new_code_cell(setup_code),
    nbf.v4.new_markdown_cell(part1_md),
    nbf.v4.new_code_cell(part1_code),
    nbf.v4.new_markdown_cell(part2_md),
    nbf.v4.new_code_cell(part2_code),
    nbf.v4.new_markdown_cell(part3_md),
    nbf.v4.new_code_cell(part3_code),
    nbf.v4.new_markdown_cell(part4_md),
    nbf.v4.new_code_cell(part4_code)
]

with open('examples/tutorial.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Updated tutorial.ipynb")
