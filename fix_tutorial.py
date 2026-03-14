import nbformat as nbf

with open('examples/tutorial.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# Part 1 Code update
nb.cells[3].source = """\
# 1. Load the raw sequences using the DeepChem molnet-style loader pattern
print("Downloading and processing Genomic Benchmarks Database...")

# We use the 'dummy_mouse_enhancers_ensembl' benchmark!
from genomics.loader import load_genomic_benchmark

tasks, datasets, transformers = load_genomic_benchmark(
    dataset_name="dummy_mouse_enhancers_ensembl", 
    featurizer="raw",
    splitter="official", # Use the benchmark's standardized Train/Test split
    reload=False
)

# Note: Genomic Benchmarks 'official' splits only contain Train and Test (No Valid Split natively)
train_dataset, test_dataset = datasets

print(f"\\nTasks: {tasks}")
print(f"Train Dataset Size: {len(train_dataset)}")
print(f"Test Dataset Size: {len(test_dataset)}")

# Notice that X is raw python strings!
print(f"\\nSample raw string sequence (X[0]):\\n{train_dataset.X[0][:80]}...")
print(f"\\nLabel (y[0]): {train_dataset.y[0]}")
"""

# Part 2 Code update
nb.cells[5].source = """\
# We define a 4-mer Featurizer (k=4)
kmer_feat = DNAKmerFeaturizer(k=4)

# Reload the EXACT same benchmark, but instantly swapping out the `featurizer`
print("Featurizing dataset using 4-mers...")
tasks, kmer_datasets, transformers = load_genomic_benchmark(
    dataset_name="dummy_mouse_enhancers_ensembl", 
    featurizer=kmer_feat,
    splitter="official",
    reload=False
)

kmer_train, kmer_test = kmer_datasets

print(f"\\nNew Train Dataset Shape: {kmer_train.X.shape}")
print(f"(Expected (N, 256) since 4^4 possible combinations exist for 4-mers)")
"""

with open('examples/tutorial.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Fixed tutorial.ipynb unpacking error and switched dataset")
