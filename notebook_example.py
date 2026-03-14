import deepchem as dc
from genomics.loader import load_human_nontata_promoters
from genomics.featurizers import DNAKmerFeaturizer

def main():
    print("Example 1: Loading raw DNA sequences for HuggingFace Transformers (like DNABert2)")
    # When featurizer is None or "raw", DeepChem wraps the raw strings using DummyFeaturizer.
    tasks, datasets, transformers = load_human_nontata_promoters(
        featurizer="raw",
        splitter="official",
        reload=False
    )
    train_dataset, valid_dataset, test_dataset = datasets
    print(f"Tasks: {tasks}")
    print(f"Train Dataset: {len(train_dataset)} sequences.")
    print(f"Sample raw sequence from Train (X[0]): {train_dataset.X[0][:50]}...\n")

    print("Example 2: Traditional Machine Learning (K-mers -> Logistic Regression)")
    # We can pass an explicit numerical featurizer, like 4-mers!
    tasks, datasets, transformers = load_human_nontata_promoters(
        featurizer=DNAKmerFeaturizer(k=4),
        splitter="random",
        reload=False
    )
    train_dataset, valid_dataset, test_dataset = datasets
    print(f"Train Dataset: {len(train_dataset)} sequences.")
    print(f"Sample K-mer feature shape from Train (X[0]): {train_dataset.X[0].shape}\n")

if __name__ == "__main__":
    main()
