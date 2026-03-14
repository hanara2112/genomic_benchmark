import argparse
import logging
import deepchem as dc
from deepchem.models import SklearnModel
from sklearn.ensemble import RandomForestClassifier

from genomics.loader import load_genomic_benchmark
from genomics.featurizers import KmerFrequencyFeaturizer, DNAOneHotFeaturizer

logger = logging.getLogger(__name__)


def get_featurizer(feat_str: str):
    if feat_str == "kmer":
        return KmerFrequencyFeaturizer(k=4)
    elif feat_str == "onehot":
        return DNAOneHotFeaturizer()
    elif feat_str == "raw":
        return "raw"
    else:
        raise ValueError(f"Unknown featurizer: {feat_str}")


def main():
    parser = argparse.ArgumentParser(description="Genomic Benchmark Script for DeepChem")
    parser.add_argument("--dataset",
                        type=str,
                        default="human_nontata_promoters",
                        help="Name of the Genomic Benchmark dataset to load.")
    parser.add_argument("--featurizer",
                        type=str,
                        default="kmer",
                        choices=["kmer", "onehot", "raw"],
                        help="Featurizer to apply to the DNA sequences.")
    parser.add_argument("--splitter",
                        type=str,
                        default="official",
                        choices=["official", "random"],
                        help="Type of data split.")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    logger.info(f"Loading '{args.dataset}' with '{args.featurizer}' featurizer via '{args.splitter}' split...")
    featurizer = get_featurizer(args.featurizer)
    
    # Load dataset using our new deepchem integration
    tasks, datasets, transformers = load_genomic_benchmark(
        dataset_name=args.dataset,
        featurizer=featurizer,
        splitter=args.splitter,
        reload=False
    )

    if len(datasets) == 2:
        train_dataset, test_dataset = datasets
        valid_dataset = None
    elif len(datasets) == 3:
        train_dataset, valid_dataset, test_dataset = datasets
    else:
        raise ValueError("Unexpected dataset split tuple length.")

    n_tasks = len(tasks)
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Train samples: {len(train_dataset)}")
    if valid_dataset: logger.info(f"Valid samples: {len(valid_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    
    if args.featurizer == "raw":
        logger.warning("Raw strings requested! This script demonstrates the traditional Sklearn Random Forest setup.")
        logger.warning("To process raw strings, you would instantiate a HuggingFaceModel (e.g. DNABert2) here instead.")
        return

    # Baseline Model: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model = dc.models.SklearnModel(rf)

    logger.info("Training a scikit-learn Random Forest model on generated numerical features...")
    model.fit(train_dataset)

    # Evaluation
    logger.info("Evaluating with ROC-AUC metric...")
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    
    train_score = model.evaluate(train_dataset, [metric], transformers)
    test_score = model.evaluate(test_dataset, [metric], transformers)
    
    logger.info(f"Train ROC-AUC: {train_score['roc_auc_score']:.3f}")
    if valid_dataset:
        valid_score = model.evaluate(valid_dataset, [metric], transformers)
        logger.info(f"Valid ROC-AUC: {valid_score['roc_auc_score']:.3f}")
    logger.info(f"Test ROC-AUC:  {test_score['roc_auc_score']:.3f}")


if __name__ == "__main__":
    main()
