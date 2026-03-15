import json

with open("examples/Tutorial.ipynb", "r") as f:
    notebook = json.load(f)

for cell in notebook["cells"]:
    if cell.get("metadata", {}).get("id") == "cnn-code":
        cell["source"] = [
            "import deepchem as dc\n",
            "import numpy as np\n",
            "\n",
            "print(\"Extracting features for DeepChem TextCNN...\")\n",
            "# TextCNNModel actually expects 1D integer categorical arrays, NOT One-Hot encoded matrices!\n",
            "# It has a built-in Embedding layer that it learns from scratch.\n",
            "from genomics.featurizers import DNAKmerFeaturizer\n",
            "\n",
            "_, token_datasets, _ = load_genomic_benchmark(\n",
            "    dataset_name=\"human_nontata_promoters\",\n",
            "    featurizer=DNAKmerFeaturizer(k=1, vocab={\"A\":1, \"C\":2, \"G\":3, \"T\":4, \"N\":5}),\n",
            "    splitter=\"official\",\n",
            "    reload=False\n",
            ")\n",
            "token_train, token_test = token_datasets[0], token_datasets[-1]\n",
            "\n",
            "# Pad to fixed length\n",
            "def pad_dataset(dataset, max_len=256):\n",
            "    X_padded = np.zeros((len(dataset), max_len), dtype=np.int32)\n",
            "    for i, x in enumerate(dataset.X):\n",
            "        length = min(len(x), max_len)\n",
            "        X_padded[i, :length] = x[:length]\n",
            "    return dc.data.NumpyDataset(X_padded, dataset.y, dataset.w, dataset.ids)\n",
            "\n",
            "padded_train = pad_dataset(token_train)\n",
            "padded_test = pad_dataset(token_test)\n",
            "\n",
            "# DeepChem's 'TextCNNModel' uses TensorFlow\n",
            "cnn_model = dc.models.TextCNNModel(\n",
            "    n_tasks=1, \n",
            "    char_dict={\"A\":1, \"C\":2, \"G\":3, \"T\":4, \"N\":5}, \n",
            "    char_dict_len=6, # (0 is reserved for padding, so length is 6)\n",
            "    seq_length=256,\n",
            "    mode=\"classification\",\n",
            "    n_classes=2,\n",
            "    batch_size=32,\n",
            "    learning_rate=1e-3\n",
            ")\n",
            "\n",
            "print(\"Training TextCNNModel for 10 epochs...\")\n",
            "cnn_model.fit(padded_train, nb_epoch=10)\n",
            "\n",
            "metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode=\"classification\")\n",
            "cnn_score = cnn_model.evaluate(padded_test, [metric], n_classes=2)\n",
            "cnn_auc = cnn_score[metric.name]\n",
            "print(f\"CNN Test ROC-AUC: {cnn_auc:.4f}\")\n"
        ]

with open("examples/Tutorial.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
