"""
DNABERT-2: Efficient Foundation Model for Multi-Species Genome.

DeepChem wrapper for DNABERT-2 following the ChemBERTa integration
pattern used in DeepChem.

Reference
---------
Zhou et al. (2023)
DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome.
arXiv:2306.15006
"""

import logging
from typing import Any, Dict, Tuple, Optional

import deepchem as dc
import torch
import torch.nn as nn
from deepchem.models.torch_models.hf_models import HuggingFaceModel

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "zhihan1996/DNABERT-2-117M"


class DNABERT2Model(HuggingFaceModel):
    """DNABERT-2 model for DNA sequence analysis.

    DNABERT-2 is a foundation model pretrained on large-scale
    multi-species genomes using byte-pair encoding (BPE) tokenization.
    This wrapper subclasses ``HuggingFaceModel`` and follows the
    same integration pattern as ChemBERTa and ProtBERT.

    BPE tokenization is handled inside ``_prepare_batch`` — raw DNA
    sequences are stored in ``X`` (via ``DummyFeaturizer``) and
    tokenized on-the-fly during training, matching the standard
    DeepChem HuggingFace model pattern.

    Parameters
    ----------
    task : str
        Learning task. One of ``'mlm'``, ``'classification'``,
        ``'regression'``, ``'mtr'``, or ``'feature_extractor'``.
    model_path : str, default 'zhihan1996/DNABERT-2-117M'
        HuggingFace Hub model ID or local checkpoint path.
    n_tasks : int, default 1
        Number of output tasks.
    config : dict or None, default None
        Extra config overrides passed to
        ``AutoConfig.from_pretrained()``.
    max_seq_length : int, default 512
        Maximum tokenised sequence length.
    kwargs : dict
        Additional arguments passed to ``HuggingFaceModel``.

    References
    ----------
    .. Zhou, Z. et al. DNABERT-2: Efficient Foundation Model and
       Benchmark for Multi-Species Genome. arXiv:2306.15006 (2023).
    """

    def __init__(
        self,
        task: str,
        model_path: str = _DEFAULT_MODEL,
        n_tasks: int = 1,
        config: Optional[Dict] = None,
        max_seq_length: int = 512,
        **kwargs,
    ):
        """Initialize a DNABERT-2 model head."""
        self.n_tasks = n_tasks
        self.max_seq_length = max_seq_length

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            model_max_length=max_seq_length,
        )

        hf_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            **(config or {}),
        )

        # Set task-specific config
        if task in ["regression", "mtr"]:
            hf_config.problem_type = "regression"
            hf_config.num_labels = n_tasks
        elif task == "classification":
            hf_config.num_labels = 2 if n_tasks == 1 else n_tasks
            hf_config.problem_type = (
                "single_label_classification"
                if n_tasks == 1
                else "multi_label_classification"
            )

        # Load appropriate model head
        if task == "mlm":
            model = AutoModelForMaskedLM.from_pretrained(
                model_path, config=hf_config, trust_remote_code=True
            )
        elif task in ["classification", "regression", "mtr"]:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, config=hf_config, trust_remote_code=True
            )
        elif task == "feature_extractor":
            model = AutoModel.from_pretrained(
                model_path, config=hf_config, trust_remote_code=True
            )
        else:
            raise ValueError(f"Invalid task: {task}")

        # The DNABERT-2 Hub model returns token-level logits (3D) even for classification.
        # We patch the model's forward to average across the sequence (Pooling).
        original_forward = model.forward

        def pooled_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            # If the output (logits) is 3D (Batch, Seq, Labels), we pool it to 2D (Batch, Labels)
            if hasattr(out, "logits") and out.logits is not None and out.logits.ndim == 3:
                out.logits = out.logits.mean(dim=1)
            elif isinstance(out, torch.Tensor) and out.ndim == 3:
                out = out.mean(dim=1)
            return out

        # Custom pooling logic
        model.forward = pooled_forward

        # Use native DeepChem loss objects
        if task == "classification":
            loss = dc.models.losses.SoftmaxCrossEntropy()
        else:
            loss = dc.models.losses.L2Loss()

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            task=task,
            loss=loss,
            **kwargs,
        )

    def _prepare_batch(
        self,
        batch: Tuple[Any, Any, Any],
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare a batch for DNABERT-2.

        Tokenizes raw DNA sequences using the BPE tokenizer loaded
        at initialization. Follows the same pattern as ChemBERTa's
        ``_prepare_batch``.

        Parameters
        ----------
        batch : tuple
            ``(inputs, labels, weights)`` from the DeepChem data
            loader. ``inputs[0]`` contains raw DNA sequence strings.

        Returns
        -------
        inputs_dict : dict
            Tokenized inputs with ``input_ids``, ``attention_mask``,
            and optionally ``labels``.
        y_tensor : Tensor or None
            Label tensor.
        w_tensor : Tensor or None
            Weight tensor.
        """
        X, y, w = batch

        # Handle potential numpy dimensionality from loader
        X_batch = X[0]
        if hasattr(X_batch, "shape") and len(X_batch.shape) > 1:
            X_batch = X_batch.flatten()
        
        sequences = [str(seq).upper() for seq in X_batch]

        tokens = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in tokens.items()}

        y_tensor = None
        if y is not None:
            # We don't pass 'labels' inside inputs to avoid buggy internal 
            # loss calculations in some Hub model versions. 
            # TorchModel handles loss using the returned y_tensor instead.
            if self.task == "classification" and self.n_tasks == 1:
                y_tensor = torch.as_tensor(y[0].flatten(), dtype=torch.long, device=self.device)
            else:
                y_tensor = torch.as_tensor(y[0], dtype=torch.float32, device=self.device)

        w_tensor = torch.as_tensor(w[0], dtype=torch.float32, device=self.device) if w is not None else None

        return inputs, y_tensor, w_tensor
