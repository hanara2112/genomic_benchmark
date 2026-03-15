"""DNABERT-2: Efficient Foundation Model for Multi-Species Genome.

DeepChem-native wrapper for DNABERT-2 [1]_. Subclasses
``HuggingFaceModel`` and follows the same pattern as ChemBERTa and
ProtBERT. BPE tokenization is handled inside ``_prepare_batch``
(no separate featurizer needed).

Target location in DeepChem:
``deepchem/models/torch_models/dnabert2.py``

References
----------
.. [1] Zhou, Z., Ji, Y., Li, W., Dutta, P., Davuluri, R., & Liu, H.
   (2023). DNABERT-2: Efficient Foundation Model and Benchmark for
   Multi-Species Genome. arXiv:2306.15006.
"""

import logging
import sys
import types
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# HuggingFaceModel import: same pattern as ProtBERT/ChemBERTa in DeepChem.
try:
    from deepchem.models.torch_models import HuggingFaceModel
except ImportError:
    try:
        from deepchem.models.torch_models.hf_models import HuggingFaceModel
    except ImportError as e:
        raise ImportError(
            "HuggingFaceModel not found. Install DeepChem with PyTorch and "
            "transformers: pip install 'deepchem[torch]' transformers"
        ) from e

_DEFAULT_MODEL_PATH = "zhihan1996/DNABERT-2-117M"


def _stub_flash_attn_if_needed() -> None:
    """Stub flash_attn so DNABERT-2 custom code can load without the optional dependency."""
    _fake = types.ModuleType("flash_attn_triton")
    _fake.flash_attn_qkvpacked_func = None  # type: ignore[attr-defined]
    for key in list(sys.modules.keys()):
        if "flash_attn_triton" in key:
            sys.modules[key] = _fake


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

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> import numpy as np
    >>> import deepchem as dc
    >>> # Prepare toy dataset (raw DNA sequences)
    >>> seqs = np.array(["ACGTACGT", "TGCATGCA", "GGGGCCCC"])
    >>> labels = np.array([0, 1, 0]).reshape(-1, 1).astype(np.float32)
    >>> dataset = dc.data.NumpyDataset(X=seqs, y=labels)
    >>> # Instantiate model
    >>> model_dir = os.path.join(tempfile.mkdtemp(), "dnabert2")
    >>> model = DNABERT2Model(
    ...     task="classification",
    ...     n_tasks=1,
    ...     model_dir=model_dir,
    ...     batch_size=2,
    ...     learning_rate=2e-5,
    ... )

    Reference
    ---------
    .. Zhou, Z. et al. DNABERT-2: Efficient Foundation Model and
       Benchmark for Multi-Species Genome. arXiv:2306.15006 (2023).
    """

    def __init__(
        self,
        task: str,
        model_path: str = _DEFAULT_MODEL_PATH,
        n_tasks: int = 1,
        config: Optional[Dict[str, Any]] = None,
        max_seq_length: int = 512,
        **kwargs: Any,
    ) -> None:
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForMaskedLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        _stub_flash_attn_if_needed()

        self.n_tasks = n_tasks
        self.max_seq_length = max_seq_length

        config_dict: Dict[str, Any] = config if config is not None else {}

        # Load BPE tokenizer (same pattern as ChemBERTa / NucleotideTransformer)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            model_max_length=max_seq_length,
        )

        # Load and patch config (fixes config_class mismatch on HF Hub)
        hf_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            **config_dict,
        )

        _config_defaults: Dict[str, Any] = {
            "is_decoder": False,
            "pad_token_id": (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None else 0
            ),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        for attr, default in _config_defaults.items():
            if not hasattr(hf_config, attr):
                setattr(hf_config, attr, default)

        if getattr(hf_config, "attention_probs_dropout_prob", 0.0) == 0.0:
            hf_config.attention_probs_dropout_prob = 1e-8

        # Task-specific config 
        if task in ("regression", "mtr"):
            hf_config.num_labels = n_tasks
            hf_config.problem_type = "regression"
        elif task == "classification":
            if n_tasks == 1:
                hf_config.problem_type = "single_label_classification"
                hf_config.num_labels = 2
            else:
                hf_config.problem_type = "multi_label_classification"
                hf_config.num_labels = n_tasks

        # Load model via transformers from_pretrained (avoids direct hf_hub/safetensors import)
        _tr = dict(trust_remote_code=True)
        if task == "mlm":
            model = AutoModelForMaskedLM.from_pretrained(
                model_path, config=hf_config, **_tr
            )
        elif task == "feature_extractor":
            model = AutoModel.from_pretrained(
                model_path, config=hf_config, **_tr
            )
        elif task in ("classification", "regression", "mtr"):
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, config=hf_config, **_tr
                )
            except (OSError, ValueError, TypeError):
                # Hub may only have base/MLM; build head from config and load backbone
                model = AutoModelForSequenceClassification.from_config(
                    hf_config, **_tr
                )
                _pretrained = AutoModelForMaskedLM.from_pretrained(
                    model_path, config=hf_config, **_tr
                )
                model.load_state_dict(_pretrained.state_dict(), strict=False)
                del _pretrained
        else:
            raise ValueError(f"Invalid task '{task}'.")

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            task=task,
            **kwargs,
        )

    def _prepare_batch(
        self,
        batch: Tuple[Any, Any, Any],
    ) -> Tuple[Dict[str, torch.Tensor], Any, Any]:
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
        inputs, labels, weights = batch
        sequences = [seq.upper().replace("N", "") for seq in inputs[0]]

        tokens = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        inputs_dict = {k: v.to(self.device) for k, v in tokens.items()}

        y_tensor = None
        if labels is not None:
            if self.task == "classification" and self.n_tasks == 1:
                y_tensor = torch.as_tensor(
                    labels[0].squeeze(-1),
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                y_tensor = torch.as_tensor(
                    labels[0], dtype=torch.float32, device=self.device
                )
            inputs_dict["labels"] = y_tensor

        w_tensor = None
        if weights is not None:
            w_tensor = torch.as_tensor(
                weights[0], dtype=torch.float32, device=self.device
            )

        return inputs_dict, y_tensor, w_tensor
