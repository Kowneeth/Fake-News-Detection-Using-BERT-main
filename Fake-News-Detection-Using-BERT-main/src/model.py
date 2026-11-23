"""
Neural Network Architectures and Model Wrappers for Fake News Detection.
Supports BERT, RoBERTa, DeBERTa, and DistilBERT with flexible layer unfreezing.
"""
import os
import json
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

# Check Transformers availability
try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    transformers = None
    AutoModel = None
    AutoTokenizer = None
    AutoConfig = None
    TRANSFORMERS_AVAILABLE = False


if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
    class TransformerClassifier(nn.Module):
        """
        Configurable Transformer Classifier for Fake News Detection.
        Supports BERT, RoBERTa, DeBERTa-v3, and DistilBERT.
        """
        def __init__(
            self,
            model_name: str = "bert-base-uncased",
            num_classes: int = 2,
            dropout_rate: float = 0.2,
            hidden_dim: int = 512,
            unfreeze_layers: int = 2
        ):
            super().__init__()
            self.model_name = model_name
            self.config = AutoConfig.from_pretrained(model_name)
            self.transformer = AutoModel.from_pretrained(model_name)
            
            # Identify embedding/hidden size
            hidden_size = getattr(self.config, "hidden_size", 768)

            # Freeze layers according to unfreeze_layers parameter
            self._configure_unfreezing(unfreeze_layers)

            # Classification Head
            self.dropout = nn.Dropout(dropout_rate)
            self.fc1 = nn.Linear(hidden_size, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, num_classes)

        def _configure_unfreezing(self, unfreeze_layers: int):
            """
            Configures layer freezing.
            - unfreeze_layers == 0: Freeze all transformer layers (feature extraction).
            - unfreeze_layers == -1: Unfreeze all layers (full fine-tuning).
            - unfreeze_layers > 0: Freeze lower layers, unfreeze top N encoder layers.
            """
            if unfreeze_layers == -1:
                logger.info("Unfreezing all transformer layers for full fine-tuning.")
                for param in self.transformer.parameters():
                    param.requires_grad = True
                return

            # First freeze all
            for param in self.transformer.parameters():
                param.requires_grad = False

            if unfreeze_layers == 0:
                logger.info("All transformer layers frozen (feature extraction mode).")
                return

            # Unfreeze top N encoder layers
            encoder = None
            if hasattr(self.transformer, "encoder"):
                encoder = self.transformer.encoder
            elif hasattr(self.transformer, "transformer"):
                encoder = self.transformer.transformer

            if encoder and hasattr(encoder, "layer"):
                layers = encoder.layer
                num_total_layers = len(layers)
                layers_to_unfreeze = list(layers)[-unfreeze_layers:]
                for layer in layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True
                logger.info(f"Unfrozen top {unfreeze_layers}/{num_total_layers} encoder layers.")
            elif encoder and hasattr(encoder, "layers"):  # e.g., DeBERTa
                layers = encoder.layers
                num_total_layers = len(layers)
                layers_to_unfreeze = list(layers)[-unfreeze_layers:]
                for layer in layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True
                logger.info(f"Unfrozen top {unfreeze_layers}/{num_total_layers} encoder layers.")
            else:
                # Fallback: unfreeze pooler or last parameters
                logger.info(f"Targeting last parameters for unfreezing.")
                for param in list(self.transformer.parameters())[-unfreeze_layers * 16:]:
                    param.requires_grad = True

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            
            # Extract pooled representation or CLS token
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                cls_repr = outputs.pooler_output
            else:
                # First token ([CLS]) of last hidden state
                cls_repr = outputs.last_hidden_state[:, 0, :]

            x = self.fc1(cls_repr)
            x = self.relu(x)
            x = self.dropout(x)
            logits = self.fc2(x)
            return logits
else:
    class TransformerClassifier:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch and Hugging Face Transformers are required to instantiate TransformerClassifier.")
