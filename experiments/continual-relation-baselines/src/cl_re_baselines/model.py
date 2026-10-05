from __future__ import annotations

import logging
from typing import Any, Sequence

from schemas.relation_sample import RelationSample
from .data_loader import SPECIAL_MARKERS, prepare_sample_tokens

logger = logging.getLogger(__name__)

# Check if PyTorch and Transformers are available
try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BERTRelationClassifier:
    """BERT-based Relation Extraction Classifier with Entity Marker representations.

    Architecture:
        1. Tokenizer with special entity boundary markers ([E1], [/E1], [E2], [/E2]).
        2. Pretrained BERT encoder backbone.
        3. Entity Representation: Concatenation of [E1] and [E2] hidden states:
               h_rep = [h_{e1}; h_{e2}] in R^{2 * hidden_size}
        4. Dropout (p=0.1) and Linear classification head (2 * hidden_size -> num_classes).
        5. Seen-class masking during inference and training to enforce Class-Incremental constraints.

    Continual Learning Properties:
        - The classifier head is fixed at 80 classes.
        - Old relation weights are permanently preserved across stages T1..T8.
        - Backbone and classifier weights are NEVER reinitialized between tasks.
    """

    def __init__(
        self,
        backbone_name: str = "bert-base-uncased",
        num_classes: int = 80,
        dropout_rate: float = 0.1,
        max_seq_length: int = 128,
        device: str = "auto",
        relation_to_class_idx: dict[str, int] | None = None,
        class_idx_to_rel_id: dict[int, int] | None = None,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch and HuggingFace Transformers are required to instantiate BERTRelationClassifier. "
                "Install them via: pip install torch transformers"
            )

        self.backbone_name = backbone_name
        self.num_classes = int(num_classes)
        self.dropout_rate = float(dropout_rate)
        self.max_seq_length = int(max_seq_length)
        self.relation_to_class_idx = relation_to_class_idx
        self.class_idx_to_rel_id = class_idx_to_rel_id

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing BERTRelationClassifier on device: {self.device}")

        # Initialize Tokenizer and add entity markers
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)
        special_tokens_dict = {"additional_special_tokens": SPECIAL_MARKERS}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        # Initialize Encoder
        self.encoder = AutoModel.from_pretrained(self.backbone_name)
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        # Classifier Head: Concatenated entity states [h_e1; h_e2]
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(hidden_size * 2, self.num_classes)

        # Move to target device
        self.encoder.to(self.device)
        self.dropout.to(self.device)
        self.classifier.to(self.device)

        # Record tokens IDs for markers
        self.e1_token_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        self.e2_token_id = self.tokenizer.convert_tokens_to_ids("[E2]")

        # Current number of seen classes for masking (default: all classes)
        self.seen_classes_count = self.num_classes

    def set_seen_classes(self, num_seen_classes: int) -> None:
        """Set the number of visible classes for prediction masking.

        Classes in [0, num_seen_classes - 1] are visible;
        Classes in [num_seen_classes, num_classes - 1] are masked out (-1e9).
        """
        if not (1 <= num_seen_classes <= self.num_classes):
            raise ValueError(f"num_seen_classes must be in [1, {self.num_classes}], got {num_seen_classes}")
        self.seen_classes_count = num_seen_classes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_indices: torch.Tensor,
        e2_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass extracting entity representations and computing class logits."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]

        batch_size = input_ids.size(0)
        batch_indices = torch.arange(batch_size, device=input_ids.device)

        # Extract hidden states at [E1] and [E2] positions
        # e1_indices: [batch_size], e2_indices: [batch_size]
        h_e1 = sequence_output[batch_indices, e1_indices]  # [batch_size, hidden_size]
        h_e2 = sequence_output[batch_indices, e2_indices]  # [batch_size, hidden_size]

        # Concatenate entity vectors
        h_entity = torch.cat([h_e1, h_e2], dim=-1)  # [batch_size, 2 * hidden_size]
        h_entity = self.dropout(h_entity)

        logits = self.classifier(h_entity)  # [batch_size, num_classes]

        # Mask unseen classes
        if self.seen_classes_count < self.num_classes:
            mask_value = -1e9
            logits[:, self.seen_classes_count :] = mask_value

        return logits

    def _tokenize_sample(self, sample: RelationSample) -> tuple[list[int], int, int]:
        """Convert a RelationSample into subword token IDs and locate [E1], [E2] token indices."""
        tokens, _, _ = prepare_sample_tokens(sample)

        # Tokenize preserving subword alignment
        subwords: list[str] = [self.tokenizer.cls_token]
        e1_idx: int | None = None
        e2_idx: int | None = None

        for token in tokens:
            if token == "[E1]":
                e1_idx = len(subwords)
                subwords.append("[E1]")
            elif token == "[E2]":
                e2_idx = len(subwords)
                subwords.append("[E2]")
            elif token in {"[/E1]", "[/E2]"}:
                subwords.append(token)
            else:
                pieces = self.tokenizer.tokenize(token)
                subwords.extend(pieces)

        subwords.append(self.tokenizer.sep_token)

        # Truncate if exceeding max_seq_length
        if len(subwords) > self.max_seq_length:
            subwords = subwords[: self.max_seq_length]
            if subwords[-1] != self.tokenizer.sep_token:
                subwords[-1] = self.tokenizer.sep_token

        input_ids = self.tokenizer.convert_tokens_to_ids(subwords)

        # Fallback if markers were truncated
        if e1_idx is None or e1_idx >= len(input_ids):
            e1_idx = 0
        if e2_idx is None or e2_idx >= len(input_ids):
            e2_idx = 0

        return input_ids, e1_idx, e2_idx

    def encode_batch(self, samples: Sequence[RelationSample]) -> dict[str, torch.Tensor]:
        """Tokenize and pad a batch of RelationSample objects into PyTorch tensors."""
        batch_input_ids: list[list[int]] = []
        batch_e1: list[int] = []
        batch_e2: list[int] = []
        batch_labels: list[int] = []

        for s in samples:
            input_ids, e1_idx, e2_idx = self._tokenize_sample(s)
            batch_input_ids.append(input_ids)
            batch_e1.append(e1_idx)
            batch_e2.append(e2_idx)
            if self.relation_to_class_idx is not None:
                batch_labels.append(self.relation_to_class_idx[s.relation])
            else:
                batch_labels.append(s.relation_id)

        max_len = max(len(ids) for ids in batch_input_ids)
        padded_ids: list[list[int]] = []
        attn_masks: list[list[int]] = []

        pad_id = self.tokenizer.pad_token_id or 0
        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            attn_masks.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long, device=self.device),
            "e1_indices": torch.tensor(batch_e1, dtype=torch.long, device=self.device),
            "e2_indices": torch.tensor(batch_e2, dtype=torch.long, device=self.device),
            "labels": torch.tensor(batch_labels, dtype=torch.long, device=self.device),
        }

    @torch.no_grad()
    def predict(self, samples: Sequence[RelationSample]) -> Sequence[int]:
        """Implement RelationPredictor protocol for Evaluator compatibility."""
        if not samples:
            return []

        self.encoder.eval()
        self.classifier.eval()

        predictions: list[int] = []
        batch_size = 64

        for i in range(0, len(samples), batch_size):
            chunk = samples[i : i + batch_size]
            batch = self.encode_batch(chunk)
            logits = self.forward(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                e1_indices=batch["e1_indices"],
                e2_indices=batch["e2_indices"],
            )
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            if self.class_idx_to_rel_id is not None:
                preds = [self.class_idx_to_rel_id[p] for p in preds]
            predictions.extend(preds)

        return predictions

    def get_state_dict(self) -> dict[str, Any]:
        """Return combined state dict for checkpointing."""
        return {
            "encoder": self.encoder.state_dict(),
            "classifier": self.classifier.state_dict(),
            "num_classes": self.num_classes,
            "seen_classes_count": self.seen_classes_count,
            "backbone_name": self.backbone_name,
            "relation_to_class_idx": self.relation_to_class_idx,
            "class_idx_to_rel_id": self.class_idx_to_rel_id,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.encoder.load_state_dict(state["encoder"])
        self.classifier.load_state_dict(state["classifier"])
        self.num_classes = state["num_classes"]
        self.seen_classes_count = state["seen_classes_count"]
        self.relation_to_class_idx = state.get("relation_to_class_idx")
        self.class_idx_to_rel_id = state.get("class_idx_to_rel_id")
