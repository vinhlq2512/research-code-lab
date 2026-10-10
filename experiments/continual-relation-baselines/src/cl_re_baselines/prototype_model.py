from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from schemas.relation_sample import RelationSample
from .data_loader import SPECIAL_MARKERS, prepare_sample_tokens

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BERTRelationPrototypeClassifier(nn.Module):
    """Frozen BERT Relation Extractor with Non-Parametric Prototype (Centroid) Classifier.

    Architecture & Continual Mechanics:
        1. Tokenizer with entity markers ([E1], [/E1], [E2], [/E2]).
        2. Pretrained BERT encoder backbone (100% frozen, requires_grad=False).
        3. Feature Representation: Concatenation of [E1] and [E2] marker hidden states:
               h_rep = [h_{e1}; h_{e2}] in R^{2 * hidden_size} = R^{1536}
        4. Class Prototypes: For each class k, a static centroid c_k is computed once
           as the arithmetic mean of its training support embeddings:
               c_k = (1 / |S_k|) * sum_{x in S_k} h_rep(x)
        5. Metric Inference: Test representations are matched against observed prototypes
           using Cosine Similarity with Seen-Class Masking:
               logits_k = cos(h_rep, c_k)  for k < seen_classes
               logits_k = -1e9             for k >= seen_classes
    """

    def __init__(
        self,
        backbone_name: str = "bert-base-uncased",
        num_classes: int = 80,
        metric: str = "cosine",
        max_seq_length: int = 128,
        device: str = "auto",
        relation_to_class_idx: dict[str, int] | None = None,
        class_idx_to_rel_id: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch and HuggingFace Transformers are required to instantiate BERTRelationPrototypeClassifier."
            )

        self.backbone_name = backbone_name
        self.num_classes = int(num_classes)
        self.metric = str(metric).lower()
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

        logger.info(f"Initializing BERTRelationPrototypeClassifier on device: {self.device}")

        # Initialize Tokenizer and add entity markers
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)
        special_tokens_dict = {"additional_special_tokens": SPECIAL_MARKERS}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        # Initialize Encoder
        self.encoder = AutoModel.from_pretrained(self.backbone_name)
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        # Freeze 100% of the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.hidden_size = self.encoder.config.hidden_size
        self.feature_dim = self.hidden_size * 2

        # Prototype Bank: [num_classes, feature_dim]
        self.register_buffer("prototypes", torch.zeros(self.num_classes, self.feature_dim))
        self.register_buffer("prototype_initialized", torch.zeros(self.num_classes, dtype=torch.bool))

        # Move model to target device
        self.to(self.device)

        # Record token IDs for entity boundary markers
        self.e1_token_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        self.e2_token_id = self.tokenizer.convert_tokens_to_ids("[E2]")

        self.seen_classes_count = self.num_classes

    def set_seen_classes(self, num_seen_classes: int) -> None:
        """Set upper bound on active classes for seen-class masking."""
        self.seen_classes_count = min(int(num_seen_classes), self.num_classes)

    def set_prototype(self, class_idx: int, prototype_vector: torch.Tensor) -> None:
        """Directly store precomputed prototype vector for class_idx."""
        if not (0 <= class_idx < self.num_classes):
            raise IndexError(f"class_idx {class_idx} out of range [0, {self.num_classes})")
        self.prototypes[class_idx] = prototype_vector.detach().to(self.device, dtype=self.prototypes.dtype)
        self.prototype_initialized[class_idx] = True

    def compute_and_set_prototype(self, class_idx: int, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute centroid from embeddings [N, D] and store in prototype bank."""
        if embeddings.size(0) == 0:
            raise ValueError(f"Cannot compute prototype for class {class_idx}: empty embeddings tensor.")
        centroid = embeddings.mean(dim=0)
        self.set_prototype(class_idx, centroid)
        return centroid

    def classify_features(
        self,
        features: torch.Tensor,
        seen_classes: int | None = None,
    ) -> torch.Tensor:
        """Classify feature vectors [B, D] via nearest prototype.

        Returns similarity logits [B, num_classes] with unobserved classes masked to -1e9.
        """
        active_seen = self.seen_classes_count if seen_classes is None else seen_classes
        feat = features.to(self.device)

        if self.metric == "cosine":
            feat_norm = F.normalize(feat, p=2, dim=-1)
            # Avoid division by zero for uninitialized prototypes by adding eps
            proto_norm = F.normalize(self.prototypes, p=2, dim=-1, eps=1e-8)
            logits = torch.matmul(feat_norm, proto_norm.t())
        elif self.metric == "euclidean":
            # Negative Euclidean distance so higher is closer
            dist = torch.cdist(feat, self.prototypes)
            logits = -dist
        else:
            raise ValueError(f"Unsupported metric: {self.metric}. Choose 'cosine' or 'euclidean'.")

        if active_seen < self.num_classes:
            logits[:, active_seen:] = -1e9

        return logits

    def extract_features_from_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract [h_e1; h_e2] entity representation from encoded token batch."""
        self.encoder.eval()
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            seq_out = outputs[0]  # [B, L, H]
            batch_indices = torch.arange(batch["input_ids"].size(0), device=self.device)
            h_e1 = seq_out[batch_indices, batch["e1_indices"]]
            h_e2 = seq_out[batch_indices, batch["e2_indices"]]
            h_entity = torch.cat([h_e1, h_e2], dim=-1)
        return h_entity

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
        """Convert RelationSample sequence into padded PyTorch tensors."""
        batch_input_ids: list[list[int]] = []
        batch_e1: list[int] = []
        batch_e2: list[int] = []
        batch_labels: list[int] = []

        for s in samples:
            input_ids, e1_idx, e2_idx = self._tokenize_sample(s)
            batch_input_ids.append(input_ids)
            batch_e1.append(e1_idx)
            batch_e2.append(e2_idx)
            if self.relation_to_class_idx is not None and s.relation in self.relation_to_class_idx:
                batch_labels.append(self.relation_to_class_idx[s.relation])
            else:
                batch_labels.append(getattr(s, "relation_id", -1))

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_indices: torch.Tensor,
        e2_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass from tokens to nearest prototype similarity logits."""
        self.encoder.eval()
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            seq_out = outputs[0]
            batch_indices = torch.arange(input_ids.size(0), device=self.device)
            h_e1 = seq_out[batch_indices, e1_indices]
            h_e2 = seq_out[batch_indices, e2_indices]
            h_entity = torch.cat([h_e1, h_e2], dim=-1)

        return self.classify_features(h_entity)

    @torch.no_grad()
    def predict(self, samples: Sequence[RelationSample]) -> Sequence[int]:
        """Implement RelationPredictor protocol for benchmark Evaluator compatibility."""
        if not samples:
            return []

        self.encoder.eval()
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
        """Return complete state dict for full checkpoint packaging."""
        return {
            "encoder": self.encoder.state_dict(),
            "prototypes": self.prototypes.cpu(),
            "prototype_initialized": self.prototype_initialized.cpu(),
            "num_classes": self.num_classes,
            "metric": self.metric,
            "seen_classes_count": self.seen_classes_count,
            "backbone_name": self.backbone_name,
            "relation_to_class_idx": self.relation_to_class_idx,
            "class_idx_to_rel_id": self.class_idx_to_rel_id,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.encoder.load_state_dict(state["encoder"])
        self.prototypes.copy_(state["prototypes"].to(self.device))
        self.prototype_initialized.copy_(state["prototype_initialized"].to(self.device))
        self.num_classes = state["num_classes"]
        self.metric = state.get("metric", "cosine")
        self.seen_classes_count = state["seen_classes_count"]
        self.relation_to_class_idx = state.get("relation_to_class_idx")
        self.class_idx_to_rel_id = state.get("class_idx_to_rel_id")

    def save_prototypes_only(self, path: Path | str) -> None:
        """Save lightweight prototype dictionary."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prototypes": self.prototypes.cpu(),
                "prototype_initialized": self.prototype_initialized.cpu(),
                "seen_classes_count": self.seen_classes_count,
                "metric": self.metric,
                "relation_to_class_idx": self.relation_to_class_idx,
                "class_idx_to_rel_id": self.class_idx_to_rel_id,
            },
            p,
        )
