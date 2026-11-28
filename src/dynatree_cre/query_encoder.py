"""Stable Frozen Query Encoder and Prompted Relation Encoder for DynaTree-CRE."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel

from src.data.protocol_v1_loader import SPECIAL_MARKERS


class SyntheticMarkerTokenizer:
    """Minimal deterministic tokenizer for fast offline unit/integration tests."""

    def __init__(self, vocab_size: int = 512) -> None:
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.unk_token_id = 3
        self.vocab: dict[str, int] = {
            "[PAD]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[UNK]": 3,
        }
        self._base_vocab_limit = max(64, vocab_size - 16)
        for i, marker in enumerate(SPECIAL_MARKERS):
            self.vocab[marker] = self._base_vocab_limit + i
        self._total_vocab = self._base_vocab_limit + len(SPECIAL_MARKERS)

    def __len__(self) -> int:
        return self._total_vocab

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def _token_to_id(self, tok: str) -> int:
        if tok in self.vocab:
            return self.vocab[tok]
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        offset = int.from_bytes(digest[:4], "little") % (self._base_vocab_limit - 8)
        return 8 + offset

    def __call__(
        self,
        texts: Sequence[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 128,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        seqs: list[list[int]] = []
        for text in texts:
            ids = [self.cls_token_id] + [self._token_to_id(tok) for tok in text.split()] + [self.sep_token_id]
            if truncation and len(ids) > max_length:
                ids = ids[: max_length - 1] + [self.sep_token_id]
            seqs.append(ids)

        max_len = max(len(s) for s in seqs) if padding else len(seqs[0])
        input_ids = torch.full((len(seqs), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, : len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class StableQueryEncoder(nn.Module):
    """Frozen PLM serving both stationary query extraction and prompted encoding."""

    def __init__(
        self,
        backbone_name: str = "bert-base-uncased",
        representation_strategy: str = "entity_pair",
        device: str | torch.device = "cpu",
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.representation_strategy = representation_strategy
        self.target_device = self._resolve_device(device)

        if backbone_name == "synthetic_tiny":
            self.tokenizer: Any = SyntheticMarkerTokenizer(vocab_size=512)
            cfg = BertConfig(
                vocab_size=len(self.tokenizer),
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
                max_position_embeddings=256,
            )
            self.encoder: nn.Module = BertModel(cfg)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_name, local_files_only=local_files_only)
            self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_MARKERS})
            self.encoder = AutoModel.from_pretrained(backbone_name, local_files_only=local_files_only)
            # Resize and initialize marker embeddings deterministically BEFORE freezing
            old_vocab = self.encoder.get_input_embeddings().weight.size(0)
            self.encoder.resize_token_embeddings(len(self.tokenizer))
            new_vocab = self.encoder.get_input_embeddings().weight.size(0)
            if new_vocab > old_vocab:
                with torch.no_grad():
                    emb_w = self.encoder.get_input_embeddings().weight
                    mean_vec = emb_w[:old_vocab].mean(dim=0, keepdim=True)
                    emb_w[old_vocab:].copy_(mean_vec.expand(new_vocab - old_vocab, -1))

        # Freeze 100% of backbone parameters permanently
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.hidden_size = int(self.encoder.config.hidden_size)
        if representation_strategy == "entity_pair":
            self.rep_dim = 2 * self.hidden_size
        elif representation_strategy in ("cls", "mean_pool"):
            self.rep_dim = self.hidden_size
        else:
            raise ValueError(f"Unsupported representation_strategy: {representation_strategy}")

        self.stationary_forward_samples: int = 0
        self.prompted_forward_samples: int = 0
        self.to(self.target_device)

    def reset_forward_counters(self) -> None:
        self.stationary_forward_samples = 0
        self.prompted_forward_samples = 0

    @property
    def total_forward_samples(self) -> int:
        return int(self.stationary_forward_samples + self.prompted_forward_samples)

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def train(self, mode: bool = True) -> StableQueryEncoder:
        """Keep the frozen backbone strictly in eval() mode to avoid stochastic dropout."""
        super().train(mode)
        self.encoder.eval()
        return self

    def _pool_sequence(
        self,
        seq_out: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_indices: torch.Tensor,
        e2_indices: torch.Tensor,
        prompt_offset: int = 0,
    ) -> torch.Tensor:
        batch_size = seq_out.size(0)
        b_idx = torch.arange(batch_size, device=seq_out.device)

        if self.representation_strategy == "entity_pair":
            h_e1 = seq_out[b_idx, prompt_offset + e1_indices]
            h_e2 = seq_out[b_idx, prompt_offset + e2_indices]
            raw = torch.cat([h_e1, h_e2], dim=-1)
        elif self.representation_strategy == "cls":
            raw = seq_out[:, prompt_offset, :]
        else:
            # Mean pool over non-prompt token positions masked by attention_mask
            token_out = seq_out[:, prompt_offset:, :]
            mask = attention_mask.unsqueeze(-1).float()
            raw = (token_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        if torch.isnan(raw).any() or torch.isinf(raw).any():
            raise FloatingPointError("NaN or Inf encountered in encoder representation.")
        return raw

    @torch.no_grad()
    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_indices: torch.Tensor,
        e2_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L2-normalized stationary query q_raw(x) without prompts or dropout."""
        self.encoder.eval()
        self.stationary_forward_samples += int(input_ids.size(0))
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        raw = self._pool_sequence(
            outputs.last_hidden_state,
            attention_mask=attention_mask,
            e1_indices=e1_indices,
            e2_indices=e2_indices,
            prompt_offset=0,
        )
        return F.normalize(raw, p=2, dim=-1, eps=1e-12)

    def forward_prompted(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_indices: torch.Tensor,
        e2_indices: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Run frozen backbone with prepended soft prompts while keeping gradients to prompt_embeds."""
        self.encoder.eval()
        self.prompted_forward_samples += int(input_ids.size(0))
        batch_size, prompt_len, hidden_dim = prompt_embeds.shape
        if hidden_dim != self.hidden_size:
            raise ValueError(f"Prompt hidden dim {hidden_dim} != encoder hidden_size {self.hidden_size}")

        word_embeds = self.encoder.embeddings.word_embeddings(input_ids)
        inputs_embeds = torch.cat([prompt_embeds, word_embeds], dim=1)
        prompt_mask = torch.ones(
            batch_size, prompt_len, device=input_ids.device, dtype=attention_mask.dtype
        )
        full_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=full_mask)
        raw = self._pool_sequence(
            outputs.last_hidden_state,
            attention_mask=attention_mask,
            e1_indices=e1_indices,
            e2_indices=e2_indices,
            prompt_offset=prompt_len,
        )
        return F.normalize(raw, p=2, dim=-1, eps=1e-12)
