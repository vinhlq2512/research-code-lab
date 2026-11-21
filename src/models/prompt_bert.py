"""Soft-Prompted Frozen BERT Model for Continual Relation Extraction.

Supports:
- P0: Single Shared Soft Prompt across all tasks.
- P1: Oracle-Task Soft Prompt Pool with Oracle Task ID routing.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.data.protocol_v1_loader import SPECIAL_MARKERS

logger = logging.getLogger(__name__)


class PromptedBERTRelationClassifier(nn.Module):
    """Frozen BERT with soft-prompt prepending and relation classification heads."""

    def __init__(
        self,
        backbone_name: str = "bert-base-uncased",
        prompt_mode: str = "shared",  # "shared" (P0) or "oracle_pool" (P1)
        prompt_length: int = 8,
        num_tasks: int = 8,
        num_classes: int = 80,
        classifier_dropout: float = 0.1,
        device: str | torch.device = "auto",
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.prompt_mode = prompt_mode
        self.prompt_length = int(prompt_length)
        self.num_tasks = int(num_tasks)
        self.num_classes = int(num_classes)
        self.classifier_dropout = float(classifier_dropout)

        # Device setup
        if device == "auto":
            if torch.cuda.is_available():
                self.target_device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.target_device = torch.device("mps")
            else:
                self.target_device = torch.device("cpu")
        else:
            self.target_device = torch.device(device)

        # Tokenizer with special entity markers
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)
        self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_MARKERS})

        # Pretrained BERT Backbone
        self.encoder = AutoModel.from_pretrained(self.backbone_name)
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        # Freeze 100% of the BERT Backbone
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        hidden_size = self.encoder.config.hidden_size
        self.rep_dim = hidden_size * 2  # 1536 for concatenated [h_e1; h_e2]

        # Soft Prompts
        if self.prompt_mode == "shared":
            # P0: Single shared prompt
            self.prompt = nn.Parameter(torch.empty(1, self.prompt_length, hidden_size))
            nn.init.normal_(self.prompt, mean=0.0, std=0.02)
        elif self.prompt_mode in ("oracle_pool", "routed_pool"):
            # P1 / P2: Pool of task prompts (one per task)
            self.prompt_pool = nn.ParameterList([
                nn.Parameter(torch.empty(1, self.prompt_length, hidden_size))
                for _ in range(self.num_tasks)
            ])
            for p in self.prompt_pool:
                nn.init.normal_(p, mean=0.0, std=0.02)
        else:
            raise ValueError(
                f"Unknown prompt_mode: {self.prompt_mode}. Choose 'shared', 'oracle_pool', or 'routed_pool'."
            )

        # Classification Head: Linear projection (1536 -> num_classes)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.rep_dim, self.num_classes)

        # Prototype memory storage for Nearest Prototype inference
        self.register_buffer("prototypes", torch.zeros(self.num_classes, self.rep_dim))
        self.register_buffer("prototype_counts", torch.zeros(self.num_classes, dtype=torch.long))

        self.seen_classes = self.num_classes

        self.to(self.target_device)

    def set_seen_classes(self, num_seen_classes: int) -> None:
        """Set number of observed classes for incremental masking."""
        self.seen_classes = min(max(num_seen_classes, 1), self.num_classes)

    def get_trainable_params(self, task_id: int | None = None) -> list[nn.Parameter]:
        """Return parameters that receive gradients during current task training."""
        params: list[nn.Parameter] = list(self.classifier.parameters())
        if self.prompt_mode == "shared":
            params.append(self.prompt)
        elif self.prompt_mode in ("oracle_pool", "routed_pool"):
            if task_id is None:
                raise ValueError(f"task_id must be provided when getting trainable params for {self.prompt_mode}")
            params.append(self.prompt_pool[task_id])
        return params

    def get_prompt_for_batch(
        self,
        batch_size: int,
        task_id: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Retrieve and expand prompt tensor for the current batch."""
        if self.prompt_mode == "shared":
            return self.prompt.expand(batch_size, -1, -1)

        if self.prompt_mode in ("oracle_pool", "routed_pool"):
            if task_id is None:
                raise ValueError(f"task_id is required in {self.prompt_mode} prompt mode.")

            if isinstance(task_id, int):
                return self.prompt_pool[task_id].expand(batch_size, -1, -1)
            elif isinstance(task_id, torch.Tensor):
                if task_id.dim() == 0:
                    return self.prompt_pool[int(task_id.item())].expand(batch_size, -1, -1)
                else:
                    # Mixed batch of task IDs (e.g., per-sample routed task predictions)
                    prompts = [self.prompt_pool[int(t.item())] for t in task_id]
                    return torch.cat(prompts, dim=0)
            else:
                raise TypeError(f"Unsupported task_id type: {type(task_id)}")

        raise RuntimeError(f"Invalid prompt_mode: {self.prompt_mode}")

    @torch.no_grad()
    def extract_unprompted_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_indices: torch.Tensor,
        e2_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Extract stationary [h_e1; h_e2] query representations from 100% frozen unprompted BERT."""
        batch_size = input_ids.size(0)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state  # [B, N, d]

        b_idx = torch.arange(batch_size, device=input_ids.device)
        q_e1 = seq_out[b_idx, e1_indices]
        q_e2 = seq_out[b_idx, e2_indices]
        q_r = torch.cat([q_e1, q_e2], dim=-1)  # [B, 1536]
        return q_r

    def extract_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_indices: torch.Tensor,
        e2_indices: torch.Tensor,
        task_id: int | torch.Tensor | None = None,
        custom_prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract concatenated entity representations [h_e1; h_e2] with soft prompt prepended."""
        batch_size = input_ids.size(0)
        if custom_prompt is not None:
            prompt_embeds = custom_prompt
        else:
            prompt_embeds = self.get_prompt_for_batch(batch_size, task_id=task_id)

        # Get word embeddings from frozen BERT
        word_embeds = self.encoder.embeddings.word_embeddings(input_ids)
        inputs_embeds = torch.cat([prompt_embeds, word_embeds], dim=1)

        # Extend attention mask
        prompt_mask = torch.ones(
            batch_size,
            self.prompt_length,
            device=input_ids.device,
            dtype=attention_mask.dtype,
        )
        full_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Forward pass through frozen BERT encoder
        outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=full_mask)
        seq_out = outputs.last_hidden_state  # [B, L + N, d]

        # Shift marker indices by prompt_length L
        b_idx = torch.arange(batch_size, device=input_ids.device)
        h_e1 = seq_out[b_idx, self.prompt_length + e1_indices]
        h_e2 = seq_out[b_idx, self.prompt_length + e2_indices]

        h_r = torch.cat([h_e1, h_e2], dim=-1)  # [B, 2*d] = [B, 1536]
        return h_r

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_indices: torch.Tensor,
        e2_indices: torch.Tensor,
        task_id: int | torch.Tensor | None = None,
        custom_prompt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute relation representations and class logits."""
        h_r = self.extract_features(
            input_ids,
            attention_mask,
            e1_indices,
            e2_indices,
            task_id=task_id,
            custom_prompt=custom_prompt,
        )
        h_drop = self.dropout(h_r)
        logits = self.classifier(h_drop)

        # Mask unseen classes
        if self.seen_classes < self.num_classes:
            logits = logits.clone()
            logits[:, self.seen_classes:] = -1e9

        return {
            "logits": logits,
            "representations": h_r,
        }

    def predict_prototype(
        self,
        h_r: torch.Tensor,
        seen_classes_only: bool = True,
    ) -> torch.Tensor:
        """Compute cosine similarity logits against stored class prototypes."""
        n_classes = self.seen_classes if seen_classes_only else self.num_classes
        active_protos = self.prototypes[:n_classes]  # [C, D]

        # Cosine similarity: [B, C]
        norm_h = F.normalize(h_r, p=2, dim=-1)
        norm_p = F.normalize(active_protos, p=2, dim=-1)
        sim = torch.mm(norm_h, norm_p.t())

        if seen_classes_only and n_classes < self.num_classes:
            full_sim = torch.full((h_r.size(0), self.num_classes), -1e9, device=h_r.device)
            full_sim[:, :n_classes] = sim
            return full_sim

        return sim

    def predict_routed_prototype(
        self,
        h_r: torch.Tensor,
        pred_task_ids: torch.Tensor,
        relations_per_task: int = 10,
    ) -> torch.Tensor:
        """Compute prototype similarity gated by each sample's routed task prediction."""
        sim = self.predict_prototype(h_r, seen_classes_only=True).clone()
        batch_size = h_r.size(0)
        gated_sim = torch.full_like(sim, -1e9)

        for i in range(batch_size):
            t_id = int(pred_task_ids[i].item())
            c_start = t_id * relations_per_task
            c_end = min(c_start + relations_per_task, self.seen_classes)
            gated_sim[i, c_start:c_end] = sim[i, c_start:c_end]

        return gated_sim

    def predict_topk_gated_prototype(
        self,
        h_r: torch.Tensor,
        topk_task_ids: torch.Tensor,
        relations_per_task: int = 10,
    ) -> torch.Tensor:
        """Compute prototype similarity gated to the union of relations of Top-K candidate tasks."""
        sim = self.predict_prototype(h_r, seen_classes_only=True).clone()
        batch_size = h_r.size(0)
        k = topk_task_ids.size(1)
        gated_sim = torch.full_like(sim, -1e9)

        for i in range(batch_size):
            for k_idx in range(k):
                t_id = int(topk_task_ids[i, k_idx].item())
                c_start = t_id * relations_per_task
                c_end = min(c_start + relations_per_task, self.seen_classes)
                gated_sim[i, c_start:c_end] = sim[i, c_start:c_end]

        return gated_sim

    def compute_orthogonality_loss(self, task_id: int) -> torch.Tensor:
        """Compute cosine orthogonality/diversity loss between P_t and previous prompts {P_0..P_{t-1}}."""
        if self.prompt_mode not in ("oracle_pool", "routed_pool") or task_id <= 0:
            return torch.tensor(0.0, device=self.target_device)

        current_p = self.prompt_pool[task_id]  # [1, L, H]
        current_mean = F.normalize(current_p.mean(dim=1), p=2, dim=-1)  # [1, H]

        prev_prompts = torch.stack(
            [self.prompt_pool[j].detach().mean(dim=1) for j in range(task_id)], dim=0
        )  # [task_id, 1, H]
        prev_means = F.normalize(prev_prompts.squeeze(1), p=2, dim=-1)  # [task_id, H]

        # Cosine similarities: [task_id]
        sims = torch.mm(current_mean, prev_means.t()).squeeze(0)
        orth_loss = torch.mean(torch.abs(sims))
        return orth_loss

    def update_prototypes(
        self,
        class_idx: int,
        support_representations: torch.Tensor,
    ) -> None:
        """Compute and update centroid representation for a given class."""
        mean_rep = support_representations.mean(dim=0)
        self.prototypes[class_idx] = mean_rep
        self.prototype_counts[class_idx] = support_representations.size(0)
