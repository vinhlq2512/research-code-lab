"""Task-Aware Router Modules for Continual Relation Extraction (P2a & P2b).

Implements:
- P2a (Centroid Cosine Router): Training-free routing in frozen BERT entity-marker space
  combining task-level centroids and relation-level sub-centroids.
- P2b (Lightweight Learned Router): 2-layer MLP + Cosine Task Key Router trained in
  stationary frozen query space with Gaussian feature statistics (strict M=0, no raw text).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskRouter(nn.Module):
    """Routes input samples to task-specific soft prompts without Oracle Task ID."""

    def __init__(
        self,
        input_dim: int = 1536,
        num_tasks: int = 8,
        relations_per_task: int = 10,
        router_type: str = "centroid",  # "centroid" (P2a) or "learned_mlp" (P2b)
        subcentroid_weight: float = 0.7,
        hidden_dim: int = 512,
        temperature: float = 16.0,
        residual_anchor_weight: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_tasks = int(num_tasks)
        self.relations_per_task = int(relations_per_task)
        self.router_type = router_type
        self.subcentroid_weight = float(subcentroid_weight)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.residual_anchor_weight = float(residual_anchor_weight)

        if self.router_type not in ("centroid", "learned_mlp"):
            raise ValueError(f"Unsupported router_type: {self.router_type}. Choose 'centroid' or 'learned_mlp'.")

        # Stationary buffers in 100% frozen BERT query space (M=0 raw text)
        self.register_buffer("task_centroids", torch.zeros(self.num_tasks, self.input_dim))
        self.register_buffer(
            "relation_subcentroids",
            torch.zeros(self.num_tasks, self.relations_per_task, self.input_dim),
        )
        self.register_buffer(
            "relation_variances",
            torch.full((self.num_tasks, self.relations_per_task, self.input_dim), 1e-4),
        )
        self.register_buffer("task_initialized", torch.zeros(self.num_tasks, dtype=torch.bool))

        # Trainable components for P2b (learned_mlp)
        if self.router_type == "learned_mlp":
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(self.hidden_dim),
            )
            self.task_keys = nn.Parameter(torch.empty(self.num_tasks, self.hidden_dim))
            nn.init.normal_(self.task_keys, mean=0.0, std=0.02)

    @torch.no_grad()
    def register_task_queries(
        self,
        task_id: int,
        query_features: torch.Tensor,
        class_indices: Sequence[int],
        sample_labels: torch.Tensor,
    ) -> None:
        """Compute and store task centroid and per-relation query statistics for task_id."""
        norm_q = F.normalize(query_features, p=2, dim=-1)

        # Task-level centroid
        task_mean = norm_q.mean(dim=0)
        self.task_centroids[task_id] = F.normalize(task_mean, p=2, dim=-1)

        # Relation-level sub-centroids and diagonal variances in frozen query space
        for r_offset, c_idx in enumerate(class_indices):
            if r_offset >= self.relations_per_task:
                break
            mask = sample_labels == int(c_idx)
            if mask.any():
                rel_q = norm_q[mask]
                rel_mean = F.normalize(rel_q.mean(dim=0), p=2, dim=-1)
                self.relation_subcentroids[task_id, r_offset] = rel_mean
                if rel_q.size(0) > 1:
                    rel_var = rel_q.var(dim=0, unbiased=True).clamp(min=1e-5, max=0.1)
                else:
                    rel_var = torch.full_like(rel_mean, 1e-3)
                self.relation_variances[task_id, r_offset] = rel_var
            else:
                self.relation_subcentroids[task_id, r_offset] = self.task_centroids[task_id]

        self.task_initialized[task_id] = True

    def compute_centroid_scores(
        self,
        query_features: torch.Tensor,
        seen_tasks: int,
    ) -> torch.Tensor:
        """Compute P2a non-parametric similarity scores [B, seen_tasks]."""
        norm_q = F.normalize(query_features, p=2, dim=-1)  # [B, D]

        # 1. Task-level centroid cosine similarity: [B, seen_tasks]
        active_task_centroids = F.normalize(self.task_centroids[:seen_tasks], p=2, dim=-1)
        task_sim = torch.mm(norm_q, active_task_centroids.t())

        # 2. Max relation sub-centroid cosine similarity per task: [B, seen_tasks]
        active_subcentroids = F.normalize(
            self.relation_subcentroids[:seen_tasks], p=2, dim=-1
        )  # [T, R, D]
        # einsum: [B, D] x [T, R, D] -> [B, T, R]
        rel_sims = torch.einsum("bd,trd->btr", norm_q, active_subcentroids)
        max_rel_sim, _ = rel_sims.max(dim=-1)  # [B, T]

        alpha = self.subcentroid_weight
        combined_scores = (1.0 - alpha) * task_sim + alpha * max_rel_sim
        return combined_scores

    def compute_learned_scores(
        self,
        query_features: torch.Tensor,
        seen_tasks: int,
    ) -> torch.Tensor:
        """Compute P2b learned MLP cosine scores [B, seen_tasks]."""
        norm_q = F.normalize(query_features, p=2, dim=-1)
        h = self.mlp(norm_q)  # [B, H]
        norm_h = F.normalize(h, p=2, dim=-1)
        norm_keys = F.normalize(self.task_keys[:seen_tasks], p=2, dim=-1)  # [T, H]
        logits = self.temperature * torch.mm(norm_h, norm_keys.t())
        return logits

    def fit_learned_router(
        self,
        current_task_id: int,
        current_queries: torch.Tensor,
        epochs: int = 30,
        lr: float = 2.0e-3,
        weight_decay: float = 0.01,
        synthetic_samples_per_rel: int = 12,
        batch_size: int = 32,
    ) -> float:
        """Train P2b learned router on current queries + stationary Gaussian feature replay (M=0)."""
        if self.router_type != "learned_mlp":
            return 0.0

        device = current_queries.device
        seen_tasks = current_task_id + 1

        # Build feature-level training dataset across all seen tasks 0..current_task_id
        features_list: list[torch.Tensor] = [F.normalize(current_queries.detach(), p=2, dim=-1)]
        labels_list: list[torch.Tensor] = [
            torch.full((current_queries.size(0),), current_task_id, dtype=torch.long, device=device)
        ]

        with torch.no_grad():
            for t_id in range(seen_tasks):
                if not bool(self.task_initialized[t_id].item()):
                    continue
                for r_idx in range(self.relations_per_task):
                    mean_vec = self.relation_subcentroids[t_id, r_idx]  # [D]
                    std_vec = torch.sqrt(self.relation_variances[t_id, r_idx])  # [D]
                    noise = torch.randn(synthetic_samples_per_rel, self.input_dim, device=device)
                    synth_q = F.normalize(mean_vec.unsqueeze(0) + noise * std_vec.unsqueeze(0) * 0.5, p=2, dim=-1)
                    features_list.append(synth_q)
                    labels_list.append(
                        torch.full((synthetic_samples_per_rel,), t_id, dtype=torch.long, device=device)
                    )

        all_x = torch.cat(features_list, dim=0)
        all_y = torch.cat(labels_list, dim=0)
        n_total = all_x.size(0)

        self.train()
        optimizer = torch.optim.AdamW(
            list(self.mlp.parameters()) + [self.task_keys],
            lr=lr,
            weight_decay=weight_decay,
        )

        final_loss = 0.0
        for _ in range(epochs):
            perm = torch.randperm(n_total, device=device)
            epoch_loss = 0.0
            steps = 0
            for i in range(0, n_total, batch_size):
                idx = perm[i : i + batch_size]
                bx = all_x[idx]
                by = all_y[idx]

                optimizer.zero_grad()
                logits = self.compute_learned_scores(bx, seen_tasks=seen_tasks)
                loss = F.cross_entropy(logits, by)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                steps += 1
            final_loss = epoch_loss / max(1, steps)

        self.eval()
        return final_loss

    @torch.no_grad()
    def predict_task(
        self,
        query_features: torch.Tensor,
        seen_tasks: int,
    ) -> dict[str, torch.Tensor]:
        """Predict task IDs [B] in 0..seen_tasks-1 without Oracle Task ID."""
        self.eval()
        centroid_scores = self.compute_centroid_scores(query_features, seen_tasks=seen_tasks)

        if self.router_type == "centroid":
            scores = centroid_scores
        else:
            learned_logits = self.compute_learned_scores(query_features, seen_tasks=seen_tasks)
            learned_probs = F.softmax(learned_logits, dim=-1)
            centroid_probs = F.softmax(self.temperature * centroid_scores, dim=-1)
            w = self.residual_anchor_weight
            scores = (1.0 - w) * learned_probs + w * centroid_probs

        pred_task_ids = torch.argmax(scores, dim=-1)
        return {
            "pred_task_ids": pred_task_ids,
            "task_scores": scores,
            "centroid_scores": centroid_scores,
        }

    @torch.no_grad()
    def compose_soft_prompt(
        self,
        query_features: torch.Tensor,
        prompt_pool: nn.ParameterList | list[torch.Tensor] | torch.Tensor,
        seen_tasks: int,
        top_k: int = 2,
        temperature: float = 0.2,
    ) -> dict[str, torch.Tensor]:
        """Compute sample-specific softly weighted prompt composition for a batch of queries.

        Args:
            query_features: [B, D] unprompted query embeddings from frozen BERT
            prompt_pool: ParameterList of length num_tasks, each [1, L, H] or tensor [num_tasks, 1, L, H]
            seen_tasks: number of observed tasks (1..num_tasks)
            top_k: number of candidate tasks to blend (K >= 1)
            temperature: temperature parameter tau for sharpening/smoothing softmax weights

        Returns:
            dict containing:
                "composed_prompts": [B, L, H] softly weighted prompts
                "topk_indices": [B, K] task indices chosen for each sample
                "topk_weights": [B, K] softmax weights assigned to top-K tasks
                "pred_task_ids": [B] hard top-1 task predictions
                "task_scores": [B, seen_tasks] raw router scores
        """
        self.eval()
        scores = self.compute_centroid_scores(query_features, seen_tasks=seen_tasks)
        if self.router_type != "centroid":
            learned_logits = self.compute_learned_scores(query_features, seen_tasks=seen_tasks)
            learned_probs = F.softmax(learned_logits, dim=-1)
            centroid_probs = F.softmax(self.temperature * scores, dim=-1)
            w = self.residual_anchor_weight
            scores = (1.0 - w) * learned_probs + w * centroid_probs

        device = query_features.device
        k = max(1, min(int(top_k), int(seen_tasks)))

        topk_scores, topk_indices = torch.topk(scores, k=k, dim=-1)  # [B, K]

        # Temperature-scaled softmax over top-K candidate scores
        temp = max(float(temperature), 1e-4)
        topk_weights = F.softmax(topk_scores / temp, dim=-1)  # [B, K]

        # Stack seen task prompts: [seen_tasks, L, H]
        if isinstance(prompt_pool, (nn.ParameterList, list)):
            p_list = [prompt_pool[t].squeeze(0).to(device) for t in range(seen_tasks)]
            p_stack = torch.stack(p_list, dim=0)  # [seen_tasks, L, H]
        elif isinstance(prompt_pool, torch.Tensor):
            p_stack = prompt_pool[:seen_tasks].squeeze(1).to(device)
        else:
            raise TypeError(f"Unsupported prompt_pool type: {type(prompt_pool)}")

        # Gather prompts for top-K tasks: [B, K, L, H]
        topk_prompts = p_stack[topk_indices]

        # Weighted composition: [B, L, H]
        weights_expanded = topk_weights.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        composed_prompts = torch.sum(weights_expanded * topk_prompts, dim=1)  # [B, L, H]

        pred_task_ids = topk_indices[:, 0]  # [B] Top-1 prediction

        return {
            "composed_prompts": composed_prompts,
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
            "pred_task_ids": pred_task_ids,
            "task_scores": scores,
        }
