"""Continual Training & Evaluation Orchestrator (Phases A-G) for DynaTree-CRE."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from src.data.protocol_v1_loader import ProtocolV1Sample, ProtocolV1Task, collate_samples_for_bert
from src.dynatree_cre.benchmark import TaskStreamBundle
from src.dynatree_cre.checkpoint import (
    load_dynatree_checkpoint,
    save_dynatree_checkpoint,
    write_json,
)
from src.dynatree_cre.config import DynaTreeConfig
from src.dynatree_cre.metrics import (
    PerformanceMatrix,
    compute_accuracy,
    compute_confusion_diagnostics,
    compute_continual_summary_metrics,
    compute_macro_f1,
    compute_routing_diagnostics,
)
from src.dynatree_cre.model import DynaTreeCREModel


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


class DynaTreeContinualTrainer:
    """Executes Phases A-G per continual task with strict invariant enforcement and resume."""

    def __init__(
        self,
        model: DynaTreeCREModel,
        config: DynaTreeConfig,
        stream: TaskStreamBundle,
        output_dir: str | Path | None = None,
        run_id: str = "run_001",
        git_info: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.stream = stream
        self.output_dir = Path(output_dir or config.experiment.output_dir)
        self.run_id = run_id
        self.git_info = git_info or {}

        num_tasks = len(stream.tasks)
        self.cosine_matrix = PerformanceMatrix(
            num_tasks=num_tasks,
            metric="accuracy",
            dataset=config.experiment.dataset,
            seed=config.experiment.seed,
        )
        self.ncm_matrix = PerformanceMatrix(
            num_tasks=num_tasks,
            metric="ncm_accuracy",
            dataset=config.experiment.dataset,
            seed=config.experiment.seed,
        )
        self.stage_records: list[dict[str, Any]] = []
        self._seen_train_sample_ids: set[str] = set()

    def _collate_with_rel_ids(
        self, samples: list[ProtocolV1Sample]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        batch = collate_samples_for_bert(
            samples,
            tokenizer=self.model.query_encoder.tokenizer,
            device=self.model.device,
            max_seq_length=self.config.prototype.max_seq_length,
        )
        rel_ids = torch.tensor([s.relation_id for s in samples], dtype=torch.long, device=self.model.device)
        return batch, rel_ids

    def _phase_a_validate_task(self, task: ProtocolV1Task) -> None:
        if len(set(task.relations)) != len(task.relations):
            raise ValueError(f"Duplicate relations in task {task.task_id}.")
        for r_id in task.relations:
            if r_id in self.model.prototype_bank.prototypes:
                raise ValueError(f"Future/overlap violation: relation {r_id} already observed.")

    @torch.no_grad()
    def _phase_b_build_prototypes(self, task: ProtocolV1Task) -> dict[int, Any]:
        samples = list(task.train_samples)
        by_rel_q: dict[int, list[torch.Tensor]] = {r: [] for r in task.relations}
        rel_names = {r: name for r, name in zip(task.relations, task.relation_names)}

        bs = self.config.training.eval_batch_size
        for i in range(0, len(samples), bs):
            chunk = samples[i : i + bs]
            batch, rel_ids = self._collate_with_rel_ids(chunk)
            q = self.model.encode_stationary_queries(
                batch["input_ids"], batch["attention_mask"], batch["e1_indices"], batch["e2_indices"]
            )
            for row_idx, r_val in enumerate(rel_ids.tolist()):
                by_rel_q[int(r_val)].append(q[row_idx : row_idx + 1].cpu())

        rel_tensors = {r: torch.cat(chunks, dim=0) for r, chunks in by_rel_q.items()}
        return self.model.prototype_bank.register_task_prototypes(
            relation_queries=rel_tensors,
            task_id=task.task_id,
            relation_names=rel_names,
        )

    @torch.no_grad()
    def _compute_prompted_class_means(
        self, task: ProtocolV1Task, use_effective_leaf_prompt: bool = False
    ) -> dict[int, torch.Tensor]:
        """Compute prompted representations per relation strictly from current-task train samples."""
        by_rel: dict[int, list[torch.Tensor]] = {r: [] for r in task.relations}
        for s in task.train_samples:
            by_rel.setdefault(s.relation_id, [])

        bs = self.config.training.eval_batch_size
        for rel_id in sorted(task.relations):
            rel_samples = [s for s in task.train_samples if s.relation_id == rel_id]
            for i in range(0, len(rel_samples), bs):
                chunk = rel_samples[i : i + bs]
                batch, rel_tensor = self._collate_with_rel_ids(chunk)
                if use_effective_leaf_prompt:
                    eff_p = self.model.prompt_bank.get_effective_prompt(rel_id, self.model.tree)
                    prompt_embeds = eff_p.unsqueeze(0).expand(len(chunk), -1, -1)
                    z = self.model.query_encoder.forward_prompted(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["e1_indices"],
                        batch["e2_indices"],
                        prompt_embeds=prompt_embeds,
                    )
                else:
                    out = self.model.forward_batch(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["e1_indices"],
                        batch["e2_indices"],
                        gold_relation_ids=None,
                    )
                    z = out["representations"]
                by_rel[rel_id].append(z.detach())

        return {r: torch.cat(chunks, dim=0) for r, chunks in by_rel.items()}

    def _phase_d_prepare_model(self, task: ProtocolV1Task) -> None:
        if self.model.is_prototype_only:
            return
        word_emb = self.model.query_encoder.encoder.get_input_embeddings().weight
        self.model.prompt_bank.synchronize_with_tree(
            self.model.tree, task_id=task.task_id, word_embedding_weight=word_emb
        )
        self.model.prompt_bank.apply_freeze_policy(task_id=task.task_id)

        init_vecs: torch.Tensor | None = None
        if self.config.classifier.init_from_prompted_prototype:
            rel_feats = self._compute_prompted_class_means(task, use_effective_leaf_prompt=True)
            ordered = sorted(task.relations)
            init_vecs = torch.stack(
                [F.normalize(rel_feats[r].mean(dim=0), p=2, dim=-1, eps=1e-12) for r in ordered],
                dim=0,
            )
        self.model.classifier.expand_for_task(sorted(task.relations), init_vectors=init_vecs)

    def _phase_e_train_task(self, task: ProtocolV1Task) -> dict[str, Any]:
        if self.model.is_prototype_only:
            return {
                "final_loss": 0.0,
                "trainable_param_count": 0,
                "train_target_in_topk_rate": 1.0,
                "train_new_leaf_selected_rate": 1.0,
                "train_zero_prompt_grad_steps": 0,
                "total_train_steps": 0,
            }

        param_groups = self.model.trainable_parameter_groups()
        trainable_tensors = [p for g in param_groups for p in g["params"]]
        trainable_param_count = int(sum(p.numel() for p in trainable_tensors))

        optimizer = torch.optim.AdamW(param_groups)
        samples = list(task.train_samples)
        bs = self.config.training.batch_size
        epochs = max(1, int(self.config.training.epochs_per_task))
        steps_per_epoch = max(1, (len(samples) + bs - 1) // bs)
        total_steps = epochs * steps_per_epoch
        warmup_steps = int(total_steps * float(self.config.training.warmup_ratio))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        frozen_snapshot = self.model.snapshot_frozen_state()
        current_rel_set = set(task.relations)
        current_leaf_ids = [self.model.tree.relation_to_node[r] for r in task.relations]

        total_samples_seen = 0
        target_in_topk_hits = 0
        new_leaf_selected_hits = 0
        zero_prompt_grad_steps = 0
        total_exec_steps = 0
        last_loss = 0.0

        self.model.prompt_bank.train()
        self.model.classifier.train()
        self.model.query_encoder.eval()

        for epoch in range(epochs):
            perm = list(range(len(samples)))
            random.shuffle(perm)
            epoch_loss = 0.0
            epoch_steps = 0

            strategy_override: str | None = None
            if self.config.routing.train_mode == "dense_soft":
                strategy_override = "dense_soft"
            elif (
                self.config.routing.train_mode == "warmup_target_assist"
                and epoch < int(self.config.routing.warmup_epochs)
            ):
                strategy_override = "oracle_route"

            for start in range(0, len(samples), bs):
                batch_samples = [samples[idx] for idx in perm[start : start + bs]]
                batch, rel_ids = self._collate_with_rel_ids(batch_samples)
                gold_arg = rel_ids if strategy_override == "oracle_route" else None

                optimizer.zero_grad()
                out = self.model.forward_batch(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["e1_indices"],
                    batch["e2_indices"],
                    strategy_override=strategy_override,
                    gold_relation_ids=gold_arg,
                )
                targets = self.model.classifier.relation_ids_to_slots(rel_ids)
                loss = F.cross_entropy(out["logits"], targets)
                loss.backward()

                # Track routing recall and zero-grad prompt steps during training
                routing = out["routing"]
                if routing is not None:
                    topk_lists = routing.selected_relation_ids.detach().cpu().tolist()
                    for r_true, topk_r in zip(rel_ids.tolist(), topk_lists):
                        total_samples_seen += 1
                        if r_true in topk_r:
                            target_in_topk_hits += 1
                        if any(cand in current_rel_set for cand in topk_r):
                            new_leaf_selected_hits += 1

                    has_leaf_grad = any(
                        self.model.prompt_bank.leaf_prompts[lid].grad is not None
                        and float(self.model.prompt_bank.leaf_prompts[lid].grad.abs().sum().item()) > 0.0
                        for lid in current_leaf_ids
                    )
                    if not has_leaf_grad:
                        zero_prompt_grad_steps += 1

                nn.utils.clip_grad_norm_(trainable_tensors, float(self.config.training.gradient_clip_norm))
                optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())
                epoch_steps += 1
                total_exec_steps += 1

            last_loss = epoch_loss / max(1, epoch_steps)

        # Verify frozen parameters (encoder, old prompts, old classifier weights) remained bit-identical
        self.model.assert_frozen_unchanged(frozen_snapshot)

        # Populate prompted NCM class prototypes after task training finishes
        self.model.eval()
        ncm_feats = self._compute_prompted_class_means(task, use_effective_leaf_prompt=False)
        self.model.classifier.update_ncm_prototypes(ncm_feats)

        return {
            "final_loss": float(last_loss),
            "trainable_param_count": trainable_param_count,
            "train_target_in_topk_rate": (
                float(target_in_topk_hits / total_samples_seen) if total_samples_seen > 0 else 0.0
            ),
            "train_new_leaf_selected_rate": (
                float(new_leaf_selected_hits / total_samples_seen) if total_samples_seen > 0 else 0.0
            ),
            "train_zero_prompt_grad_steps": int(zero_prompt_grad_steps),
            "total_train_steps": int(total_exec_steps),
        }

    @torch.no_grad()
    def _evaluate_single_task(
        self,
        eval_task: ProtocolV1Task,
        strategy_override: str | None = None,
    ) -> dict[str, Any]:
        self.model.eval()
        samples = list(eval_task.test_samples)
        bs = self.config.training.eval_batch_size

        true_rels: list[int] = []
        pred_cos_rels: list[int] = []
        pred_ncm_rels: list[int] = []
        topk_selected_all: list[list[int]] = []
        entropies: list[float] = []

        t0 = time.perf_counter()
        for i in range(0, len(samples), bs):
            chunk = samples[i : i + bs]
            batch, rel_ids = self._collate_with_rel_ids(chunk)
            gold_arg = rel_ids if strategy_override == "oracle_route" else None
            out = self.model.forward_batch(
                batch["input_ids"],
                batch["attention_mask"],
                batch["e1_indices"],
                batch["e2_indices"],
                strategy_override=strategy_override,
                gold_relation_ids=gold_arg,
            )
            ordered = out["ordered_relation_ids"]
            cos_slots = torch.argmax(out["logits"], dim=-1).cpu().tolist()
            pred_cos_rels.extend(ordered[s] for s in cos_slots)

            if out["ncm_logits"] is not None:
                ncm_slots = torch.argmax(out["ncm_logits"], dim=-1).cpu().tolist()
                pred_ncm_rels.extend(ordered[s] for s in ncm_slots)
            else:
                pred_ncm_rels.extend(ordered[s] for s in cos_slots)

            true_rels.extend(rel_ids.cpu().tolist())
            if out["routing"] is not None:
                topk_selected_all.extend(out["routing"].selected_relation_ids.cpu().tolist())
                entropies.extend(out["routing"].routing_entropy.cpu().tolist())

        elapsed = time.perf_counter() - t0
        return {
            "true_rels": true_rels,
            "pred_cos_rels": pred_cos_rels,
            "pred_ncm_rels": pred_ncm_rels,
            "topk_selected": topk_selected_all,
            "entropies": entropies,
            "cosine_acc": compute_accuracy(true_rels, pred_cos_rels),
            "cosine_macro_f1": compute_macro_f1(true_rels, pred_cos_rels),
            "ncm_acc": compute_accuracy(true_rels, pred_ncm_rels),
            "ncm_macro_f1": compute_macro_f1(true_rels, pred_ncm_rels),
            "eval_time_sec": float(elapsed),
            "num_samples": len(true_rels),
        }

    def _phase_f_evaluate_seen(self, current_task_id: int) -> dict[str, Any]:
        per_task_results: dict[int, dict[str, Any]] = {}
        all_true: list[int] = []
        all_cos_pred: list[int] = []
        all_ncm_pred: list[int] = []
        all_topk: list[list[int]] = []
        all_entropies: list[float] = []
        total_eval_time = 0.0

        for j in range(current_task_id + 1):
            eval_res = self._evaluate_single_task(self.stream.tasks[j])
            per_task_results[j] = eval_res
            self.cosine_matrix.set_score(current_task_id, j, eval_res["cosine_acc"])
            self.ncm_matrix.set_score(current_task_id, j, eval_res["ncm_acc"])

            all_true.extend(eval_res["true_rels"])
            all_cos_pred.extend(eval_res["pred_cos_rels"])
            all_ncm_pred.extend(eval_res["pred_ncm_rels"])
            all_topk.extend(eval_res["topk_selected"])
            all_entropies.extend(eval_res["entropies"])
            total_eval_time += eval_res["eval_time_sec"]

        # Optional oracle routing diagnostic on seen tasks (strictly separated)
        oracle_acc: float | None = None
        if not self.model.is_prototype_only:
            oracle_true: list[int] = []
            oracle_pred: list[int] = []
            for j in range(current_task_id + 1):
                ores = self._evaluate_single_task(
                    self.stream.tasks[j], strategy_override="oracle_route"
                )
                oracle_true.extend(ores["true_rels"])
                oracle_pred.extend(ores["pred_cos_rels"])
            oracle_acc = compute_accuracy(oracle_true, oracle_pred)

        cos_whole = compute_accuracy(all_true, all_cos_pred)
        ncm_whole = compute_accuracy(all_true, all_ncm_pred)

        routing_diag = (
            compute_routing_diagnostics(
                true_relation_ids=all_true,
                selected_topk_relation_ids=all_topk,
                routing_entropies=all_entropies,
                tree=self.model.tree,
                relation_to_task=self.stream.relation_to_task,
                oracle_acc=oracle_acc,
                routed_acc=cos_whole,
            )
            if all_topk
            else {}
        )
        current_rels = set(self.stream.tasks[current_task_id].relations)
        confusion_cos = compute_confusion_diagnostics(
            true_relation_ids=all_true,
            pred_relation_ids=all_cos_pred,
            current_task_relation_ids=current_rels,
            tree=self.model.tree,
            relation_to_task=self.stream.relation_to_task,
        )
        confusion_ncm = compute_confusion_diagnostics(
            true_relation_ids=all_true,
            pred_relation_ids=all_ncm_pred,
            current_task_relation_ids=current_rels,
            tree=self.model.tree,
            relation_to_task=self.stream.relation_to_task,
        )

        return {
            "cosine_whole_accuracy": cos_whole,
            "cosine_whole_macro_f1": compute_macro_f1(all_true, all_cos_pred),
            "ncm_whole_accuracy": ncm_whole,
            "ncm_whole_macro_f1": compute_macro_f1(all_true, all_ncm_pred),
            "oracle_cosine_whole_accuracy": oracle_acc,
            "per_task": {
                str(j): {
                    "cosine_acc": r["cosine_acc"],
                    "ncm_acc": r["ncm_acc"],
                    "num_samples": r["num_samples"],
                }
                for j, r in per_task_results.items()
            },
            "routing_diagnostics": routing_diag,
            "confusion_cosine": confusion_cos,
            "confusion_ncm": confusion_ncm,
            "eval_time_sec": float(total_eval_time),
            "inference_latency_ms_per_sample": (
                float(1000.0 * total_eval_time / len(all_true)) if all_true else 0.0
            ),
        }

    def run(self, resume_from: str | Path | None = None) -> dict[str, Any]:
        """Run full continual task stream (with optional clean checkpoint resume)."""
        set_deterministic_seed(self.config.experiment.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_yaml(self.output_dir / "resolved_config.yaml")

        start_task = 0
        if resume_from is not None:
            payload = load_dynatree_checkpoint(
                ckpt_path=Path(resume_from),
                config=self.config,
                model=self.model,
                dataset_fingerprint=self.stream.dataset_fingerprint,
                task_order_fingerprint=self.stream.task_order_fingerprint,
            )
            start_task = int(payload["completed_task_id"]) + 1
            m_state = payload.get("matrices_state", {})
            if "cosine" in m_state:
                self.cosine_matrix = PerformanceMatrix.from_dict(m_state["cosine"])
            if "ncm" in m_state:
                self.ncm_matrix = PerformanceMatrix.from_dict(m_state["ncm"])
            self.stage_records = list(payload.get("stage_records", []))

        prev_cluster_protos: dict[str, np.ndarray] = {}
        if self.model.tree.initialized:
            prev_cluster_protos = {
                cid: self.model.tree.nodes[cid].prototype.copy()
                for cid in self.model.tree.cluster_ids()
                if self.model.tree.nodes[cid].prototype is not None
            }

        for task_id in range(start_task, len(self.stream.tasks)):
            task = self.stream.tasks[task_id]
            t_start = time.perf_counter()

            # Phase A: Validate current task
            self._phase_a_validate_task(task)

            # Phase B: Build new relation prototypes
            new_protos = self._phase_b_build_prototypes(task)

            # Phase C: Update Dynamic Semantic Prompt Tree
            topo_event = self.model.tree.insert_task_relations(
                new_protos, task_id=task.task_id, config=self.config.tree
            )
            tree_metrics = self.model.tree.compute_metrics(prev_cluster_protos or None)
            prev_cluster_protos = {
                cid: self.model.tree.nodes[cid].prototype.copy()
                for cid in self.model.tree.cluster_ids()
                if self.model.tree.nodes[cid].prototype is not None
            }

            # Phase D: Prepare prompts & classifier
            self._phase_d_prepare_model(task)

            # Phase E: Train current task
            train_stats = self._phase_e_train_task(task)
            train_time = time.perf_counter() - t_start

            # Phase F: Evaluate all seen tasks
            eval_stats = self._phase_f_evaluate_seen(task_id)

            # Phase G: Save tree snapshot, checkpoint, and stage record
            write_json(
                self.output_dir / "trees" / f"task_{task_id + 1:02d}.json",
                self.model.tree.to_dict(),
            )
            stage_rec = {
                "task_id": task_id,
                "relations": task.relations,
                "relation_names": task.relation_names,
                "topology_event": topo_event,
                "tree_metrics": tree_metrics,
                "train_stats": train_stats,
                "train_time_sec": float(train_time),
                "eval_stats": eval_stats,
                "memory_bytes": self.model.memory_breakdown_bytes(),
            }
            self.stage_records.append(stage_rec)

            ckpt_path = save_dynatree_checkpoint(
                output_dir=self.output_dir,
                completed_task_id=task_id,
                config=self.config,
                model=self.model,
                dataset_fingerprint=self.stream.dataset_fingerprint,
                task_order_fingerprint=self.stream.task_order_fingerprint,
                git_info=self.git_info,
                run_id=self.run_id,
                matrices_state={
                    "cosine": self.cosine_matrix.to_dict(),
                    "ncm": self.ncm_matrix.to_dict(),
                },
                stage_records=self.stage_records,
            )
            stage_rec["checkpoint_bytes"] = int(ckpt_path.stat().st_size)

        last_stage = len(self.stream.tasks) - 1
        cosine_summary = compute_continual_summary_metrics(self.cosine_matrix, current_task=last_stage)
        ncm_summary = compute_continual_summary_metrics(self.ncm_matrix, current_task=last_stage)

        self.cosine_matrix.export_csv(self.output_dir / "performance_matrix.csv")
        self.cosine_matrix.export_json(self.output_dir / "performance_matrix.json")
        self.ncm_matrix.export_csv(self.output_dir / "ncm_performance_matrix.csv")

        final_stage = self.stage_records[-1]
        metrics_payload = {
            "experiment_id": self.config.experiment.experiment_id,
            "variant": self.config.method.variant,
            "dataset": self.config.experiment.dataset,
            "task_distribution": self.config.experiment.task_distribution,
            "seed": self.config.experiment.seed,
            "num_tasks_executed": len(self.stream.tasks),
            "cosine_summary": cosine_summary,
            "ncm_summary": ncm_summary,
            "final_cosine_whole_accuracy": final_stage["eval_stats"]["cosine_whole_accuracy"],
            "final_ncm_whole_accuracy": final_stage["eval_stats"]["ncm_whole_accuracy"],
            "final_oracle_cosine_whole_accuracy": final_stage["eval_stats"][
                "oracle_cosine_whole_accuracy"
            ],
            "final_tree_metrics": final_stage["tree_metrics"],
            "stages": self.stage_records,
        }
        write_json(self.output_dir / "metrics.json", metrics_payload)
        write_json(
            self.output_dir / "routing_metrics.json",
            {
                "stages": [
                    {
                        "task_id": r["task_id"],
                        "train_routing": {
                            "target_in_topk_rate": r["train_stats"]["train_target_in_topk_rate"],
                            "new_leaf_selected_rate": r["train_stats"]["train_new_leaf_selected_rate"],
                            "zero_prompt_grad_steps": r["train_stats"]["train_zero_prompt_grad_steps"],
                        },
                        "eval_routing": r["eval_stats"]["routing_diagnostics"],
                    }
                    for r in self.stage_records
                ]
            },
        )
        total_persistent_params = int(
            sum(p.numel() for p in self.model.prompt_bank.parameters())
            + sum(p.numel() for p in self.model.classifier.parameters())
            + self.model.classifier.old_weights.numel()
            + self.model.classifier.ncm_prototypes.numel()
            + sum(
                p.prototype_vector.size + p.raw_prototype_vector.size
                for p in self.model.prototype_bank.prototypes.values()
            )
        )
        resource_payload = {
            "backbone_params_frozen": int(sum(p.numel() for p in self.model.query_encoder.parameters())),
            "total_persistent_adapter_and_state_params": total_persistent_params,
            "memory_bytes": final_stage["memory_bytes"],
            "final_checkpoint_bytes": final_stage["checkpoint_bytes"],
            "total_train_time_sec": float(sum(r["train_time_sec"] for r in self.stage_records)),
            "total_eval_time_sec": float(sum(r["eval_stats"]["eval_time_sec"] for r in self.stage_records)),
            "final_inference_latency_ms_per_sample": final_stage["eval_stats"][
                "inference_latency_ms_per_sample"
            ],
        }
        write_json(self.output_dir / "resource_metrics.json", resource_payload)
        write_json(
            self.output_dir / "manifest.json",
            {
                "run_id": self.run_id,
                "dataset_fingerprint": self.stream.dataset_fingerprint,
                "task_order_fingerprint": self.stream.task_order_fingerprint,
                "git_info": self.git_info,
                "memory_policy": self.config.memory.__dict__,
            },
        )
        return metrics_payload
