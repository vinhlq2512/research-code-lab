"""Benchmark loaders, task-distribution builders (Exp A/B/C), and synthetic test dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src.data.protocol_v1_loader import (
    ProtocolV1Dataset,
    ProtocolV1Sample,
    ProtocolV1Task,
    insert_entity_markers,
)


@dataclass
class TaskStreamBundle:
    tasks: list[ProtocolV1Task]
    dataset_fingerprint: str
    task_order_fingerprint: str
    relation_to_task: dict[int, int]


def _compute_fingerprints(tasks: list[ProtocolV1Task]) -> tuple[str, str]:
    ds_h = hashlib.sha256()
    order_obj = []
    for t in tasks:
        order_obj.append({"task_id": t.task_id, "relations": t.relations})
        for s in t.train_samples:
            ds_h.update(f"train:{t.task_id}:{s.relation_id}:{s.marked_text}\n".encode("utf-8"))
        for s in t.test_samples:
            ds_h.update(f"test:{t.task_id}:{s.relation_id}:{s.marked_text}\n".encode("utf-8"))
    order_h = hashlib.sha256(json.dumps(order_obj, sort_keys=True).encode("utf-8")).hexdigest()
    return ds_h.hexdigest(), order_h


def _balanced_semantic_groups(
    rel_ids: list[int],
    rel_features: np.ndarray,
    num_tasks: int,
) -> list[list[int]]:
    """Partition relations into balanced semantic groups offline for stress-test orders B/C."""
    norms = np.linalg.norm(rel_features, axis=1, keepdims=True).clip(min=1e-12)
    norm_f = rel_features / norms
    sim = np.clip(norm_f @ norm_f.T, -1.0, 1.0)
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(z, t=num_tasks, criterion="maxclust")

    # Sort relations by (cluster_label, relation_id) then slice evenly into num_tasks buckets
    paired = sorted(zip(labels.tolist(), rel_ids), key=lambda x: (x[0], x[1]))
    ordered_rels = [r for _, r in paired]
    per_task = max(1, len(ordered_rels) // num_tasks)
    groups: list[list[int]] = []
    idx = 0
    for t in range(num_tasks):
        end = len(ordered_rels) if t == num_tasks - 1 else idx + per_task
        groups.append(ordered_rels[idx:end])
        idx = end
    return groups


def _reorder_tasks_by_relation_groups(
    base_tasks: list[ProtocolV1Task],
    groups: list[list[int]],
) -> list[ProtocolV1Task]:
    """Re-pack existing train/test samples into new task buckets without leaking test/train."""
    rel_train_map: dict[int, list[ProtocolV1Sample]] = {}
    rel_test_map: dict[int, list[ProtocolV1Sample]] = {}
    rel_name_map: dict[int, str] = {}

    for task in base_tasks:
        for s in task.train_samples:
            rel_train_map.setdefault(s.relation_id, []).append(s)
            rel_name_map[s.relation_id] = s.relation_name
        for s in task.test_samples:
            rel_test_map.setdefault(s.relation_id, []).append(s)
            rel_name_map[s.relation_id] = s.relation_name

    new_tasks: list[ProtocolV1Task] = []
    global_cidx = 0
    for task_id, rel_group in enumerate(groups):
        train_samples: list[ProtocolV1Sample] = []
        test_samples: list[ProtocolV1Sample] = []
        rel_names: list[str] = []
        class_indices: list[int] = []
        for rel_id in rel_group:
            cidx = global_cidx
            global_cidx += 1
            class_indices.append(cidx)
            rel_names.append(rel_name_map[rel_id])
            for s in rel_train_map[rel_id]:
                train_samples.append(
                    ProtocolV1Sample(
                        relation_id=s.relation_id,
                        relation_name=s.relation_name,
                        class_idx=cidx,
                        task_id=task_id,
                        words=list(s.words),
                        marked_text=s.marked_text,
                        head_text=s.head_text,
                        tail_text=s.tail_text,
                        head_indices=list(s.head_indices),
                        tail_indices=list(s.tail_indices),
                    )
                )
            for s in rel_test_map[rel_id]:
                test_samples.append(
                    ProtocolV1Sample(
                        relation_id=s.relation_id,
                        relation_name=s.relation_name,
                        class_idx=cidx,
                        task_id=task_id,
                        words=list(s.words),
                        marked_text=s.marked_text,
                        head_text=s.head_text,
                        tail_text=s.tail_text,
                        head_indices=list(s.head_indices),
                        tail_indices=list(s.tail_indices),
                    )
                )
        new_tasks.append(
            ProtocolV1Task(
                task_id=task_id,
                cluster_id=-1,
                relations=list(rel_group),
                relation_names=rel_names,
                class_indices=class_indices,
                train_samples=train_samples,
                test_samples=test_samples,
            )
        )
    return new_tasks


def load_protocol_v1_task_stream(
    base_dir: str | Path,
    split_subdir: str = "CFRLdata_10_100_10_5",
    task_order_strategy: str = "conpl_seed_shuffle",
    task_distribution: str = "standard_random",
    seed: int = 2021,
    train_shot: int = 5,
    test_shot: int = 100,
    max_seq_length: int = 128,
    max_tasks: int | None = None,
    max_test_samples_per_task: int | None = None,
) -> TaskStreamBundle:
    """Load Protocol v1 tasks cleanly with optional Experiment B/C distribution construction."""
    base_path = Path(base_dir)
    dataset = ProtocolV1Dataset(
        base_dir=base_path,
        split_subdir=split_subdir,
        task_order_strategy=task_order_strategy,
        seed=seed,
        train_shot=train_shot,
        test_shot=test_shot,
        max_seq_length=max_seq_length,
    )
    tasks = dataset.build_tasks()

    if task_distribution in ("semantic_homogeneous", "cross_task_dispersed"):
        rel_index = np.load(base_path / "rel_index.npy").astype(int).tolist()
        rel_feature = np.load(base_path / "rel_feature.npy").astype(np.float32)
        groups_homo = _balanced_semantic_groups(rel_index, rel_feature, num_tasks=len(tasks))
        if task_distribution == "semantic_homogeneous":
            groups = groups_homo
        else:
            # Round-robin across homogeneous clusters to maximize cross-task semantic dispersion
            groups = [[] for _ in range(len(tasks))]
            t_ptr = 0
            for grp in groups_homo:
                for rel_id in grp:
                    groups[t_ptr % len(tasks)].append(rel_id)
                    t_ptr += 1
        tasks = _reorder_tasks_by_relation_groups(tasks, groups)
    elif task_distribution != "standard_random":
        raise ValueError(f"Unsupported task_distribution: {task_distribution}")

    if max_tasks is not None:
        tasks = tasks[: int(max_tasks)]

    if max_test_samples_per_task is not None:
        limit = int(max_test_samples_per_task)
        for task in tasks:
            # Stratified deterministic slice across relations in the task
            by_rel: dict[int, list[ProtocolV1Sample]] = {}
            for s in task.test_samples:
                by_rel.setdefault(s.relation_id, []).append(s)
            per_rel = max(1, limit // max(1, len(by_rel)))
            sliced: list[ProtocolV1Sample] = []
            for r_id in sorted(by_rel.keys()):
                sliced.extend(by_rel[r_id][:per_rel])
            task.test_samples = sliced[:limit]

    rel_to_task: dict[int, int] = {}
    for task in tasks:
        for r_id in task.relations:
            rel_to_task[int(r_id)] = int(task.task_id)

    ds_fp, order_fp = _compute_fingerprints(tasks)
    return TaskStreamBundle(
        tasks=tasks,
        dataset_fingerprint=ds_fp,
        task_order_fingerprint=order_fp,
        relation_to_task=rel_to_task,
    )


def build_synthetic_task_stream(
    num_tasks: int = 3,
    relations_per_task: int = 2,
    train_shot: int = 4,
    test_shot: int = 4,
) -> TaskStreamBundle:
    """Create a deterministic synthetic continual RE benchmark with clear semantic clusters."""
    domain_templates = [
        ("family_parent", "alice", "bob", ["mother", "parent", "family", "birth"]),
        ("company_leader", "acme", "carol", ["founder", "ceo", "company", "board"]),
        ("geo_capital", "paris", "france", ["capital", "city", "located", "country"]),
    ]
    tasks: list[ProtocolV1Task] = []
    rel_to_task: dict[int, int] = {}
    rel_counter = 1
    cidx_counter = 0

    for t in range(num_tasks):
        train_samples: list[ProtocolV1Sample] = []
        test_samples: list[ProtocolV1Sample] = []
        rels: list[int] = []
        rel_names: list[str] = []
        class_indices: list[int] = []
        for r_slot in range(relations_per_task):
            rel_id = rel_counter
            rel_counter += 1
            cidx = cidx_counter
            cidx_counter += 1
            rels.append(rel_id)
            class_indices.append(cidx)
            rel_to_task[rel_id] = t
            # Rotate domains across tasks so similar relations appear in different tasks!
            dom_name, e1_tok, e2_tok, keywords = domain_templates[(t + r_slot) % len(domain_templates)]
            rel_name = f"{dom_name}_t{t}_r{r_slot}"
            rel_names.append(rel_name)

            for split_name, shot, target in (
                ("train", train_shot, train_samples),
                ("test", test_shot, test_samples),
            ):
                for s_i in range(shot):
                    kw = keywords[s_i % len(keywords)]
                    words = [
                        e1_tok,
                        "is",
                        "the",
                        kw,
                        f"rel{rel_id}",
                        "of",
                        e2_tok,
                        f"{split_name}{s_i}",
                    ]
                    _, marked = insert_entity_markers(words, [0], [6])
                    target.append(
                        ProtocolV1Sample(
                            relation_id=rel_id,
                            relation_name=rel_name,
                            class_idx=cidx,
                            task_id=t,
                            words=words,
                            marked_text=marked,
                            head_text=e1_tok,
                            tail_text=e2_tok,
                            head_indices=[0],
                            tail_indices=[6],
                        )
                    )
        tasks.append(
            ProtocolV1Task(
                task_id=t,
                cluster_id=-1,
                relations=rels,
                relation_names=rel_names,
                class_indices=class_indices,
                train_samples=train_samples,
                test_samples=test_samples,
            )
        )

    ds_fp, order_fp = _compute_fingerprints(tasks)
    return TaskStreamBundle(
        tasks=tasks,
        dataset_fingerprint=ds_fp,
        task_order_fingerprint=order_fp,
        relation_to_task=rel_to_task,
    )
