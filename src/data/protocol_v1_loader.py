"""Protocol v1 Dataset Loader for FewRel Continual Relation Extraction.

Implements strict Protocol v1 Track A (FewRel 8 tasks x 10 relations, 5-shot)
with entity markers [E1], [/E1], [E2], [/E2], canonical cluster groupings,
and deterministic task orders.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase


SPECIAL_MARKERS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]


def insert_entity_markers(
    words: list[str],
    e1_indices: list[int],
    e2_indices: list[int],
) -> tuple[list[str], str]:
    """Insert entity marker tokens [E1]...[/E1] and [E2]...[/E2] into word list.

    Inserts tokens in descending index order so earlier offsets are preserved.
    Returns (marked_words, marked_text_string).
    """
    insertions: list[tuple[int, str, int]] = [
        (min(e1_indices), "[E1]", 0),
        (max(e1_indices) + 1, "[/E1]", 1),
        (min(e2_indices), "[E2]", 2),
        (max(e2_indices) + 1, "[/E2]", 3),
    ]
    # Sort descending by index; if same index, put [/E*] before [E*]
    insertions.sort(key=lambda x: (x[0], 0 if "/" in x[1] else 1), reverse=True)

    marked = list(words)
    for idx, tag, _ in insertions:
        safe_idx = min(idx, len(marked))
        marked.insert(safe_idx, tag)

    return marked, " ".join(marked)


@dataclass
class ProtocolV1Sample:
    relation_id: int  # 1-based relation id (1..80)
    relation_name: str
    class_idx: int  # 0-based continual class index (0..79)
    task_id: int  # 0-based task index (0..7)
    words: list[str]
    marked_text: str
    head_text: str
    tail_text: str
    head_indices: list[int]
    tail_indices: list[int]


@dataclass
class ProtocolV1Task:
    task_id: int  # 0..7
    cluster_id: int
    relations: list[int]  # 10 relations
    relation_names: list[str]
    class_indices: list[int]  # [10*task_id .. 10*task_id + 9]
    train_samples: list[ProtocolV1Sample]  # 5-shot * 10 = 50 samples
    test_samples: list[ProtocolV1Sample]  # 100 * 10 = 1000 samples


class ProtocolV1Dataset:
    """Loads and formats Protocol v1 FewRel dataset for continual learning."""

    def __init__(
        self,
        base_dir: str | Path = "data/protocol_v1/fewrel",
        split_subdir: str = "CFRLdata_10_100_10_5",
        task_order_strategy: str = "conpl_seed_shuffle",
        seed: int = 2021,
        train_shot: int = 5,
        test_shot: int = 100,
        max_seq_length: int = 128,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.split_dir = self.base_dir / split_subdir
        self.task_order_strategy = task_order_strategy
        self.seed = int(seed)
        self.train_shot = int(train_shot)
        self.test_shot = int(test_shot)
        self.max_seq_length = int(max_seq_length)

        # Load relation names
        rel_name_file = self.base_dir / "relation_name.txt"
        with rel_name_file.open("r", encoding="utf-8") as f:
            self.relation_names = [line.strip() for line in f]
        assert len(self.relation_names) == 80, f"Expected 80 relation names, found {len(self.relation_names)}"

        # Load relation indices (1..80) and cluster labels (0..7)
        rel_index_file = self.base_dir / "rel_index.npy"
        cluster_label_file = self.split_dir / "rel_cluster_label_0.npy"
        self.rel_index = np.load(rel_index_file)
        self.cluster_labels = np.load(cluster_label_file)

        # Build map: rel_id (1..80) -> cluster_id (0..7)
        self.rel_to_cluster: dict[int, int] = {}
        for idx, r_id in enumerate(self.rel_index):
            self.rel_to_cluster[int(r_id)] = int(self.cluster_labels[idx])

        # Group relations by cluster
        self.cluster_to_rels: dict[int, list[int]] = {c: [] for c in range(8)}
        for r_id, c_id in self.rel_to_cluster.items():
            self.cluster_to_rels[c_id].append(r_id)

        for c in range(8):
            assert len(self.cluster_to_rels[c]) == 10, f"Cluster {c} has {len(self.cluster_to_rels[c])} relations, expected 10"

    def _determine_cluster_order(self) -> list[int]:
        """Compute ordering of the 8 clusters for continual learning."""
        if self.task_order_strategy == "sequential":
            return list(range(8))
        elif self.task_order_strategy == "conpl_seed_shuffle":
            rng = random.Random(self.seed)
            shuffled = list(range(8))
            rng.shuffle(shuffled)
            order = np.argsort(shuffled).tolist()
            return [int(x) for x in order]
        else:
            raise ValueError(f"Unknown task_order_strategy: {self.task_order_strategy}")

    def _parse_lines(self, file_path: Path) -> dict[int, list[dict[str, Any]]]:
        """Parse TSV format from CFRL/ConPL dataset."""
        rel_to_raw: dict[int, list[dict[str, Any]]] = {r: [] for r in range(1, 81)}
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if not parts or not parts[0]:
                    continue
                rel_id = int(parts[0])
                words = parts[2].split(" ")
                e1_idx = [int(x) for x in parts[4].split(" ")]
                e2_idx = [int(x) for x in parts[6].split(" ")]
                head_text = parts[3]
                tail_text = parts[5]

                _, marked_text = insert_entity_markers(words, e1_idx, e2_idx)
                rel_to_raw[rel_id].append({
                    "rel_id": rel_id,
                    "words": words,
                    "marked_text": marked_text,
                    "head_text": head_text,
                    "tail_text": tail_text,
                    "head_indices": e1_idx,
                    "tail_indices": e2_idx,
                })
        return rel_to_raw

    def build_tasks(self) -> list[ProtocolV1Task]:
        """Construct the 8 continual tasks with strict shots and class indices."""
        cluster_order = self._determine_cluster_order()

        train_raw = self._parse_lines(self.split_dir / "train_0.txt")
        test_raw = self._parse_lines(self.split_dir / "test_0.txt")

        tasks: list[ProtocolV1Task] = []
        global_class_idx = 0

        for task_id, cluster_id in enumerate(cluster_order):
            cluster_rels = list(self.cluster_to_rels[cluster_id])
            # Deterministic relation order within cluster
            cluster_rels.sort()

            task_train_samples: list[ProtocolV1Sample] = []
            task_test_samples: list[ProtocolV1Sample] = []
            task_relation_names: list[str] = []
            task_class_indices: list[int] = []

            for rel_id in cluster_rels:
                rel_name = self.relation_names[rel_id - 1]
                class_idx = global_class_idx
                global_class_idx += 1

                task_relation_names.append(rel_name)
                task_class_indices.append(class_idx)

                # Slice exact train shot (5-shot)
                rel_train = train_raw[rel_id][: self.train_shot]
                for item in rel_train:
                    sample = ProtocolV1Sample(
                        relation_id=rel_id,
                        relation_name=rel_name,
                        class_idx=class_idx,
                        task_id=task_id,
                        words=item["words"],
                        marked_text=item["marked_text"],
                        head_text=item["head_text"],
                        tail_text=item["tail_text"],
                        head_indices=item["head_indices"],
                        tail_indices=item["tail_indices"],
                    )
                    task_train_samples.append(sample)

                # Slice exact test shot (up to test_shot, e.g. 100)
                rel_test = test_raw[rel_id][: self.test_shot]
                for item in rel_test:
                    sample = ProtocolV1Sample(
                        relation_id=rel_id,
                        relation_name=rel_name,
                        class_idx=class_idx,
                        task_id=task_id,
                        words=item["words"],
                        marked_text=item["marked_text"],
                        head_text=item["head_text"],
                        tail_text=item["tail_text"],
                        head_indices=item["head_indices"],
                        tail_indices=item["tail_indices"],
                    )
                    task_test_samples.append(sample)

            tasks.append(
                ProtocolV1Task(
                    task_id=task_id,
                    cluster_id=cluster_id,
                    relations=cluster_rels,
                    relation_names=task_relation_names,
                    class_indices=task_class_indices,
                    train_samples=task_train_samples,
                    test_samples=task_test_samples,
                )
            )

        return tasks


def collate_samples_for_bert(
    samples: Sequence[ProtocolV1Sample],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    max_seq_length: int = 128,
) -> dict[str, torch.Tensor]:
    """Tokenize a batch of ProtocolV1Sample and locate [E1] and [E2] marker positions."""
    texts = [s.marked_text for s in samples]
    labels = [s.class_idx for s in samples]
    task_ids = [s.task_id for s in samples]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    e1_tok_id = tokenizer.convert_tokens_to_ids("[E1]")
    e2_tok_id = tokenizer.convert_tokens_to_ids("[E2]")

    batch_size = input_ids.size(0)
    e1_positions = []
    e2_positions = []

    for b in range(batch_size):
        row = input_ids[b]
        p1 = (row == e1_tok_id).nonzero(as_tuple=True)[0]
        p2 = (row == e2_tok_id).nonzero(as_tuple=True)[0]
        if len(p1) == 0:
            e1_positions.append(0)
        else:
            e1_positions.append(p1[0].item())
        if len(p2) == 0:
            e2_positions.append(0)
        else:
            e2_positions.append(p2[0].item())

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "e1_indices": torch.tensor(e1_positions, dtype=torch.long, device=device),
        "e2_indices": torch.tensor(e2_positions, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "task_ids": torch.tensor(task_ids, dtype=torch.long, device=device),
    }
