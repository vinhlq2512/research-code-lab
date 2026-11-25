"""Scalable FewRel Dataset Loader with Nested Sampling and Zero Leakage.

Supports:
- Configurable train shots: 5, 10, 50, 100, and 'full' (canonical 420 train samples / relation).
- Strict Zero Leakage: Holds out 100 test samples per relation and strictly filters out
  any test sentence from the train pool using normalized content hashing.
- Deterministic Nested Sampling: S_5 subset S_10 subset S_50 subset S_100 subset S_full for any given seed.
- 100% Compatible with ProtocolV1Task, ProtocolV1Sample, and collate_samples_for_bert.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import re
from typing import Any, Sequence

import numpy as np

from src.data.protocol_v1_loader import (
    ProtocolV1Sample,
    ProtocolV1Task,
    insert_entity_markers,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def normalize_sentence_for_hash(words: Sequence[str]) -> str:
    """Normalize a sequence of words/tokens for content hashing and duplicate detection."""
    clean_words = [
        w.lower()
        for w in words
        if w not in ("[E1]", "[/E1]", "[E2]", "[/E2]", "[E11]", "[E12]", "[E21]", "[E22]")
    ]
    # Remove punctuation-only artifacts and trim spaces
    text = " ".join(clean_words)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ScalableFewRelDataset:
    """Dataset loader supporting arbitrary train shots with deterministic nested sampling."""

    def __init__(
        self,
        data_json_path: str | Path | None = None,
        base_dir: str | Path = "data/protocol_v1/fewrel",
        split_subdir: str = "CFRLdata_10_100_10_5",
        id2rel_path: str | Path | None = None,
        train_shot: int | str = 5,
        test_shot: int = 100,
        full_train_limit: int = 420,
        seed: int = 2021,
        task_order_strategy: str = "conpl_seed_shuffle",
    ) -> None:
        self.base_dir = Path(base_dir) if Path(base_dir).is_absolute() else REPO_ROOT / base_dir
        self.split_dir = self.base_dir / split_subdir
        self.seed = int(seed)
        self.test_shot = int(test_shot)
        self.full_train_limit = int(full_train_limit)
        self.task_order_strategy = task_order_strategy

        # Parse train shot
        if isinstance(train_shot, str):
            train_shot_lower = train_shot.strip().lower()
            if train_shot_lower == "full":
                self.train_shot: int | str = "full"
            else:
                self.train_shot = int(train_shot_lower)
        else:
            self.train_shot = int(train_shot)

        # Locate WAVE-CRE data_with_marker.json
        if data_json_path is None:
            self.data_json_path = (
                REPO_ROOT / "paper-reproductions" / "WAVE-CRE" / "datasets" / "data_with_marker.json"
            )
        else:
            p = Path(data_json_path)
            self.data_json_path = p if p.is_absolute() else REPO_ROOT / p

        if not self.data_json_path.exists():
            raise FileNotFoundError(f"FewRel source json file not found: {self.data_json_path}")

        # Locate id2rel.json
        if id2rel_path is None:
            self.id2rel_path = (
                REPO_ROOT / "paper-reproductions" / "WAVE-CRE" / "datasets" / "id2rel.json"
            )
        else:
            p = Path(id2rel_path)
            self.id2rel_path = p if p.is_absolute() else REPO_ROOT / p

        # Load relation names
        rel_name_file = self.base_dir / "relation_name.txt"
        with rel_name_file.open("r", encoding="utf-8") as f:
            self.relation_names = [line.strip() for line in f if line.strip()]
        self.total_relations = len(self.relation_names)
        self.num_tasks = 8

        # Load relation indices and cluster labels
        rel_index_file = self.base_dir / "rel_index.npy"
        cluster_label_file = self.split_dir / "rel_cluster_label_0.npy"
        self.rel_index = np.load(rel_index_file)
        self.cluster_labels = np.load(cluster_label_file)

        # Build map: rel_id (1..80) -> cluster_id (0..7)
        self.rel_to_cluster: dict[int, int] = {}
        for idx, r_id in enumerate(self.rel_index):
            self.rel_to_cluster[int(r_id)] = int(self.cluster_labels[idx])

        self.cluster_to_rels: dict[int, list[int]] = {c: [] for c in range(self.num_tasks)}
        for r_id, c_id in self.rel_to_cluster.items():
            self.cluster_to_rels[c_id].append(r_id)

        # Load id2rel mapping (index 0..79 -> PID string)
        with self.id2rel_path.open("r", encoding="utf-8") as f:
            self.id2rel = json.load(f)
        assert len(self.id2rel) == self.total_relations, (
            f"id2rel length {len(self.id2rel)} != total_relations {self.total_relations}"
        )

        # Load test set and full clean train pool
        self._test_samples_by_rel: dict[int, list[ProtocolV1Sample]] = {}
        self._clean_train_pool_by_rel: dict[int, list[ProtocolV1Sample]] = {}
        self._load_and_partition_data()

    def _determine_cluster_order(self) -> list[int]:
        """Compute ordering of the clusters for continual learning."""
        if self.task_order_strategy == "sequential":
            return list(range(self.num_tasks))
        elif self.task_order_strategy == "conpl_seed_shuffle":
            rng = random.Random(self.seed)
            shuffled = list(range(self.num_tasks))
            rng.shuffle(shuffled)
            order = np.argsort(shuffled).tolist()
            return [int(x) for x in order]
        else:
            raise ValueError(f"Unknown task_order_strategy: {self.task_order_strategy}")

    def _load_and_partition_data(self) -> None:
        """Load test_0.txt as official test set, then load clean non-overlapping train pool from JSON."""
        # 1. Load official test set from test_0.txt
        test_file = self.split_dir / "test_0.txt"
        test_hashes_by_rel: dict[int, set[str]] = {r: set() for r in range(1, self.total_relations + 1)}
        self._test_samples_by_rel = {r: [] for r in range(1, self.total_relations + 1)}

        with test_file.open("r", encoding="utf-8") as f:
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
                norm_hash = normalize_sentence_for_hash(words)
                test_hashes_by_rel[rel_id].add(norm_hash)

                sample = ProtocolV1Sample(
                    relation_id=rel_id,
                    relation_name=self.relation_names[rel_id - 1],
                    class_idx=-1,  # will assign in build_tasks
                    task_id=-1,
                    words=words,
                    marked_text=marked_text,
                    head_text=head_text,
                    tail_text=tail_text,
                    head_indices=e1_idx,
                    tail_indices=e2_idx,
                )
                self._test_samples_by_rel[rel_id].append(sample)

        # Slice exact test_shot (e.g. 100)
        for r in range(1, self.total_relations + 1):
            self._test_samples_by_rel[r] = self._test_samples_by_rel[r][: self.test_shot]

        # 2. Build global set of test hashes across ALL 80 relations to prevent cross-relation leakage
        global_test_hashes: set[str] = set()
        for r_hashes in test_hashes_by_rel.values():
            global_test_hashes.update(r_hashes)

        # 3. Load WAVE-CRE data_with_marker.json and filter out ANY test hash globally
        with self.data_json_path.open("r", encoding="utf-8") as f:
            wave_data = json.load(f)

        self._clean_train_pool_by_rel = {r: [] for r in range(1, self.total_relations + 1)}

        for rel_idx, pid in enumerate(self.id2rel):
            rel_id = rel_idx + 1
            rel_name = self.relation_names[rel_idx]

            raw_items = wave_data[pid]
            clean_samples: list[ProtocolV1Sample] = []

            for item in raw_items:
                tokens = item["tokens"]
                # Convert WAVE-CRE tokens [E11] -> [E1], [E12] -> [/E1], [E21] -> [E2], [E22] -> [/E2]
                converted_words: list[str] = []
                e1_indices: list[int] = []
                e2_indices: list[int] = []
                in_e1 = False
                in_e2 = False

                for w in tokens:
                    if w == "[E11]":
                        converted_words.append("[E1]")
                        in_e1 = True
                    elif w == "[E12]":
                        converted_words.append("[/E1]")
                        in_e1 = False
                    elif w == "[E21]":
                        converted_words.append("[E2]")
                        in_e2 = True
                    elif w == "[E22]":
                        converted_words.append("[/E2]")
                        in_e2 = False
                    else:
                        curr_idx = len(converted_words)
                        if in_e1:
                            e1_indices.append(curr_idx)
                        if in_e2:
                            e2_indices.append(curr_idx)
                        converted_words.append(w)

                norm_hash = normalize_sentence_for_hash(converted_words)
                if norm_hash in global_test_hashes:
                    # Strictly filter out test overlap GLOBALLY (Zero Leakage)
                    continue

                marked_text = " ".join(converted_words)
                head_text = " ".join([converted_words[i] for i in e1_indices]) if e1_indices else ""
                tail_text = " ".join([converted_words[i] for i in e2_indices]) if e2_indices else ""

                sample = ProtocolV1Sample(
                    relation_id=rel_id,
                    relation_name=rel_name,
                    class_idx=-1,
                    task_id=-1,
                    words=converted_words,
                    marked_text=marked_text,
                    head_text=head_text,
                    tail_text=tail_text,
                    head_indices=e1_indices,
                    tail_indices=e2_indices,
                )
                clean_samples.append(sample)

            # Deterministic nested shuffle per relation using fixed seed
            rng = random.Random(self.seed + rel_id * 1000)
            rng.shuffle(clean_samples)
            self._clean_train_pool_by_rel[rel_id] = clean_samples

    def build_tasks(self) -> list[ProtocolV1Task]:
        """Construct the 8 continual tasks with deterministic nested sampling."""
        cluster_order = self._determine_cluster_order()
        tasks: list[ProtocolV1Task] = []
        global_class_idx = 0

        for task_id, cluster_id in enumerate(cluster_order):
            cluster_rels = list(self.cluster_to_rels[cluster_id])
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

                # 1. Deterministic Nested Train Sampling
                clean_pool = self._clean_train_pool_by_rel[rel_id]
                if self.train_shot == "full":
                    n_samples = min(self.full_train_limit, len(clean_pool))
                    selected_train = clean_pool[:n_samples]
                else:
                    n_samples = int(self.train_shot)
                    if n_samples > len(clean_pool):
                        raise ValueError(
                            f"Requested {n_samples} train samples for relation {rel_id} ({rel_name}), "
                            f"but only {len(clean_pool)} clean non-overlapping samples available!"
                        )
                    # S_k is strictly the first k elements of the deterministically shuffled clean pool
                    selected_train = clean_pool[:n_samples]

                for item in selected_train:
                    sample = ProtocolV1Sample(
                        relation_id=rel_id,
                        relation_name=rel_name,
                        class_idx=class_idx,
                        task_id=task_id,
                        words=item.words,
                        marked_text=item.marked_text,
                        head_text=item.head_text,
                        tail_text=item.tail_text,
                        head_indices=item.head_indices,
                        tail_indices=item.tail_indices,
                    )
                    task_train_samples.append(sample)

                # 2. Test Set
                raw_test = self._test_samples_by_rel[rel_id]
                for item in raw_test:
                    sample = ProtocolV1Sample(
                        relation_id=rel_id,
                        relation_name=rel_name,
                        class_idx=class_idx,
                        task_id=task_id,
                        words=item.words,
                        marked_text=item.marked_text,
                        head_text=item.head_text,
                        tail_text=item.tail_text,
                        head_indices=item.head_indices,
                        tail_indices=item.tail_indices,
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
