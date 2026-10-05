#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PIPELINE_SRC = REPO_ROOT / "dataset-pipelines" / "continual-relation-extraction" / "src"
BASELINES_SRC = SCRIPT_DIR / "continual-relation-baselines" / "src"

for p in [PIPELINE_SRC, BASELINES_SRC]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from datasets.fewrel import FewRelDataset
from evaluation.evaluator import Evaluator
from task_generation.task_builder import ContinualTaskBuilder
from task_generation.task_order import TaskOrder

from cl_re_baselines.mock_model import MockRelationClassifier
from cl_re_baselines.sequential_trainer import SequentialFTTrainer

# Check for PyTorch
try:
    import torch
    from cl_re_baselines.model import BERTRelationClassifier
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def resolve_path(candidate: str | Path, base_dir: Path = REPO_ROOT) -> Path:
    p = Path(candidate)
    if p.is_absolute() and p.exists():
        return p
    if (base_dir / p).exists():
        return base_dir / p
    if (Path.cwd() / p).exists():
        return Path.cwd() / p
    return base_dir / p


def load_config(config_path: Path | str) -> dict:
    resolved = resolve_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Check for yaml or json
    if resolved.suffix in {".yaml", ".yml"}:
        try:
            import yaml
            with resolved.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            # Fallback to json if json version exists
            json_candidate = resolved.with_suffix(".json")
            if json_candidate.exists():
                with json_candidate.open("r", encoding="utf-8") as f:
                    return json.load(f)
            raise ImportError(
                f"PyYAML is not installed to read {resolved.name}. "
                f"Please provide {json_candidate.name} or install pyyaml."
            )
    else:
        with resolved.open("r", encoding="utf-8") as f:
            return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="B0 Sequential Fine-Tuning Runner for Continual Relation Extraction."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/b0_sequential_ft_fewrel_5shot_seed2021.json",
        help="Path to experiment config file (JSON or YAML)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast simulation using MockRelationClassifier without GPU/PyTorch",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Override number of tasks (e.g. 2 for smoke test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override execution device (e.g. cpu, cuda, mps)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (e.g. results/fewrel/5shot/B0_smoke_test/seed_2021)",
    )
    parser.add_argument(
        "--require-real-model",
        action="store_true",
        help="Strict mode: abort immediately if PyTorch or deep learning weights are missing. Disallow silent mock fallback.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    seed = int(config.get("seed", 2021))
    dataset_cfg = config.get("dataset", {})
    output_cfg = config.get("output", {})

    print("==================================================================")
    print("B0: Sequential Fine-Tuning Baseline (FewRel Track A)")
    print("==================================================================")
    print(f"Experiment ID: {config.get('experiment', {}).get('id')}")
    print(f"Seed:          {seed}")
    print(f"Train Shot:    {dataset_cfg.get('train_shot')}-shot")
    print(f"Val Shot:      {dataset_cfg.get('val_shot')}-shot")
    print(f"Config:        {args.config}")
    print("==================================================================")

    # 1. Load Dataset
    raw_data_dir = resolve_path(dataset_cfg.get("raw_data_dir", "dataset-pipelines/continual-relation-extraction/data/raw/fewrel"))
    print(f"\n[1/5] Loading FewRel dataset from {raw_data_dir}...")
    dataset = FewRelDataset(
        data_dir=raw_data_dir,
        train_val_test_counts=(
            int(dataset_cfg.get("train_shot", 5)),
            int(dataset_cfg.get("val_shot", 5)),
            140,
        ),
    )
    relations = dataset.get_relations()
    print(f"  Loaded {len(relations)} relations.")

    # 2. Load Task Order
    task_order_path = resolve_path(dataset_cfg.get("task_order_path", "dataset-pipelines/continual-relation-extraction/task-orders/fewrel/order_seed_2021.json"))
    print(f"\n[2/5] Loading Task Order from {task_order_path}...")
    task_order = TaskOrder.load(task_order_path)
    task_order.validate_against_expected_relations(relations)
    print(f"  Validated task order: {task_order.num_tasks} tasks, seed {task_order.seed}")

    # 3. Build Continual Tasks
    print("\n[3/5] Building continual tasks...")
    builder = ContinualTaskBuilder()
    tasks = builder.build_tasks(dataset, task_order)

    if args.num_tasks is not None:
        tasks = tasks[: args.num_tasks]
        print(f"  Restricted tasks to first {len(tasks)} tasks (Smoke Test mode).")

    for t in tasks:
        print(f"  Task {t.task_id + 1} (T{t.task_id + 1}): {len(t.relations)} rels | train={t.train_count}, val={t.val_count}, test={t.test_count}")

    # 4. Instantiate Model
    print("\n[4/5] Initializing model...")
    device_override = args.device or config.get("training", {}).get("device", "auto")

    if args.dry_run:
        print("  Running in simulation mode with MockRelationClassifier (--dry-run specified).")
        model = MockRelationClassifier(
            num_classes=int(config.get("model", {}).get("num_classes", 80)),
            seed=seed,
        )
    else:
        if not HAS_TORCH:
            error_msg = (
                "FATAL: PyTorch/Transformers is not installed in the environment.\n"
                "Real BERT training cannot proceed without PyTorch.\n"
                "To run a fast zero-dependency simulation, pass --dry-run explicitly.\n"
                "Otherwise, install dependencies via: pip install torch transformers"
            )
            print(f"  [ERROR] {error_msg}")
            raise RuntimeError(error_msg)

        relation_to_id = dataset.get_relation_to_id()
        relation_to_class_idx = {rel: idx for idx, rel in enumerate(task_order.all_relations)}
        class_idx_to_rel_id = {idx: relation_to_id[rel] for rel, idx in relation_to_class_idx.items()}

        print(f"  Initializing BERTRelationClassifier ({config.get('model', {}).get('backbone')}) on device '{device_override}'...")
        model = BERTRelationClassifier(
            backbone_name=config.get("model", {}).get("backbone", "bert-base-uncased"),
            num_classes=int(config.get("model", {}).get("num_classes", 80)),
            dropout_rate=float(config.get("model", {}).get("classifier_dropout", 0.1)),
            max_seq_length=int(config.get("model", {}).get("max_seq_length", 128)),
            device=device_override,
            relation_to_class_idx=relation_to_class_idx,
            class_idx_to_rel_id=class_idx_to_rel_id,
        )

    # 5. Run Sequential Fine-Tuning Trainer
    print("\n[5/5] Executing Sequential Fine-Tuning across stages...")
    results_dir_str = args.output_dir or output_cfg.get("results_dir", "results/fewrel/5shot/B0_sequential_ft/seed_2021")
    results_dir = resolve_path(results_dir_str)
    trainer = SequentialFTTrainer(
        config=config,
        tasks=tasks,
        model=model,
        evaluator=Evaluator(),
        results_dir=results_dir,
    )

    summary = trainer.run()

    print("\n==================================================================")
    print("B0 Sequential Fine-Tuning Completed Successfully!")
    print("==================================================================")
    print(f"Final Average Accuracy (Final AA): {summary['final_average_accuracy'] * 100:.2f}%")
    print(f"Final Macro-F1:                    {summary.get('final_macro_f1', 0.0) * 100:.2f}%")
    print(f"Average Incremental Accuracy (AIA):{summary['average_incremental_accuracy'] * 100:.2f}%")
    print(f"Average Catastrophic Forgetting:   {summary['average_forgetting'] * 100:.2f}%")
    print(f"Backward Transfer (BWT):           {summary['backward_transfer'] * 100:.2f}%")
    most_f = summary.get("most_forgotten_task", {})
    if most_f.get("task_name"):
        print(f"Most Forgotten Task:               {most_f['task_name']} (drop: {float(most_f.get('forgetting', 0.0)) * 100:.2f}%)")
    print("==================================================================")
    print(f"Artifacts exported to: {results_dir}")
    print(f"  - Performance Matrix (CSV): {results_dir / 'performance_matrix.csv'}")
    print(f"  - Performance Matrix (JSON):{results_dir / 'performance_matrix.json'}")
    print(f"  - Metrics Log:              {results_dir / 'metrics.jsonl'}")
    print(f"  - Summary:                  {results_dir / 'summary.json'}")
    print(f"  - Conclusion:               {results_dir / 'conclusion.md'}")
    print(f"  - Checkpoints:              {results_dir / 'checkpoints'}")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
