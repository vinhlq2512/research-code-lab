#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PIPELINE_SRC = REPO_ROOT / "dataset-pipelines" / "continual-relation-extraction" / "src"
BASELINES_SRC = SCRIPT_DIR / "continual-relation-baselines" / "src"

for p in [PIPELINE_SRC, BASELINES_SRC]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

from datasets.fewrel import FewRelDataset
from evaluation.evaluator import compute_accuracy, compute_macro_f1
from evaluation.metrics import compute_continual_summary_metrics
from evaluation.performance_matrix import PerformanceMatrix
from task_generation.task_builder import ContinualTask, ContinualTaskBuilder
from task_generation.task_order import TaskOrder
from schemas.relation_sample import RelationSample

from cl_re_baselines.checkpoint import CheckpointManager
from cl_re_baselines.logger import StreamingMetricsLogger, generate_conclusion_report
from cl_re_baselines.prototype_model import BERTRelationPrototypeClassifier
from generate_plots_svg import generate_svg_and_html

logger = logging.getLogger(__name__)


def resolve_path(candidate: str | Path, base_dir: Path = REPO_ROOT) -> Path:
    p = Path(candidate)
    if p.is_absolute() and p.exists():
        return p
    if (base_dir / p).exists():
        return base_dir / p
    if (Path.cwd() / p).exists():
        return Path.cwd() / p
    return base_dir / p


def load_config(config_path: Path | str) -> dict[str, Any]:
    resolved = resolve_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if resolved.suffix in {".yaml", ".yml"}:
        try:
            import yaml
            with resolved.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            json_candidate = resolved.with_suffix(".json")
            if json_candidate.exists():
                with json_candidate.open("r", encoding="utf-8") as f:
                    return json.load(f)
            raise
    else:
        with resolved.open("r", encoding="utf-8") as f:
            return json.load(f)


def extract_features(
    model: BERTRelationPrototypeClassifier,
    samples: Sequence[RelationSample],
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract concatenated entity marker representations [h_e1; h_e2] using frozen BERT."""
    if not samples:
        return (
            torch.empty((0, model.feature_dim), device=model.device),
            torch.empty((0,), dtype=torch.long, device=model.device),
        )

    all_feats: list[torch.Tensor] = []
    all_labels: list[int] = []

    model.encoder.eval()
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            chunk = samples[i : i + batch_size]
            batch = model.encode_batch(chunk)
            feats = model.extract_features_from_batch(batch)
            all_feats.append(feats)
            all_labels.extend(batch["labels"].cpu().tolist())

    return torch.cat(all_feats, dim=0), torch.tensor(all_labels, dtype=torch.long, device=model.device)


def load_or_extract_features(
    tasks: list[ContinualTask],
    model: BERTRelationPrototypeClassifier,
    cache_path: Path | None,
    use_disk_cache: bool = True,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Load cached entity features from disk if available, otherwise extract via BERT and cache."""
    num_tasks = len(tasks)

    if use_disk_cache and cache_path and cache_path.exists():
        print(f"\n[Feature Cache] Found disk feature cache at {cache_path}. Loading...")
        try:
            data = torch.load(cache_path, map_location=model.device)
            cached_train = data.get("train", [])
            cached_test = data.get("test", [])
            if len(cached_train) >= num_tasks and len(cached_test) >= num_tasks:
                print(f"[Feature Cache] Successfully loaded features for {num_tasks} tasks from disk in <1s.")
                return cached_train[:num_tasks], cached_test[:num_tasks]
            else:
                print(f"[Feature Cache] Disk cache had {len(cached_train)} tasks, but {num_tasks} needed. Re-extracting...")
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}. Re-extracting...")

    # Extract features with BERT
    print("\n------------------------------------------------------------------")
    print(f"Feature Extraction: Extracting [h_e1; h_e2] entity markers with Frozen BERT on {model.device}...")
    start_time = time.time()

    cached_train: list[tuple[torch.Tensor, torch.Tensor]] = []
    cached_test: list[tuple[torch.Tensor, torch.Tensor]] = []

    for t_idx, task in enumerate(tasks):
        t_feat, t_lbl = extract_features(model, task.train_samples)
        cached_train.append((t_feat, t_lbl))

        te_feat, te_lbl = extract_features(model, task.test_samples)
        cached_test.append((te_feat, te_lbl))
        print(f"  Task {t_idx + 1}/{num_tasks} extracted: train={t_feat.size(0)} vectors, test={te_feat.size(0)} vectors")

    elapsed = time.time() - start_time
    total_vectors = sum(x.size(0) + y.size(0) for (x, _), (y, _) in zip(cached_train, cached_test))
    print(f"Feature extraction finished in {elapsed:.2f}s ({total_vectors} vectors extracted).")
    print("------------------------------------------------------------------")

    # Save to disk cache if path provided
    if use_disk_cache and cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Save cpu tensors to keep cache portable
            cpu_train = [(f.cpu(), l.cpu()) for f, l in cached_train]
            cpu_test = [(f.cpu(), l.cpu()) for f, l in cached_test]
            torch.save({"train": cpu_train, "test": cpu_test, "num_tasks": num_tasks}, cache_path)
            print(f"[Feature Cache] Saved extracted features to disk cache at {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save disk cache: {e}")

    return cached_train, cached_test


def run_nearest_prototype_experiment(
    config: dict[str, Any],
    tasks: list[ContinualTask],
    model: BERTRelationPrototypeClassifier,
    results_dir: Path,
    cache_path: Path | None = None,
    use_disk_cache: bool = True,
) -> dict[str, Any]:
    """Execute Continual Nearest Prototype Benchmark on Frozen BERT."""
    num_tasks = len(tasks)
    seed = int(config.get("seed", 2021))
    task_order_path = str(config.get("dataset", {}).get("task_order_path", ""))
    metric_name = str(config.get("model", {}).get("metric", "cosine")).lower()

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(results_dir / "checkpoints")
    metrics_logger = StreamingMetricsLogger(results_dir)

    acc_matrix = PerformanceMatrix(
        num_tasks=num_tasks,
        metric="accuracy",
        dataset=config.get("dataset", {}).get("name", "fewrel"),
        seed=seed,
        task_order_path=task_order_path,
    )
    f1_matrix = PerformanceMatrix(
        num_tasks=num_tasks,
        metric="macro_f1",
        dataset=config.get("dataset", {}).get("name", "fewrel"),
        seed=seed,
        task_order_path=task_order_path,
    )

    # 1. Feature Caching Phase
    cached_train, cached_test = load_or_extract_features(
        tasks=tasks,
        model=model,
        cache_path=cache_path,
        use_disk_cache=use_disk_cache,
    )

    # 2. Sequential Continual Prototype Loop
    for stage in range(num_tasks):
        task = tasks[stage]
        relations_per_task = len(task.relations)
        seen_classes = (stage + 1) * relations_per_task
        model.set_seen_classes(seen_classes)

        print(f"\n>>> Stage {stage + 1}/{num_tasks}: Forming Prototypes for Task {stage + 1} (seen_classes={seen_classes})")
        train_x, train_y = cached_train[stage]
        train_x = train_x.to(model.device)
        train_y = train_y.to(model.device)

        # Compute centroid c_k = (1 / |S_k|) * sum_{x in S_k} f(x) for each relation in this task
        stage_start_class = stage * relations_per_task
        for rel_local_idx in range(relations_per_task):
            class_global_idx = stage_start_class + rel_local_idx
            mask = train_y == class_global_idx
            class_samples = train_x[mask]
            if class_samples.size(0) == 0:
                raise ValueError(f"No training samples found for class {class_global_idx} in Task {stage + 1}")
            centroid = model.compute_and_set_prototype(class_global_idx, class_samples)

        print(f"  Formed {relations_per_task} class prototypes. Total active prototypes in bank: {seen_classes}")

        # Save stage checkpoint (full model.pt + lightweight prototypes.pt)
        checkpoint_dir = checkpoint_manager.save_checkpoint(
            stage=stage,
            model=model,
            metadata={
                "task_id": task.task_id,
                "relations": task.relations,
                "seen_classes": seen_classes,
                "frozen_encoder": True,
                "metric": metric_name,
                "classifier_type": "nearest_prototype",
            },
        )
        model.save_prototypes_only(checkpoint_dir / "prototypes.pt")
        print(f"  Saved Stage {stage + 1} checkpoint at {checkpoint_dir.name}/ (model.pt + prototypes.pt)")

        # Evaluate on all observed tasks j in [0..stage]
        with torch.no_grad():
            for j in range(stage + 1):
                test_x, test_y = cached_test[j]
                test_x = test_x.to(model.device)
                test_y = test_y.to(model.device)

                logits = model.classify_features(test_x, seen_classes=seen_classes)
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                labels = test_y.cpu().tolist()

                acc = compute_accuracy(labels, preds)
                f1 = compute_macro_f1(labels, preds)

                acc_matrix.set_score(trained_until=stage, evaluated_task=j, score=acc)
                f1_matrix.set_score(trained_until=stage, evaluated_task=j, score=f1)

                metrics_logger.log_evaluation(
                    stage=stage,
                    test_task=j,
                    accuracy=acc,
                    macro_f1=f1,
                    sample_count=len(labels),
                )
                print(f"    Eval on T{j + 1}: Acc = {acc * 100:.2f}% | Macro-F1 = {f1 * 100:.2f}%")

        # Stream update matrices to disk immediately
        acc_matrix.export_csv(results_dir / "performance_matrix.csv")
        acc_matrix.export_json(results_dir / "performance_matrix.json")
        f1_matrix.export_csv(results_dir / "f1_performance_matrix.csv")
        f1_matrix.export_json(results_dir / "f1_performance_matrix.json")

    # 3. Continual Summary Metrics & Final Reports
    summary = compute_continual_summary_metrics(acc_matrix)
    f1_summary = compute_continual_summary_metrics(f1_matrix)
    summary["final_macro_f1"] = f1_summary["final_average_accuracy"]
    summary["experiment"] = config.get("experiment", {}).get("id", "E004")
    summary["method"] = config.get("experiment", {}).get("name", "frozen_bert_nearest_prototype")
    summary["baseline_role"] = config.get("experiment", {}).get("baseline_role", "non_parametric_probing")
    summary["seed"] = seed
    summary["shot"] = int(config.get("dataset", {}).get("train_shot", 5))

    metrics_logger.save_summary(summary)

    # Generate conclusion report
    conclusion_path = results_dir / "conclusion.md"
    generate_conclusion_report(summary, acc_matrix.as_list(), conclusion_path)

    # Generate publication vector & raster plots
    try:
        generate_svg_and_html(results_dir)
        print("  Generated SVG/PNG plots and dashboard.html successfully.")
    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E004: Frozen BERT + Nearest Prototype Non-Parametric Benchmark Runner."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/frozen_bert_nearest_prototype_fewrel_5shot_seed2021.json",
        help="Path to experiment config file (JSON or YAML)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Override number of tasks (useful for smoke tests)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'mps', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--cache-features-path",
        type=str,
        default=None,
        help="Path to feature disk cache file",
    )
    parser.add_argument(
        "--no-disk-cache",
        action="store_true",
        help="Disable loading and saving feature disk cache",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config = load_config(args.config)
    seed = int(config.get("seed", 2021))
    torch.manual_seed(seed)

    # Output directory
    results_dir = Path(args.output_dir) if args.output_dir else resolve_path(config.get("output", {}).get("results_dir", "results/E004"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Device configuration
    device = args.device if args.device != "auto" else config.get("training", {}).get("device", "auto")

    # Cache path
    cache_path_str = args.cache_features_path or config.get("training", {}).get("cache_features_path")
    cache_path = resolve_path(cache_path_str) if cache_path_str else None
    use_disk_cache = not args.no_disk_cache and bool(config.get("training", {}).get("feature_caching", True))

    # Dataset & Task generation
    ds_cfg = config.get("dataset", {})
    raw_data_dir = resolve_path(ds_cfg.get("raw_data_dir", "dataset-pipelines/continual-relation-extraction/data/raw/fewrel"))
    task_order_path = resolve_path(ds_cfg.get("task_order_path", "dataset-pipelines/continual-relation-extraction/task-orders/fewrel/order_seed_2021.json"))

    print(f"Loading FewRel dataset from: {raw_data_dir}")
    dataset = FewRelDataset(
        data_dir=raw_data_dir,
        train_val_test_counts=(
            int(ds_cfg.get("train_shot", 5)),
            int(ds_cfg.get("val_shot", 5)),
            140,
        ),
    )
    relations = dataset.get_relations()
    print(f"  Loaded {len(relations)} relations.")

    print(f"Loading task order from: {task_order_path}")
    task_order = TaskOrder.load(task_order_path)
    task_order.validate_against_expected_relations(relations)
    print(f"  Validated task order: {task_order.num_tasks} tasks, seed {task_order.seed}")

    builder = ContinualTaskBuilder()
    tasks = builder.build_tasks(dataset, task_order)

    if args.num_tasks is not None:
        tasks = tasks[: args.num_tasks]
        print(f"  Restricted tasks to first {len(tasks)} tasks (Smoke Test mode).")

    for t in tasks:
        print(f"  Task {t.task_id + 1} (T{t.task_id + 1}): {len(t.relations)} rels | train={t.train_count}, val={t.val_count}, test={t.test_count}")

    # Model initialization
    model_cfg = config.get("model", {})
    backbone = model_cfg.get("backbone", "bert-base-uncased")
    num_classes = int(model_cfg.get("num_classes", 80))
    metric = str(model_cfg.get("metric", "cosine"))
    max_seq_length = int(model_cfg.get("max_seq_length", 128))

    # Build relation mappings
    relation_to_id = dataset.get_relation_to_id()
    relation_to_class_idx = {rel: idx for idx, rel in enumerate(task_order.all_relations)}
    class_idx_to_rel_id = {idx: relation_to_id[rel] for rel, idx in relation_to_class_idx.items()}

    print(f"Initializing BERTRelationPrototypeClassifier with {backbone} on {device} (metric={metric})...")
    model = BERTRelationPrototypeClassifier(
        backbone_name=backbone,
        num_classes=num_classes,
        metric=metric,
        max_seq_length=max_seq_length,
        device=device,
        relation_to_class_idx=relation_to_class_idx,
        class_idx_to_rel_id=class_idx_to_rel_id,
    )

    # Execute experiment
    summary = run_nearest_prototype_experiment(
        config=config,
        tasks=tasks,
        model=model,
        results_dir=results_dir,
        cache_path=cache_path,
        use_disk_cache=use_disk_cache,
    )

    print("\n==================================================================")
    print(f"E004 Nearest Prototype Execution Completed Successfully!")
    print(f"  Final Average Accuracy (AA_{len(tasks)}): {summary.get('final_average_accuracy', 0.0) * 100:.2f}%")
    print(f"  Final Macro-F1:                     {summary.get('final_macro_f1', 0.0) * 100:.2f}%")
    print(f"  Average Incremental Accuracy (AIA): {summary.get('average_incremental_accuracy', 0.0) * 100:.2f}%")
    print(f"  Average Forgetting (AF):            {summary.get('average_forgetting', 0.0) * 100:.2f}%")
    print(f"  Backward Transfer (BWT):            {summary.get('backward_transfer', 0.0) * 100:.2f}%")
    print(f"  Results saved in:                   {results_dir}")
    print("==================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
