from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class CheckpointManager:
    """Manages saving and loading of model checkpoints across continual learning stages."""

    def __init__(self, checkpoint_base_dir: Path | str) -> None:
        self.base_dir = Path(checkpoint_base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_stage_dir(self, stage: int) -> Path:
        """Get checkpoint directory for stage t (1-indexed name: after_T{stage + 1})."""
        stage_dir = self.base_dir / f"after_T{stage + 1}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir

    def save_checkpoint(
        self,
        stage: int,
        model: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save model checkpoint and stage metadata."""
        stage_dir = self.get_stage_dir(stage)
        meta_file = stage_dir / "stage_metadata.json"

        meta = dict(metadata or {})
        meta["stage"] = stage
        meta["task_name"] = f"T{stage + 1}"

        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Save model weights
        if hasattr(model, "get_state_dict"):
            state = model.get_state_dict()
            if HAS_TORCH and any(isinstance(v, (torch.Tensor, dict)) for v in state.values()):
                torch_file = stage_dir / "model.pt"
                torch.save(state, torch_file)
            else:
                json_file = stage_dir / "model_state.json"
                with json_file.open("w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)

        return stage_dir

    def load_checkpoint(self, stage: int, model: Any) -> dict[str, Any]:
        """Load model state and return stage metadata."""
        stage_dir = self.get_stage_dir(stage)
        meta_file = stage_dir / "stage_metadata.json"
        meta: dict[str, Any] = {}

        if meta_file.exists():
            with meta_file.open("r", encoding="utf-8") as f:
                meta = json.load(f)

        torch_file = stage_dir / "model.pt"
        json_file = stage_dir / "model_state.json"

        if torch_file.exists() and HAS_TORCH:
            state = torch.load(torch_file, map_location="cpu")
            if hasattr(model, "load_state_dict"):
                model.load_state_dict(state)
        elif json_file.exists():
            with json_file.open("r", encoding="utf-8") as f:
                state = json.load(f)
            if hasattr(model, "load_state_dict"):
                model.load_state_dict(state)

        return meta
