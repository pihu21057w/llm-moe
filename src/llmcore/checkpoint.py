from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, scheduler: Any | None, step: int, config: dict, extra_state: dict[str, Any] | None = None, save_optimizer_state: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None and save_optimizer_state else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra_state": extra_state or {},
    }
    torch.save(state, path)


def load_checkpoint(path: str | Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, scheduler: Any | None = None, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if checkpoint.get("rng_state") is not None:
        torch.set_rng_state(checkpoint["rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return checkpoint


def write_config(path: str | Path, config: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
