from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None


@dataclass(slots=True)
class LogRecord:
    step: int
    metrics: dict[str, float]


class TrainingMonitor:
    def __init__(self, output_dir: str | Path, tensorboard: bool = True, csv_logging: bool = True, wandb_enabled: bool = False, wandb_project: str = "llmcore", wandb_run_name: str = "") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()
        self.tensorboard = SummaryWriter(str(self.output_dir / "tensorboard")) if tensorboard and SummaryWriter is not None else None
        self.csv_file = self.output_dir / "metrics.csv" if csv_logging else None
        self.csv_handle = self.csv_file.open("a", newline="", encoding="utf-8") if self.csv_file is not None else None
        self.csv_writer = None
        if self.csv_handle is not None:
            self.csv_writer = csv.DictWriter(self.csv_handle, fieldnames=["step", "metric", "value"])
            if self.csv_file.stat().st_size == 0:
                self.csv_writer.writeheader()
        self.wandb = None
        if wandb_enabled:
            try:
                import wandb

                self.wandb = wandb.init(project=wandb_project, name=wandb_run_name or None, dir=str(self.output_dir), reinit=True)
            except Exception:
                self.wandb = None

    def log(self, step: int, metrics: dict[str, float]) -> None:
        if self.tensorboard is not None:
            for key, value in metrics.items():
                self.tensorboard.add_scalar(key, value, step)
        if self.csv_writer is not None:
            for key, value in metrics.items():
                self.csv_writer.writerow({"step": step, "metric": key, "value": value})
            self.csv_handle.flush()
        if self.wandb is not None:
            self.wandb.log({**metrics, "step": step}, step=step)

    def report_step(self, step: int, metrics: dict[str, float]) -> None:
        table = Table(title=f"Step {step}")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for key, value in metrics.items():
            table.add_row(key, f"{value:.6f}")
        self.console.print(table)

    def close(self) -> None:
        if self.tensorboard is not None:
            self.tensorboard.close()
        if self.csv_handle is not None:
            self.csv_handle.close()
        if self.wandb is not None:
            self.wandb.finish()
