"""Experiment tracking: MLflow (local file store), W&B offline, or nothing.

Every run also appends metrics to ``<output_dir>/metrics.jsonl`` so results survive without any
tracking backend installed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from toolforge.config import TrackingConfig


class Tracker:
    def __init__(self, cfg: TrackingConfig, run_name: str, output_dir: str | Path) -> None:
        self.cfg = cfg
        self.backend = cfg.backend
        self.local = Path(output_dir) / "metrics.jsonl"
        self.local.parent.mkdir(parents=True, exist_ok=True)
        self._mlflow: Any = None
        self._wandb: Any = None
        if self.backend == "mlflow":
            try:
                import mlflow

                mlflow.set_tracking_uri(cfg.mlflow_uri)
                mlflow.set_experiment(cfg.experiment)
                mlflow.start_run(run_name=run_name)
                self._mlflow = mlflow
            except Exception as exc:  # tracking must never break training
                print(f"[toolforge] mlflow unavailable ({exc}); logging to {self.local} only")
                self.backend = "none"
        elif self.backend == "wandb":
            try:
                import wandb

                self._wandb = wandb.init(project=cfg.wandb_project, name=run_name, mode=cfg.wandb_mode)
            except Exception as exc:
                print(f"[toolforge] wandb unavailable ({exc}); logging to {self.local} only")
                self.backend = "none"

    def log_params(self, params: dict[str, Any]) -> None:
        flat = _flatten(params)
        if self._mlflow is not None:
            for k, v in flat.items():
                self._mlflow.log_param(k[:250], str(v)[:500])
        if self._wandb is not None:
            self._wandb.config.update(flat, allow_val_change=True)
        self._write({"params": flat})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if self._mlflow is not None:
            self._mlflow.log_metrics({_safe(k): v for k, v in clean.items()}, step=step)
        if self._wandb is not None:
            self._wandb.log(clean, step=step)
        self._write({"step": step, "metrics": clean})

    def log_artifact(self, path: str | Path) -> None:
        if self._mlflow is not None and Path(path).exists():
            self._mlflow.log_artifact(str(path))

    def close(self) -> None:
        if self._mlflow is not None:
            self._mlflow.end_run()
        if self._wandb is not None:
            self._wandb.finish()

    def _write(self, row: dict[str, Any]) -> None:
        row = {"time": time.time(), **row}
        with self.local.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


def _safe(key: str) -> str:
    return "".join(c if c.isalnum() or c in "_-./ " else "_" for c in key)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def hf_callback(tracker: Tracker) -> Any:
    """A transformers TrainerCallback forwarding trainer logs to ``tracker``."""
    from transformers import TrainerCallback

    class _Cb(TrainerCallback):  # type: ignore[misc]
        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kw: Any) -> None:
            if logs:
                tracker.log_metrics(logs, step=state.global_step)

    return _Cb()
