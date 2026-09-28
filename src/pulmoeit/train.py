"""Training with early stopping, MLflow tracking and measurement-QC fitting."""

from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from . import dataset
from .grid import image_mask
from .metrics import pearson
from .models import build_model, n_parameters, predict, save_checkpoint
from .monitoring import MeasurementQC
from .recon import LinearReconstructor
from .signals import features


def _mlflow_run(params: dict):
    if os.getenv("PULMOEIT_NO_MLFLOW"):
        return nullcontext(None)
    try:
        import mlflow
    except ImportError:
        return nullcontext(None)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("pulmoeit")
    run = mlflow.start_run(tags={"git_commit": _git_commit()})
    mlflow.log_params(_flatten(params))
    return run


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def train(params: dict, data_dir: Path, model_dir: Path, report_dir: Path) -> dict:
    cfg, mcfg = params["train"], params["model"]
    torch.manual_seed(params["seed"])
    # All logical cores oversubscribe small conv nets badly (25x slower under WSL2).
    torch.set_num_threads(cfg.get("threads") or max(1, min(8, (os.cpu_count() or 2) // 2)))
    rng = np.random.default_rng(params["seed"])

    tr, va = dataset.load(data_dir / "train.npz"), dataset.load(data_dir / "val.npz")
    x_tr, y_tr = features(tr["v_ref"], tr["v_insp"]), tr["target"]
    x_va, y_va = features(va["v_ref"], va["v_insp"]), va["target"]
    baseline = LinearReconstructor.load(model_dir / "baseline.npz")

    kwargs = dict(mcfg.get("kwargs") or {})
    model = build_model(
        mcfg["name"],
        mask=image_mask(),
        mean=x_tr.mean(0),
        std=x_tr.std(0) + 1e-6,
        recon_matrix=baseline.matrix,
        **kwargs,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    x_scale = torch.from_numpy(x_tr.std(0))
    history, best = [], (np.inf, None, -1)
    t0 = time.perf_counter()

    with _mlflow_run(params) as run:
        for epoch in range(cfg["epochs"]):
            model.train()
            perm = torch.from_numpy(rng.permutation(len(xt)))
            losses = []
            for i in range(0, len(perm), cfg["batch_size"]):
                idx = perm[i : i + cfg["batch_size"]]
                xb = xt[idx]
                if cfg.get("input_noise", 0) > 0:
                    xb = xb + cfg["input_noise"] * x_scale * torch.randn_like(xb)
                loss = torch.mean((model(xb) - yt[idx]) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            sched.step()
            pred = predict(model, x_va)
            val_loss = float(np.mean((pred - y_va) ** 2))
            val_corr = float(np.mean(pearson(pred, y_va)))
            history.append(
                {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_corr": val_corr}
            )
            print(f"epoch {epoch:3d}  train {history[-1]['train_loss']:.5f}  val {val_loss:.5f}  corr {val_corr:.3f}")
            if run is not None:
                import mlflow

                mlflow.log_metrics({k: v for k, v in history[-1].items() if k != "epoch"}, step=epoch)
            if val_loss < best[0]:
                best = (val_loss, {k: v.clone() for k, v in model.state_dict().items()}, epoch)
            elif epoch - best[2] >= cfg["patience"]:
                break

        model.load_state_dict(best[1])
        save_checkpoint(model_dir / "model.pt", model, mcfg["name"], kwargs)
        qc = MeasurementQC.fit(x_tr, x_va)
        qc.save(model_dir / "qc.npz")

        summary = {
            "model": mcfg["name"],
            "n_parameters": n_parameters(model),
            "best_epoch": best[2],
            "val_loss": best[0],
            "val_corr": history[best[2]]["val_corr"],
            "train_seconds": round(time.perf_counter() - t0, 1),
            "qc_components": int(len(qc.components)),
            "qc_threshold": qc.threshold,
        }
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "train.json").write_text(json.dumps(summary, indent=2))
        (report_dir / "train_history.json").write_text(json.dumps(history, indent=2))
        if run is not None:
            import mlflow

            mlflow.log_metrics({"best_val_loss": best[0], "n_parameters": summary["n_parameters"]})
            mlflow.log_artifact(str(model_dir / "model.pt"))
    return summary
