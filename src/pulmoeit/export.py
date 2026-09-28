"""Export to ONNX (edge runtime), parity check and model card generation."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .models import load_checkpoint, n_parameters
from .protocol import n_measurements
from .train import _git_commit

ONNX_NAME = "pulmoeit.onnx"


def export(params: dict, model_dir: Path, report_dir: Path) -> dict:
    model = load_checkpoint(model_dir / "model.pt")
    onnx_path = model_dir / ONNX_NAME
    dummy = torch.zeros(2, n_measurements())
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["dv"],
        output_names=["image"],
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=4096)},),
        dynamo=True,
        external_data=False,
        verbose=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(0, 0.05, (8, n_measurements())).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    max_err = float(np.abs(sess.run(None, {"dv": x})[0] - ref).max())
    if max_err > 1e-4:
        raise RuntimeError(f"ONNX parity check failed: max abs error {max_err:.2e}")

    metrics_path = report_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    card = {
        "name": "pulmoeit",
        "version": _git_commit(),
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "architecture": params["model"]["name"],
        "n_parameters": n_parameters(model),
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "onnx_parity_max_abs_error": max_err,
        "input": {
            "name": "dv",
            "shape": ["batch", n_measurements()],
            "description": "(v_insp - v_ref) / |v_ref|, 16 electrodes, adjacent protocol",
        },
        "output": {
            "name": "image",
            "shape": ["batch", 32, 32],
            "description": "tidal ventilation image, row 0 anterior, column 0 patient right",
        },
        "training_data": {
            "source": "simulation (CEM FEM)",
            "acquisition": params["data"]["acquisition"],
            "seed": params["seed"],
        },
        "evaluation": {k: metrics.get(k) for k in ("test", "test_shift", "qc", "gates")},
        "intended_use": "Research prototype. Not a medical device, not for clinical decisions.",
        "limitations": [
            "2D model of a 3D current flow; trained on simulations only (sim-to-real gap not yet measured).",
            "Single protocol (16 electrodes, adjacent drive).",
            "Clinical indices not validated against CT or clinical reference.",
        ],
    }
    (model_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    return card
