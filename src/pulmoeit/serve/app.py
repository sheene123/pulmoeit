"""Inference API: ONNX Runtime reconstruction + bedside indices + measurement QC + drift.

Depends only on numpy / onnxruntime / fastapi so the serving image stays small.
"""

from __future__ import annotations

import json
import os
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..grid import image_mask
from ..metrics import QUADRANTS, center_of_ventilation, global_inhomogeneity, regional_distribution
from ..monitoring import MeasurementQC, drift_report
from ..protocol import n_measurements
from ..signals import features

N_MEAS = n_measurements()
MIN_DRIFT_WINDOW = 50


class Frames(BaseModel):
    v_ref: list[float] = Field(min_length=N_MEAS, max_length=N_MEAS, description="end-expiration frame (V)")
    v_insp: list[float] = Field(min_length=N_MEAS, max_length=N_MEAS, description="end-inspiration frame (V)")


class Runtime:
    def __init__(self, model_dir: Path):
        self.session = ort.InferenceSession(str(model_dir / "pulmoeit.onnx"), providers=["CPUExecutionProvider"])
        self.qc = MeasurementQC.load(model_dir / "qc.npz")
        card = model_dir / "model_card.json"
        self.card = json.loads(card.read_text()) if card.exists() else {"version": "unknown"}
        self.recent_scores: deque[float] = deque(maxlen=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rt = Runtime(Path(os.getenv("PULMOEIT_MODEL_DIR", "models")))
    yield


app = FastAPI(title="PulmoEIT", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": app.state.rt.card.get("version")}


@app.get("/v1/model")
def model_card() -> dict:
    return app.state.rt.card


@app.post("/v1/reconstruct")
def reconstruct(frames: Frames) -> dict:
    rt: Runtime = app.state.rt
    v_ref = np.asarray(frames.v_ref)
    if not np.all(np.isfinite(v_ref)) or np.min(np.abs(v_ref)) < 1e-9:
        raise HTTPException(422, "v_ref contains zero or non-finite values (open electrode?)")
    x = features(v_ref, frames.v_insp)
    if not np.all(np.isfinite(x)):
        raise HTTPException(422, "non-finite measurements")

    score = float(rt.qc.score(x)[0])
    rt.recent_scores.append(score)
    flagged = score > rt.qc.threshold
    image = rt.session.run(None, {"dv": x})[0]
    regions = regional_distribution(image)[0]
    return {
        "model_version": rt.card.get("version"),
        "image": [
            [round(float(v), 5) if inside else None for v, inside in zip(row, mrow, strict=True)]
            for row, mrow in zip(image[0], image_mask(), strict=True)
        ],
        "indices": {
            "global_inhomogeneity": _num(global_inhomogeneity(image)[0]),
            "center_of_ventilation_pct": _num(center_of_ventilation(image)[0]),
            "regional_distribution_pct": {q: _num(v) for q, v in zip(QUADRANTS, regions, strict=True)},
        },
        "qc": {
            "score": score,
            "threshold": rt.qc.threshold,
            "flagged": bool(flagged),
            "suspect_electrode": int(rt.qc.suspect_electrode(x)[0]) if flagged else None,
        },
    }


@app.get("/v1/monitoring/drift")
def drift() -> dict:
    rt: Runtime = app.state.rt
    if len(rt.recent_scores) < MIN_DRIFT_WINDOW:
        return {"status": "insufficient_data", "n_current": len(rt.recent_scores), "min_required": MIN_DRIFT_WINDOW}
    return {"status": "ok", **drift_report(rt.qc.reference_scores, np.array(rt.recent_scores))}


def _num(v: float) -> float | None:
    return None if not np.isfinite(v) else round(float(v), 4)
