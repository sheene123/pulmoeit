"""Classical one-step linear reconstruction (Tikhonov / NOSER prior) on the pixel grid.

This is the family of algorithms embedded in commercial EIT monitors (GREIT-like) and
the baseline every learned method has to beat.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .fem import ForwardModel
from .grid import GRID, image_mask, pixel_centres
from .mesh import build_mesh
from .metrics import pearson


@dataclass
class LinearReconstructor:
    matrix: np.ndarray  # (GRID*GRID, n_meas), sign and gain included
    lam: float
    prior: str

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x @ self.matrix.T).reshape(-1, GRID, GRID).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, matrix=self.matrix.astype(np.float32), lam=self.lam, prior=self.prior)

    @classmethod
    def load(cls, path: Path) -> LinearReconstructor:
        with np.load(path) as f:
            return cls(f["matrix"], float(f["lam"]), str(f["prior"]))


def pixel_jacobian(n_boundary: int = 160, contact_impedance: float = 0.01) -> np.ndarray:
    """Sensitivity of normalised difference data to pixel conductivity changes,
    linearised around a homogeneous reference thorax. Shape (n_meas, n_mask_pixels)."""
    mesh = build_mesh(n_boundary=n_boundary)
    v0, jac = ForwardModel(mesh, contact_impedance).jacobian(np.ones(len(mesh.elements)))
    jac = jac / np.abs(v0)[:, None]
    mask = image_mask().ravel()
    centres = pixel_centres()[:, :, 0].reshape(-1, 2)[mask]
    _, pixel_of_elem = cKDTree(centres).query(mesh.centroids)
    jp = np.zeros((jac.shape[0], mask.sum()))
    np.add.at(jp.T, pixel_of_elem, jac.T)
    return jp


def fit_linear(
    x_val: np.ndarray,
    y_val: np.ndarray,
    lambdas: list[float],
    prior: str = "noser",
    n_boundary: int = 160,
) -> tuple[LinearReconstructor, dict]:
    """Pick the regularisation weight on validation data and calibrate the image gain."""
    jp = pixel_jacobian(n_boundary)
    jtj = jp.T @ jp
    diag = np.diag(jtj)
    if prior == "noser":
        gamma = np.diag(diag + 1e-3 * diag.mean())
    elif prior == "tikhonov":
        gamma = np.eye(len(diag)) * diag.mean()
    else:
        raise ValueError(f"unknown prior {prior!r}")

    mask = image_mask().ravel()
    targets = y_val.reshape(len(y_val), -1)[:, mask]
    best, scores = None, {}
    for lam in lambdas:
        r = -np.linalg.solve(jtj + lam * gamma, jp.T)  # ventilation = conductivity drop
        pred = x_val @ r.T
        gain = float((pred * targets).sum() / (pred * pred).sum())
        mse = float(np.mean((gain * pred - targets) ** 2))
        scores[lam] = mse
        if best is None or mse < best[0]:
            best = (mse, lam, gain * r)
    _, lam, r = best
    full = np.zeros((GRID * GRID, jp.shape[0]))
    full[mask] = r
    recon = LinearReconstructor(full, lam, prior)
    report = {
        "lambda": lam,
        "val_mse": scores[lam],
        "val_corr": float(np.mean(pearson(recon(x_val), y_val))),
        "lambda_scan": {f"{k:g}": v for k, v in scores.items()},
    }
    return recon, report
