"""Measurement quality control and drift monitoring (numpy only, runs on-device).

A learned reconstruction must know when not to be trusted. We model the subspace of
plausible normalised difference frames with PCA on training data: frames far from it
(detached electrode, motion, unseen anatomy) get a high residual score, and the residual
energy per electrode points at the faulty contact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .protocol import electrode_involvement


@dataclass
class MeasurementQC:
    mean: np.ndarray
    components: np.ndarray  # (k, n_meas), orthonormal rows
    threshold: float
    reference_scores: np.ndarray  # held-out score sample, reference for drift tests

    @classmethod
    def fit(
        cls, x_fit: np.ndarray, x_calib: np.ndarray, variance: float = 0.995, quantile: float = 0.995
    ) -> MeasurementQC:
        """Learn the subspace on `x_fit`; set the alarm threshold and the drift reference
        on held-out `x_calib` (in-sample residuals would be optimistically small)."""
        x = _unit(x_fit)
        mean = x.mean(axis=0)
        _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
        k = int(np.searchsorted(np.cumsum(s**2) / np.sum(s**2), variance) + 1)
        qc = cls(mean, vt[:k], 0.0, np.empty(0))
        scores = qc.score(x_calib)
        qc.threshold = float(np.quantile(scores, quantile))
        rng = np.random.default_rng(0)
        qc.reference_scores = rng.choice(scores, min(len(scores), 2000), replace=False)
        return qc

    def residual(self, x: np.ndarray) -> np.ndarray:
        xc = _unit(x) - self.mean
        return xc - (xc @ self.components.T) @ self.components

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.residual(x), axis=1)

    def suspect_electrode(self, x: np.ndarray) -> np.ndarray:
        inv = electrode_involvement()
        energy = (self.residual(x) ** 2) @ inv / inv.sum(axis=0)
        return energy.argmax(axis=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            components=self.components,
            threshold=self.threshold,
            reference_scores=self.reference_scores,
        )

    @classmethod
    def load(cls, path: Path) -> MeasurementQC:
        with np.load(path) as f:
            return cls(f["mean"], f["components"], float(f["threshold"]), f["reference_scores"])


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def ks_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and asymptotic p-value."""
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    d = float(np.max(np.abs(np.searchsorted(a, grid, "right") / len(a) - np.searchsorted(b, grid, "right") / len(b))))
    ne = len(a) * len(b) / (len(a) + len(b))
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * d
    k = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1) ** (k - 1) * np.exp(-2 * k**2 * lam**2)), 0, 1))
    return d, p


def drift_report(reference: np.ndarray, current: np.ndarray, alpha: float = 0.01) -> dict:
    d, p = ks_test(reference, current)
    return {
        "n_current": int(len(current)),
        "ks_statistic": d,
        "p_value": p,
        "drift": bool(p < alpha),
        "median_score_ratio": float(np.median(current) / np.median(reference)),
    }
