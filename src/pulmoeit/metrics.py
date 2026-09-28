"""Image metrics and bedside ventilation indices (numpy only).

Clinical indices follow the TREND consensus (Frerichs et al., Thorax 2017):
- GI: global inhomogeneity index (Zhao et al., Intensive Care Med 2009)
- CoV: ventro-dorsal centre of ventilation, 0 % = sternum, 100 % = spine
- regional distribution: share of tidal ventilation per quadrant
"""

from __future__ import annotations

import numpy as np

from .grid import GRID, image_mask, row_depth

QUADRANTS = ("ventral_right", "ventral_left", "dorsal_right", "dorsal_left")


def _flat(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    return img.reshape(-1, GRID * GRID)[:, image_mask().ravel()]


def pearson(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    p, t = _flat(pred), _flat(target)
    p = p - p.mean(1, keepdims=True)
    t = t - t.mean(1, keepdims=True)
    den = np.sqrt((p * p).sum(1) * (t * t).sum(1))
    return np.divide((p * t).sum(1), den, out=np.zeros(len(p)), where=den > 0)


def rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((_flat(pred) - _flat(target)) ** 2, axis=1))


def global_inhomogeneity(img: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """GI = sum |DV - median(DV)| / sum DV over the functional lung region."""
    out = []
    for v in _flat(img):
        roi = v > threshold * v.max() if v.max() > 0 else np.zeros_like(v, bool)
        dv = v[roi]
        out.append(np.abs(dv - np.median(dv)).sum() / dv.sum() if dv.size else np.nan)
    return np.array(out)


def center_of_ventilation(img: np.ndarray) -> np.ndarray:
    img = np.clip(np.asarray(img, dtype=np.float64).reshape(-1, GRID, GRID), 0, None) * image_mask()
    rows = img.sum(axis=2)
    total = rows.sum(axis=1)
    return np.divide(rows @ row_depth(), total, out=np.full(len(img), np.nan), where=total > 0)


def regional_distribution(img: np.ndarray) -> np.ndarray:
    """Percent of tidal ventilation in each quadrant, shape (n, 4), see QUADRANTS."""
    img = np.clip(np.asarray(img, dtype=np.float64).reshape(-1, GRID, GRID), 0, None) * image_mask()
    h = GRID // 2
    q = np.stack([img[:, :h, :h], img[:, :h, h:], img[:, h:, :h], img[:, h:, h:]], axis=1).sum(axis=(2, 3))
    total = q.sum(axis=1, keepdims=True)
    return np.divide(100 * q, total, out=np.full_like(q, np.nan), where=total > 0)


def summarize(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.mean(rmse(pred, target))),
        "corr": float(np.mean(pearson(pred, target))),
        "gi_mae": float(np.nanmean(np.abs(global_inhomogeneity(pred) - global_inhomogeneity(target)))),
        "cov_mae": float(np.nanmean(np.abs(center_of_ventilation(pred) - center_of_ventilation(target)))),
        "regional_mae": float(np.nanmean(np.abs(regional_distribution(pred) - regional_distribution(target)))),
    }


def roc_auc(scores_neg: np.ndarray, scores_pos: np.ndarray) -> float:
    """Probability that a positive sample scores higher than a negative one."""
    s = np.concatenate([scores_neg, scores_pos])
    ranks = np.empty(len(s))
    ranks[np.argsort(s)] = np.arange(1, len(s) + 1)
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    return float((ranks[n_neg:].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
