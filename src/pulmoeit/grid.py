"""Pixel grid of the reconstructed images (numpy only)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .geometry import REF_ASPECT, REFERENCE

GRID = 32


@lru_cache
def pixel_centres(n: int = GRID, supersample: int = 1) -> np.ndarray:
    """Pixel centre coordinates, shape (n, n, supersample**2, 2).

    Row 0 is anterior (+y), column 0 is the patient's right (-x).
    """
    step = 2.0 / n
    sub = (np.arange(supersample) + 0.5) / supersample
    xs = -1.0 + (np.arange(n)[:, None] + sub[None, :]) * step  # (n, s)
    ys = 1.0 - (np.arange(n)[:, None] + sub[None, :]) * step
    X = np.broadcast_to(xs[None, :, None, :], (n, n, supersample, supersample))
    Y = np.broadcast_to(ys[:, None, :, None], (n, n, supersample, supersample))
    return np.stack([X, Y], axis=-1).reshape(n, n, supersample**2, 2)


@lru_cache
def image_mask(n: int = GRID) -> np.ndarray:
    centres = pixel_centres(n)[:, :, 0].reshape(-1, 2)
    return REFERENCE.contains(centres).reshape(n, n)


@lru_cache
def row_depth(n: int = GRID) -> np.ndarray:
    """Ventro-dorsal position of each pixel row, 0 (sternum) to 100 (spine) percent."""
    y = pixel_centres(n)[:, 0, 0, 1]
    return (REF_ASPECT - y) / (2 * REF_ASPECT) * 100.0
