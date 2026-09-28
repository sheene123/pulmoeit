"""Thorax boundary shapes (numpy only, safe to import from the serving image)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Antero-posterior / lateral ratio of the reference thorax used for imaging.
REF_ASPECT = 0.72


@dataclass(frozen=True)
class Geometry:
    """Star-shaped thorax cross-section.

    r(theta) = ellipse(theta; 1, aspect) * (1 + sum_k a_k cos(k theta) + b_k sin(k theta)),
    with k starting at 2 (k = 1 would only translate the shape).
    +y is anterior (sternum), +x is the patient's left (radiological convention).
    """

    aspect: float = REF_ASPECT
    fourier: tuple[tuple[float, float], ...] = ()

    def radius(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        b = self.aspect
        r = b / np.sqrt((b * np.cos(theta)) ** 2 + np.sin(theta) ** 2)
        scale = np.ones_like(theta)
        for k, (ak, bk) in enumerate(self.fourier, start=2):
            scale = scale + ak * np.cos(k * theta) + bk * np.sin(k * theta)
        return r * scale

    def point(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        return self.radius(theta)[:, None] * np.c_[np.cos(theta), np.sin(theta)]

    def contains(self, points: np.ndarray) -> np.ndarray:
        theta = np.arctan2(points[:, 1], points[:, 0])
        return np.hypot(points[:, 0], points[:, 1]) < self.radius(theta)

    def map_to(self, points: np.ndarray, other: Geometry) -> np.ndarray:
        """Radially map points of this thorax onto `other` (anatomy-preserving warp)."""
        theta = np.arctan2(points[:, 1], points[:, 0])
        return points * (other.radius(theta) / self.radius(theta))[:, None]


REFERENCE = Geometry(REF_ASPECT)
