"""Signal preprocessing shared by training and serving (numpy only)."""

from __future__ import annotations

import numpy as np


def features(v_ref: np.ndarray, v_insp: np.ndarray) -> np.ndarray:
    """Normalised difference data (v_insp - v_ref) / |v_ref|, the standard input of
    time-difference EIT. Insensitive to global gain and to most contact effects."""
    v_ref = np.atleast_2d(np.asarray(v_ref, dtype=np.float64))
    v_insp = np.atleast_2d(np.asarray(v_insp, dtype=np.float64))
    return ((v_insp - v_ref) / np.abs(v_ref)).astype(np.float32)
