"""Score officiel du KTC2023 : moyenne des SSIM des classes « conducteur » et « résistif ».

`score_officiel` appelle le code des organisateurs (convolutions directes, lent) ; `score_rapide`
calcule la même chose par FFT, pour la validation pendant l'entraînement (écart vérifié dans les
tests).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

_C1, _C2, _R = 1e-4, 9e-4, 80


def _noyau() -> np.ndarray:
    w = np.arange(-np.ceil(2 * _R), np.ceil(2 * _R) + 1)
    X, Y = np.meshgrid(w, w)
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (X**2 + Y**2) / _R**2)


_NOYAU = _noyau()
_CORRECTION = fftconvolve(np.ones((256, 256)), _NOYAU, mode="same")


def _flou(x: np.ndarray) -> np.ndarray:
    return fftconvolve(x, _NOYAU, mode="same") / _CORRECTION


def _ssim(verite: np.ndarray, reco: np.ndarray) -> float:
    gt, gr = _flou(verite), _flou(reco)
    st = _flou(verite**2) - gt**2
    sr = _flou(reco**2) - gr**2
    str_ = _flou(verite * reco) - gt * gr
    carte = ((2 * gt * gr + _C1) * (2 * str_ + _C2)) / ((gt**2 + gr**2 + _C1) * (st + sr + _C2))
    return float(np.mean(carte))


def score_rapide(verite: np.ndarray, reco: np.ndarray) -> float:
    """Score officiel (0 à 1 par cuve) calculé par FFT ; images de classes 256 × 256."""
    if reco.shape != (256, 256):
        return 0.0
    v, r = verite.astype(float), reco.astype(float)
    return 0.5 * (_ssim((v == 2) * 1.0, (r == 2) * 1.0) + _ssim((v == 1) * 1.0, (r == 1) * 1.0))


def score_officiel(verite: np.ndarray, reco: np.ndarray) -> float:
    """Fonction de score des organisateurs (code KTCScoring importable)."""
    import KTCScoring

    return float(KTCScoring.scoringFunction(verite, reco))
