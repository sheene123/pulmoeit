"""Indices cliniques d'un essai de PEEP en EIT (numpy seulement).

- Méthode de Costa et al. (Intensive Care Med 2009;35:1132-1137) : lors d'un essai de PEEP
  décroissante, la compliance de chaque pixel vaut ΔZ / ΔP ; on en déduit, à chaque palier, la part
  de poumon collabée (perte de compliance sous la PEEP où le pixel était le meilleur) et la part
  surdistendue (perte au-dessus) ; la PEEP proposée est le croisement des deux courbes.
- Espaces silencieux (Ukere et al., Intensive Care Med 2016 ; Spadaro et al., Crit Care 2018) :
  pixels du poumon qui ventilent moins de 10 % du maximum, séparés en dépendants (dos) et non
  dépendants (avant) par la ligne horizontale du centre de ventilation.

Le modèle de poumon des essais simulés est une courbe pression-volume sigmoïde par région
(Venegas et al., J Appl Physiol 1998), dont la pression d'ouverture augmente vers le dos (poids
du poumon sus-jacent) : les régions dorsales s'ouvrent quand la PEEP monte, les régions ventrales
se distendent.
"""

from __future__ import annotations

import numpy as np

from .grid import GRID, image_mask, row_depth

SEUIL_ROI = 0.2  # pixels « poumon fonctionnel » : compliance max ≥ 20 % du maximum de l'image
SEUIL_SILENCE = 0.1  # espaces silencieux : ventilation < 10 % du maximum
SEUIL_CLASSE = 0.5  # pixel collabé / surdistendu : plus de la moitié de sa meilleure compliance perdue


# ---------------------------------------------------------------- physiologie simulée


def volume_sigmoide(p: np.ndarray, p_inflexion: np.ndarray, pente: float) -> np.ndarray:
    """Fraction de volume d'une région (0 = collabée, 1 = pleinement distendue) à la pression p."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(p, dtype=float) - p_inflexion) / pente))


def ventilation_regionale(peep: float, delta_p: float, p_inflexion: np.ndarray, pente: float) -> np.ndarray:
    """Volume courant relatif de chaque région pour une PEEP et une pression motrice données."""
    return volume_sigmoide(peep + delta_p, p_inflexion, pente) - volume_sigmoide(peep, p_inflexion, pente)


# ---------------------------------------------------------------- méthode de Costa


def _roi(compliances: np.ndarray, masque: np.ndarray | None) -> np.ndarray:
    best = compliances.max(axis=0)
    roi = best >= SEUIL_ROI * best.max() if best.max() > 0 else np.zeros_like(best, bool)
    return roi & (image_mask() if masque is None else masque)


def costa(images: np.ndarray, peeps: np.ndarray, delta_p: np.ndarray, masque: np.ndarray | None = None) -> dict:
    """images : (n_paliers, GRID, GRID) images courantes (ΔZ) ; peeps, delta_p : (n_paliers,).

    Retourne les courbes de collapsus et de surdistension (en % de la compliance totale), la PEEP
    proposée (croisement, interpolé linéairement) et la carte des pixels utilisés. Invariant à un
    facteur d'échelle global de l'image : chaque méthode peut avoir ses propres unités."""
    peeps = np.asarray(peeps, dtype=float)
    c = np.clip(np.asarray(images, dtype=float), 0, None) / np.asarray(delta_p, dtype=float)[:, None, None]
    roi = _roi(c, masque)
    c = c[:, roi]  # (n_paliers, n_pixels)
    best = c.max(axis=0)
    peep_best = peeps[np.argmax(c, axis=0)]
    perte = np.where(best > 0, (best - c) / np.where(best > 0, best, 1), 0.0)  # (n, p) dans [0, 1]
    sous = peeps[:, None] < peep_best[None]
    poids = best / max(best.sum(), 1e-12)
    collapsus = 100 * (perte * sous * poids).sum(axis=1)
    surdistension = 100 * (perte * (peeps[:, None] > peep_best[None]) * poids).sum(axis=1)
    # classe de chaque pixel à chaque palier : 0 hors poumon fonctionnel, 1 normal, 2 collabé,
    # 3 surdistendu (plus de la moitié de sa meilleure compliance perdue, sous ou au-dessus de sa
    # meilleure PEEP)
    classes = np.zeros((len(peeps), GRID, GRID), np.int8)
    perdu = perte > SEUIL_CLASSE
    classes[:, roi] = np.where(perdu & sous, 2, np.where(perdu & ~sous & (peeps[:, None] != peep_best[None]), 3, 1))
    return {
        "collapsus": collapsus,
        "surdistension": surdistension,
        "peep_proposee": croisement(peeps, collapsus, surdistension),
        "roi": roi,
        "classes": classes,
    }


def croisement(peeps: np.ndarray, collapsus: np.ndarray, surdistension: np.ndarray) -> float:
    """PEEP où le collapsus (qui baisse quand la PEEP monte) croise la surdistension (qui monte)."""
    ordre = np.argsort(peeps)
    p, d = peeps[ordre], (collapsus - surdistension)[ordre]
    for i in range(len(p) - 1):
        if d[i] >= 0 >= d[i + 1]:
            return float(p[i] + (p[i + 1] - p[i]) * d[i] / (d[i] - d[i + 1])) if d[i] != d[i + 1] else float(p[i])
    return float(p[np.argmin(np.abs(d))])  # pas de croisement dans l'essai : palier le plus proche


# ---------------------------------------------------------------- espaces silencieux


def espaces_silencieux(image: np.ndarray, poumon: np.ndarray) -> dict:
    """Part (%) du poumon qui ventile < 10 % du maximum, dépendante (dos) et non dépendante (avant).

    `poumon` : masque anatomique du poumon sur la grille d'image (connu en simulation ; sur un
    patient, il vient d'un scanner ou d'un contour estimé)."""
    v = np.clip(np.asarray(image, dtype=float), 0, None) * image_mask()
    if v.max() <= 0 or not poumon.any():
        return {"dependants": float("nan"), "non_dependants": float("nan")}
    silence = poumon & (v < SEUIL_SILENCE * v.max())
    profondeur = np.broadcast_to(row_depth()[:, None], (GRID, GRID))
    cov = float((v.sum(axis=1) @ row_depth()) / v.sum())  # centre de ventilation, 0 % avant, 100 % dos
    n = poumon.sum()
    return {
        "dependants": float(100 * (silence & (profondeur > cov)).sum() / n),
        "non_dependants": float(100 * (silence & (profondeur <= cov)).sum() / n),
    }
