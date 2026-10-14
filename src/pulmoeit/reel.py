"""Vrais enregistrements EIT : lecture, conventions, respirations, cycle moyen (numpy et scipy).

Appareil : Goe-MF II (Viasys), 16 électrodes, protocole adjacent. Le fichier .get contient 256 réels
par image, dont 208 mesures (16 injections × 13), dans l'ordre « décalé » : pour l'injection d, les
paires de mesure partent de la paire qui suit l'injection. Cet ordre est vérifié pour chaque jeu sur
le profil des tensions (`verifier_ordre`).

Sens des électrodes : l'électrode 1 est sur le sternum ; selon le jeu, l'électrode 5 est du côté
gauche du patient (« gauche ») ou du côté droit (« droite »). Dans PulmoEIT, l'électrode 0 est
devant et la numérotation tourne vers la droite du patient : le sens « droite » correspond donc
directement, le sens « gauche » est un miroir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .protocol import N_ELECTRODES, adjacent_protocol

N_MESURES = N_ELECTRODES * (N_ELECTRODES - 3)


def lire_get(chemin: Path) -> np.ndarray:
    """Fichier .get du Goe-MF II : (n_images, 208) mesures brutes, ordre de l'appareil."""
    brut = np.fromfile(chemin, dtype="<f4")
    return brut.reshape(-1, 256)[:, :N_MESURES].astype(np.float64)


def vers_protocole(mesures: np.ndarray, sens: str = "gauche") -> np.ndarray:
    """Mesures de l'appareil (n, 208) -> ordre et conventions de PulmoEIT (n, 208).

    sens : côté du patient où se trouve l'électrode 5 (« gauche » ou « droite »). Les amplitudes
    de l'appareil sont positives ; les tensions de PulmoEIT sont négatives (le potentiel décroît
    de la source au puits) ; le changement de sens de parcours inverse le signe à l'injection et à
    la mesure, soit aucun changement de signe."""
    if sens not in {"gauche", "droite"}:
        raise ValueError(f"sens inconnu : {sens!r}")
    n = N_ELECTRODES
    appareil = mesures.reshape(len(mesures), n, n - 3)
    _, paires = adjacent_protocol(n)
    sortie = np.empty((len(mesures), len(paires)))
    for k, (d, m) in enumerate(paires):
        d_app, m_app = ((-d - 1) % n, (-m - 1) % n) if sens == "gauche" else (d, m)
        sortie[:, k] = appareil[:, d_app, (m_app - d_app - 2) % n]
    return -sortie


def verifier_ordre(mesures: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Corrélation du profil moyen des tensions avec celui d'un modèle homogène, pour l'ordre
    « décalé » (celui de l'appareil) et pour un ordre par numéro d'électrode. reference : (208,)
    tensions du modèle dans l'ordre de PulmoEIT."""
    n = N_ELECTRODES
    appareil = np.abs(mesures).mean(axis=0).reshape(n, n - 3)
    _, paires = adjacent_protocol(n)
    decale, par_numero = np.empty(len(paires)), np.empty(len(paires))
    for k, (d, m) in enumerate(paires):
        decale[k] = appareil[d, (m - d - 2) % n]
        par_numero[k] = appareil[d, [x for x in range(n) if x not in {(d - 1) % n, d, (d + 1) % n}].index(m)]
    ref = np.abs(reference)
    return {
        "decale": float(np.corrcoef(decale, ref)[0, 1]),
        "par_numero": float(np.corrcoef(par_numero, ref)[0, 1]),
    }


def signal_global(mesures: np.ndarray) -> np.ndarray:
    """Somme des amplitudes : augmente quand l'air entre (l'impédance monte)."""
    return np.abs(mesures).sum(axis=1)


def respirations(signal: np.ndarray, images_par_seconde: float, fenetre_s: float = 4.0) -> list[tuple[int, int]]:
    """(fin d'expiration, fin d'inspiration suivante) pour chaque respiration complète.

    La dérive lente (changement de PEEP, d'air dans la plèvre…) est retirée par une médiane glissante
    avant la détection, pour que les grands sauts ne masquent pas les respirations."""
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks

    largeur = max(3, int(fenetre_s * images_par_seconde) | 1)
    s = signal - median_filter(signal, size=largeur, mode="nearest")
    s = np.convolve(s, np.ones(3) / 3, mode="same")
    ecart = max(2, int(0.5 * images_par_seconde))  # au plus 2 respirations par seconde
    prominence = 0.5 * np.std(s)
    hauts, _ = find_peaks(s, distance=ecart, prominence=prominence)
    bas, _ = find_peaks(-s, distance=ecart, prominence=prominence)
    paires: list[tuple[int, int]] = []
    for b in bas:
        suivants = hauts[hauts > b]
        if suivants.size and (not paires or b > paires[-1][1]):
            paires.append((int(b), int(suivants[0])))
    return paires


def cycle_moyen(mesures: np.ndarray, resp: list[tuple[int, int]], n_phases: int = 16) -> np.ndarray:
    """Cycle respiratoire moyen : chaque respiration (d'une fin d'expiration à la suivante) est
    rééchantillonnée sur n_phases instants, puis les mesures sont moyennées phase par phase.
    Retourne (n_phases, 208) ; la phase 0 est la fin d'expiration."""
    debuts = [b for b, _ in resp]
    cycles = []
    for a, b in zip(debuts[:-1], debuts[1:], strict=True):
        t = np.linspace(a, b, n_phases, endpoint=False)
        i0 = np.floor(t).astype(int)
        f = (t - i0)[:, None]
        cycles.append((1 - f) * mesures[i0] + f * mesures[np.minimum(i0 + 1, len(mesures) - 1)])
    if not cycles:
        raise ValueError("au moins deux respirations sont nécessaires")
    return np.mean(cycles, axis=0)


def bruit_relatif(mesures: np.ndarray, resp: list[tuple[int, int]]) -> float:
    """Niveau de bruit des données moyennées, relatif à l'amplitude respiratoire : écart-type des
    fins d'expiration d'une respiration à l'autre, divisé par la racine du nombre de respirations,
    rapporté à la variation moyenne entre fin d'expiration et fin d'inspiration."""
    fe = np.array([mesures[b] for b, _ in resp])
    fi = np.array([mesures[h] for _, h in resp])
    bruit = np.linalg.norm(fe.std(axis=0)) / np.sqrt(len(resp))
    return float(bruit / max(np.linalg.norm((fi - fe).mean(axis=0)), 1e-30))
