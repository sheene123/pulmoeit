"""Cuves simulées : de 1 à 4 objets résistifs ou conducteurs, de formes variées.

Les objets sont dessinés sur la grille 256 × 256 de la vérité terrain officielle (classes 0 :
eau, 1 : résistif, 2 : conducteur), puis la conductivité est lue aux nœuds du maillage de
simulation. Valeurs de conductivité tirées comme l'équipe gagnante du KTC2023 (Denker et al.,
2024) : résistif 0,025 à 0,125 S/m, conducteur 5 à 6 S/m.
"""

from __future__ import annotations

import numpy as np

from .officiel import COTE_VERITE, RAYON

RESISTIF, CONDUCTEUR = (0.025, 0.125), (5.0, 6.0)


def centres_pixels(cote: int = COTE_VERITE) -> tuple[np.ndarray, np.ndarray]:
    """Coordonnées (m) des centres de pixels, dans la convention du code officiel : ligne 0 en
    haut (y maximal), colonne 0 à gauche (x minimal)."""
    largeur = 2 * RAYON / cote
    c = -RAYON + largeur / 2 + largeur * np.arange(cote)
    return np.meshgrid(c, c[::-1])  # X, Y


def _forme(rng: np.random.Generator, X: np.ndarray, Y: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    """Masque d'un objet de taille caractéristique r centré en (cx, cy)."""
    angle = rng.uniform(0, np.pi)
    u = (X - cx) * np.cos(angle) + (Y - cy) * np.sin(angle)
    v = -(X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)
    type_ = rng.choice(["disque", "ellipse", "rectangle", "polygone", "etoile"], p=[0.25, 0.2, 0.2, 0.2, 0.15])
    if type_ == "disque":
        return u**2 + v**2 <= r**2
    if type_ == "ellipse":
        e = rng.uniform(0.4, 0.9)
        return (u / r) ** 2 + (v / (e * r)) ** 2 <= 1
    if type_ == "rectangle":
        e = rng.uniform(0.3, 1.0)
        return (np.abs(u) <= r) & (np.abs(v) <= e * r)
    # polygone convexe ou forme étoilée : rayon fonction de l'angle
    theta = np.arctan2(v, u)
    rho = np.hypot(u, v)
    n = int(rng.integers(3, 8))
    if type_ == "polygone":
        sommets = np.sort(rng.uniform(0, 2 * np.pi, n))
        rayons = r * rng.uniform(0.7, 1.0, n)
    else:
        sommets = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
        rayons = r * np.where(np.arange(2 * n) % 2 == 0, 1.0, rng.uniform(0.4, 0.7))
    rayon_angle = np.interp(
        theta % (2 * np.pi), np.append(sommets, sommets[0] + 2 * np.pi), np.append(rayons, rayons[0])
    )
    return rho <= rayon_angle


def tirer(rng: np.random.Generator, cote: int = COTE_VERITE) -> np.ndarray:
    """Classes (cote, cote) uint8 d'une cuve tirée au hasard, objets sans chevauchement."""
    X, Y = centres_pixels(cote)
    classes = np.zeros((cote, cote), np.uint8)
    occupe = np.zeros((cote, cote), bool)
    n_objets = int(rng.choice([1, 2, 3, 4], p=[0.25, 0.3, 0.25, 0.2]))
    for _ in range(n_objets):
        for _essai in range(30):
            r = RAYON * rng.uniform(0.15, 0.4)  # rayon équivalent des vrais objets : 0,2 à 0,3
            distance = rng.uniform(0, RAYON * 0.9 - r)
            angle = rng.uniform(0, 2 * np.pi)
            masque = _forme(rng, X, Y, distance * np.cos(angle), distance * np.sin(angle), r)
            masque &= X**2 + Y**2 <= (0.95 * RAYON) ** 2
            marge = masque.copy()
            for decalage in (-3, 3):  # quelques pixels d'eau entre deux objets
                marge |= np.roll(masque, decalage, 0) | np.roll(masque, decalage, 1)
            if masque.sum() > 30 and not (marge & occupe).any():
                classes[masque] = rng.choice([1, 2])
                occupe |= masque
                break
    return classes


def conductivite_noeuds(classes: np.ndarray, noeuds: np.ndarray, fond: float, rng: np.random.Generator) -> np.ndarray:
    """Conductivité (n, 1) aux nœuds du maillage : lue dans la grille des classes, une valeur
    tirée par objet."""
    cote = classes.shape[0]
    largeur = 2 * RAYON / cote
    col = np.clip(((noeuds[:, 0] + RAYON) / largeur).astype(int), 0, cote - 1)
    lig = np.clip(cote - 1 - ((noeuds[:, 1] + RAYON) / largeur).astype(int), 0, cote - 1)
    c = classes[lig, col]
    sigma = np.full(len(noeuds), fond)
    sigma[c == 1] = rng.uniform(*RESISTIF)
    sigma[c == 2] = rng.uniform(*CONDUCTEUR)
    return sigma[:, None]


def reduire(classes: np.ndarray, cote: int) -> np.ndarray:
    """Grille de classes réduite par échantillonnage du centre de chaque bloc."""
    pas = classes.shape[0] // cote
    return classes[pas // 2 :: pas, pas // 2 :: pas]


def agrandir(classes: np.ndarray, cote: int = COTE_VERITE) -> np.ndarray:
    """Grille de classes agrandie au plus proche voisin (pour le score officiel, en 256 × 256)."""
    pas = cote // classes.shape[0]
    return np.repeat(np.repeat(classes, pas, axis=0), pas, axis=1)
