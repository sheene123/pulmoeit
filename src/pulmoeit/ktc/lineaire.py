"""Reconstruction linéaire de différence (celle des organisateurs) mise sous forme matricielle.

Pour un niveau donné, la reconstruction officielle vaut
    delta_sigma = (Jᵀ G J + Lᵀ L)⁻¹ Jᵀ G delta_U[mesures gardées],
avec J la jacobienne au point de linéarisation, G l'inverse de la covariance du bruit et L le
préconditionneur de l'a priori de régularité. Tout est fixe sauf delta_U : on calcule une fois
la matrice R (pixels × 2 356 mesures, zéros sur les mesures retirées) et chaque reconstruction
devient un produit matrice-vecteur, sur GPU pendant l'entraînement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fantomes import centres_pixels
from .officiel import N_ELECTRODES, NIVEAUX, charger_maillage, importer_code, mesures_gardees

COTE_ENTREE = 128  # grille de l'image d'entrée et de sortie du réseau (plafond du score : 20,6 sur 21)


def operateurs(dossier_code: Path, injections: np.ndarray, motif: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
    """Matrices R (7, COTE_ENTREE², 2356) des sept niveaux, en float32."""
    importer_code(dossier_code)
    import KTCAux
    import KTCFwd
    import KTCRegularization

    m1, m2 = charger_maillage(dossier_code / "Mesh_sparse.mat")
    sigma0 = np.ones((len(m1.g), 1))
    z = 1e-6 * np.ones((N_ELECTRODES, 1))
    a_priori = KTCRegularization.SMPrior(m1.g, 0.115, 0.05**2, sigma0)
    X, Y = centres_pixels(COTE_ENTREE)
    points = np.column_stack([X.ravel(), Y.ravel()])
    _, interpolation, _ = KTCAux.Interpolate2Newmesh2DNode(m1.g, m1.H, m1.Node, np.zeros(len(m1.g)), points, [])
    sortie = np.zeros((len(NIVEAUX), COTE_ENTREE**2, u_ref.size), np.float32)
    for i, niveau in enumerate(NIVEAUX):
        garde = mesures_gardees(injections, niveau)
        solveur = KTCFwd.EITFEM(m2, injections, motif, garde)
        solveur.SetInvGamma(0.05, 0.01, u_ref)
        J = solveur.Jacobian(sigma0, z)
        masque = garde.T.flatten()
        G = solveur.InvGamma_n[np.ix_(masque, masque)]
        A = np.linalg.solve(J.T @ G @ J + a_priori.L.T @ a_priori.L, (G @ J).T)  # (nœuds, mesures gardées)
        sortie[i][:, masque] = (interpolation @ A).astype(np.float32)
    return sortie


def entree_reseau(R: np.ndarray, delta_u: np.ndarray, niveau: int) -> np.ndarray:
    """Image (COTE_ENTREE, COTE_ENTREE) de la reconstruction linéaire ; les mesures absentes
    (NaN) comptent pour zéro, comme les mesures retirées dans R."""
    x = R[niveau - 1] @ np.nan_to_num(delta_u.ravel())
    return x.reshape(COTE_ENTREE, COTE_ENTREE)
