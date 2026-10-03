"""Accès au code, aux maillages et aux données officiels du KTC2023 (téléchargés, jamais
redistribués), et conventions du défi."""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ZENODO = "https://zenodo.org/api/records/10986692/files/{}/content"
N_ELECTRODES = 32
NIVEAUX = range(1, 8)
COTE_VERITE = 256  # vérité terrain et score officiel : images 256 × 256
RAYON = 0.115  # rayon de la cuve (m)


def telecharger(cache: Path) -> Path:
    """Code Python officiel et données d'évaluation, dans `cache` ; renvoie le dossier du code."""
    cache.mkdir(parents=True, exist_ok=True)
    for archive in ("Codes_Python.zip", "EvaluationData.zip"):
        if not any(cache.glob(archive.removesuffix(".zip") + "*")):
            with urllib.request.urlopen(ZENODO.format(archive), timeout=120) as r:
                zipfile.ZipFile(io.BytesIO(r.read())).extractall(cache)
    return cache / "Codes_Python"


def importer_code(dossier_code: Path) -> None:
    """Rend importables KTCFwd, KTCMeshing, KTCRegularization, KTCAux, KTCScoring."""
    if str(dossier_code) not in sys.path:
        sys.path.insert(0, str(dossier_code))


def charger_maillage(chemin: Path):
    """(maillage d'ordre 1, maillage d'ordre 2) au format des organisateurs (repris de leur
    code, factorisé)."""
    import KTCMeshing
    import scipy.io

    m = scipy.io.loadmat(chemin)
    sortie = []
    for s in ("", "2"):
        topo = [t[0].flatten() for t in m["Element" + s]["Topology"].tolist()]
        elec = [
            [e[0][0][0], e[0][0][1 : len(e[0][0])]] if len(e[0]) > 0 else []
            for e in m["Element2E" if s else "ElementE"].tolist()
        ]
        coords, connexions = m["Node" + s]["Coordinate"], m["Node" + s]["ElementConnection"]
        noeuds = [KTCMeshing.NODE(c[0].flatten(), []) for c in coords]
        for k in range(coords.shape[0]):
            noeuds[k].ElementConnection = connexions[k][0].flatten()
        elements = [KTCMeshing.ELEMENT(ind, []) for ind in topo]
        for k, e in enumerate(elec):
            elements[k].Electrode = e
        sortie.append(KTCMeshing.Mesh(m["H" + s], m["g" + s], m["elfaces" + s][0].tolist(), noeuds, elements))
    return sortie


def mesures_gardees(injections: np.ndarray, niveau: int) -> np.ndarray:
    """Masque (31, 76) des mesures disponibles au niveau donné : au niveau n, les électrodes
    0 à 2(n-1)-1 sont retirées, ainsi que les injections qui les utilisent (règle des
    organisateurs)."""
    garde = np.ones((N_ELECTRODES - 1, injections.shape[1]), bool)
    retirees = np.arange(2 * (niveau - 1))
    for i in range(injections.shape[1] - 1):
        if np.any(injections[retirees, i]):
            garde[:, i] = False
    garde[retirees, :] = False
    return garde


def ajuster_fond(u_ref_mesure: np.ndarray, u_ref_unitaire: np.ndarray) -> float:
    """Conductivité de l'eau : les tensions d'un milieu homogène varient comme 1 / sigma
    (impédance de contact négligeable), donc sigma = <U(1), U(1)> / <U(1), U_ref>."""
    a, b = np.asarray(u_ref_unitaire).ravel(), np.asarray(u_ref_mesure).ravel()
    return float(a @ a / (a @ b))
