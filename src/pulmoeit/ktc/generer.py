"""Génère les cuves simulées d'entraînement avec le simulateur officiel (maillage dense, ordre 2 ;
la reconstruction utilise le maillage clairsemé, ce qui évite le « crime inverse »).

Chaque exemple garde la différence de tensions delta_U (2 356 mesures, toutes gardées) et la
vérité 128 × 128 : les sept niveaux de difficulté se calculent ensuite à partir du même delta_U,
en masquant les mesures (lineaire.operateurs).

    python -m pulmoeit.ktc.generer --code <Codes_Python> --sortie data/ktc --n 30000 --processus 4
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .fantomes import conductivite_noeuds, reduire, tirer
from .lineaire import COTE_ENTREE
from .officiel import N_ELECTRODES, ajuster_fond, charger_maillage, importer_code

_ETAT: dict = {}
TAILLE_LOT = 500
N_MESURES = 76 * (N_ELECTRODES - 1)
IMPEDANCES_CONTACT = (1e-6, 1e-5, 3e-5, 1e-4)  # valeurs tirées ; la référence est calculée une fois pour chacune


def _initialiser(dossier_code: str) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import scipy.io

    code = Path(dossier_code)
    importer_code(code)
    import KTCFwd

    ref = scipy.io.loadmat(code / "TrainingData" / "ref.mat")
    m1, m2 = charger_maillage(code / "Mesh_dense.mat")
    garde = np.ones((N_ELECTRODES - 1, ref["Injref"].shape[1]), bool)
    solveur = KTCFwd.EITFEM(m2, ref["Injref"], ref["Mpat"], garde)
    z = 1e-6 * np.ones((N_ELECTRODES, 1))
    u_unitaire = solveur.SolveForward(np.ones((len(m1.g), 1)), z)
    fond = ajuster_fond(ref["Uelref"], u_unitaire)
    references = {
        z: np.asarray(solveur.SolveForward(np.full((len(m1.g), 1), fond), z * np.ones((N_ELECTRODES, 1)))).ravel()
        for z in IMPEDANCES_CONTACT
    }
    _ETAT.update(solveur=solveur, noeuds=m1.g, fond=fond, references=references)


def _bruit(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Modèle de bruit des organisateurs : 0,05 % de chaque tension + 0,01 % de la plus grande."""
    ecart = 5e-4 * np.abs(u) + 1e-4 * np.abs(u).max()
    return u + ecart * rng.standard_normal(u.shape)


def _lot(tache: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    graine, debut, n = tache
    rng = np.random.default_rng([graine, debut])
    solveur, noeuds, fond = _ETAT["solveur"], _ETAT["noeuds"], _ETAT["fond"]
    delta_u = np.empty((n, N_MESURES), np.float32)
    verites = np.empty((n, COTE_ENTREE, COTE_ENTREE), np.uint8)
    for i in range(n):
        classes = tirer(rng)
        z = float(rng.choice(IMPEDANCES_CONTACT))
        derive = 1 + rng.normal(0, 0.005)  # l'eau change un peu entre la référence et la mesure
        sigma = conductivite_noeuds(classes, noeuds, fond * derive, rng)
        u = np.asarray(solveur.SolveForward(sigma, z * np.ones((N_ELECTRODES, 1)))).ravel()
        u_ref = _ETAT["references"][z]
        delta_u[i] = (_bruit(u, rng) - _bruit(u_ref, rng)).ravel()
        verites[i] = reduire(classes, COTE_ENTREE)
    return delta_u, verites


def generer(dossier_code: Path, sortie: Path, n: int, processus: int, graine: int = 0) -> None:
    from concurrent.futures import as_completed

    sortie.mkdir(parents=True, exist_ok=True)
    taches = [(graine, debut, min(TAILLE_LOT, n - debut)) for debut in range(0, n, TAILLE_LOT)]
    debut_t = time.perf_counter()
    fait = 0
    with ProcessPoolExecutor(processus, initializer=_initialiser, initargs=(str(dossier_code),)) as pool:
        futurs = {pool.submit(_lot, tache): k for k, tache in enumerate(taches)}
        for futur in as_completed(futurs):  # chaque lot est écrit dès qu'il est prêt
            delta_u, verites = futur.result()
            np.savez_compressed(sortie / f"lot_{futurs[futur]:04d}.npz", delta_u=delta_u, verites=verites)
            fait += len(delta_u)
            ecoule = time.perf_counter() - debut_t
            print(
                f"{fait}/{n} cuves ({ecoule / 60:.0f} min, reste ≈ {ecoule / fait * (n - fait) / 60:.0f} min)",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--sortie", type=Path, required=True)
    parser.add_argument("--n", type=int, default=30000)
    parser.add_argument("--processus", type=int, default=4)
    parser.add_argument("--graine", type=int, default=0)
    args = parser.parse_args()
    generer(args.code, args.sortie, args.n, args.processus, args.graine)


if __name__ == "__main__":
    main()
