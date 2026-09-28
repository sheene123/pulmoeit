"""Pont entre la page web et le paquet `pulmoeit`, exécuté dans le navigateur par Pyodide.

La simulation physique, la reconstruction linéaire NOSER, le contrôle qualité et les indices
cliniques tournent ici ; le réseau PostUNet est exécuté par ONNX Runtime Web côté JavaScript,
à partir du même fichier ONNX que l'API.
"""

import csv
import io
import json

import numpy as np

from pulmoeit.fem import ForwardModel
from pulmoeit.geometry import REFERENCE, Geometry
from pulmoeit.grid import GRID, image_mask
from pulmoeit.mesh import build_mesh
from pulmoeit.metrics import QUADRANTS, center_of_ventilation, global_inhomogeneity, pearson, regional_distribution
from pulmoeit.monitoring import MeasurementQC
from pulmoeit.phantom import SCENARIOS, sample_phantom
from pulmoeit.protocol import N_ELECTRODES, electrode_involvement, n_measurements
from pulmoeit.recon import LinearReconstructor
from pulmoeit.signals import features

N_BORD = 192  # maillage de simulation (plus léger que hors ligne, le calcul se fait dans le navigateur)
_etat: dict = {}


def initialiser() -> str:
    _etat["reconstructeur"] = LinearReconstructor.load("baseline.npz")
    _etat["qc"] = MeasurementQC.load("qc.npz")
    _etat["masque"] = image_mask()
    return json.dumps({"scenarios": list(SCENARIOS), "mesures": n_measurements(), "seuil_qc": _etat["qc"].threshold})


def _electrodes(mesh, geometrie: Geometry) -> list[list[float]]:
    """Centre de chaque électrode, ramené dans le repère de l'image (colonne, ligne en pixels)."""
    centres = np.array([mesh.nodes[aretes].reshape(-1, 2).mean(axis=0) for aretes in mesh.electrode_edges])
    centres = geometrie.map_to(centres, REFERENCE)
    return [[(x + 1) / 2 * GRID, (1 - y) / 2 * GRID] for x, y in centres]


def _electrodes_nominales() -> list[list[float]]:
    if "nominales" not in _etat:
        _etat["nominales"] = _electrodes(build_mesh(REFERENCE, N_BORD), REFERENCE)
    return _etat["nominales"]


def _analyser(v_ref, v_insp, cible, electrodes, decollee: int) -> str:
    x = features(v_ref, v_insp)
    qc = _etat["qc"]
    score = float(qc.score(x)[0])
    alerte = score > qc.threshold
    noser = _etat["reconstructeur"](x)[0]
    _etat.update(cible=cible, image_noser=noser)
    return json.dumps(
        {
            "x": x[0].tolist(),
            "noser": noser.ravel().tolist(),
            "cible": None if cible is None else cible.ravel().tolist(),
            "masque": _etat["masque"].ravel().astype(int).tolist(),
            "electrodes": electrodes,
            "qc": {
                "score": score,
                "seuil": qc.threshold,
                "alerte": bool(alerte),
                "suspecte": int(qc.suspect_electrode(x)[0]) if alerte else None,
                "decollee": decollee if decollee >= 0 else None,
            },
            "mesures": {"v_ref": np.asarray(v_ref).tolist(), "v_insp": np.asarray(v_insp).tolist()},
        }
    )


def simuler(
    scenario: str, graine: int, snr_db: float, deplacement: float, forme: float, aspect: float, decollee: int
) -> str:
    """Simule un patient (même anatomie pour une même graine, quel que soit le scénario) et une
    acquisition : bruit, placement de la ceinture, forme du thorax, électrode éventuellement décollée."""
    graine = int(graine)
    rng = np.random.default_rng(graine)
    geometrie = Geometry(
        aspect=float(aspect),
        fourier=tuple((rng.normal(0, forme), rng.normal(0, forme)) for _ in range(3)),
    )
    mesh = build_mesh(geometrie, N_BORD, electrode_shift=rng.normal(0, deplacement, N_ELECTRODES))
    fwd = ForwardModel(mesh, rng.uniform(0.005, 0.03, N_ELECTRODES))
    fantome = sample_phantom(np.random.default_rng(graine + 1), scenario)
    s_ref, s_insp, _ = fantome.evaluate(geometrie.map_to(mesh.centroids, REFERENCE))
    v_ref, v_insp = fwd.measure(s_ref), fwd.measure(s_insp)

    bruit_rng = np.random.default_rng(graine + 2)
    bruit = np.sqrt(np.mean(v_ref**2)) * 10 ** (-float(snr_db) / 20)
    v_ref = v_ref + bruit_rng.normal(0, bruit, v_ref.shape)
    v_insp = v_insp + bruit_rng.normal(0, bruit, v_insp.shape)
    decollee = int(decollee)
    if decollee >= 0:
        touchees = electrode_involvement()[:, decollee]
        n = int(touchees.sum())
        v_insp[touchees] = v_insp[touchees] * bruit_rng.uniform(0.6, 1.4, n) + bruit_rng.normal(0, 20 * bruit, n)
    return _analyser(v_ref, v_insp, fantome.render(), _electrodes(mesh, geometrie), decollee)


def depuis_fichier(texte: str) -> str:
    """Mesures fournies par l'utilisateur : CSV à colonnes v_ref,v_insp ou JSON {"v_ref": [...], "v_insp": [...]}."""
    texte = texte.strip()
    try:
        if texte.startswith("{"):
            donnees = json.loads(texte)
            v_ref, v_insp = donnees["v_ref"], donnees["v_insp"]
        else:
            lignes = list(csv.DictReader(io.StringIO(texte)))
            v_ref = [float(ligne["v_ref"]) for ligne in lignes]
            v_insp = [float(ligne["v_insp"]) for ligne in lignes]
        v_ref, v_insp = np.asarray(v_ref, dtype=float), np.asarray(v_insp, dtype=float)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as erreur:
        return json.dumps({"erreur": f"fichier illisible ({erreur}). Attendu : colonnes v_ref,v_insp."})
    n = n_measurements()
    if v_ref.shape != (n,) or v_insp.shape != (n,):
        return json.dumps(
            {"erreur": f"{n} valeurs attendues par trame (16 électrodes, protocole adjacent), reçu {len(v_ref)}."}
        )
    if not (np.all(np.isfinite(v_ref)) and np.all(np.isfinite(v_insp))) or np.min(np.abs(v_ref)) < 1e-12:
        return json.dumps({"erreur": "valeurs nulles ou non finies dans la trame de référence (électrode ouverte ?)."})
    return _analyser(v_ref, v_insp, None, _electrodes_nominales(), -1)


def _nombre(valeur) -> float | None:
    """JSON n'accepte pas NaN : une valeur non définie devient null."""
    valeur = float(valeur)
    return valeur if np.isfinite(valeur) else None


def _indices(image: np.ndarray) -> dict:
    regions = regional_distribution(image[None])[0]
    return {
        "gi": _nombre(global_inhomogeneity(image[None])[0]),
        "cov": _nombre(center_of_ventilation(image[None])[0]),
        "regions": {q: _nombre(v) for q, v in zip(QUADRANTS, regions, strict=True)},
    }


def evaluer(image_modele_json: str) -> str:
    """Indices cliniques des trois images, et corrélation avec la vérité quand elle est connue."""
    modele = np.asarray(json.loads(image_modele_json), dtype=float).reshape(GRID, GRID)
    resultat = {"noser": _indices(_etat["image_noser"]), "modele": _indices(modele)}
    cible = _etat.get("cible")
    if cible is not None:
        resultat["cible"] = _indices(cible)
        resultat["noser"]["correlation"] = _nombre(pearson(_etat["image_noser"][None], cible[None])[0])
        resultat["modele"]["correlation"] = _nombre(pearson(modele[None], cible[None])[0])
    return json.dumps(resultat)
