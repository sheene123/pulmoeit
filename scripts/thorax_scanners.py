"""Thorax issus de vrais scanners : le test hors du générateur de fantômes.

Les fantômes d'entraînement sont des ellipses (src/pulmoeit/phantom.py) : un test sur des fantômes du
même générateur surestime la performance. Ici, la coupe de chaque thorax vient d'un vrai scanner
(TotalSegmentator, Wasserthal et al. 2023, Zenodo doi:10.5281/zenodo.10047292, CC BY 4.0) : contour
du corps, poumons (lobes), cœur et aorte, os (vertèbres, côtes, sternum) et graisse, à la hauteur
de la ceinture (coupe où les poumons sont les plus grands, parmi celles où le cœur atteint
au moins 40 % de sa section maximale). Les mêmes situations cliniques que
l'entraînement y sont simulées (sain, SDRA, un seul poumon, pneumothorax, épanchement), avec le
même simulateur, les mêmes conductivités, le même bruit, et le modèle actuel est évalué sans
réentraînement.

Deux variantes : « formes » (tissu uniforme autour des organes, comme à l'entraînement) et
« formes + graisse » (la graisse, peu conductrice, est séparée du muscle d'après les unités
Hounsfield).

    python scripts/thorax_scanners.py extraire --n 40        # télécharge et garde une coupe par patient
    python scripts/thorax_scanners.py evaluer                # simule et évalue NOSER et PostUNet
    python scripts/thorax_scanners.py preparer               # jeu d'entraînement ellipses + scanners
    python scripts/thorax_scanners.py web                    # données de la page « Thorax réels »

Les patients sont séparés une fois pour toutes : 75 % pour l'entraînement, 25 % pour le test
(`evaluer --patients test`), pour comparer l'ancien et le nouveau modèle sur les mêmes patients
jamais vus.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from pulmoeit.fem import ForwardModel  # noqa: E402
from pulmoeit.geometry import REFERENCE, Geometry  # noqa: E402
from pulmoeit.grid import GRID, image_mask, pixel_centres  # noqa: E402
from pulmoeit.mesh import build_mesh  # noqa: E402
from pulmoeit.metrics import (  # noqa: E402
    center_of_ventilation,
    global_inhomogeneity,
    pearson,
    regional_distribution,
)
from pulmoeit.phantom import SCENARIO_PROBS, SCENARIOS  # noqa: E402
from pulmoeit.protocol import N_ELECTRODES  # noqa: E402
from pulmoeit.signals import features  # noqa: E402

ARCHIVE = "https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip?download=1"
DOSSIER = RACINE / "data" / "scanners"

# étiquettes de la coupe
DEHORS, TISSU, GRAISSE, OS, COEUR, POUMON_D, POUMON_G = range(7)
LOBES_D = ("lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right")
LOBES_G = ("lung_upper_lobe_left", "lung_lower_lobe_left")
COEUR_STRUCTURES = ("heart", "aorta")
# os qui croisent une coupe à mi-thorax (moins de requêtes : Zenodo limite leur nombre)
OS_STRUCTURES = (
    ("sternum",)
    + tuple(f"vertebrae_T{k}" for k in range(4, 12))
    + tuple(f"rib_{c}_{k}" for c in ("left", "right") for k in range(3, 12))
)
N_FOURIER = 10  # termes de Fourier du contour (k = 2 … 11)


# ---------------------------------------------------------------- extraction des coupes


def _nifti(octets: bytes):
    import nibabel as nib

    image = nib.Nifti1Image.from_bytes(gzip.decompress(octets))
    return nib.as_closest_canonical(image)  # axes RAS+ : +x droite du patient, +y avant, +z haut


def _lire(z, chemin: str) -> bytes:
    """Lecture d'un fichier de l'archive distante, avec pause et nouvelles tentatives si Zenodo
    limite le débit (erreur 429)."""
    import time

    from remotezip import RemoteIOError

    for essai in range(8):
        try:
            return z.read(chemin)
        except RemoteIOError as e:
            if "429" not in str(e) or essai == 7:
                raise
            time.sleep(10 * (essai + 1))
    raise RuntimeError(chemin)


def extraire(n: int, sortie: Path) -> None:
    """Télécharge, pour n patients dont le scanner couvre le thorax, la coupe à hauteur de ceinture."""
    from remotezip import RemoteZip
    from scipy import ndimage

    sortie.mkdir(parents=True, exist_ok=True)
    with RemoteZip(ARCHIVE) as z:
        meta = list(csv.DictReader(io.StringIO(z.read("meta.csv").decode("utf-8-sig")), delimiter=";"))
        candidats = sorted(
            m["image_id"] for m in meta if "thorax" in m["study_type"] and "angiography" not in m["study_type"]
        )
        rng = np.random.default_rng(0)
        rng.shuffle(candidats)
        noms = set(z.namelist())
        gardes = 0
        for sujet in candidats:
            if gardes >= n:
                break
            fichier = sortie / f"{sujet}.npz"
            if fichier.exists():
                gardes += 1
                continue
            masques = {}
            for nom in LOBES_D + LOBES_G + COEUR_STRUCTURES + OS_STRUCTURES:
                chemin = f"{sujet}/segmentations/{nom}.nii.gz"
                if chemin in noms:
                    masques[nom] = _nifti(_lire(z, chemin))
            coeur = (np.asarray(masques["heart"].dataobj) > 0).sum(axis=(0, 1))
            if coeur.max() == 0:
                print(f"{sujet} : pas de cœur dans le champ, ignoré")
                continue
            poumons = sum(
                (np.asarray(masques[n].dataobj) > 0).sum(axis=(0, 1)) for n in LOBES_D + LOBES_G if n in masques
            )
            # hauteur de ceinture : poumons les plus grands, parmi les coupes où le cœur est bien présent
            k = int(np.argmax(np.where(coeur >= 0.4 * coeur.max(), poumons, -1)))
            if k in (0, len(coeur) - 1):  # coupe au bord du volume : la hauteur de ceinture est hors du champ
                print(f"{sujet} : scanner coupé au milieu du thorax, ignoré")
                continue
            coupe = {nom: np.asarray(m.dataobj[:, :, k]) > 0 for nom, m in masques.items()}
            droit = np.any([coupe[n] for n in LOBES_D if n in coupe], axis=0)
            gauche = np.any([coupe[n] for n in LOBES_G if n in coupe], axis=0)
            if droit.sum() < 500 or gauche.sum() < 500:
                print(f"{sujet} : poumons trop petits sur la coupe du cœur, ignoré")
                continue
            ct = _nifti(_lire(z, f"{sujet}/ct.nii.gz"))
            hu = np.asarray(ct.dataobj[:, :, k], dtype=np.float32)
            corps, nb = ndimage.label(ndimage.binary_opening(hu > -300, iterations=2))
            if nb == 0:
                continue
            corps = corps == (np.argmax(np.bincount(corps.ravel())[1:]) + 1)  # plus grande composante
            corps = ndimage.binary_fill_holes(corps)
            etiquettes = np.where(corps, TISSU, DEHORS).astype(np.int8)
            etiquettes[corps & (hu > -200) & (hu < -30)] = GRAISSE
            etiquettes[np.any([coupe[n] for n in OS_STRUCTURES if n in coupe], axis=0)] = OS
            etiquettes[np.any([coupe[n] for n in COEUR_STRUCTURES if n in coupe], axis=0)] = COEUR
            etiquettes[droit] = POUMON_D
            etiquettes[gauche] = POUMON_G
            etiquettes[~corps] = DEHORS
            np.savez_compressed(
                fichier,
                etiquettes=etiquettes,
                hu=np.clip(hu, -1024, 1500).astype(np.int16),
                affine=ct.affine,
                coupe=k,
            )
            gardes += 1
            print(f"{sujet} : coupe {k} gardée ({gardes}/{n})", flush=True)


# ---------------------------------------------------------------- géométrie et fantôme


class ThoraxScanner:
    """Coupe de scanner ramenée dans le repère du simulateur : x vers la gauche du patient,
    y vers l'avant, demi-largeur du thorax = 1."""

    def __init__(self, fichier: Path):
        d = np.load(fichier)
        self.nom = fichier.stem
        self.etiquettes = d["etiquettes"]
        self.hu = d["hu"]
        affine = d["affine"]
        i, j = np.nonzero(self.etiquettes != DEHORS)
        monde = (affine[:2, :2] @ np.vstack([i, j]) + affine[:2, 3:4]).T  # (x, y) RAS en mm
        self.xmin, self.xmax = monde[:, 0].min(), monde[:, 0].max()
        self.ymin, self.ymax = monde[:, 1].min(), monde[:, 1].max()
        self.cx, self.cy = (self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2
        self.demi = (self.xmax - self.xmin) / 2
        self.affine_inv = np.linalg.inv(affine[:2, :2])
        self.origine = affine[:2, 3]
        self.geometrie = self._ajuster_contour()

    def vers_mm(self, p: np.ndarray) -> np.ndarray:
        return np.c_[self.cx - p[:, 0] * self.demi, self.cy + p[:, 1] * self.demi]

    def etiquette(self, p: np.ndarray) -> np.ndarray:
        """Étiquette du voxel le plus proche de chaque point (repère du simulateur)."""
        ij = np.rint((self.affine_inv @ (self.vers_mm(p) - self.origine).T).T).astype(int)
        n0, n1 = self.etiquettes.shape
        dedans = (ij[:, 0] >= 0) & (ij[:, 0] < n0) & (ij[:, 1] >= 0) & (ij[:, 1] < n1)
        sortie = np.full(len(p), DEHORS, dtype=np.int8)
        sortie[dedans] = self.etiquettes[ij[dedans, 0], ij[dedans, 1]]
        return sortie

    def _ajuster_contour(self) -> Geometry:
        """Contour du corps -> rapport d'aspect et termes de Fourier de la classe Geometry."""
        theta = np.linspace(-np.pi, np.pi, 720, endpoint=False)
        rayons = np.linspace(0, 1.6, 1600)
        r = np.empty_like(theta)
        for n, t in enumerate(theta):  # dernier point du corps le long de chaque rayon
            pts = rayons[:, None] * np.array([np.cos(t), np.sin(t)])
            dedans = np.nonzero(self.etiquette(pts) != DEHORS)[0]
            r[n] = rayons[dedans.max()] if dedans.size else rayons[1]
        aspect = (self.ymax - self.ymin) / 2 / self.demi
        ellipse = Geometry(aspect).radius(theta)
        echelle = r / ellipse - 1
        colonnes = [f(k * theta) for k in range(2, 2 + N_FOURIER) for f in (np.cos, np.sin)]
        coef, *_ = np.linalg.lstsq(np.stack(colonnes, axis=1), echelle, rcond=None)
        return Geometry(float(aspect), tuple(zip(coef[0::2], coef[1::2], strict=True)))


def _parts(y: np.ndarray, masque: np.ndarray) -> np.ndarray:
    """Position verticale relative dans un poumon : 0 au bord dorsal, 1 au bord ventral."""
    if not masque.any():
        return np.zeros_like(y)
    bas, haut = y[masque].min(), y[masque].max()
    return (y - bas) / max(haut - bas, 1e-9)


def fantome(thorax: ThoraxScanner, rng: np.random.Generator, scenario: str, graisse: bool):
    """Retourne une fonction p -> (sigma_expiration, sigma_inspiration, ventilation), avec les
    mêmes conductivités et les mêmes lois de ventilation que src/pulmoeit/phantom.py."""
    u, g = rng.uniform, rng.normal
    p = {
        "sigma_bg": u(0.3, 0.45),
        "sigma_poumon": u(0.1, 0.16),
        "sigma_coeur": u(0.5, 0.7),
        "cardiaque": g(0, 0.02),
        "tidal": u(0.15, 0.45),
        "gravite": u(-0.3, 0.5),
        "gain": (u(0.8, 1.2), u(0.8, 1.2)),
        "cote": int(rng.integers(2)),
        "coupe_sdra": (u(0.2, 0.6) + g(0, 0.04), u(0.2, 0.6) + g(0, 0.04)),
        "sigma_sdra": u(0.3, 0.4),
        "coupe_pnx": u(0.55, 0.8),
        "coupe_epanchement": u(0.1, 0.35),
        "sigma_epanchement": u(1.2, 1.6),
    }

    def evaluer(points: np.ndarray):
        e = thorax.etiquette(points)
        y = points[:, 1]
        sigma = np.full(len(points), p["sigma_bg"])
        if graisse:
            sigma[e == GRAISSE] = 0.04
        sigma[e == OS] = 0.03
        sigma[e == COEUR] = p["sigma_coeur"]
        vent = np.zeros(len(points))
        poumons = (e == POUMON_D, e == POUMON_G)  # côté 0 = droite du patient, comme phantom.py
        for cote, m in enumerate(poumons):
            sigma[m] = p["sigma_poumon"]
            vent[m] = p["tidal"] * p["gain"][cote] * (1 + p["gravite"] * (-y[m]) / thorax.geometrie.aspect)
        if scenario == "dorsal_atelectasis":
            for cote, m in enumerate(poumons):
                z = m & (_parts(y, m) < p["coupe_sdra"][cote])
                sigma[z], vent[z] = p["sigma_sdra"], 0.0
        elif scenario == "one_lung":
            vent[poumons[p["cote"]]] = 0.0
        elif scenario == "pneumothorax":
            m = poumons[p["cote"]]
            z = m & (_parts(y, m) > p["coupe_pnx"])
            sigma[z], vent[z] = 0.01, 0.0
        elif scenario == "pleural_effusion":
            m = poumons[p["cote"]]
            z = m & (_parts(y, m) < p["coupe_epanchement"])
            sigma[z], vent[z] = p["sigma_epanchement"], 0.0
        vent = np.clip(vent, 0.0, 0.9)
        sigma_insp = sigma * (1.0 - vent)
        sigma_insp[e == COEUR] *= 1.0 + p["cardiaque"]
        return sigma, sigma_insp, vent

    return evaluer


def simuler(thorax: ThoraxScanner, rng: np.random.Generator, scenario: str, graisse: bool, brut: bool = False) -> dict:
    """Une respiration : mesures bruitées (même acquisition que AcquisitionConfig) et vérité 32 × 32."""
    maillage = build_mesh(thorax.geometrie, 224, electrode_width=0.3, electrode_shift=rng.normal(0, 0.03, N_ELECTRODES))
    z_ref = rng.uniform(0.005, 0.03, N_ELECTRODES)
    z_insp = z_ref * np.clip(1 + rng.normal(0, 0.05, N_ELECTRODES), 0.5, 2.0)
    evaluer = fantome(thorax, rng, scenario, graisse)
    s_ref, s_insp, _ = evaluer(maillage.centroids)
    v_ref, v_insp = ForwardModel(maillage, z_ref).measure(s_ref), ForwardModel(maillage, z_insp).measure(s_insp)
    bruit = np.sqrt(np.mean(v_ref**2)) * 10 ** (-rng.uniform(50.0, 70.0) / 20)
    v_ref = v_ref + rng.normal(0, bruit, v_ref.shape)
    v_insp = v_insp + rng.normal(0, bruit, v_insp.shape)
    # vérité sur la grille de référence : chaque pixel est ramené dans le thorax du patient
    pts = pixel_centres(GRID, 3).reshape(-1, 2)
    vent = evaluer(REFERENCE.map_to(pts, thorax.geometrie))[2].reshape(GRID, GRID, -1).mean(axis=-1)
    verite = (vent * image_mask()).astype(np.float32)
    if brut:
        return {"v_ref": v_ref, "v_insp": v_insp, "verite": verite}
    return {"x": features(v_ref, v_insp)[0], "verite": verite}


# ---------------------------------------------------------------- séparation et jeux de données


def separer(fichiers: list[Path]) -> dict[str, list[Path]]:
    """Séparation fixe par patient (jamais le même patient dans deux ensembles)."""
    fichiers = sorted(fichiers)
    ordre = np.random.default_rng(12345).permutation(len(fichiers))
    n_test = max(1, round(0.25 * len(fichiers)))
    n_val = max(1, round(0.1 * len(fichiers)))
    test = [fichiers[i] for i in ordre[:n_test]]
    val = [fichiers[i] for i in ordre[n_test : n_test + n_val]]
    train = [fichiers[i] for i in ordre[n_test + n_val :]]
    return {"train": train, "val": val, "test": test}


def _patient(args: tuple) -> list[dict]:
    fichier, graine, n = args
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(graine)
    thorax = ThoraxScanner(fichier)
    sorties = []
    for _ in range(n):
        scenario = str(rng.choice(SCENARIOS, p=SCENARIO_PROBS))
        graisse = bool(rng.random() < 0.5)
        sorties.append(_brut(thorax, rng, scenario, graisse) | {"scenario": SCENARIOS.index(scenario)})
    return sorties


def _brut(thorax: ThoraxScanner, rng: np.random.Generator, scenario: str, graisse: bool) -> dict:
    s = simuler(thorax, rng, scenario, graisse, brut=True)
    return {"v_ref": s["v_ref"], "v_insp": s["v_insp"], "target": s["verite"]}


def jeu(fichiers: list[Path], par_patient: int, graine: int, processus: int = 8) -> dict[str, np.ndarray]:
    """Exemples au format de pulmoeit.dataset (v_ref, v_insp, target, scenario, fault)."""
    from concurrent.futures import ProcessPoolExecutor

    taches = [(f, [graine, i], par_patient) for i, f in enumerate(fichiers)]
    with ProcessPoolExecutor(processus) as pool:
        exemples = [e for lot in pool.map(_patient, taches) for e in lot]
    return {
        "v_ref": np.stack([e["v_ref"] for e in exemples]),
        "v_insp": np.stack([e["v_insp"] for e in exemples]),
        "target": np.stack([e["target"] for e in exemples]),
        "scenario": np.array([e["scenario"] for e in exemples], dtype=np.int8),
        "fault": np.full(len(exemples), -1, dtype=np.int8),
    }


def preparer(coupes: Path, par_patient: int) -> None:
    """data_scanners/ : jeux d'entraînement et de validation ellipses + scanners (les jeux de test
    restent ceux des ellipses, pour que `pulmoeit evaluate` reste comparable) ; models_scanners/ :
    NOSER et contrôle qualité inchangés ; configs/params.scanners.yaml."""
    import shutil

    import yaml

    from pulmoeit import dataset

    parts = separer(list(coupes.glob("*.npz")))
    donnees, modeles = RACINE / "data_scanners", RACINE / "models_scanners"
    for nom, graine in (("train", 1), ("val", 2)):
        ellipses = dataset.load(RACINE / "data" / f"{nom}.npz")
        scanners = jeu(parts[nom], par_patient if nom == "train" else par_patient // 2, graine)
        dataset.save(donnees / f"{nom}.npz", {k: np.concatenate([ellipses[k], scanners[k]]) for k in ellipses})
        print(
            f"{nom} : {len(ellipses['target'])} ellipses + {len(scanners['target'])} scanners ({len(parts[nom])} patients)"
        )
    # copyfile et non copy : les fichiers suivis par DVC sont en lecture seule, et l'entraînement
    # doit pouvoir réécrire qc.npz dans models_scanners/
    for nom in ("test.npz", "test_shift.npz", "test_fault.npz"):
        shutil.copyfile(RACINE / "data" / nom, donnees / nom)
    modeles.mkdir(exist_ok=True)
    for nom in ("baseline.npz", "qc.npz"):
        shutil.copyfile(RACINE / "models" / nom, modeles / nom)
    params = yaml.safe_load((RACINE / "params.yaml").read_text())
    params["paths"] = {"data": "data_scanners", "models": "models_scanners", "reports": "reports/modele_scanners"}
    (RACINE / "configs" / "params.scanners.yaml").write_text(
        yaml.safe_dump(params, sort_keys=False, allow_unicode=True)
    )
    (donnees / "patients.json").write_text(json.dumps({k: [f.stem for f in v] for k, v in parts.items()}, indent=1))


# ---------------------------------------------------------------- évaluation


def _scores(pred: np.ndarray, verite: np.ndarray) -> dict:
    return {
        "corr": float(np.mean(pearson(pred, verite))),
        "gi_mae": float(np.nanmean(np.abs(global_inhomogeneity(pred) - global_inhomogeneity(verite)))),
        "cov_mae": float(np.nanmean(np.abs(center_of_ventilation(pred) - center_of_ventilation(verite)))),
        "regional_mae": float(np.nanmean(np.abs(regional_distribution(pred) - regional_distribution(verite)))),
    }


def evaluer(dossier: Path, modeles: Path, sortie: Path, par_patient: int, patients: str = "tous") -> dict:
    import onnxruntime as ort

    from pulmoeit.monitoring import MeasurementQC
    from pulmoeit.recon import LinearReconstructor

    fichiers = sorted(dossier.glob("*.npz"))
    if patients == "test":
        fichiers = separer(fichiers)["test"]
    noser = LinearReconstructor.load(modeles / "baseline.npz")
    session = ort.InferenceSession(str(modeles / "pulmoeit.onnx"), providers=["CPUExecutionProvider"])
    qc = MeasurementQC.load(modeles / "qc.npz")
    rng = np.random.default_rng(2026)
    resultats: dict = {
        "donnees": {
            "source": "TotalSegmentator v2.0.1, Wasserthal et al., Radiology: AI 2023, doi:10.5281/zenodo.10047292 (CC BY 4.0)",
            "patients": len(fichiers),
            "ensemble": patients,
            "modele": str(modeles.relative_to(RACINE) if modeles.is_relative_to(RACINE) else modeles),
            "exemples_par_patient_et_variante": par_patient,
        }
    }
    exemples = {}
    for variante, graisse in (("formes", False), ("formes_graisse", True)):
        x, verites, scenarios = [], [], []
        for f in fichiers:
            thorax = ThoraxScanner(f)
            for _ in range(par_patient):
                scenario = str(rng.choice(SCENARIOS, p=SCENARIO_PROBS))
                s = simuler(thorax, rng, scenario, graisse)
                x.append(s["x"])
                verites.append(s["verite"])
                scenarios.append(scenario)
        x, verites = np.stack(x), np.stack(verites)
        img_noser = noser(x)
        img_reseau = session.run(None, {"dv": x.astype(np.float32)})[0].reshape(-1, GRID, GRID)
        scores_qc = qc.score(x)
        resultats[variante] = {
            "n": len(x),
            "baseline": _scores(img_noser, verites),
            "model": _scores(img_reseau, verites),
            "par_scenario": {
                sc: {
                    "n": int(sum(s == sc for s in scenarios)),
                    "baseline_corr": float(
                        np.mean(pearson(*(a[[s == sc for s in scenarios]] for a in (img_noser, verites))))
                    ),
                    "model_corr": float(
                        np.mean(pearson(*(a[[s == sc for s in scenarios]] for a in (img_reseau, verites))))
                    ),
                }
                for sc in SCENARIOS
            },
            "qc_fausses_alarmes": float(np.mean(scores_qc > qc.threshold)),
        }
        exemples[variante] = (verites, img_noser, img_reseau, scenarios)
        print(
            variante, json.dumps(resultats[variante]["baseline"]), json.dumps(resultats[variante]["model"]), flush=True
        )
    reference = json.loads((RACINE / "reports" / "metrics.json").read_text())
    resultats["rappel_fantomes_ellipses"] = {
        "meme_distribution": {k: reference["test"][k] for k in ("baseline", "model")},
        "decale": {k: reference["test_shift"][k] for k in ("baseline", "model")},
    }
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(resultats, indent=1, ensure_ascii=False))
    figure(sortie.with_suffix(".png"), fichiers, exemples["formes_graisse"])
    return resultats


def figure(chemin: Path, fichiers: list[Path], exemples) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verites, img_noser, img_reseau, scenarios = exemples
    masque = image_mask()
    noms = {
        "healthy": "sain",
        "dorsal_atelectasis": "SDRA",
        "one_lung": "un poumon",
        "pneumothorax": "pneumothorax",
        "pleural_effusion": "épanchement",
    }
    par_patient = len(scenarios) // len(fichiers)
    choix, vus = [], set()
    for sc in ("healthy", "dorsal_atelectasis", "pneumothorax", "pleural_effusion"):
        k = [i for i, s in enumerate(scenarios) if s == sc and i // par_patient not in vus]
        if k:  # un patient différent par ligne
            choix.append(k[0])
            vus.add(k[0] // par_patient)
    fig, axes = plt.subplots(len(choix), 4, figsize=(11, 2.9 * len(choix)))
    for ligne, k in enumerate(choix):
        thorax = ThoraxScanner(fichiers[k // par_patient])
        coupe = np.fliplr(
            np.rot90(thorax.hu)
        )  # RAS -> affichage radiologique (avant en haut, droite du patient à gauche)
        axes[ligne, 0].imshow(coupe, cmap="gray", vmin=-1000, vmax=400)
        axes[ligne, 0].set_title(f"scanner {thorax.nom}" if ligne == 0 else thorax.nom, fontsize=9)
        for col, (img, titre) in enumerate(
            ((verites[k], "vérité"), (img_noser[k], "NOSER"), (img_reseau[k], "PostUNet")), start=1
        ):
            v = np.where(masque, img / max(np.abs(img).max(), 1e-12), np.nan)
            axes[ligne, col].imshow(v, cmap="Blues", vmin=0, vmax=1)
            axes[ligne, col].set_title(f"{titre} ({noms[scenarios[k]]})", fontsize=9)
        for ax in axes[ligne]:
            ax.set(xticks=[], yticks=[])
    fig.suptitle(
        "Thorax issus de vrais scanners (TotalSegmentator), avec graisse : modèle actuel, sans réentraînement",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(chemin, dpi=110)
    plt.close(fig)


def exporter_web(coupes: Path, sortie: Path) -> None:
    """Données de la page « Thorax réels » de la démo web : pour les patients de test, 5 situations ×
    avec ou sans graisse, les mesures (le navigateur fait tourner les deux réseaux), le corrigé, NOSER
    et la coupe du scanner ; plus le nouveau modèle et les résultats globaux."""
    import base64
    import shutil

    import onnxruntime as ort
    from PIL import Image

    from pulmoeit.recon import LinearReconstructor

    def octets(img: np.ndarray) -> str:  # 0 = pas d'air, 255 = maximum de l'image (négatifs à 0)
        img = np.clip(img, 0, None) * image_mask()
        return base64.b64encode((img / max(img.max(), 1e-12) * 255).round().astype(np.uint8).tobytes()).decode()

    sortie.mkdir(parents=True, exist_ok=True)
    noser = LinearReconstructor.load(RACINE / "models" / "baseline.npz")
    sessions = {
        nom: ort.InferenceSession(str(RACINE / dossier / "pulmoeit.onnx"), providers=["CPUExecutionProvider"])
        for nom, dossier in (("ancien", "models"), ("nouveau", "models_scanners"))
    }
    patients = []
    for i, f in enumerate(separer(sorted(coupes.glob("*.npz")))["test"]):
        thorax = ThoraxScanner(f)
        # coupe du scanner : affichage radiologique, fenêtre poumon-médiastin, recadrée sur le corps
        hu = np.fliplr(np.rot90(thorax.hu)).astype(np.float32)
        corps = np.fliplr(np.rot90(thorax.etiquettes)) != DEHORS
        lignes, colonnes = np.nonzero(corps)
        hu = hu[max(lignes.min() - 6, 0) : lignes.max() + 7, max(colonnes.min() - 6, 0) : colonnes.max() + 7]
        gris = (np.clip((hu + 1150) / 1500, 0, 1) * 255).astype(np.uint8)
        image = Image.fromarray(gris)
        image.thumbnail((300, 300))
        image.save(sortie / f"P{i + 1}.png", optimize=True)
        cas = {}
        for j, scenario in enumerate(SCENARIOS):
            for graisse in (False, True):
                s = simuler(thorax, np.random.default_rng([2026, i, j, int(graisse)]), scenario, graisse)
                x = s["x"][None].astype(np.float32)
                images = {
                    "noser": noser(x)[0],
                    **{n: se.run(None, {"dv": x})[0].reshape(GRID, GRID) for n, se in sessions.items()},
                }
                cas[f"{scenario}|{int(graisse)}"] = {
                    "x": base64.b64encode(x[0].tobytes()).decode(),
                    "verite": octets(s["verite"]),
                    "noser": octets(images["noser"]),
                    "scores": {n: round(float(pearson(im[None], s["verite"][None])[0]), 3) for n, im in images.items()},
                }
        patients.append({"nom": f"P{i + 1}", "image": f"P{i + 1}.png", "cas": cas})
        print(f"P{i + 1} ({f.stem})", flush=True)
    ancien = json.loads((RACINE / "reports" / "thorax_scanners_test_ancien.json").read_text())
    nouveau = json.loads((RACINE / "reports" / "thorax_scanners_test_nouveau.json").read_text())
    resultats = {
        v: {"noser": ancien[v]["baseline"], "ancien": ancien[v]["model"], "nouveau": nouveau[v]["model"]}
        for v in ("formes", "formes_graisse")
    }
    maillage = build_mesh(REFERENCE, 192)
    centres = np.array([maillage.nodes[a].reshape(-1, 2).mean(axis=0) for a in maillage.electrode_edges])
    (sortie / "cas.json").write_text(
        json.dumps(
            {
                "masque": image_mask().ravel().astype(int).tolist(),
                "electrodes": [[round((x + 1) / 2 * GRID, 3), round((1 - y) / 2 * GRID, 3)] for x, y in centres],
                "patients": patients,
                "resultats": resultats,
                "n_test": ancien["formes"]["n"],
            },
            ensure_ascii=False,
        )
    )
    shutil.copyfile(RACINE / "models_scanners" / "pulmoeit.onnx", sortie / "pulmoeit_scanners.onnx")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sous = parser.add_subparsers(dest="etape", required=True)
    e = sous.add_parser("extraire")
    e.add_argument("--n", type=int, default=40)
    e.add_argument("--sortie", type=Path, default=DOSSIER / "coupes")
    v = sous.add_parser("evaluer")
    v.add_argument("--coupes", type=Path, default=DOSSIER / "coupes")
    v.add_argument("--modeles", type=Path, default=RACINE / "models")
    v.add_argument("--par-patient", type=int, default=10)
    v.add_argument("--sortie", type=Path, default=RACINE / "reports" / "thorax_scanners.json")
    v.add_argument("--patients", choices=("tous", "test"), default="tous")
    pr = sous.add_parser("preparer")
    pr.add_argument("--coupes", type=Path, default=DOSSIER / "coupes")
    pr.add_argument("--par-patient", type=int, default=100)
    w = sous.add_parser("web")
    w.add_argument("--coupes", type=Path, default=DOSSIER / "coupes")
    w.add_argument("--sortie", type=Path, default=RACINE / "demo" / "web" / "scanners")
    args = parser.parse_args()
    if args.etape == "web":
        exporter_web(args.coupes, args.sortie)
    elif args.etape == "extraire":
        extraire(args.n, args.sortie)
    elif args.etape == "evaluer":
        evaluer(args.coupes.resolve(), args.modeles.resolve(), args.sortie, args.par_patient, args.patients)
    else:
        preparer(args.coupes, args.par_patient)
    return 0


if __name__ == "__main__":
    sys.exit(main())
