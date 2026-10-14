"""Essais de PEEP simulés : de combien chaque méthode de reconstruction se trompe sur ce que le
réanimateur utilise (zones collabées, zones surdistendues, PEEP proposée), et pas sur l'image.

Thorax : coupes de vrais scanners (scripts/thorax_scanners.py, mêmes patients de validation et de
test). Poumon : courbe pression-volume sigmoïde par région, pression d'inflexion croissante vers le
dos (src/pulmoeit/peep.py). Essai de PEEP décroissante de 24 à 6 cmH2O, pression motrice constante.
À chaque palier, l'image courante est reconstruite par NOSER, l'ancien et le nouveau PostUNet, et
par les algorithmes d'EIDORS (GREIT, Gauss-Newton en une étape, Gauss-Newton itératif, variation
totale) sur le même maillage et le même protocole ; puis la méthode de Costa est appliquée à chaque
série. Le réglage de chaque méthode EIDORS est choisi sur les patients de validation (erreur
quadratique sur l'image, comme pour NOSER).

    python scripts/essai_peep.py simuler       # essais de validation et de test + entrée EIDORS
    python scripts/essai_peep.py eidors        # reconstructions EIDORS (GNU Octave dans Docker)
    python scripts/essai_peep.py evaluer       # Costa, espaces silencieux, rapport
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
sys.path.insert(0, str(RACINE / "scripts"))

from thorax_scanners import COEUR, GRAISSE, OS, POUMON_D, POUMON_G, ThoraxScanner, separer  # noqa: E402

from pulmoeit.fem import ForwardModel  # noqa: E402
from pulmoeit.geometry import REFERENCE  # noqa: E402
from pulmoeit.grid import GRID, image_mask, pixel_centres  # noqa: E402
from pulmoeit.mesh import build_mesh  # noqa: E402
from pulmoeit.metrics import (  # noqa: E402
    center_of_ventilation,
    global_inhomogeneity,
    pearson,
    regional_distribution,
)
from pulmoeit.peep import costa, espaces_silencieux, volume_sigmoide  # noqa: E402
from pulmoeit.protocol import N_ELECTRODES, adjacent_protocol  # noqa: E402
from pulmoeit.signals import features  # noqa: E402

DOSSIER = RACINE / "data" / "peep"
COUPES = RACINE / "data" / "scanners" / "coupes"
EIDORS = Path.home() / "outils_eit" / "eidors-v3.12"
IMAGE_OCTAVE = "gnuoctave/octave:9.2.0"
PEEPS = np.arange(24.0, 5.0, -2.0)  # essai décroissant, 10 paliers


# ---------------------------------------------------------------- simulation


def _profondeur(y: np.ndarray, haut: float, bas: float) -> np.ndarray:
    """0 au sommet ventral des poumons, 1 au bas dorsal."""
    return np.clip((haut - y) / max(haut - bas, 1e-9), 0, 1)


def essai(thorax: ThoraxScanner, rng: np.random.Generator) -> dict:
    """Un essai de PEEP décroissante sur un thorax : mesures, vérité et paramètres."""
    maillage = build_mesh(thorax.geometrie, 224, electrode_width=0.3, electrode_shift=rng.normal(0, 0.03, N_ELECTRODES))
    direct = ForwardModel(maillage, rng.uniform(0.005, 0.03, N_ELECTRODES))
    p = {
        "p_avant": rng.uniform(4, 10),  # pression d'inflexion des régions ventrales
        "p_dos": rng.uniform(22, 32),  # … et dorsales (poids du poumon sus-jacent, SDRA)
        "pente": rng.uniform(2.0, 3.5),
        "delta_p": rng.uniform(10, 14),  # pression motrice, constante pendant l'essai
        "ecart_cotes": rng.normal(0, 2),  # un poumon un peu plus atteint que l'autre
        "sigma_collabe": rng.uniform(0.3, 0.45),
        "sigma_distendu": rng.uniform(0.06, 0.12),
        "sigma_bg": rng.uniform(0.3, 0.45),
        "sigma_coeur": rng.uniform(0.5, 0.7),
        "graisse": bool(rng.random() < 0.5),
        "snr_db": rng.uniform(50, 70),
    }

    def tissus(points: np.ndarray):
        e = thorax.etiquette(points)
        poumon = (e == POUMON_D) | (e == POUMON_G)
        sigma = np.full(len(points), p["sigma_bg"])
        if p["graisse"]:
            sigma[e == GRAISSE] = 0.04
        sigma[e == OS] = 0.03
        sigma[e == COEUR] = p["sigma_coeur"]
        return e, poumon, sigma

    e, poumon, sigma_fixe = tissus(maillage.centroids)
    y = maillage.centroids[:, 1]
    haut, bas = y[poumon].max(), y[poumon].min()

    def inflexion(y_pts: np.ndarray, etiquettes: np.ndarray) -> np.ndarray:
        pinf = p["p_avant"] + (p["p_dos"] - p["p_avant"]) * _profondeur(y_pts, haut, bas)
        return pinf + np.where(etiquettes == POUMON_G, p["ecart_cotes"], 0.0)

    pinf = inflexion(y[poumon], e[poumon])

    def sigma_a(pression: float) -> np.ndarray:
        s = sigma_fixe.copy()
        v = volume_sigmoide(pression, pinf, p["pente"])
        s[poumon] = p["sigma_collabe"] + (p["sigma_distendu"] - p["sigma_collabe"]) * v
        return s

    v_exp, v_insp = [], []
    for peep in PEEPS:
        a, b = direct.measure(sigma_a(peep)), direct.measure(sigma_a(peep + p["delta_p"]))
        bruit = np.sqrt(np.mean(a**2)) * 10 ** (-p["snr_db"] / 20)
        v_exp.append(a + rng.normal(0, bruit, a.shape))
        v_insp.append(b + rng.normal(0, bruit, b.shape))
    # vérité sur la grille d'image : volume courant de chaque pixel (moyenne sur 3 × 3 sous-points)
    pts = REFERENCE.map_to(pixel_centres(GRID, 3).reshape(-1, 2), thorax.geometrie)
    e_pts, poumon_pts, _ = tissus(pts)
    pinf_pts = inflexion(pts[:, 1], e_pts)
    verite = np.stack(
        [
            (
                np.where(
                    poumon_pts,
                    volume_sigmoide(peep + p["delta_p"], pinf_pts, p["pente"])
                    - volume_sigmoide(peep, pinf_pts, p["pente"]),
                    0.0,
                )
                .reshape(GRID, GRID, -1)
                .mean(axis=-1)
                * image_mask()
            )
            for peep in PEEPS
        ]
    ).astype(np.float32)
    return {
        "v_exp": np.stack(v_exp),
        "v_insp": np.stack(v_insp),
        "verite": verite,
        "poumon": (poumon_pts.reshape(GRID, GRID, -1).mean(axis=-1) >= 0.5) & image_mask(),
        "params": p,
    }


def _liste(tableaux: list) -> np.ndarray:
    """Tableau d'objets 1D (cellule MATLAB), même si tous les éléments ont la même taille."""
    sortie = np.empty(len(tableaux), dtype=object)
    for i, a in enumerate(tableaux):
        sortie[i] = a
    return sortie


def simuler(par_patient: int) -> None:

    parts = separer(sorted(COUPES.glob("*.npz")))
    DOSSIER.mkdir(parents=True, exist_ok=True)
    for ensemble in ("val", "test"):
        essais = []
        for i, f in enumerate(parts[ensemble]):
            thorax = ThoraxScanner(f)
            for j in range(par_patient):
                essais.append(
                    essai(thorax, np.random.default_rng([31, i, j, ensemble == "test"])) | {"patient": f.stem}
                )
            print(f"{ensemble} : {f.stem}", flush=True)
        np.savez_compressed(
            DOSSIER / f"essais_{ensemble}.npz",
            v_exp=np.stack([e["v_exp"] for e in essais]),
            v_insp=np.stack([e["v_insp"] for e in essais]),
            verite=np.stack([e["verite"] for e in essais]),
            poumon=np.stack([e["poumon"] for e in essais]),
            delta_p=np.array([e["params"]["delta_p"] for e in essais]),
            patient=np.array([e["patient"] for e in essais]),
        )
        (DOSSIER / f"params_{ensemble}.json").write_text(
            json.dumps([{"patient": e["patient"]} | e["params"] for e in essais], indent=1)
        )
    exporter_eidors()


# ---------------------------------------------------------------- EIDORS


def exporter_eidors() -> None:
    """Entrée d'EIDORS : le maillage de reconstruction de NOSER (n_boundary 160), un maillage plus
    grossier pour les solveurs itératifs (plus lents ; 1 440 éléments, toujours plus fin que la
    grille de 580 pixels), le protocole, et toutes les mesures (validation puis test)."""
    from scipy.io import savemat

    injections, paires = adjacent_protocol()
    vh = np.concatenate([np.load(DOSSIER / f"essais_{e}.npz")["v_exp"].reshape(-1, 208) for e in ("val", "test")])
    vi = np.concatenate([np.load(DOSSIER / f"essais_{e}.npz")["v_insp"].reshape(-1, 208) for e in ("val", "test")])
    contenu = {
        "injections": injections,
        "paires": paires + 1,
        "vh": vh.T,
        "vi": vi.T,
        "pixels": pixel_centres(GRID)[:, :, 0].reshape(-1, 2)[image_mask().ravel()],
    }
    for suffixe, n in (("", 160), ("_g", 96)):
        m = build_mesh(n_boundary=n)
        contenu |= {
            f"noeuds{suffixe}": m.nodes,
            f"elements{suffixe}": m.elements + 1,
            f"electrodes{suffixe}": _liste([np.unique(a) + 1 for a in m.electrode_edges]),
        }
    savemat(DOSSIER / "eidors_entree.mat", contenu)


METHODES_EIDORS = ("GREIT", "Gauss-Newton une étape", "Gauss-Newton itératif", "Variation totale")


def eidors(limite: int = 0) -> None:
    """Une méthode EIDORS par conteneur, en parallèle. limite > 0 : premières images seulement."""
    from scipy.io import savemat

    val = np.load(DOSSIER / "essais_val.npz")
    verite = val["verite"].reshape(-1, GRID * GRID)[:, image_mask().ravel()].T  # (n_pixels, n_val)
    n_val = verite.shape[1] if limite <= 0 else min(verite.shape[1], limite)
    exporter_eidors()
    cache = Path.home() / "outils_eit" / "cache_eidors"
    cache.mkdir(exist_ok=True)
    processus = []
    t0 = time.perf_counter()
    for indice, methode in enumerate(METHODES_EIDORS, start=1):  # même ordre que eidors_reconstruire.m
        cle = "".join(c for c in methode.lower().replace(" ", "_") if c.isascii())  # noms de fichiers sans accents
        savemat(
            DOSSIER / f"verite_val_{cle}.mat",
            {
                "verite": verite[:, :n_val],
                "n_val": n_val,
                "limite": limite,
                "indice": indice,
                "fichier": f"eidors_sortie_{cle}.mat",
            },
        )
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{EIDORS}:/eidors",
            "-v", f"{RACINE / 'scripts'}:/scripts:ro",
            "-v", f"{DOSSIER}:/donnees",
            "-v", f"{cache}:/cache",
            "-v", f"{DOSSIER / f'verite_val_{cle}.mat'}:/donnees/verite_val.mat:ro",
            IMAGE_OCTAVE, "octave-cli", "--no-gui", "--eval", "addpath('/scripts'); eidors_reconstruire",
        ]  # fmt: skip
        journal = open(DOSSIER / f"eidors_{cle}.log", "w")  # noqa: SIM115
        processus.append((methode, subprocess.Popen(cmd, stdout=journal, stderr=subprocess.STDOUT), journal))
        print(f"lancé : {methode}", flush=True)
    for methode, p, journal in processus:
        p.wait()
        journal.close()
        print(f"{methode} : terminé (code {p.returncode}, {time.perf_counter() - t0:.0f} s)", flush=True)


# ---------------------------------------------------------------- évaluation


def _pixels(valeurs: np.ndarray) -> np.ndarray:
    """(n_pixels_masque, n_images) -> (n_images, GRID, GRID)."""
    img = np.zeros((valeurs.shape[1], GRID * GRID), np.float32)
    img[:, image_mask().ravel()] = valeurs.T
    return img.reshape(-1, GRID, GRID)


def couches(img: np.ndarray) -> np.ndarray:
    """Part (%) de la ventilation dans 4 couches horizontales de l'avant (sternum) vers l'arrière,
    comme les régions d'intérêt « en couches » des moniteurs EIT. img : (n, GRID, GRID)."""
    v = np.clip(img, 0, None) * image_mask()
    par_couche = v.reshape(len(v), 4, GRID // 4, GRID).sum(axis=(2, 3))
    total = par_couche.sum(axis=1, keepdims=True)
    return np.divide(100 * par_couche, total, out=np.full_like(par_couche, np.nan), where=total > 0)


def mesures_appareil(img: np.ndarray, poumon: np.ndarray) -> dict:
    """Les indices affichés par un moniteur EIT, pour une série d'images courantes (n, GRID, GRID)."""
    ss = [espaces_silencieux(im, poumon) for im in img]
    return {
        "quadrants": regional_distribution(img),
        "couches": couches(img),
        "centre": center_of_ventilation(img),
        "gi": global_inhomogeneity(img),
        "ss_dos": np.array([s["dependants"] for s in ss]),
        "ss_avant": np.array([s["non_dependants"] for s in ss]),
    }


def evaluer() -> dict:
    import onnxruntime as ort
    from scipy.io import loadmat

    from pulmoeit.recon import LinearReconstructor

    val, test = (np.load(DOSSIER / f"essais_{e}.npz") for e in ("val", "test"))
    n_val = val["v_exp"].shape[0] * val["v_exp"].shape[1]
    images = {}
    x = features(test["v_exp"].reshape(-1, 208), test["v_insp"].reshape(-1, 208))
    images["NOSER (méthode actuelle)"] = LinearReconstructor.load(RACINE / "models" / "baseline.npz")(x)
    for nom, dossier in (
        ("PostUNet ancien (ellipses)", "models"),
        ("PostUNet nouveau (ellipses + scanners)", "models_scanners"),
    ):
        s = ort.InferenceSession(str(RACINE / dossier / "pulmoeit.onnx"), providers=["CPUExecutionProvider"])
        images[nom] = s.run(None, {"dv": x.astype(np.float32)})[0].reshape(-1, GRID, GRID)
    # EIDORS : réglage choisi sur la validation
    sortie = []
    for f in sorted(DOSSIER.glob("eidors_sortie_*.mat")):
        m = loadmat(f, simplify_cells=True)["methodes"]
        sortie += m if isinstance(m, list) else [m]
    reglages = {}
    for m in sortie:  # réglage déjà choisi sur la validation par eidors_reconstruire.m
        reglages[m["nom"]] = {
            "reglage": float(m["hp"]),
            "erreurs_validation": np.atleast_1d(m["scores"]).tolist(),
            "secondes_par_image": float(m["secondes_par_image"]),
        }
        images[f"EIDORS {m['nom']}"] = -_pixels(np.atleast_2d(m["pixels"]))[
            n_val:
        ]  # ventilation = baisse de conductivité
    n_essais, n_paliers = test["v_exp"].shape[:2]
    verite = test["verite"]
    resultats: dict = {
        "essais_test": int(n_essais),
        "paliers": PEEPS.tolist(),
        "reglages_eidors": reglages,
        "methodes": {},
    }
    ref = [costa(verite[k], PEEPS, np.full(n_paliers, test["delta_p"][k])) for k in range(n_essais)]
    exemples = {}
    for nom, imgs in images.items():
        imgs = imgs.reshape(n_essais, n_paliers, GRID, GRID)
        err_peep, err_col, err_sur, accord, ss_err, corr = [], [], [], [], [], []
        app = {k: [] for k in ("quadrants", "couches", "centre", "gi", "ss_dos", "ss_avant")}
        for k in range(n_essais):
            dp = np.full(n_paliers, test["delta_p"][k])
            r = costa(imgs[k], PEEPS, dp)
            err_peep.append(r["peep_proposee"] - ref[k]["peep_proposee"])
            err_col.append(np.mean(np.abs(r["collapsus"] - ref[k]["collapsus"])))
            err_sur.append(np.mean(np.abs(r["surdistension"] - ref[k]["surdistension"])))
            # pixel par pixel : même classe que la vérité (normal / collabé / surdistendu) dans le poumon fonctionnel
            zone = ref[k]["roi"]  # poumon fonctionnel de la vérité ; un pixel hors de celui de la méthode compte comme désaccord
            accord.append(float(np.mean(r["classes"][:, zone] == ref[k]["classes"][:, zone])))
            poumon = test["poumon"][k]
            ss_err.append(
                np.mean(
                    [
                        abs(
                            espaces_silencieux(imgs[k, i], poumon)["dependants"]
                            - espaces_silencieux(verite[k, i], poumon)["dependants"]
                        )
                        for i in range(n_paliers)
                    ]
                )
            )
            corr.append(float(np.mean(pearson(imgs[k], verite[k]))))
            m_meth, m_ref = mesures_appareil(imgs[k], poumon), mesures_appareil(verite[k], poumon)
            for cle in app:
                app[cle].append(float(np.nanmean(np.abs(m_meth[cle] - m_ref[cle]))))
        err_peep = np.array(err_peep)
        resultats["methodes"][nom] = {
            # mesures affichées par l'appareil (erreur absolue moyenne sur tous les paliers)
            "erreur_quadrants_pts": float(np.mean(app["quadrants"])),
            "erreur_couches_pts": float(np.mean(app["couches"])),
            "erreur_centre_ventilation_pts": float(np.mean(app["centre"])),
            "erreur_gi": float(np.mean(app["gi"])),
            "erreur_zones_silencieuses_dos_pts": float(np.mean(app["ss_dos"])),
            "erreur_zones_silencieuses_avant_pts": float(np.mean(app["ss_avant"])),
            # outil d'essai de PEEP (méthode de Costa) et pixels
            "erreur_peep_moyenne_cmH2O": float(np.mean(np.abs(err_peep))),
            "erreur_peep_mediane_cmH2O": float(np.median(np.abs(err_peep))),
            "biais_peep_cmH2O": float(np.mean(err_peep)),
            "part_peep_a_2_cmH2O_pres": float(np.mean(np.abs(err_peep) <= 2)),
            "erreur_collapsus_pts": float(np.mean(err_col)),
            "erreur_surdistension_pts": float(np.mean(err_sur)),
            "accord_pixels": float(np.mean(accord)),
            "erreur_espaces_silencieux_dependants_pts": float(np.nanmean(ss_err)),
            "correlation_image": float(np.mean(corr)),
        }
        exemples[nom] = imgs[0]
        print(nom, json.dumps({k: round(v, 3) for k, v in resultats["methodes"][nom].items()}), flush=True)
    resultats["peep_verite"] = [r["peep_proposee"] for r in ref]
    (RACINE / "reports").mkdir(exist_ok=True)
    (RACINE / "reports" / "essai_peep.json").write_text(json.dumps(resultats, indent=1, ensure_ascii=False))
    figure(RACINE / "reports" / "essai_peep.png", verite[0], test["delta_p"][0], exemples)
    return resultats


def figure(chemin: Path, verite: np.ndarray, delta_p: float, exemples: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dp = np.full(len(PEEPS), delta_p)
    series = {"Vérité": verite} | exemples
    fig, axes = plt.subplots(
        len(series), 4, figsize=(12, 2.6 * len(series)), gridspec_kw={"width_ratios": [1, 1, 1, 1.5]}
    )
    masque = image_mask()
    for ligne, (nom, imgs) in enumerate(series.items()):
        r = costa(imgs, PEEPS, dp)
        for col, i in enumerate((0, 5, 9)):  # PEEP 24, 14, 6
            v = np.clip(imgs[i], 0, None)
            axes[ligne, col].imshow(np.where(masque, v / max(v.max(), 1e-12), np.nan), cmap="Blues", vmin=0, vmax=1)
            axes[ligne, col].set(xticks=[], yticks=[])
            axes[ligne, col].set_title(
                f"{nom}\nPEEP {PEEPS[i]:.0f}" if col == 0 else f"PEEP {PEEPS[i]:.0f}", fontsize=8
            )
        ax = axes[ligne, 3]
        ax.plot(PEEPS, r["collapsus"], "-o", ms=3, label="collapsus")
        ax.plot(PEEPS, r["surdistension"], "-o", ms=3, label="surdistension")
        ax.axvline(r["peep_proposee"], color="k", ls="--", lw=1)
        ax.set_title(f"PEEP proposée {r['peep_proposee']:.1f} cmH₂O", fontsize=8)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=7)
        if ligne == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(chemin, dpi=90)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sous = parser.add_subparsers(dest="etape", required=True)
    s = sous.add_parser("simuler")
    s.add_argument("--par-patient", type=int, default=3)
    ei = sous.add_parser("eidors")
    ei.add_argument("--limite", type=int, default=0, help="essai rapide sur les premières images")
    sous.add_parser("evaluer")
    args = parser.parse_args()
    if args.etape == "simuler":
        simuler(args.par_patient)
    elif args.etape == "eidors":
        eidors(args.limite)
    else:
        evaluer()
    return 0


if __name__ == "__main__":
    sys.exit(main())
