"""Vraies données : titration de la PEEP chez un enfant en SDRA (EIDORS, jeu « cg-2012-ards-recruitment »).

Données : GK Wolf, C Gómez-Laberge, JN Kheir, D Zurakowski, BK Walsh, A Adler, JH Arnold,
Children's Hospital Boston, 2009 (NCT00830284) ; C Gómez-Laberge, JH Arnold, GK Wolf, IEEE Trans Med
Imag 2012;31:834-842 ; GK Wolf et al., Pediatr Crit Care Med 2012;13:509-515. Licence Creative
Commons avec attribution. Appareil Viasys (Goe-MF II), 16 électrodes, 13 images/s.

Titration décroissante après recrutement (fichiers d1 à d4, plages d'images et pressions du script
des auteurs) : PEEP 20, 18, 16, 14 cmH2O, pression motrice 6, 6, 5, 4 cmH2O. Pour chaque palier,
les respirations sont détectées et moyennées (fin d'expiration, fin d'inspiration) ; l'image courante
est reconstruite par chaque méthode (EIDORS : réglage choisi en simulation, scripts/essai_peep.py),
puis la méthode de Costa et les mesures d'un moniteur sont calculées.

Il n'y a pas de vérité terrain : on compare les méthodes entre elles et à la physiologie attendue.
Le sens droite / gauche des électrodes n'est pas documenté pour ce jeu (l'ordre des mesures, lui, est
vérifié sur le profil des tensions) ; les indices utilisés (collapsus, surdistension, couches de
l'avant vers l'arrière) n'en dépendent pas.

    python scripts/peep_reel.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
sys.path.insert(0, str(RACINE / "scripts"))

from essai_peep import DOSSIER, EIDORS, IMAGE_OCTAVE, METHODES_EIDORS, _liste, _pixels, couches  # noqa: E402
from evaluer_donnees_reelles import cycles, lire_get, vers_protocole  # noqa: E402

from pulmoeit.grid import GRID, image_mask, pixel_centres  # noqa: E402
from pulmoeit.metrics import center_of_ventilation, pearson  # noqa: E402
from pulmoeit.peep import costa  # noqa: E402

DONNEES = Path.home() / "pulmoeit_reel" / "peep" / "cg" / "DATA" / "STUDYNAME" / "SUBJECT_1" / "YYYYMMDD" / "Eit" / "Viasys"
URL = "https://eidors3d.sourceforge.net/data_contrib/cg-2012-ards-recruitment/"
# (fichier, première et dernière image (1-indexées, incluses ; None = tout), PEEP, pression motrice)
PALIERS = [
    ("1001_d1.get", (1, 765), 20.0, 6.0),
    ("1001_d2.get", None, 18.0, 6.0),
    ("1001_d3.get", (368, 780), 16.0, 5.0),
    ("1001_d4.get", (411, 780), 14.0, 4.0),
]
IMAGES_PAR_SECONDE = 13


def paliers() -> list[dict]:
    """Pour chaque palier : tensions moyennes en fin d'expiration et en fin d'inspiration."""
    sortie = []
    for fichier, plage, peep, dp in PALIERS:
        v = vers_protocole(lire_get(DONNEES / fichier))
        if plage is not None:
            v = v[plage[0] - 1 : plage[1]]
        signal = np.abs(v).sum(axis=1)
        resp = cycles(signal, IMAGES_PAR_SECONDE)
        sortie.append(
            {
                "fichier": fichier,
                "peep": peep,
                "delta_p": dp,
                "respirations": len(resp),
                "vh": np.mean([v[b] for b, _ in resp], axis=0),
                "vi": np.mean([v[h] for _, h in resp], axis=0),
            }
        )
        print(f"{fichier} : PEEP {peep:.0f}, {len(resp)} respirations", flush=True)
    return sortie


def eidors_reel(vh: np.ndarray, vi: np.ndarray) -> dict[str, np.ndarray]:
    """Reconstruit les paliers avec chaque méthode EIDORS, au réglage choisi en simulation."""
    from scipy.io import loadmat, savemat

    from pulmoeit.mesh import build_mesh
    from pulmoeit.protocol import adjacent_protocol

    injections, paires = adjacent_protocol()
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
    savemat(DOSSIER / "eidors_entree_reel.mat", contenu)
    images = {}
    for indice, methode in enumerate(METHODES_EIDORS, start=1):
        cle = "".join(c for c in methode.lower().replace(" ", "_") if c.isascii())
        reglage = float(loadmat(DOSSIER / f"eidors_sortie_{cle}.mat", simplify_cells=True)["methodes"]["hp"])
        controle = DOSSIER / f"reel_{cle}.mat"
        savemat(
            controle,
            {"n_val": 0, "limite": 0, "indice": indice, "hp_fixe": reglage, "entree": "eidors_entree_reel.mat", "fichier": f"reel_sortie_{cle}.mat"},
        )
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{EIDORS}:/eidors",
            "-v", f"{RACINE / 'scripts'}:/scripts:ro",
            "-v", f"{DOSSIER}:/donnees",
            "-v", f"{Path.home() / 'outils_eit' / 'cache_eidors'}:/cache",
            "-v", f"{controle}:/donnees/verite_val.mat:ro",
            IMAGE_OCTAVE, "octave-cli", "--no-gui", "--eval", "addpath('/scripts'); eidors_reconstruire",
        ]  # fmt: skip
        subprocess.run(cmd, check=True, capture_output=True)
        m = loadmat(DOSSIER / f"reel_sortie_{cle}.mat", simplify_cells=True)["methodes"]
        images[f"EIDORS {methode}"] = -_pixels(np.atleast_2d(m["pixels"]))
        print(f"EIDORS {methode} : réglage {reglage:g}", flush=True)
    return images


def main() -> int:
    import onnxruntime as ort

    from pulmoeit.recon import LinearReconstructor
    from pulmoeit.signals import features

    pal = paliers()
    vh, vi = np.stack([p["vh"] for p in pal]), np.stack([p["vi"] for p in pal])
    x = features(vh, vi)
    images = {"NOSER (méthode actuelle)": LinearReconstructor.load(RACINE / "models" / "baseline.npz")(x)}
    for nom, dossier in (("PostUNet ancien (ellipses)", "models"), ("PostUNet nouveau (ellipses + scanners)", "models_scanners")):
        s = ort.InferenceSession(str(RACINE / dossier / "pulmoeit.onnx"), providers=["CPUExecutionProvider"])
        images[nom] = s.run(None, {"dv": x.astype(np.float32)})[0].reshape(-1, GRID, GRID)
    images |= eidors_reel(vh, vi)
    peeps = np.array([p["peep"] for p in pal])
    dps = np.array([p["delta_p"] for p in pal])
    resultats: dict = {
        "donnees": {"source": URL, "paliers": [{k: p[k] for k in ("fichier", "peep", "delta_p", "respirations")} for p in pal]},
        "methodes": {},
    }
    for nom, imgs in images.items():
        r = costa(imgs, peeps, dps)
        resultats["methodes"][nom] = {
            "peep_proposee": r["peep_proposee"],
            "collapsus_pct": r["collapsus"].round(1).tolist(),
            "surdistension_pct": r["surdistension"].round(1).tolist(),
            "couches_pct": couches(imgs).round(1).tolist(),  # avant -> arrière, par palier
            "centre_ventilation_pct": center_of_ventilation(imgs).round(1).tolist(),
        }
        print(nom, json.dumps(resultats["methodes"][nom]), flush=True)
    # accord entre méthodes (pas de vérité) : corrélation moyenne des images avec les autres méthodes
    noms = list(images)
    for a in noms:
        resultats["methodes"][a]["accord_avec_les_autres"] = float(
            np.mean([np.mean(pearson(images[a], images[b])) for b in noms if b != a])
        )
    (RACINE / "reports" / "peep_reel.json").write_text(json.dumps(resultats, indent=1, ensure_ascii=False))
    figure(RACINE / "reports" / "peep_reel.png", images, peeps, dps)
    return 0


def figure(chemin: Path, images: dict, peeps: np.ndarray, dps: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masque = image_mask()
    fig, axes = plt.subplots(len(images), len(peeps) + 1, figsize=(2.4 * len(peeps) + 3.6, 2.4 * len(images)),
                             gridspec_kw={"width_ratios": [1] * len(peeps) + [1.5]})  # fmt: skip
    for ligne, (nom, imgs) in enumerate(images.items()):
        for col, peep in enumerate(peeps):
            v = np.clip(imgs[col], 0, None)
            axes[ligne, col].imshow(np.where(masque, v / max(v.max(), 1e-12), np.nan), cmap="Blues", vmin=0, vmax=1)
            axes[ligne, col].set(xticks=[], yticks=[])
            axes[ligne, col].set_title(f"{nom}\nPEEP {peep:.0f}" if col == 0 else f"PEEP {peep:.0f}", fontsize=7)
        r = costa(imgs, peeps, dps)
        ax = axes[ligne, -1]
        ax.plot(peeps, r["collapsus"], "-o", ms=3, label="collapsus")
        ax.plot(peeps, r["surdistension"], "-o", ms=3, label="surdistension")
        ax.axvline(r["peep_proposee"], color="k", ls="--", lw=1)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=7)
        ax.set_title(f"PEEP proposée {r['peep_proposee']:.1f}", fontsize=7)
        if ligne == 0:
            ax.legend(fontsize=6)
    fig.suptitle("Enfant en SDRA, titration de la PEEP (EIDORS cg-2012-ards-recruitment) : vraies mesures, pas de vérité", fontsize=9)
    fig.tight_layout()
    fig.savefig(chemin, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
