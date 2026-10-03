"""Première confrontation à des mesures réelles : un nouveau-né en respiration spontanée.

Données : EIDORS, jeu « if-neonate-spontaneous » (I. Frerichs ; Heinrich et al., Intensive Care
Med. 32:1392-1398, 2006). Nouveau-né de 10 jours en décubitus ventral, tête tournée à gauche,
appareil Goe-MF II, 16 électrodes, protocole adjacent, 13 images/s, 220 images. L'électrode n° 1
est à l'avant du thorax, la n° 5 sur le côté gauche, la n° 9 dans le dos, la n° 13 à droite.

Il n'y a pas de vérité terrain chez un patient. Ce qui est vérifié, fixé avant de regarder les
images :

1. localisation : la ventilation se trouve sur les côtés (poumons) plus qu'au centre (cœur,
   médiastin) ;
2. reproductibilité : l'image d'un cycle ressemble à celle des autres cycles ;
3. contrôle qualité : des mesures réelles correctes ne doivent pas être rejetées ;
4. cohérence : le réseau et la reconstruction classique (NOSER) placent la ventilation au même
   endroit.

    python scripts/evaluer_donnees_reelles.py            # télécharge les données si besoin
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from pulmoeit.grid import GRID, image_mask, pixel_centres  # noqa: E402
from pulmoeit.metrics import (  # noqa: E402
    QUADRANTS,
    center_of_ventilation,
    global_inhomogeneity,
    pearson,
    regional_distribution,
)
from pulmoeit.monitoring import MeasurementQC, drift_report  # noqa: E402
from pulmoeit.protocol import N_ELECTRODES, adjacent_protocol  # noqa: E402
from pulmoeit.recon import LinearReconstructor  # noqa: E402
from pulmoeit.signals import features  # noqa: E402

URL = "https://eidors3d.sourceforge.net/data_contrib/if-neonate-spontaneous/if-neonate-spontaneous.zip"
FICHIER = "P04P-1016.get"
IMAGES_PAR_SECONDE = 13
BANDE_CENTRALE = 0.2  # |x| < 0,2 (thorax de largeur 2) : cœur et médiastin


def telecharger(dossier: Path) -> Path:
    chemin = dossier / FICHIER
    if not chemin.exists():
        dossier.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as reponse:
            zipfile.ZipFile(io.BytesIO(reponse.read())).extractall(dossier)
    return chemin


def lire_get(chemin: Path) -> np.ndarray:
    """Fichier .get du Goe-MF II : 256 réels par image, dont 208 mesures (16 injections × 13)
    puis 48 valeurs annexes."""
    brut = np.fromfile(chemin, dtype="<f4")
    return brut.reshape(-1, 256)[:, : N_ELECTRODES * (N_ELECTRODES - 3)].astype(np.float64)


def vers_protocole(mesures: np.ndarray) -> np.ndarray:
    """Mesures Goe-MF II (n, 208) -> ordre et conventions du simulateur PulmoEIT.

    - ordre : pour l'injection d, l'appareil donne les paires m = d+2, d+3, …, d+14 ; le simulateur
      range les paires par numéro d'électrode croissant ;
    - sens de numérotation : électrode 1 devant et 5 à gauche chez le patient, alors que dans le
      simulateur l'électrode 0 est devant et la numérotation tourne vers la droite du patient. Une
      électrode j du simulateur est donc l'électrode (-j) mod 16 de l'appareil ; une paire
      adjacente (j, j+1) devient la paire (-j-1, -j) parcourue en sens inverse, ce qui inverse le
      signe à l'injection et à la mesure, soit aucun changement de signe au total ;
    - signe : l'appareil enregistre des amplitudes positives, le simulateur des tensions
      négatives (le potentiel décroît de la source au puits le long du bord)."""
    n = N_ELECTRODES
    appareil = mesures.reshape(len(mesures), n, n - 3)
    _, paires = adjacent_protocol(n)
    sortie = np.empty((len(mesures), len(paires)))
    for k, (d, m) in enumerate(paires):
        d_app, m_app = (-d - 1) % n, (-m - 1) % n
        sortie[:, k] = appareil[:, d_app, (m_app - d_app - 2) % n]
    return -sortie


def cycles(signal: np.ndarray, images_par_seconde: int) -> list[tuple[int, int]]:
    """(fin d'expiration, fin d'inspiration suivante) pour chaque respiration complète."""
    from scipy.signal import find_peaks

    lisse = np.convolve(signal, np.ones(3) / 3, mode="same")
    ecart = max(2, int(0.5 * images_par_seconde))  # au plus 2 respirations par seconde
    prominence = 0.3 * np.std(lisse)
    hauts, _ = find_peaks(lisse, distance=ecart, prominence=prominence)
    bas, _ = find_peaks(-lisse, distance=ecart, prominence=prominence)
    paires = []
    for b in bas:
        suivants = hauts[hauts > b]
        if suivants.size and (not paires or b > paires[-1][1]):
            paires.append((int(b), int(suivants[0])))
    return paires


def part_centrale(images: np.ndarray) -> np.ndarray:
    """Part de la ventilation (valeurs positives) située dans la bande centrale |x| < 0,2."""
    x = pixel_centres(GRID)[:, :, 0, 0]
    centre = (np.abs(x) < BANDE_CENTRALE) & image_mask()
    v = np.clip(images, 0, None) * image_mask()
    total = v.sum(axis=(1, 2))
    return np.divide(v[:, centre].sum(axis=1), total, out=np.full(len(v), np.nan), where=total > 0)


def reproductibilite(images: np.ndarray) -> float:
    """Corrélation moyenne entre les images de deux respirations différentes."""
    n = len(images)
    if n < 2:
        return float("nan")
    paires = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean([pearson(images[i : i + 1], images[j : j + 1])[0] for i, j in paires]))


def resume(nom: str, images: np.ndarray) -> dict:
    moyenne = images.mean(axis=0, keepdims=True)
    return {
        "methode": nom,
        "part_ventilation_bande_centrale": float(np.nanmean(part_centrale(images))),
        "part_surface_bande_centrale": float(
            ((np.abs(pixel_centres(GRID)[:, :, 0, 0]) < BANDE_CENTRALE) & image_mask()).sum() / image_mask().sum()
        ),
        "reproductibilite_entre_cycles": reproductibilite(images),
        "gi": float(global_inhomogeneity(moyenne)[0]),
        "centre_ventilation_pct": float(center_of_ventilation(moyenne)[0]),
        "repartition_pct": dict(zip(QUADRANTS, map(float, regional_distribution(moyenne)[0]), strict=True)),
        "image_moyenne_positive": bool(moyenne.sum() > 0),
    }


def figure(chemin: Path, signal: np.ndarray, paires: list, noser: np.ndarray, reseau: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masque = image_mask()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    t = np.arange(len(signal)) / IMAGES_PAR_SECONDE
    axes[0].plot(t, signal, color="0.3", lw=1)
    for b, h in paires:
        axes[0].plot(t[b], signal[b], "v", color="#2a6fb0")
        axes[0].plot(t[h], signal[h], "^", color="#c0392b")
    axes[0].set(xlabel="temps (s)", ylabel="somme des tensions (u. a.)", title="Respiration détectée")
    for ax, img, titre in ((axes[1], noser, "NOSER (classique)"), (axes[2], reseau, "PostUNet (appris)")):
        m = np.where(masque, img / max(np.abs(img).max(), 1e-12), np.nan)
        ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set(title=titre, xticks=[], yticks=[])
        ax.text(1, 1, "D", va="top", fontsize=9)  # droite du patient à gauche de l'image
        ax.text(GRID - 2, 1, "G", va="top", ha="right", fontsize=9)
        ax.text(GRID / 2, 0.5, "avant", va="top", ha="center", fontsize=8)
    fig.suptitle("Nouveau-né en respiration spontanée (EIDORS) : ventilation moyenne d'un cycle")
    fig.tight_layout()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin, dpi=130)
    plt.close(fig)


def main() -> int:
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--donnees", type=Path, default=RACINE / "data" / "reel" / "neonate")
    parser.add_argument("--modeles", type=Path, default=RACINE / "models")
    parser.add_argument("--sortie", type=Path, default=RACINE / "reports" / "donnees_reelles.json")
    args = parser.parse_args()

    v = vers_protocole(lire_get(telecharger(args.donnees)))
    signal = np.abs(v).sum(axis=1)  # l'air de l'inspiration augmente l'impédance, donc les tensions
    paires = cycles(signal, IMAGES_PAR_SECONDE)
    x = np.concatenate([features(v[b], v[h]) for b, h in paires])

    noser = LinearReconstructor.load(args.modeles / "baseline.npz")(x)
    session = ort.InferenceSession(str(args.modeles / "pulmoeit.onnx"), providers=["CPUExecutionProvider"])
    reseau = session.run(None, {"dv": x.astype(np.float32)})[0].reshape(-1, GRID, GRID)

    qc = MeasurementQC.load(args.modeles / "qc.npz")
    scores = qc.score(x)
    sortie = {
        "donnees": {
            "source": URL,
            "images": int(len(v)),
            "duree_s": round(len(v) / IMAGES_PAR_SECONDE, 1),
            "respirations": len(paires),
            "frequence_respiratoire_par_min": round(
                60 * IMAGES_PAR_SECONDE / float(np.median(np.diff([h for _, h in paires]))), 1
            )
            if len(paires) > 1
            else None,
        },
        "noser": resume("NOSER", noser),
        "postunet": resume("PostUNet", reseau),
        "coherence_noser_postunet": float(
            pearson(noser.mean(axis=0, keepdims=True), reseau.mean(axis=0, keepdims=True))[0]
        ),
        "controle_qualite": {
            "seuil": qc.threshold,
            "score_median": float(np.median(scores)),
            "part_rejetee": float(np.mean(scores > qc.threshold)),
            "electrode_suspecte_la_plus_frequente": int(np.bincount(qc.suspect_electrode(x)).argmax()),
            "derive": drift_report(qc.reference_scores, scores),
        },
    }
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    args.sortie.write_text(json.dumps(sortie, indent=1, ensure_ascii=False))
    figure(args.sortie.with_suffix(".png"), signal, paires, noser.mean(axis=0), reseau.mean(axis=0))
    print(json.dumps(sortie, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
