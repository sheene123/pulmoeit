"""Animations de la ventilation pendant la respiration (GIF), image par image.

1. Nouveau-né réel (EIDORS, voir evaluer_donnees_reelles.py) : NOSER et PostUNet sur les 220 images
   de l'enregistrement, à vitesse réelle (13 images/s). Chaque image est la différence avec la fin
   d'expiration précédente : elle s'allume à l'inspiration et s'éteint à l'expiration. Pas de
   vérité terrain.
2. Patient simulé en SDRA (atélectasie dorsale) sur deux respirations : vérité, NOSER et PostUNet.

Chaque méthode a sa propre échelle de couleurs (leurs unités diffèrent), fixe sur toute l'animation.

    python scripts/animer_respiration.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
sys.path.insert(0, str(RACINE / "scripts"))

from evaluer_donnees_reelles import (  # noqa: E402
    IMAGES_PAR_SECONDE,
    cycles,
    lire_get,
    telecharger,
    vers_protocole,
)

from pulmoeit.grid import GRID, image_mask  # noqa: E402
from pulmoeit.recon import LinearReconstructor  # noqa: E402
from pulmoeit.signals import features  # noqa: E402


def reconstruire(modeles: Path, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    noser = LinearReconstructor.load(modeles / "baseline.npz")(x).reshape(-1, GRID, GRID)
    session = ort.InferenceSession(str(modeles / "pulmoeit.onnx"), providers=["CPUExecutionProvider"])
    reseau = session.run(None, {"dv": x.astype(np.float32)})[0].reshape(-1, GRID, GRID)
    return noser, reseau


def _echelle(images: np.ndarray) -> float:
    return max(float(np.percentile(np.abs(images[:, image_mask()]), 99.5)), 1e-12)


def animer(
    chemin: Path,
    titre: str,
    temps: np.ndarray,
    courbe: np.ndarray,
    etiquette_courbe: str,
    panneaux: list[tuple[str, np.ndarray]],
    images_par_seconde: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    masque = image_mask()
    n = len(panneaux)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.3 * n + 4.2, 3.7), gridspec_kw={"width_ratios": [1.5] + [1] * n})
    axes[0].plot(temps, courbe, color="0.35", lw=1.2)
    curseur = axes[0].axvline(temps[0], color="#c0392b", lw=1.5)
    point = axes[0].plot([temps[0]], [courbe[0]], "o", color="#c0392b")[0]
    axes[0].set(xlabel="temps (s)", ylabel=etiquette_courbe, yticks=[])
    axes[0].set_title("Respiration", fontsize=10)

    rendus = []
    for ax, (nom, images) in zip(axes[1:], panneaux, strict=True):
        e = _echelle(images)
        rendu = ax.imshow(np.where(masque, images[0] / e, np.nan), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(nom, fontsize=10, pad=16)
        ax.set(xticks=[], yticks=[])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(GRID / 2 - 0.5, -0.8, "avant", ha="center", va="bottom", fontsize=8, color="0.3")
        ax.text(GRID / 2 - 0.5, GRID - 0.2, "dos", ha="center", va="top", fontsize=8, color="0.3")
        ax.text(-0.8, GRID / 2, "D", ha="right", va="center", fontsize=9, color="0.3")
        ax.text(GRID - 0.2, GRID / 2, "G", ha="left", va="center", fontsize=9, color="0.3")
        rendus.append((rendu, images, e))

    fig.suptitle(titre, fontsize=11)
    fig.text(
        0.5,
        0.015,
        "rouge : l'air entre (plus d'impédance) ; bleu : effet inverse",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    def image(k: int):
        curseur.set_xdata([temps[k], temps[k]])
        point.set_data([temps[k]], [courbe[k]])
        for rendu, images, e in rendus:
            rendu.set_data(np.where(masque, images[k] / e, np.nan))
        return [curseur, point] + [r for r, _, _ in rendus]

    animation = FuncAnimation(fig, image, frames=len(temps), blit=True)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    animation.save(chemin, writer=PillowWriter(fps=images_par_seconde), dpi=80)
    plt.close(fig)
    print(f"{chemin} ({chemin.stat().st_size / 1e6:.1f} Mo, {len(temps)} images)")


def nouveau_ne(donnees: Path, modeles: Path, sortie: Path) -> None:
    v = vers_protocole(lire_get(telecharger(donnees)))
    signal = np.abs(v).sum(axis=1)
    fins_expiration = [b for b, _ in cycles(signal, IMAGES_PAR_SECONDE)]
    # l'animation commence à la première fin d'expiration ; chaque image est comparée à la dernière
    # fin d'expiration qui la précède
    debut = fins_expiration[0]
    images = np.arange(debut, len(v))
    ref = [max(b for b in fins_expiration if b <= k) for k in images]
    x = np.concatenate([features(v[r], v[k]) for k, r in zip(images, ref, strict=True)])
    noser, reseau = reconstruire(modeles, x)
    animer(
        sortie / "respiration_nouveau_ne.gif",
        f"Nouveau-né réel en respiration spontanée (EIDORS) : {len(images) / IMAGES_PAR_SECONDE:.0f} s à vitesse réelle",
        (images - debut) / IMAGES_PAR_SECONDE,
        signal[images],
        "impédance globale",
        [("NOSER (classique)", noser), ("PostUNet (appris)", reseau)],
        IMAGES_PAR_SECONDE,
    )


def sdra_simule(modeles: Path, sortie: Path, graine: int = 7) -> None:
    from pulmoeit.fem import ForwardModel
    from pulmoeit.geometry import REFERENCE, Geometry
    from pulmoeit.mesh import build_mesh
    from pulmoeit.phantom import sample_phantom
    from pulmoeit.protocol import N_ELECTRODES

    rng = np.random.default_rng(graine)
    geometrie = Geometry(aspect=0.72, fourier=((0.01, -0.01), (0.0, 0.01), (0.005, 0.0)))
    maillage = build_mesh(geometrie, 224, electrode_width=0.3, electrode_shift=rng.normal(0, 0.03, N_ELECTRODES))
    direct = ForwardModel(maillage, rng.uniform(0.005, 0.03, N_ELECTRODES))
    fantome = sample_phantom(rng, "dorsal_atelectasis")
    sigma_exp, _, ventilation = fantome.evaluate(geometrie.map_to(maillage.centroids, REFERENCE))

    par_cycle, n_cycles = 26, 2
    t = np.arange(par_cycle * n_cycles) / par_cycle  # en respirations
    phase = (1 - np.cos(2 * np.pi * t)) / 2  # 0 en fin d'expiration, 1 en fin d'inspiration
    v_exp = direct.measure(sigma_exp)
    bruit = np.sqrt(np.mean(v_exp**2)) * 10 ** (-60 / 20)
    v_exp = v_exp + rng.normal(0, bruit, v_exp.shape)
    x = np.concatenate(
        [
            features(v_exp, direct.measure(sigma_exp * (1 - a * ventilation)) + rng.normal(0, bruit, v_exp.shape))
            for a in phase
        ]
    )
    noser, reseau = reconstruire(modeles, x)
    verite = phase[:, None, None] * fantome.render()[None]
    animer(
        sortie / "respiration_sdra_simule.gif",
        "Patient simulé, SDRA : l'arrière des poumons est écrasé et ne reçoit plus d'air",
        t * 3.0,  # 3 s par respiration
        phase,
        "volume d'air",
        [("Vérité (simulation)", verite), ("NOSER (classique)", noser), ("PostUNet (appris)", reseau)],
        par_cycle / 3.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--donnees", type=Path, default=RACINE / "data" / "reel" / "neonate")
    parser.add_argument("--modeles", type=Path, default=RACINE / "models")
    parser.add_argument("--sortie", type=Path, default=RACINE / "reports" / "animations")
    args = parser.parse_args()
    nouveau_ne(args.donnees, args.modeles, args.sortie)
    sdra_simule(args.modeles, args.sortie)
    return 0


if __name__ == "__main__":
    sys.exit(main())
