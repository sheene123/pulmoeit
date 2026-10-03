"""Test final sur les 21 cuves d'évaluation du KTC2023, avec le score officiel.

Compare le U-Net à la méthode de référence des organisateurs (différence linéarisée + seuillage
d'Otsu), recalculée ici sur les mêmes cuves, et rappelle les scores publiés des équipes. Les
cuves d'évaluation n'ont servi ni à l'entraînement ni au choix du modèle. Tout tourne sur le
processeur.

    python -m pulmoeit.ktc.evaluer --code <Codes_Python> --evaluation <EvaluationData> \\
        --modele models/ktc/unet_ktc.pt --reference <scores de la référence (json)>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .officiel import NIVEAUX
from .score import score_officiel, score_rapide

# scores officiels publiés (Denker et al., 2024, tableau des résultats du KTC2023)
PUBLIES = {
    "Bremen, post-traitement (1er)": [2.76, 2.56, 2.54, 1.71, 2.06, 1.92, 1.69],
    "Bremen, FC U-Net": [2.72, 2.64, 2.31, 1.80, 2.06, 2.07, 1.53],
    "Team ABC": [2.75, 2.37, 2.07, 1.74, 1.08, 1.53, 1.22],
    "Team DTU": [2.28, 2.30, 1.87, 1.55, 1.34, 1.44, 1.60],
}


def evaluer(code: Path, evaluation: Path, modele_chemin: Path, verifier_officiel: int = 3) -> dict:
    import scipy.io
    import torch

    from .entrainer import UNetKTC, predire
    from .lineaire import operateurs
    from .officiel import importer_code

    importer_code(code)
    ref = scipy.io.loadmat(code / "TrainingData" / "ref.mat")
    R = torch.from_numpy(operateurs(code, ref["Injref"], ref["Mpat"], ref["Uelref"]))
    modele = UNetKTC()
    modele.load_state_dict(torch.load(modele_chemin, map_location="cpu", weights_only=True))
    scores, predictions, verifications = {}, {}, []
    for niveau in NIVEAUX:
        dossier = evaluation / "evaluation_datasets" / f"level{niveau}"
        u_ref = scipy.io.loadmat(dossier / "ref.mat")["Uelref"].ravel()
        for k, lettre in enumerate("ABC", start=1):
            du = scipy.io.loadmat(dossier / f"data{k}.mat")["Uel"].ravel() - u_ref
            verite = scipy.io.loadmat(evaluation / "GroundTruths" / f"level_{niveau}" / f"{k}_true.mat")["truth"]
            classes = predire(modele, R, du, niveau)
            cle = f"{niveau}{lettre}"
            scores[cle] = score_rapide(verite, classes)
            predictions[cle] = classes
            if len(verifications) < verifier_officiel:  # le score rapide doit égaler l'officiel
                verifications.append(abs(score_officiel(verite, classes) - scores[cle]))
    par_niveau = [sum(scores[f"{n}{c}"] for c in "ABC") for n in NIVEAUX]
    return {
        "scores": scores,
        "par_niveau": par_niveau,
        "total": float(sum(par_niveau)),
        "ecart_max_score_officiel": float(max(verifications)),
        "_predictions": predictions,
    }


def figure(chemin: Path, evaluation: Path, reference_dossier: Path, predictions: dict, cles: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy.io

    fig, axes = plt.subplots(3, len(cles), figsize=(2.3 * len(cles), 7))
    for j, cle in enumerate(cles):
        niveau, k = int(cle[0]), "ABC".index(cle[1]) + 1
        verite = scipy.io.loadmat(evaluation / "GroundTruths" / f"level_{niveau}" / f"{k}_true.mat")["truth"]
        reference = scipy.io.loadmat(reference_dossier / f"niveau{niveau}" / f"{k}.mat")["reconstruction"]
        for i, (img, titre) in enumerate(
            ((verite, "vérité"), (reference, "référence"), (predictions[cle], "PulmoEIT"))
        ):
            axes[i, j].imshow(img, cmap="viridis", vmin=0, vmax=2)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if j == 0:
                axes[i, j].set_ylabel(titre)
        axes[0, j].set_title(f"niveau {cle}")
    fig.suptitle("KTC2023 : cuves d'évaluation (jaune : conducteur, vert : résistif)")
    fig.tight_layout()
    fig.savefig(chemin, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--modele", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="scores de la méthode de référence (json)")
    parser.add_argument("--sorties-reference", type=Path, required=True, help="images de la méthode de référence")
    parser.add_argument("--sortie", type=Path, default=Path("reports/ktc2023.json"))
    args = parser.parse_args()
    r = evaluer(args.code, args.evaluation, args.modele)
    reference = json.loads(args.reference.read_text())
    resultat = {
        "pulmoeit": {k: v for k, v in r.items() if not k.startswith("_")},
        "reference_organisateurs": {
            "scores": reference["scores"],
            "par_niveau": [reference["par_niveau"][str(n)] for n in NIVEAUX],
            "total": reference["total"],
        },
        "publies": {nom: {"par_niveau": v, "total": round(sum(v), 2)} for nom, v in PUBLIES.items()},
    }
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    args.sortie.write_text(json.dumps(resultat, indent=1, ensure_ascii=False))
    figure(
        args.sortie.with_suffix(".png"),
        args.evaluation,
        args.sorties_reference,
        r["_predictions"],
        ["1A", "2B", "3C", "4A", "5B", "6C", "7A"],
    )
    print(f"PulmoEIT : {r['total']:.2f} / 21   (référence des organisateurs : {reference['total']:.2f})")
    for nom, v in resultat["publies"].items():
        print(f"   {nom} : {v['total']:.2f}")


if __name__ == "__main__":
    main()
