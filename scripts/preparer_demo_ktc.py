"""Prépare les données de la page « KTC2023 » de la démo web.

Pour chaque cuve réelle (21 d'évaluation à leur niveau, 4 d'entraînement aux 7 niveaux) : l'entrée
du réseau (reconstruction linéaire normalisée, 128 × 128), la vérité terrain et, pour l'évaluation,
la méthode de référence des organisateurs, réduites en 128 × 128 pour l'affichage, et les scores
officiels (calculés ici en 256 × 256). Le réseau lui-même tourne dans le navigateur (ONNX).

    python scripts/preparer_demo_ktc.py --ktc ~/pulmoeit_reel/ktc --modele models/ktc/unet_ktc.pt
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))


def _b64(classes: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(classes, dtype=np.uint8).tobytes()).decode()


def main() -> int:
    import scipy.io
    import torch

    from pulmoeit.ktc.entrainer import UNetKTC, predire, reconstruire
    from pulmoeit.ktc.fantomes import reduire
    from pulmoeit.ktc.lineaire import COTE_ENTREE, operateurs
    from pulmoeit.ktc.score import score_rapide

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--ktc", type=Path, required=True, help="dossier avec Codes_Python, EvaluationData, sorties_reference"
    )
    parser.add_argument("--modele", type=Path, default=RACINE / "models" / "ktc" / "unet_ktc.pt")
    parser.add_argument("--sortie", type=Path, default=RACINE / "demo" / "web" / "ktc")
    args = parser.parse_args()
    code = args.ktc / "Codes_Python"
    evaluation = args.ktc / "EvaluationData" / "EvaluationData"
    ref = scipy.io.loadmat(code / "TrainingData" / "ref.mat")
    R = torch.from_numpy(operateurs(code, ref["Injref"], ref["Mpat"], ref["Uelref"]))
    modele = UNetKTC()
    modele.load_state_dict(torch.load(args.modele, map_location="cpu", weights_only=True))
    modele.eval()

    cas, entrees = [], []

    def ajouter(nom, groupe, niveau, du, verite, reference=None):
        du_t = torch.from_numpy(np.nan_to_num(du).astype(np.float32))[None]
        n = torch.tensor([niveau])
        entrees.append(reconstruire(R, du_t, n)[0].numpy().astype(np.float32))
        prediction = predire(modele, R, du, niveau)
        element = {
            "nom": nom,
            "groupe": groupe,
            "niveau": niveau,
            "verite": _b64(reduire(verite, COTE_ENTREE)),
            "score": round(score_rapide(verite, prediction), 3),
        }
        if reference is not None:
            element["reference"] = _b64(reduire(reference, COTE_ENTREE))
            element["score_reference"] = round(score_rapide(verite, reference), 3)
        cas.append(element)

    for niveau in range(1, 8):
        dossier = evaluation / "evaluation_datasets" / f"level{niveau}"
        u_ref = scipy.io.loadmat(dossier / "ref.mat")["Uelref"].ravel()
        for k, lettre in enumerate("ABC", start=1):
            du = scipy.io.loadmat(dossier / f"data{k}.mat")["Uel"].ravel() - u_ref
            verite = scipy.io.loadmat(evaluation / "GroundTruths" / f"level_{niveau}" / f"{k}_true.mat")["truth"]
            reference = scipy.io.loadmat(args.ktc / "sorties_reference" / f"niveau{niveau}" / f"{k}.mat")[
                "reconstruction"
            ]
            ajouter(f"{niveau}{lettre}", "evaluation", niveau, du, verite, reference.astype(np.uint8))
    u_ref = ref["Uelref"].ravel()
    for k in range(1, 5):
        du = scipy.io.loadmat(code / "TrainingData" / f"data{k}.mat")["Uel"].ravel() - u_ref
        verite = scipy.io.loadmat(code / "GroundTruths" / f"true{k}.mat")["truth"]
        for niveau in range(1, 8):
            ajouter(f"E{k}", "entrainement", niveau, du, verite)

    args.sortie.mkdir(parents=True, exist_ok=True)
    np.stack(entrees).astype("<f4").tofile(args.sortie / "entrees.bin")
    (args.sortie / "cas.json").write_text(json.dumps({"cote": COTE_ENTREE, "cas": cas}, ensure_ascii=False))
    total = sum(c["score"] for c in cas if c["groupe"] == "evaluation")
    print(
        f"{len(cas)} cas, entrées {len(entrees) * COTE_ENTREE**2 * 4 / 1e6:.1f} Mo ; total évaluation {total:.2f} / 21"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
