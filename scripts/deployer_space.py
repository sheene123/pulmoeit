"""Publie la démo web sur un Space statique Hugging Face (calcul dans le navigateur).

dvc repro                          # produit models/ (ONNX, baseline NOSER, QC, model card)
hf auth login                      # une fois, avec un token « write »
python scripts/deployer_space.py --space <utilisateur>/pulmoeit
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

RACINE = Path(__file__).resolve().parents[1]
MODELES = ("pulmoeit.onnx", "baseline.npz", "qc.npz", "model_card.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="identifiant du Space, ex. utilisateur/pulmoeit")
    parser.add_argument("--message", default="Met à jour la démo web")
    args = parser.parse_args()

    manquants = [m for m in MODELES if not (RACINE / "models" / m).exists()]
    if manquants:
        raise SystemExit(f"modèles absents ({', '.join(manquants)}) : lancez d'abord `dvc repro`")

    with tempfile.TemporaryDirectory() as construction, tempfile.TemporaryDirectory() as dossier:
        # uv dépose un .gitignore « * » dans son dossier de sortie : on ne copie que la wheel.
        subprocess.run(["uv", "build", "--wheel", "-q", "-o", construction, str(RACINE)], check=True)
        site = Path(dossier)
        for roue in Path(construction).glob("*.whl"):
            shutil.copy(roue, site / roue.name)
        for fichier in ("index.html", "pont.py", "README.md"):
            shutil.copy(RACINE / "demo" / "web" / fichier, site / fichier)
        for modele in MODELES:
            shutil.copy(RACINE / "models" / modele, site / modele)
        api = HfApi()
        api.create_repo(args.space, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(
            folder_path=str(site),
            repo_id=args.space,
            repo_type="space",
            commit_message=args.message,
            delete_patterns=["style.css"],  # fichier du modèle de Space statique
        )
    print(f"https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
