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
DEPOT_KTC = "sheenee261/pulmoeit-ktc2023"  # modèle du benchmark KTC2023 (entraîné sur Kaggle)


def modele_ktc() -> Path:
    """Le modèle KTC2023 local s'il existe, sinon celui du registre Hugging Face."""
    local = RACINE / "models" / "ktc" / "ktc.onnx"
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(DEPOT_KTC, "ktc.onnx", repo_type="model"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="sheenee261/pulmoeit", help="identifiant du Space, ex. utilisateur/pulmoeit")
    parser.add_argument("--message", default="Met à jour la démo web")
    parser.add_argument("--local", type=Path, help="assemble le site dans ce dossier, sans le publier (test)")
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
        for fichier in ("index.html", "ktc.html", "respiration.html", "scanners.html", "pont.py", "README.md"):
            shutil.copy(RACINE / "demo" / "web" / fichier, site / fichier)
        shutil.copytree(RACINE / "demo" / "web" / "ktc", site / "ktc")  # données de la page KTC2023
        shutil.copytree(RACINE / "demo" / "web" / "respiration", site / "respiration")  # animations
        shutil.copytree(RACINE / "demo" / "web" / "scanners", site / "scanners")  # thorax réels + nouveau modèle
        shutil.copy(modele_ktc(), site / "ktc.onnx")
        for modele in MODELES:
            shutil.copy(RACINE / "models" / modele, site / modele)
        if args.local:
            shutil.copytree(site, args.local, dirs_exist_ok=True)
            print(f"site assemblé dans {args.local}")
            return 0
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
