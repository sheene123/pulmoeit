"""Publie la démo web (vrais enregistrements EIT) sur un Space statique Hugging Face.

python scripts/preparer_reel.py    # produit demo/web/reel/ (reconstructions des six méthodes)
hf auth login                      # une fois, avec un token « write »
python scripts/deployer_space.py --space <utilisateur>/pulmoeit
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

RACINE = Path(__file__).resolve().parents[1]
WEB = RACINE / "demo" / "web"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="sheenee261/pulmoeit", help="identifiant du Space, ex. utilisateur/pulmoeit")
    parser.add_argument("--message", default="Met à jour la démo web")
    parser.add_argument("--local", type=Path, help="assemble le site dans ce dossier, sans le publier (test)")
    args = parser.parse_args()

    if not (WEB / "reel" / "index.json").exists():
        raise SystemExit("demo/web/reel/ absent : lancez d'abord `python scripts/preparer_reel.py`")

    with tempfile.TemporaryDirectory() as dossier:
        site = Path(dossier)
        for fichier in ("index.html", "README.md"):
            shutil.copy(WEB / fichier, site / fichier)
        shutil.copytree(WEB / "reel", site / "reel")
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
            # l'ancienne démo (simulations, KTC2023, scanners) est retirée du Space
            # (motifs précis : ils ne doivent toucher aucun fichier de la nouvelle démo)
            delete_patterns=[
                "*.whl",
                "*.onnx",
                "*.npz",
                "pont.py",
                "model_card.json",
                "style.css",
                "ktc.html",
                "respiration.html",
                "scanners.html",
                "ktc/*",
                "respiration/*",
                "scanners/*",
            ],
        )
    print(f"https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
