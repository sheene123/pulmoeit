"""Publie le modèle KTC2023 dans un dépôt de modèle Hugging Face, avec sa fiche et ses résultats.

Le modèle est entraîné sur Kaggle (pulmoeit.ktc.entrainer) puis exporté en ONNX ; il n'est publié
que s'il bat la méthode de référence des organisateurs sur les 21 cuves d'évaluation (règle fixée
avant les résultats, docs/ktc2023.md).

    python scripts/publier_ktc.py --version v1.0.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
DEPOT = "sheenee261/pulmoeit-ktc2023"


def fr(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def main() -> int:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", required=True)
    parser.add_argument("--repo", default=DEPOT)
    args = parser.parse_args()
    onnx = RACINE / "models" / "ktc" / "ktc.onnx"
    resultats = json.loads((RACINE / "reports" / "ktc2023.json").read_text())
    nous, ref = resultats["pulmoeit"], resultats["reference_organisateurs"]
    if nous["total"] <= ref["total"]:
        print(f"refusé : {nous['total']:.2f} ne bat pas la référence ({ref['total']:.2f})", file=sys.stderr)
        return 2
    empreinte = hashlib.sha256(onnx.read_bytes()).hexdigest()
    lignes = "\n".join(
        f"| {n} | {fr(r)} | {fr(p)} |"
        for n, (r, p) in enumerate(zip(ref["par_niveau"], nous["par_niveau"], strict=True), 1)
    )
    publies = "\n".join(f"| {nom} | {fr(v['total'])} |" for nom, v in resultats["publies"].items())
    carte = f"""---
license: mit
library_name: onnx
tags: [electrical-impedance-tomography, eit, image-segmentation, inverse-problems, ktc2023, onnx]
---

# PulmoEIT sur le KTC2023 ({args.version})

U-Net qui transforme la reconstruction linéaire officielle du [Kuopio Tomography Challenge 2023](https://fips.fi/data-challenges/kuopio-tomography-challenge-2023/)
en segmentation à trois classes (eau, résistif, conducteur). Code : [sheene123/pulmoeit](https://github.com/sheene123/pulmoeit)
(`src/pulmoeit/ktc/`, méthode et limites dans `docs/ktc2023.md`). Démo : [Space PulmoEIT](https://huggingface.co/spaces/sheenee261/pulmoeit), page KTC2023.

**Score officiel sur les 21 cuves réelles d'évaluation : {fr(nous["total"])} sur 21**, contre {fr(ref["total"])} pour la
méthode de référence des organisateurs (mieux sur les 21 cuves).

| Niveau | Référence des organisateurs | PulmoEIT |
|---|---|---|
{lignes}

| Algorithmes publiés (Denker et al., 2024) | Total sur 21 |
|---|---|
{publies}

## Utilisation

- `ktc.onnx` : entrées `image` (N, 128, 128), la reconstruction linéaire officielle sur la grille de pixels du
  défi, divisée par son écart-type, et `niveau` (N,), entier de 1 à 7 ; sortie `logits` (N, 3, 128, 128). Classes
  0 eau, 1 résistif, 2 conducteur ; agrandir en 256 × 256 pour le score officiel. SHA-256 : `{empreinte}`.
- La reconstruction linéaire se calcule avec `pulmoeit.ktc.lineaire.operateurs` (code officiel du défi).

## Entraînement

15 000 cuves simulées avec le simulateur officiel (maillage dense ; reconstruction sur le maillage clairsemé),
1 à 4 objets de formes variées, chaque cuve servant aux 7 niveaux. GPU T4 (Kaggle). Modèle choisi sur les
4 cuves réelles d'entraînement (époque 40) ; les 21 cuves d'évaluation n'ont servi qu'au test final.

## Limites

Comparaison aux équipes faite après le défi, pas en aveugle. Vérités terrain de la version d'avril 2024 des
données. Un seul entraînement. Une cuve d'eau n'est pas un thorax : prototype de recherche, pas un dispositif
médical. Données et code du défi : Räsänen et al., doi:10.5281/zenodo.8252370 (CC BY 4.0).
"""
    with tempfile.TemporaryDirectory() as tmp:
        dossier = Path(tmp)
        shutil.copy(onnx, dossier / "ktc.onnx")
        shutil.copy(RACINE / "reports" / "ktc2023.json", dossier / "ktc2023.json")
        shutil.copy(RACINE / "reports" / "ktc2023_entrainement.json", dossier / "entrainement.json")
        shutil.copy(RACINE / "reports" / "ktc2023.png", dossier / "ktc2023.png")
        (dossier / "README.md").write_text(carte)
        api = HfApi()
        api.create_repo(args.repo, repo_type="model", exist_ok=True)
        commit = api.upload_folder(
            folder_path=str(dossier), repo_id=args.repo, repo_type="model", commit_message=f"Publie {args.version}"
        )
        api.create_tag(args.repo, tag=args.version, revision=commit.oid, repo_type="model", exist_ok=True)
    print(f"publié : https://huggingface.co/{args.repo}/tree/{args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
