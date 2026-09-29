"""Registre de modèles : publie une version du modèle sur le Hugging Face Hub, après comparaison
avec le modèle en production (champion / challenger).

    python scripts/publier_modele.py --repo <utilisateur>/pulmoeit --version v0.2.0

Le challenger (models/ et reports/ produits par `dvc repro`) n'est publié que s'il ne régresse pas
face au champion, c'est-à-dire la dernière version publiée sur la branche main du dépôt de modèle.
Chaque publication est un commit du dépôt de modèle, étiqueté par la version : on peut revenir à
n'importe quelle version, et la démo comme l'API savent quelle version elles servent.
Code de sortie : 0 si publié, 2 si refusé par la comparaison, 1 en cas d'erreur.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
FICHIERS = {
    "models/pulmoeit.onnx": "pulmoeit.onnx",
    "models/baseline.npz": "baseline.npz",
    "models/qc.npz": "qc.npz",
    "models/model_card.json": "model_card.json",
    "reports/metrics.json": "metrics.json",
    "reports/report.md": "report.md",
}
# tolérances : un challenger peut perdre un peu (bruit d'entraînement), jamais beaucoup
TOLERANCES = {"corr_test": 0.01, "corr_decale": 0.02, "auc_defaut": 0.01}


def indicateurs(metrics: dict) -> dict[str, float]:
    return {
        "corr_test": metrics["test"]["model"]["corr"],
        "corr_decale": metrics["test_shift"]["model"]["corr"],
        "auc_defaut": metrics["qc"]["fault_auc"],
    }


def champion(api, repo: str) -> dict | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        chemin = hf_hub_download(repo, "metrics.json", repo_type="model", token=api.token)
    except (RepositoryNotFoundError, EntryNotFoundError):
        return None
    return json.loads(Path(chemin).read_text())


def comparer(challenger: dict[str, float], reference: dict[str, float] | None) -> tuple[bool, list[str]]:
    if reference is None:
        return True, ["aucun modèle en production : première publication"]
    lignes, accepte = [], True
    for cle, tolerance in TOLERANCES.items():
        ecart = challenger[cle] - reference[cle]
        ok = ecart >= -tolerance
        accepte &= ok
        lignes.append(
            f"{cle} : champion {reference[cle]:.3f}, challenger {challenger[cle]:.3f} ({ecart:+.3f}) {'OK' if ok else 'RÉGRESSION'}"
        )
    return accepte, lignes


def fiche_modele(carte: dict, version: str, depot_code: str) -> str:
    test, decale, qc = carte["evaluation"]["test"], carte["evaluation"]["test_shift"], carte["evaluation"]["qc"]
    return f"""---
license: mit
library_name: onnx
tags:
- medical-imaging
- electrical-impedance-tomography
- onnx
- mlops
---

# PulmoEIT {version}

Reconstruction d'images de ventilation pulmonaire par tomographie d'impédance électrique (EIT) :
réseau PostUNet ({carte["n_parameters"]:,} paramètres) qui corrige la reconstruction linéaire NOSER.
Code, pipeline et évaluation : [{depot_code}](https://github.com/{depot_code}).
Démo : [Space PulmoEIT](https://huggingface.co/spaces/sheenee261/pulmoeit).

| Jeu de test (simulation) | Corrélation NOSER | Corrélation PostUNet |
|---|---|---|
| Même distribution | {test["baseline"]["corr"]:.3f} | **{test["model"]["corr"]:.3f}** |
| Conditions décalées | {decale["baseline"]["corr"]:.3f} | **{decale["model"]["corr"]:.3f}** |

Contrôle qualité des mesures : électrode décollée détectée avec une AUC de {qc["fault_auc"]:.3f}.

- Entrée `dv` : (lot, 208) tensions différentielles normalisées ; sortie `image` : (lot, 32, 32).
- Commit du code : `{carte["version"]}` ; empreinte ONNX : `{carte["onnx_sha256"][:16]}…`.
- {carte["intended_use"]}
- Limites : {" ".join(carte["limitations"])}
"""


def main() -> int:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", required=True, help="dépôt de modèle, ex. utilisateur/pulmoeit")
    parser.add_argument("--version", required=True, help="étiquette de version, ex. v0.2.0")
    parser.add_argument("--depot-code", default="sheene123/pulmoeit")
    parser.add_argument("--forcer", action="store_true", help="publie même en cas de régression")
    args = parser.parse_args()

    manquants = [f for f in FICHIERS if not (RACINE / f).exists()]
    if manquants:
        print(f"fichiers absents ({', '.join(manquants)}) : lancez d'abord `dvc repro`", file=sys.stderr)
        return 1
    metrics = json.loads((RACINE / "reports/metrics.json").read_text())
    if not metrics["gates"]["passed"]:
        print("quality gates en échec : rien n'est publié", file=sys.stderr)
        return 2

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    reference = champion(api, args.repo)
    accepte, lignes = comparer(indicateurs(metrics), indicateurs(reference) if reference else None)
    resume = "\n".join(f"- {ligne}" for ligne in lignes)
    print(f"Comparaison champion / challenger :\n{resume}")
    if not accepte and not args.forcer:
        print("régression face au modèle en production : publication refusée", file=sys.stderr)
        return 2

    carte = json.loads((RACINE / "models/model_card.json").read_text())
    carte["registre"] = {"version": args.version, "depot": args.repo}
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as dossier:
        dossier = Path(dossier)
        for source, cible in FICHIERS.items():
            (dossier / cible).write_bytes((RACINE / source).read_bytes())
        (dossier / "model_card.json").write_text(json.dumps(carte, indent=2, ensure_ascii=False))
        (dossier / "README.md").write_text(fiche_modele(carte, args.version, args.depot_code))
        commit = api.upload_folder(
            folder_path=str(dossier), repo_id=args.repo, repo_type="model", commit_message=f"Publie {args.version}"
        )
    api.create_tag(args.repo, tag=args.version, revision=commit.oid, repo_type="model", exist_ok=True)
    print(f"publié : https://huggingface.co/{args.repo}/tree/{args.version}")
    sortie = os.environ.get("GITHUB_STEP_SUMMARY")
    if sortie:
        with open(sortie, "a") as f:
            f.write(f"## Registre de modèles\n\nVersion **{args.version}** publiée sur `{args.repo}`.\n\n{resume}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
