"""Entraîne le U-Net qui transforme la reconstruction linéaire en segmentation à trois classes.

Entrée : reconstruction linéaire officielle 128 × 128 du niveau tiré, divisée par son écart-type,
et sept canaux constants qui indiquent le niveau (électrodes retirées). Sortie : classes eau,
résistif, conducteur en 128 × 128, agrandies en 256 × 256 pour le score. Chaque cuve simulée sert
aux sept niveaux. Le modèle retenu est celui du meilleur score officiel sur les 4 cuves RÉELLES
d'entraînement (× 7 niveaux) ; les 21 cuves d'évaluation ne servent qu'au test final.

    python -m pulmoeit.ktc.entrainer --code <Codes_Python> --donnees data/ktc --sortie models/ktc
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .fantomes import agrandir
from .lineaire import COTE_ENTREE, operateurs
from .officiel import NIVEAUX
from .score import score_rapide

N_NIVEAUX = len(NIVEAUX)


def _bloc(entree: int, sortie: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(entree, sortie, 3, padding=1),
        nn.BatchNorm2d(sortie),
        nn.ReLU(inplace=True),
        nn.Conv2d(sortie, sortie, 3, padding=1),
        nn.BatchNorm2d(sortie),
        nn.ReLU(inplace=True),
    )


class UNetKTC(nn.Module):
    def __init__(self, base: int = 32, profondeur: int = 4):
        super().__init__()
        c = [base * 2**k for k in range(profondeur + 1)]
        self.descente = nn.ModuleList([_bloc(1 + N_NIVEAUX if k == 0 else c[k - 1], c[k]) for k in range(profondeur)])
        self.fond = _bloc(c[profondeur - 1], c[profondeur])
        self.montee = nn.ModuleList(
            [nn.ConvTranspose2d(c[k + 1], c[k], 2, stride=2) for k in reversed(range(profondeur))]
        )
        self.fusion = nn.ModuleList([_bloc(2 * c[k], c[k]) for k in reversed(range(profondeur))])
        self.tete = nn.Conv2d(c[0], 3, 1)

    def forward(self, image: torch.Tensor, niveau: torch.Tensor) -> torch.Tensor:
        """image (N, 64, 64), niveau (N,) de 1 à 7 -> logits (N, 3, 64, 64)."""
        canaux = F.one_hot(niveau - 1, N_NIVEAUX).float()[:, :, None, None].expand(-1, -1, *image.shape[-2:])
        x = torch.cat([image[:, None], canaux], dim=1)
        sauts = []
        for bloc in self.descente:
            x = bloc(x)
            sauts.append(x)
            x = F.max_pool2d(x, 2)
        x = self.fond(x)
        for monte, fusion, saut in zip(self.montee, self.fusion, reversed(sauts), strict=True):
            x = fusion(torch.cat([monte(x), saut], dim=1))
        return self.tete(x)


def normaliser(images: torch.Tensor) -> torch.Tensor:
    """Divise chaque image par son écart-type (le signe, qui distingue résistif et conducteur,
    est gardé)."""
    return images / images.flatten(1).std(dim=1).clamp(min=1e-12)[:, None, None]


def reconstruire(R: torch.Tensor, delta_u: torch.Tensor, niveaux: torch.Tensor) -> torch.Tensor:
    """Reconstructions linéaires (N, 64, 64) : un produit matrice-vecteur par niveau."""
    sortie = torch.empty(len(delta_u), COTE_ENTREE * COTE_ENTREE, device=delta_u.device)
    for n in niveaux.unique():
        k = niveaux == n
        sortie[k] = delta_u[k] @ R[int(n) - 1].T
    return normaliser(sortie.view(-1, COTE_ENTREE, COTE_ENTREE))


def charger_lots(dossier: Path) -> tuple[np.ndarray, np.ndarray]:
    lots = sorted(dossier.glob("lot_*.npz"))
    du = np.concatenate([np.load(f)["delta_u"] for f in lots])
    v = np.concatenate([np.load(f)["verites"] for f in lots])
    return du, v


def cuves_reelles(dossier_code: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Les 4 cuves réelles d'entraînement : (delta_U, vérité 256 × 256)."""
    import scipy.io

    ref = scipy.io.loadmat(dossier_code / "TrainingData" / "ref.mat")["Uelref"].ravel()
    return [
        (
            scipy.io.loadmat(dossier_code / "TrainingData" / f"data{k}.mat")["Uel"].ravel() - ref,
            scipy.io.loadmat(dossier_code / "GroundTruths" / f"true{k}.mat")["truth"],
        )
        for k in range(1, 5)
    ]


@torch.no_grad()
def predire(modele: UNetKTC, R: torch.Tensor, delta_u: np.ndarray, niveau: int) -> np.ndarray:
    """Classes 256 × 256 d'une cuve ; les mesures absentes (NaN) comptent pour zéro."""
    appareil = R.device
    modele.eval()
    du = torch.from_numpy(np.nan_to_num(delta_u).astype(np.float32))[None].to(appareil)
    n = torch.tensor([niveau], device=appareil)
    classes = modele(reconstruire(R, du, n), n).argmax(1)[0].cpu().numpy().astype(np.uint8)
    return agrandir(classes)


def score_reel(modele: UNetKTC, R: torch.Tensor, reelles: list) -> tuple[float, dict]:
    """Somme des scores sur les 4 cuves réelles et les 7 niveaux (maximum 28)."""
    detail = {}
    for niveau in NIVEAUX:
        detail[niveau] = float(np.mean([score_rapide(v, predire(modele, R, du, niveau)) for du, v in reelles]))
    return float(sum(detail.values()) * len(reelles)), detail


def entrainer(
    dossier_code: Path, donnees: Path, sortie: Path, epoques: int = 60, lot: int = 64, graine: int = 0
) -> dict:
    import scipy.io

    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    ref = scipy.io.loadmat(dossier_code / "TrainingData" / "ref.mat")
    R = torch.from_numpy(operateurs(dossier_code, ref["Injref"], ref["Mpat"], ref["Uelref"])).to(appareil)
    du, verites = charger_lots(donnees)
    print(f"{len(du)} cuves simulées, appareil {appareil}", flush=True)
    du_t = torch.from_numpy(du).to(appareil)
    v_t = torch.from_numpy(verites.astype(np.int64)).to(appareil)
    echelle_bruit = float(np.median(np.abs(du)))
    reelles = cuves_reelles(dossier_code)

    modele = UNetKTC().to(appareil)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=1e-3, weight_decay=1e-4)
    pas_par_epoque = len(du) // lot
    planning = torch.optim.lr_scheduler.OneCycleLR(optimiseur, 1e-3, total_steps=epoques * pas_par_epoque)
    poids = torch.tensor([1.0, 3.0, 3.0], device=appareil)  # objets rares face à l'eau
    meilleur, historique, debut = -1.0, [], time.perf_counter()
    sortie.mkdir(parents=True, exist_ok=True)
    for epoque in range(1, epoques + 1):
        modele.train()
        ordre = torch.randperm(len(du), device=appareil)
        for k in range(pas_par_epoque):
            idx = ordre[k * lot : (k + 1) * lot]
            niveaux = torch.from_numpy(rng.integers(1, N_NIVEAUX + 1, len(idx))).to(appareil)
            x = du_t[idx]
            # robustesse au réel : gain et bruit supplémentaires
            x = x * (1 + 0.05 * torch.randn(len(x), 1, device=appareil))
            x = x + echelle_bruit * 0.05 * torch.rand(len(x), 1, device=appareil) * torch.randn_like(x)
            logits = modele(reconstruire(R, x, niveaux), niveaux)
            perte = F.cross_entropy(logits, v_t[idx], weight=poids)
            optimiseur.zero_grad(set_to_none=True)
            perte.backward()
            optimiseur.step()
            planning.step()
        score, detail = score_reel(modele, R, reelles)
        historique.append({"epoque": epoque, "perte": float(perte), "score_reel": score, "par_niveau": detail})
        print(
            f"époque {epoque:3d}  perte {float(perte):.4f}  score réel {score:.2f}/28  "
            f"({(time.perf_counter() - debut) / 60:.0f} min)",
            flush=True,
        )
        if score > meilleur:
            meilleur = score
            torch.save(modele.state_dict(), sortie / "unet_ktc.pt")
    infos = {"score_reel_retenu": meilleur, "epoques": epoques, "cuves_simulees": len(du), "historique": historique}
    (sortie / "entrainement_ktc.json").write_text(json.dumps(infos, indent=1, ensure_ascii=False))
    return infos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--donnees", type=Path, required=True)
    parser.add_argument("--sortie", type=Path, required=True)
    parser.add_argument("--epoques", type=int, default=60)
    args = parser.parse_args()
    entrainer(args.code, args.donnees, args.sortie, args.epoques)


if __name__ == "__main__":
    main()


def exporter_onnx(poids: Path, sortie: Path) -> float:
    """Exporte le U-Net en ONNX (entrées « image » (N, 128, 128) et « niveau » (N,) entier) ;
    renvoie l'écart maximal entre PyTorch et ONNX Runtime (test de parité)."""
    import onnxruntime as ort

    modele = UNetKTC()
    modele.load_state_dict(torch.load(poids, map_location="cpu", weights_only=True))
    modele.eval()
    image = torch.randn(2, COTE_ENTREE, COTE_ENTREE)
    niveau = torch.tensor([1, 7])
    torch.onnx.export(
        modele,
        (image, niveau),
        str(sortie),
        input_names=["image", "niveau"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "n"}, "niveau": {0: "n"}, "logits": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    with torch.no_grad():
        attendu = modele(image, niveau).numpy()
    obtenu = ort.InferenceSession(str(sortie), providers=["CPUExecutionProvider"]).run(
        None, {"image": image.numpy(), "niveau": niveau.numpy()}
    )[0]
    return float(np.abs(obtenu - attendu).max())
