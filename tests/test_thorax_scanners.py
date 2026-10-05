"""Conventions des thorax issus de scanners (scripts/thorax_scanners.py), sur une coupe synthétique
dont on connaît la réponse : orientation, contour, vérité du bon côté."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
ts = pytest.importorskip("thorax_scanners")


@pytest.fixture
def coupe(tmp_path):
    """Coupe RAS de 200 × 200 voxels de 2 mm : corps elliptique (demi-axes 150 et 105 mm), poumon
    droit du patient du côté +x RAS, poumon gauche du côté -x, cœur légèrement à gauche et en avant."""
    n = 200
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x, y = (i - n / 2) * 2.0, (j - n / 2) * 2.0  # mm, +x droite du patient, +y avant
    e = np.zeros((n, n), np.int8)
    e[(x / 150) ** 2 + (y / 105) ** 2 <= 1] = ts.TISSU
    e[((x - 70) / 45) ** 2 + (y / 70) ** 2 <= 1] = ts.POUMON_D
    e[((x + 70) / 45) ** 2 + (y / 70) ** 2 <= 1] = ts.POUMON_G
    e[((x + 15) / 25) ** 2 + ((y - 40) / 20) ** 2 <= 1] = ts.COEUR
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-n, -n, 0]
    f = tmp_path / "s9999.npz"
    np.savez(f, etiquettes=e, hu=np.zeros((n, n), np.int16), affine=affine, coupe=50)
    return f


def test_orientation_et_contour(coupe):
    t = ts.ThoraxScanner(coupe)
    assert t.geometrie.aspect == pytest.approx(105 / 150, abs=0.02)
    assert max(abs(a) for ab in t.geometrie.fourier for a in ab) < 0.02  # une ellipse : presque pas de Fourier
    # repère du simulateur : +x vers la gauche du patient, +y vers l'avant
    assert t.etiquette(np.array([[0.47, 0.0]]))[0] == ts.POUMON_G
    assert t.etiquette(np.array([[-0.47, 0.0]]))[0] == ts.POUMON_D
    assert t.etiquette(np.array([[0.1, 0.27]]))[0] == ts.COEUR
    assert t.etiquette(np.array([[0.0, 0.9]]))[0] == ts.DEHORS


def test_verite_du_bon_cote(coupe):
    t = ts.ThoraxScanner(coupe)
    s = ts.simuler(t, np.random.default_rng(0), "one_lung", graisse=False)
    v = s["verite"]
    gauche_image, droite_image = v[:, : v.shape[1] // 2].sum(), v[:, v.shape[1] // 2 :].sum()
    assert s["x"].shape == (208,)
    assert v.max() > 0
    # un seul poumon ventilé : toute la ventilation d'un seul côté de l'image
    assert min(gauche_image, droite_image) < 0.02 * max(gauche_image, droite_image)


def test_separation_par_patient(tmp_path):
    fichiers = [tmp_path / f"s{k:04d}.npz" for k in range(40)]
    parts = ts.separer(fichiers)
    assert sum(len(v) for v in parts.values()) == 40
    assert not set(parts["train"]) & set(parts["test"])
    assert ts.separer(list(reversed(fichiers))) == parts  # indépendante de l'ordre
