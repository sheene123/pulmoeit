"""Indices d'un essai de PEEP (src/pulmoeit/peep.py) sur un poumon synthétique dont on connaît la réponse."""

import numpy as np
import pytest

from pulmoeit.grid import GRID, image_mask, row_depth
from pulmoeit.peep import costa, croisement, espaces_silencieux, ventilation_regionale

PEEPS = np.arange(24.0, 5.0, -2.0)


@pytest.fixture
def essai():
    """Pression d'inflexion croissante de l'avant (8) vers le dos (22) : le dos se ferme à PEEP basse."""
    profondeur = np.broadcast_to(row_depth()[:, None], (GRID, GRID)) / 100
    pinf = 8 + 14 * profondeur
    return np.stack([ventilation_regionale(p, 12, pinf, 3.0) * image_mask() for p in PEEPS])


def test_courbes_de_costa(essai):
    r = costa(essai, PEEPS, np.full(len(PEEPS), 12.0))
    assert r["collapsus"][0] == pytest.approx(0, abs=0.5)  # PEEP haute : rien de collabé
    assert r["surdistension"][-1] == pytest.approx(0, abs=0.5)  # PEEP basse : rien de surdistendu
    assert np.all(np.diff(r["collapsus"]) >= -1e-9)  # le collapsus augmente quand la PEEP baisse
    assert np.all(np.diff(r["surdistension"]) <= 1e-9)
    assert PEEPS.min() < r["peep_proposee"] < PEEPS.max()


def test_invariance_a_l_echelle(essai):
    dp = np.full(len(PEEPS), 12.0)
    a, b = costa(essai, PEEPS, dp), costa(37 * essai, PEEPS, dp)
    np.testing.assert_allclose(a["collapsus"], b["collapsus"])
    assert a["peep_proposee"] == pytest.approx(b["peep_proposee"])


def test_classes_des_pixels(essai):
    r = costa(essai, PEEPS, np.full(len(PEEPS), 12.0))
    assert r["classes"].shape == (len(PEEPS), GRID, GRID)
    assert set(np.unique(r["classes"])) <= {0, 1, 2, 3}
    dos = r["classes"][-1][(row_depth() > 80)[:, None] & r["roi"]]  # PEEP 6 : le dos est collabé
    assert np.mean(dos == 2) > 0.5


def test_croisement_interpole():
    assert croisement(np.array([10.0, 12, 14]), np.array([20.0, 10, 0]), np.array([0.0, 10, 20])) == pytest.approx(12)
    assert croisement(np.array([10.0, 12]), np.array([30.0, 10]), np.array([0.0, 20])) == pytest.approx(11.5)


def test_espaces_silencieux_a_peep_haute(essai):
    s = espaces_silencieux(essai[0], image_mask())  # PEEP 24 : l'avant est surdistendu, il ne ventile plus
    assert s["non_dependants"] > s["dependants"]
