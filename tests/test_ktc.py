import numpy as np

from pulmoeit.ktc.fantomes import agrandir, centres_pixels, reduire, tirer
from pulmoeit.ktc.officiel import ajuster_fond, mesures_gardees
from pulmoeit.ktc.score import score_rapide


def test_fantomes():
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = tirer(rng)
        assert c.shape == (256, 256) and set(np.unique(c)) <= {0, 1, 2}
        assert (c > 0).any()
        X, Y = centres_pixels()
        assert not c[X**2 + Y**2 > 0.115**2].any()  # rien hors de la cuve
    assert np.array_equal(agrandir(reduire(c, 128)).shape, (256, 256))


def test_score_rapide():
    rng = np.random.default_rng(1)
    v = tirer(rng)
    assert abs(score_rapide(v, v) - 1) < 1e-9
    assert score_rapide(v, np.zeros_like(v)) < score_rapide(v, v)
    assert score_rapide(v, np.zeros((64, 64), np.uint8)) == 0.0


def test_mesures_gardees():
    injections = np.zeros((32, 76))
    for i in range(32):
        injections[i, i] = 1
        injections[(i + 1) % 32, i] = -1
    assert mesures_gardees(injections, 1).all()
    garde = mesures_gardees(injections, 3)  # électrodes 0 à 3 retirées
    assert not garde[:4].any() and garde[4:].any()


def test_ajuster_fond():
    u1 = np.linspace(1, 2, 50)
    assert abs(ajuster_fond(u1 / 0.8, u1) - 0.8) < 1e-12
