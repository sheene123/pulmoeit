"""Vrais enregistrements (src/pulmoeit/reel.py) : conventions de l'appareil, respirations, cycle moyen."""

import numpy as np
import pytest

from pulmoeit.protocol import adjacent_protocol
from pulmoeit.reel import bruit_relatif, cycle_moyen, respirations, verifier_ordre, vers_protocole

N = 16


def appareil_code():
    """Mesures de l'appareil codées par (injection d, paire mesurée m) : 100 d + m."""
    a = np.zeros((1, N, N - 3))
    for d in range(N):
        for k in range(N - 3):
            a[0, d, k] = 100 * d + (d + 2 + k) % N  # ordre « décalé » : la première paire suit l'injection
    return a.reshape(1, -1)


def test_sens_droite_garde_la_numerotation():
    v = vers_protocole(appareil_code(), "droite")[0]
    _, paires = adjacent_protocol(N)
    for valeur, (d, m) in zip(v, paires, strict=True):
        assert -valeur == 100 * d + m


def test_sens_gauche_est_un_miroir():
    v = vers_protocole(appareil_code(), "gauche")[0]
    _, paires = adjacent_protocol(N)
    for valeur, (d, m) in zip(v, paires, strict=True):
        assert -valeur == 100 * ((-d - 1) % N) + (-m - 1) % N


def test_sens_inconnu():
    with pytest.raises(ValueError):
        vers_protocole(appareil_code(), "haut")


def test_verifier_ordre_reconnait_l_ordre_decale():
    rng = np.random.default_rng(0)
    reference = -rng.uniform(1, 10, size=N * (N - 3))  # profil du modèle, ordre de PulmoEIT
    # mesures de l'appareil (amplitudes positives, ordre décalé) construites à partir du profil
    _, paires = adjacent_protocol(N)
    appareil = np.zeros((N, N - 3))
    for k, (d, m) in enumerate(paires):
        appareil[d, (m - d - 2) % N] = -reference[k]
    r = verifier_ordre(np.tile(appareil.reshape(1, -1), (3, 1)), reference)
    assert r["decale"] == pytest.approx(1.0)
    assert r["par_numero"] < 0.9


def test_respirations_malgre_un_saut_de_niveau():
    fps, t = 13, np.arange(390) / 13
    signal = np.sin(2 * np.pi * 0.5 * t) + 5 * (t > 15)  # 30 respirations par minute, palier à 15 s
    paires = respirations(signal, fps)
    assert 12 <= len(paires) <= 15
    assert all(b < h for b, h in paires)


def test_cycle_moyen_et_bruit():
    fps, t = 13, np.arange(260) / 13
    respiration = 0.5 - 0.5 * np.cos(2 * np.pi * 0.5 * t)  # 0 en fin d'expiration
    profil = np.linspace(1, 2, N * (N - 3))
    mesures = 100 + respiration[:, None] * profil
    resp = respirations(respiration, fps)
    cycle = cycle_moyen(mesures, resp, n_phases=16)
    assert cycle.shape == (16, N * (N - 3))
    assert np.argmin(cycle[:, 0]) == 0  # la phase 0 est la fin d'expiration
    assert bruit_relatif(mesures, resp) < 1e-3
    bruite = mesures + np.random.default_rng(1).normal(0, 0.05, mesures.shape)
    assert bruit_relatif(bruite, resp) > bruit_relatif(mesures, resp)


def test_cycle_moyen_demande_deux_respirations():
    with pytest.raises(ValueError):
        cycle_moyen(np.zeros((10, 4)), [(1, 4)])
