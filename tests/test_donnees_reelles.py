import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluer_donnees_reelles import cycles, vers_protocole  # noqa: E402

from pulmoeit.protocol import adjacent_protocol  # noqa: E402


def test_conversion_goe_vers_protocole():
    """Chaque mesure de l'appareil (injection d', paire m') arrive à la place de la paire miroir
    du simulateur, avec le signe inversé."""
    n = 16
    appareil = np.zeros((1, n, n - 3))
    for d in range(n):
        for k in range(n - 3):
            appareil[0, d, k] = 100 * d + (d + 2 + k) % n  # code (injection, paire mesurée)
    v = vers_protocole(appareil.reshape(1, -1))[0]
    _, paires = adjacent_protocol(n)
    for valeur, (d, m) in zip(v, paires, strict=True):
        assert -valeur == 100 * ((-d - 1) % n) + (-m - 1) % n


def test_cycles_respiratoires():
    t = np.arange(260) / 13
    signal = np.sin(2 * np.pi * t * 0.6)  # 36 respirations par minute
    paires = cycles(signal, 13)
    assert 8 <= len(paires) <= 11
    assert all(signal[h] > signal[b] for b, h in paires)
