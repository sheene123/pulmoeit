"""Adjacent ("Sheffield") stimulation / measurement protocol."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

N_ELECTRODES = 16


@lru_cache
def adjacent_protocol(n_el: int = N_ELECTRODES) -> tuple[np.ndarray, np.ndarray]:
    """Return (injections, pairs).

    injections[d] is the current pattern of stimulation d: +1 on electrode d, -1 on d+1.
    pairs[k] = (d, m): measurement k is U_m - U_{m+1} during stimulation d. Pairs that
    touch a current-carrying electrode are skipped, giving n_el * (n_el - 3) values.
    """
    injections = np.zeros((n_el, n_el))
    for d in range(n_el):
        injections[d, d] = 1.0
        injections[d, (d + 1) % n_el] = -1.0
    pairs = [(d, m) for d in range(n_el) for m in range(n_el) if m not in {(d - 1) % n_el, d, (d + 1) % n_el}]
    return injections, np.array(pairs)


def n_measurements(n_el: int = N_ELECTRODES) -> int:
    return n_el * (n_el - 3)


@lru_cache
def electrode_involvement(n_el: int = N_ELECTRODES) -> np.ndarray:
    """Boolean (n_meas, n_el): electrodes used (to inject or to measure) by each value."""
    _, pairs = adjacent_protocol(n_el)
    inv = np.zeros((len(pairs), n_el), dtype=bool)
    for k, (d, m) in enumerate(pairs):
        inv[k, [d, (d + 1) % n_el, m, (m + 1) % n_el]] = True
    return inv
