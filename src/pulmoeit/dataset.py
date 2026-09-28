"""Synthetic acquisition campaigns: random patients, belts, noise and electrode faults.

Each patient gets its own boundary shape, electrode positions and contact impedances
(domain randomisation) and is simulated on a finer mesh than the one used for
reconstruction, to avoid the "inverse crime".
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np

from .fem import ForwardModel
from .geometry import REFERENCE, Geometry
from .mesh import build_mesh
from .phantom import SCENARIOS, sample_phantom
from .protocol import N_ELECTRODES, electrode_involvement
from .signals import features  # noqa: F401  (re-exported)


@dataclass(frozen=True)
class AcquisitionConfig:
    n_boundary: int = 224
    aspect_range: tuple[float, float] = (0.64, 0.80)
    fourier_std: float = 0.02  # boundary shape variability
    n_fourier: int = 3
    electrode_jitter: float = 0.03  # std of electrode displacement, fraction of pitch
    electrode_width: float = 0.3
    contact_range: tuple[float, float] = (0.005, 0.03)
    contact_drift: float = 0.05  # relative contact impedance change between frames
    snr_db: tuple[float, float] = (50.0, 70.0)
    fault_prob: float = 0.0  # probability that one electrode detaches between frames
    breaths_per_patient: int = 4

    @classmethod
    def from_dict(cls, d: dict) -> AcquisitionConfig:
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


@dataclass
class Sample:
    v_ref: np.ndarray
    v_insp: np.ndarray
    target: np.ndarray
    scenario: int
    fault: int = -1
    meta: dict = field(default_factory=dict)


def simulate_patient(seed: np.random.SeedSequence, cfg: AcquisitionConfig) -> list[Sample]:
    rng = np.random.default_rng(seed)
    geometry = Geometry(
        aspect=rng.uniform(*cfg.aspect_range),
        fourier=tuple((rng.normal(0, cfg.fourier_std), rng.normal(0, cfg.fourier_std)) for _ in range(cfg.n_fourier)),
    )
    mesh = build_mesh(
        geometry,
        cfg.n_boundary,
        electrode_width=cfg.electrode_width,
        electrode_shift=rng.normal(0, cfg.electrode_jitter, N_ELECTRODES),
    )
    centroids_ref = geometry.map_to(mesh.centroids, REFERENCE)
    z_ref = rng.uniform(*cfg.contact_range, N_ELECTRODES)
    z_insp = z_ref * np.clip(1 + rng.normal(0, cfg.contact_drift, N_ELECTRODES), 0.5, 2.0)
    fwd_ref, fwd_insp = ForwardModel(mesh, z_ref), ForwardModel(mesh, z_insp)
    involved = electrode_involvement()

    samples = []
    for _ in range(cfg.breaths_per_patient):
        phantom = sample_phantom(rng)
        s_ref, s_insp, _ = phantom.evaluate(centroids_ref)
        v_ref, v_insp = fwd_ref.measure(s_ref), fwd_insp.measure(s_insp)
        noise = np.sqrt(np.mean(v_ref**2)) * 10 ** (-rng.uniform(*cfg.snr_db) / 20)
        v_ref = v_ref + rng.normal(0, noise, v_ref.shape)
        v_insp = v_insp + rng.normal(0, noise, v_insp.shape)
        fault = -1
        if rng.random() < cfg.fault_prob:
            fault = int(rng.integers(N_ELECTRODES))
            bad = involved[:, fault]
            v_insp[bad] = v_insp[bad] * rng.uniform(0.6, 1.4, bad.sum()) + rng.normal(0, 20 * noise, bad.sum())
        samples.append(
            Sample(
                v_ref=v_ref,
                v_insp=v_insp,
                target=phantom.render(),
                scenario=SCENARIOS.index(phantom.scenario),
                fault=fault,
            )
        )
    return samples


def generate(n: int, cfg: AcquisitionConfig, seed: int, workers: int = 1) -> dict[str, np.ndarray]:
    n_patients = math.ceil(n / cfg.breaths_per_patient)
    seeds = np.random.SeedSequence(seed).spawn(n_patients)
    job = partial(simulate_patient, cfg=cfg)
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            batches = list(pool.map(job, seeds, chunksize=4))
    else:
        batches = [job(s) for s in seeds]
    samples = [s for batch in batches for s in batch][:n]
    return {
        "v_ref": np.stack([s.v_ref for s in samples]),
        "v_insp": np.stack([s.v_insp for s in samples]),
        "target": np.stack([s.target for s in samples]),
        "scenario": np.array([s.scenario for s in samples], dtype=np.int8),
        "fault": np.array([s.fault for s in samples], dtype=np.int8),
    }


def save(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
