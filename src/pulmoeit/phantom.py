"""Parametric thorax phantoms for ventilation EIT, defined in reference coordinates.

Conductivities are in S/m at ~100 kHz (order of magnitude from Gabriel et al. 1996).
`ventilation` is the relative conductivity drop between end-expiration and end-inspiration,
i.e. the tidal image an EIT monitor displays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import REF_ASPECT
from .grid import GRID, image_mask, pixel_centres

SCENARIOS = ("healthy", "dorsal_atelectasis", "one_lung", "pneumothorax", "pleural_effusion")
SCENARIO_PROBS = (0.35, 0.2, 0.1, 0.15, 0.2)


@dataclass(frozen=True)
class Ellipse:
    cx: float
    cy: float
    ax: float
    ay: float

    def inside(self, p: np.ndarray, scale: float = 1.0) -> np.ndarray:
        return ((p[:, 0] - self.cx) / (self.ax * scale)) ** 2 + ((p[:, 1] - self.cy) / (self.ay * scale)) ** 2 <= 1.0


@dataclass(frozen=True)
class ThoraxPhantom:
    scenario: str
    sigma_bg: float
    lungs: tuple[Ellipse, Ellipse]  # (patient right = image left, patient left)
    sigma_lung: float
    heart: Ellipse
    sigma_heart: float
    cardiac: float  # relative heart conductivity change between the two frames
    spine: Ellipse
    sigma_spine: float
    tidal: float
    gravity: float  # > 0: more ventilation in dependent (dorsal) regions
    side_gain: tuple[float, float]
    lesion: dict = field(default_factory=dict)

    def evaluate(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (sigma_expiration, sigma_inspiration, ventilation) at points p."""
        y = p[:, 1]
        sigma = np.full(len(p), self.sigma_bg)
        vent = np.zeros(len(p))
        for side, lung in enumerate(self.lungs):
            m = lung.inside(p)
            sigma[m] = self.sigma_lung
            vent[m] = self.tidal * self.side_gain[side] * (1 + self.gravity * (-y[m]) / REF_ASPECT)

        les = self.lesion
        if self.scenario == "dorsal_atelectasis":
            for side, lung in enumerate(self.lungs):
                m = lung.inside(p) & (y < les["y_cut"][side])
                sigma[m], vent[m] = les["sigma"], 0.0
        elif self.scenario == "one_lung":
            vent[self.lungs[les["side"]].inside(p)] = 0.0
        elif self.scenario == "pneumothorax":
            m = self.lungs[les["side"]].inside(p, 1.08) & (y > les["y_cut"])
            sigma[m], vent[m] = les["sigma"], 0.0
        elif self.scenario == "pleural_effusion":
            m = self.lungs[les["side"]].inside(p, 1.1) & (y < les["y_cut"])
            sigma[m], vent[m] = les["sigma"], 0.0

        heart = self.heart.inside(p)
        sigma[heart], vent[heart] = self.sigma_heart, 0.0
        spine = self.spine.inside(p)
        sigma[spine], vent[spine] = self.sigma_spine, 0.0

        vent = np.clip(vent, 0.0, 0.9)
        sigma_insp = sigma * (1.0 - vent)
        sigma_insp[heart] *= 1.0 + self.cardiac
        return sigma, sigma_insp, vent

    def render(self, n: int = GRID, supersample: int = 3) -> np.ndarray:
        """Ground-truth tidal image on the reference pixel grid (anti-aliased)."""
        pts = pixel_centres(n, supersample).reshape(-1, 2)
        vent = self.evaluate(pts)[2].reshape(n, n, -1).mean(axis=-1)
        return (vent * image_mask(n)).astype(np.float32)


def sample_phantom(rng: np.random.Generator, scenario: str | None = None) -> ThoraxPhantom:
    if scenario is None:
        scenario = str(rng.choice(SCENARIOS, p=SCENARIO_PROBS))
    u, g = rng.uniform, rng.normal
    lungs = (
        Ellipse(-0.47 + g(0, 0.03), -0.03 + g(0, 0.03), 0.30 * u(0.9, 1.1), 0.45 * u(0.9, 1.1)),
        Ellipse(0.49 + g(0, 0.03), -0.06 + g(0, 0.03), 0.27 * u(0.9, 1.1), 0.41 * u(0.9, 1.1)),
    )
    side = int(rng.integers(2))
    lesion: dict = {
        "healthy": {},
        "dorsal_atelectasis": {
            "y_cut": tuple(u(-0.3, 0.05) + g(0, 0.04, size=2)),
            "sigma": u(0.3, 0.4),
        },
        "one_lung": {"side": side},
        "pneumothorax": {"side": side, "y_cut": u(0.0, 0.25), "sigma": 0.01},
        "pleural_effusion": {"side": side, "y_cut": u(-0.4, -0.15), "sigma": u(1.2, 1.6)},
    }[scenario]
    return ThoraxPhantom(
        scenario=scenario,
        sigma_bg=u(0.3, 0.45),
        lungs=lungs,
        sigma_lung=u(0.1, 0.16),
        heart=Ellipse(0.1 + g(0, 0.03), 0.25 + g(0, 0.03), 0.22 * u(0.9, 1.1), 0.18 * u(0.9, 1.1)),
        sigma_heart=u(0.5, 0.7),
        cardiac=g(0, 0.02),
        spine=Ellipse(0.0, -0.55, 0.1, 0.1),
        sigma_spine=0.03,
        tidal=u(0.15, 0.45),
        gravity=u(-0.3, 0.5),
        side_gain=(u(0.8, 1.2), u(0.8, 1.2)),
        lesion=lesion,
    )
