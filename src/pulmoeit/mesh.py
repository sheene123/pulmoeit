"""2D triangular meshes of the thorax with a belt of electrodes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.spatial import Delaunay

from .geometry import REFERENCE, Geometry
from .protocol import N_ELECTRODES

_N_DENSE = 4096


@dataclass
class Mesh:
    nodes: np.ndarray  # (n_nodes, 2)
    elements: np.ndarray  # (n_elem, 3), counter-clockwise
    electrode_edges: list[np.ndarray]  # per electrode: (n_edges, 2) boundary node pairs
    geometry: Geometry

    @property
    def n_electrodes(self) -> int:
        return len(self.electrode_edges)

    @cached_property
    def centroids(self) -> np.ndarray:
        return self.nodes[self.elements].mean(axis=1)

    @cached_property
    def areas(self) -> np.ndarray:
        p = self.nodes[self.elements]
        return 0.5 * _cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]


def build_mesh(
    geometry: Geometry = REFERENCE,
    n_boundary: int = 192,
    n_electrodes: int = N_ELECTRODES,
    electrode_width: float = 0.3,
    electrode_shift: np.ndarray | None = None,
    first_angle: float = np.pi / 2,
) -> Mesh:
    """Mesh a thorax section with concentric node layers + Delaunay.

    Electrodes are equally spaced along the perimeter (as on a textile belt), electrode 0
    centred on the anterior midline, numbered counter-clockwise in image space.
    `electrode_width` and `electrode_shift` are fractions of the electrode pitch.
    """
    theta_d = np.linspace(0.0, 2 * np.pi, _N_DENSE, endpoint=False)
    pts_d = geometry.point(theta_d)
    seg = np.linalg.norm(np.roll(pts_d, -1, axis=0) - pts_d, axis=1)
    s_d = np.concatenate([[0.0], np.cumsum(seg)])
    theta_ext = np.concatenate([theta_d, [2 * np.pi]])
    perimeter = s_d[-1]

    h = perimeter / n_boundary
    s0 = np.interp(first_angle, theta_ext, s_d)
    s_nodes = s0 - h / 2 + h * np.arange(n_boundary)
    boundary = geometry.point(np.interp(np.mod(s_nodes, perimeter), s_d, theta_ext))

    layers = [boundary]
    n_layers = max(2, round(float(np.linalg.norm(pts_d, axis=1).mean()) / h))
    for j in range(1, n_layers):
        frac = 1.0 - j / n_layers
        n_j = max(6, round(n_boundary * frac))
        theta_j = np.linspace(0.0, 2 * np.pi, n_j, endpoint=False) + (j % 2) * np.pi / n_j
        layers.append(frac * geometry.point(theta_j))
    layers.append(np.zeros((1, 2)))
    nodes = np.vstack(layers)

    tri = Delaunay(nodes).simplices
    tri = tri[geometry.contains(nodes[tri].mean(axis=1))]
    p = nodes[tri]
    signed = _cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    tri = tri[np.abs(signed) > 1e-12]
    signed = signed[np.abs(signed) > 1e-12]
    tri[signed < 0] = tri[signed < 0][:, [0, 2, 1]]

    pitch = perimeter / n_electrodes
    shift = np.zeros(n_electrodes) if electrode_shift is None else np.asarray(electrode_shift)
    centres = s0 + (np.arange(n_electrodes) + shift) * pitch
    edge_mid = s_nodes + h / 2
    edges = np.c_[np.arange(n_boundary), (np.arange(n_boundary) + 1) % n_boundary]
    electrode_edges = []
    for c in centres:
        dist = np.abs(np.mod(edge_mid - c + perimeter / 2, perimeter) - perimeter / 2)
        sel = np.flatnonzero(dist <= electrode_width * pitch / 2)
        if sel.size == 0:
            sel = np.array([np.argmin(dist)])
        electrode_edges.append(edges[sel])

    return Mesh(nodes=nodes, elements=tri, electrode_edges=electrode_edges, geometry=geometry)
