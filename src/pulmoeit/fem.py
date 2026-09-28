"""Complete electrode model (CEM) forward solver with P1 finite elements.

System (Vauhkonen formulation), grounded by sum(U) = 0 through a Lagrange multiplier:

    [ K(sigma) + B   C  0 ] [u]   [0]
    [ C^T            D  1 ] [U] = [I]
    [ 0              1^T 0] [l]   [0]
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .mesh import Mesh
from .protocol import adjacent_protocol


class ForwardModel:
    def __init__(self, mesh: Mesh, contact_impedance: float | np.ndarray = 0.01):
        self.mesh = mesh
        n_el = mesh.n_electrodes
        self.n_nodes = len(mesh.nodes)
        self.n_el = n_el
        self.injections, self.pairs = adjacent_protocol(n_el)
        z = np.broadcast_to(np.asarray(contact_impedance, dtype=float), (n_el,))

        p = mesh.nodes[mesh.elements]
        x, y = p[..., 0], p[..., 1]
        area2 = 2.0 * mesh.areas
        b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
        c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
        self.grads = np.stack([b, c], axis=-1) / area2[:, None, None]  # (n_elem, 3, 2)
        self.areas = mesh.areas
        self._kloc = self.areas[:, None, None] * np.einsum("eik,ejk->eij", self.grads, self.grads)
        self._k_rows = np.repeat(mesh.elements, 3, axis=1).ravel()
        self._k_cols = np.tile(mesh.elements, (1, 3)).ravel()

        rows, cols, vals = [], [], []
        n = self.n_nodes
        for l, edges in enumerate(mesh.electrode_edges):
            i, j = edges[:, 0], edges[:, 1]
            w = np.linalg.norm(mesh.nodes[i] - mesh.nodes[j], axis=1) / z[l]
            e = np.full_like(i, n + l)
            rows += [i, j, i, j, i, j, e, e, [n + l]]
            cols += [i, j, j, i, e, e, i, j, [n + l]]
            vals += [w / 3, w / 3, w / 6, w / 6, -w / 2, -w / 2, -w / 2, -w / 2, [w.sum()]]
        ground = n + n_el
        rows += [np.full(n_el, ground), n + np.arange(n_el)]
        cols += [n + np.arange(n_el), np.full(n_el, ground)]
        vals += [np.ones(n_el), np.ones(n_el)]
        self._s_rows = np.concatenate(rows)
        self._s_cols = np.concatenate(cols)
        self._s_vals = np.concatenate(vals)
        self.size = n + n_el + 1

    def _system(self, sigma: np.ndarray) -> sp.csc_matrix:
        data = np.concatenate([(sigma[:, None, None] * self._kloc).ravel(), self._s_vals])
        rows = np.concatenate([self._k_rows, self._s_rows])
        cols = np.concatenate([self._k_cols, self._s_cols])
        return sp.csc_matrix((data, (rows, cols)), shape=(self.size, self.size))

    def solve(self, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Solve all stimulations. Returns nodal potentials (n_stim, n_nodes) and
        electrode potentials (n_stim, n_el)."""
        sigma = np.asarray(sigma, dtype=float)
        rhs = np.zeros((self.size, self.n_el))
        rhs[self.n_nodes : self.n_nodes + self.n_el] = self.injections.T
        sol = splu(self._system(sigma)).solve(rhs)
        return sol[: self.n_nodes].T, sol[self.n_nodes : self.n_nodes + self.n_el].T

    def _voltages(self, electrode_potentials: np.ndarray) -> np.ndarray:
        d, m = self.pairs.T
        return electrode_potentials[d, m] - electrode_potentials[d, (m + 1) % self.n_el]

    def measure(self, sigma: np.ndarray) -> np.ndarray:
        return self._voltages(self.solve(sigma)[1])

    def jacobian(self, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (voltages, dV/dsigma_elem) using the adjoint / reciprocity identity
        dV_(d,m)/dsigma_e = -area_e * grad(u_d) . grad(u_m)."""
        u, potentials = self.solve(sigma)
        grad_u = np.einsum("pei,eik->pek", u[:, self.mesh.elements], self.grads)
        d, m = self.pairs.T
        jac = -self.areas * np.einsum("mek,mek->me", grad_u[d], grad_u[m])
        return self._voltages(potentials), jac
