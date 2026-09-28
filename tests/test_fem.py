import numpy as np
import pytest

from pulmoeit.fem import ForwardModel
from pulmoeit.geometry import Geometry
from pulmoeit.mesh import build_mesh


@pytest.fixture(scope="module")
def fwd():
    return ForwardModel(build_mesh(n_boundary=128))


def test_mesh_is_valid(fwd):
    mesh = fwd.mesh
    assert mesh.n_electrodes == 16
    assert np.all(mesh.areas > 0)
    edges = {frozenset(e) for t in mesh.elements for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    assert all(frozenset(e) in edges for el in mesh.electrode_edges for e in el)


def test_reciprocity(fwd):
    v = fwd.measure(np.ones(len(fwd.mesh.elements)))
    index = {tuple(p): k for k, p in enumerate(fwd.pairs)}
    for (d, m), k in index.items():
        assert v[k] == pytest.approx(v[index[(m, d)]], rel=1e-9, abs=1e-12)


def test_rotational_symmetry_on_a_disc():
    fwd = ForwardModel(build_mesh(Geometry(1.0), n_boundary=128))
    v = fwd.measure(np.ones(len(fwd.mesh.elements)))
    offset = (fwd.pairs[:, 1] - fwd.pairs[:, 0]) % 16
    for r in np.unique(offset):
        vals = v[offset == r]
        assert np.ptp(vals) < 5e-3 * np.abs(vals).mean()


def test_jacobian_matches_finite_differences(fwd):
    rng = np.random.default_rng(0)
    sigma = rng.uniform(0.5, 1.5, len(fwd.mesh.elements))
    _, jac = fwd.jacobian(sigma)
    for e in rng.choice(len(sigma), 3, replace=False):
        ds = np.zeros_like(sigma)
        ds[e] = 1e-6
        fd = (fwd.measure(sigma + ds) - fwd.measure(sigma - ds)) / 2e-6
        assert np.linalg.norm(fd - jac[:, e]) < 1e-4 * np.linalg.norm(fd)


def test_conductivity_scaling():
    mesh = build_mesh(n_boundary=128)
    sigma = np.ones(len(mesh.elements))
    v1 = ForwardModel(mesh, 0.02).measure(sigma)
    v2 = ForwardModel(mesh, 0.01).measure(2 * sigma)
    np.testing.assert_allclose(v2, v1 / 2, rtol=1e-9)
