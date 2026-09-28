import numpy as np
import pytest

from pulmoeit import metrics
from pulmoeit.grid import GRID, image_mask
from pulmoeit.phantom import SCENARIOS, sample_phantom


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_phantom_targets_are_bounded(scenario):
    img = sample_phantom(np.random.default_rng(1), scenario).render()
    assert img.shape == (GRID, GRID)
    assert img.min() >= 0 and img.max() <= 0.9
    assert np.all(img[~image_mask()] == 0)
    assert img.max() > 0


def test_one_lung_ventilation_is_unilateral():
    rng = np.random.default_rng(3)
    ph = sample_phantom(rng, "one_lung")
    img = ph.render()
    right, left = img[:, : GRID // 2].sum(), img[:, GRID // 2 :].sum()
    assert min(right, left) < 0.05 * max(right, left)


def test_indices_on_uniform_image():
    img = image_mask().astype(float)[None]
    assert metrics.global_inhomogeneity(img)[0] == pytest.approx(0.0)
    assert metrics.center_of_ventilation(img)[0] == pytest.approx(50.0, abs=0.5)
    reg = metrics.regional_distribution(img)[0]
    assert reg.sum() == pytest.approx(100.0)
    np.testing.assert_allclose(reg, 25.0, atol=0.5)


def test_dorsal_shift_moves_center_of_ventilation():
    img = image_mask().astype(float)
    dorsal = img.copy()
    dorsal[: GRID // 2] *= 0.2
    assert metrics.center_of_ventilation(dorsal[None])[0] > metrics.center_of_ventilation(img[None])[0] + 10


def test_pearson_and_auc():
    rng = np.random.default_rng(0)
    a = rng.random((4, GRID, GRID))
    np.testing.assert_allclose(metrics.pearson(a, a), 1.0)
    assert metrics.roc_auc(np.array([0.1, 0.2]), np.array([0.3, 0.4])) == 1.0
    assert metrics.roc_auc(np.array([0.3, 0.4]), np.array([0.1, 0.2])) == 0.0
