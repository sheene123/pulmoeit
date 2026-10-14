"""Indices de ventilation (src/pulmoeit/metrics.py)."""

import numpy as np
import pytest

from pulmoeit import metrics
from pulmoeit.grid import GRID, image_mask


def test_indices_sur_une_image_uniforme():
    img = image_mask().astype(float)[None]
    assert metrics.global_inhomogeneity(img)[0] == pytest.approx(0.0)
    assert metrics.center_of_ventilation(img)[0] == pytest.approx(50.0, abs=0.5)
    reg = metrics.regional_distribution(img)[0]
    assert reg.sum() == pytest.approx(100.0)
    np.testing.assert_allclose(reg, 25.0, atol=0.5)


def test_ventilation_vers_le_dos_deplace_le_centre():
    img = image_mask().astype(float)
    dorsal = img.copy()
    dorsal[: GRID // 2] *= 0.2
    assert metrics.center_of_ventilation(dorsal[None])[0] > metrics.center_of_ventilation(img[None])[0] + 10


def test_pearson_et_auc():
    a = np.random.default_rng(0).random((4, GRID, GRID))
    np.testing.assert_allclose(metrics.pearson(a, a), 1.0)
    assert metrics.roc_auc(np.array([0.1, 0.2]), np.array([0.3, 0.4])) == 1.0
    assert metrics.roc_auc(np.array([0.3, 0.4]), np.array([0.1, 0.2])) == 0.0
