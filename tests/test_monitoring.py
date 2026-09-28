import numpy as np

from pulmoeit.dataset import AcquisitionConfig, generate
from pulmoeit.monitoring import MeasurementQC, drift_report, ks_test
from pulmoeit.protocol import electrode_involvement, n_measurements
from pulmoeit.signals import features


def _low_rank(rng, n, basis):
    return rng.normal(size=(n, len(basis))) @ basis + 1e-3 * rng.normal(size=(n, basis.shape[1]))


def test_qc_flags_and_localises_a_faulty_electrode():
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(10, n_measurements()))
    qc = MeasurementQC.fit(_low_rank(rng, 500, basis), _low_rank(rng, 200, basis))
    clean = _low_rank(rng, 50, basis)
    faulty = clean.copy()
    bad = electrode_involvement()[:, 5]
    faulty[:, bad] += rng.normal(0, 2.0, (50, bad.sum()))
    assert np.mean(qc.score(clean) > qc.threshold) < 0.1
    assert np.all(qc.score(faulty) > qc.threshold)
    assert np.mean(qc.suspect_electrode(faulty) == 5) > 0.9


def test_ks_drift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=1000)
    assert ks_test(ref, rng.normal(size=500))[1] > 0.01
    assert drift_report(ref, rng.normal(0.5, 1, size=500))["drift"]


def test_generate_small_dataset():
    d = generate(6, AcquisitionConfig(n_boundary=128, breaths_per_patient=3, fault_prob=0.5), seed=0)
    assert d["v_ref"].shape == (6, n_measurements())
    assert d["target"].shape == (6, 32, 32)
    x = features(d["v_ref"], d["v_insp"])
    assert np.all(np.isfinite(x))
    assert np.all((d["fault"] >= -1) & (d["fault"] < 16))
