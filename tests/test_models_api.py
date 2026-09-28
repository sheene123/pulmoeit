import numpy as np
import pytest
import torch

from pulmoeit.grid import GRID, image_mask
from pulmoeit.models import build_model, load_checkpoint, save_checkpoint
from pulmoeit.protocol import n_measurements

N = n_measurements()


@pytest.mark.parametrize("name", ["direct", "postunet"])
def test_models_output_masked_images(name):
    model = build_model(
        name, mask=image_mask(), recon_matrix=np.random.default_rng(0).normal(0, 0.01, (GRID * GRID, N))
    )
    out = model(torch.randn(3, N))
    assert out.shape == (3, GRID, GRID)
    assert torch.all(out[:, ~torch.from_numpy(image_mask())] == 0)


def test_checkpoint_roundtrip(tmp_path):
    model = build_model("direct", mask=image_mask(), mean=np.full(N, 0.1), std=np.full(N, 2.0)).eval()
    save_checkpoint(tmp_path / "m.pt", model, "direct", {})
    x = torch.randn(2, N)
    torch.testing.assert_close(load_checkpoint(tmp_path / "m.pt")(x), model(x))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from fastapi.testclient import TestClient

    from pulmoeit.dataset import AcquisitionConfig, generate
    from pulmoeit.export import export
    from pulmoeit.monitoring import MeasurementQC
    from pulmoeit.signals import features

    model_dir = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(0)
    model = build_model("postunet", mask=image_mask(), recon_matrix=rng.normal(0, 0.01, (GRID * GRID, N)))
    save_checkpoint(model_dir / "model.pt", model, "postunet", {})
    params = {"seed": 0, "model": {"name": "postunet"}, "data": {"acquisition": {}}}
    export(params, model_dir, model_dir)

    data = generate(24, AcquisitionConfig(n_boundary=128, breaths_per_patient=4), seed=1)
    x = features(data["v_ref"], data["v_insp"])
    MeasurementQC.fit(x[:16], x[16:], quantile=1.0).save(model_dir / "qc.npz")

    import os

    os.environ["PULMOEIT_MODEL_DIR"] = str(model_dir)
    from pulmoeit.serve.app import app

    with TestClient(app) as c:
        c.frames = data
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/model").json()["n_parameters"] > 0


def test_reconstruct(client):
    payload = {"v_ref": client.frames["v_ref"][0].tolist(), "v_insp": client.frames["v_insp"][0].tolist()}
    r = client.post("/v1/reconstruct", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["image"]) == GRID and len(body["image"][0]) == GRID
    assert body["image"][0][0] is None  # corner pixel is outside the thorax
    assert set(body["indices"]["regional_distribution_pct"]) == {
        "ventral_right",
        "ventral_left",
        "dorsal_right",
        "dorsal_left",
    }
    assert isinstance(body["qc"]["flagged"], bool)


def test_rejects_open_electrode(client):
    v_ref = client.frames["v_ref"][0].copy()
    v_ref[3] = 0.0
    r = client.post("/v1/reconstruct", json={"v_ref": v_ref.tolist(), "v_insp": client.frames["v_insp"][0].tolist()})
    assert r.status_code == 422


def test_rejects_wrong_length(client):
    assert client.post("/v1/reconstruct", json={"v_ref": [1.0] * 10, "v_insp": [1.0] * 10}).status_code == 422


def test_drift_needs_a_window(client):
    assert client.get("/v1/monitoring/drift").json()["status"] == "insufficient_data"
