"""Reconstruction networks. Preprocessing lives inside the modules so that the exported
ONNX graph maps raw normalised difference data straight to a tidal image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .grid import GRID
from .protocol import n_measurements


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.GroupNorm(4, c_out),
        nn.GELU(),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.GroupNorm(4, c_out),
        nn.GELU(),
    )


def _up(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1), _block(c_out, c_out))


class DirectNet(nn.Module):
    """Fully learned inverse map: measurements -> FC encoder -> conv decoder -> image."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, mask: np.ndarray, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))
        self.fc = nn.Sequential(
            nn.Linear(len(mean), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64 * 4 * 4),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(_up(64, 32), _up(32, 16), _up(16, 16), nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc((x - self.mean) / self.std).view(-1, 64, 4, 4)
        return self.decoder(h)[:, 0] * self.mask


class PostUNet(nn.Module):
    """Model-based + learned: the linear NOSER image is refined by a small residual U-Net.
    Keeps the physics in the loop, which helps under distribution shift."""

    def __init__(self, recon_matrix: np.ndarray, mask: np.ndarray, base: int = 16):
        super().__init__()
        self.register_buffer("recon", torch.as_tensor(recon_matrix, dtype=torch.float32))
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))
        b = base
        self.enc1, self.enc2, self.enc3 = _block(1, b), _block(b, 2 * b), _block(2 * b, 4 * b)
        self.pool = nn.MaxPool2d(2)
        self.up2, self.dec2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2), _block(4 * b, 2 * b)
        self.up1, self.dec1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2), _block(2 * b, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = (x @ self.recon.T).view(-1, 1, GRID, GRID)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return (x0 + self.head(d1))[:, 0] * self.mask


def build_model(
    name: str,
    *,
    mask: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    recon_matrix: np.ndarray | None = None,
    **kwargs,
) -> nn.Module:
    n = n_measurements()
    if name == "direct":
        mean = np.zeros(n) if mean is None else mean
        std = np.ones(n) if std is None else std
        return DirectNet(mean, std, mask, **kwargs)
    if name == "postunet":
        recon_matrix = np.zeros((GRID * GRID, n)) if recon_matrix is None else recon_matrix
        return PostUNet(recon_matrix, mask, **kwargs)
    raise ValueError(f"unknown model {name!r}")


def save_checkpoint(path: Path, model: nn.Module, name: str, kwargs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"name": name, "kwargs": kwargs, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    mask = ckpt["state_dict"]["mask"].numpy()
    model = build_model(ckpt["name"], mask=mask, **ckpt["kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(x[i : i + batch_size])).numpy() for i in range(0, len(x), batch_size)]
    return np.concatenate(out)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
