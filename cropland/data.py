from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r["filename"] for r in rows}) != len(rows):
        raise ValueError(f"Empty manifest or duplicate filenames: {path}")
    for row in rows:
        if Path(row["filename"]).name != row["filename"]:
            raise ValueError("Manifest filenames must not include directories")
    return rows


def scale_rgb(array, prep):
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB; got {array.shape}")
    lower, upper = prep["lower"], prep["upper"]
    if upper <= lower:
        raise ValueError("Invalid radiometric scaling range")
    return np.clip((array.astype(np.float32) - lower) / (upper - lower), 0, 1)


def normalize(x, prep):
    mean = x.new_tensor(prep["mean"])[:, None, None]
    std = x.new_tensor(prep["std"])[:, None, None]
    return (x - mean) / std


def paired_geometry(image, mask, horizontal, vertical, rotations):
    if horizontal:
        image, mask = image.flip(-1), mask.flip(-1)
    if vertical:
        image, mask = image.flip(-2), mask.flip(-2)
    return torch.rot90(image, rotations, (-2, -1)), torch.rot90(mask, rotations, (-2, -1))


class FarmDataset(Dataset):
    def __init__(self, root, manifest, preprocessing, image_size, augment=False):
        self.root, self.rows = Path(root), read_rows(manifest)
        self.prep, self.size, self.augment = preprocessing, image_size, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        name = self.rows[index]["filename"]
        with rasterio.open(self.root / "train/images" / name) as src:
            a = src.read()
        with rasterio.open(self.root / "train/masks" / name) as src:
            m = src.read(1)
        if not set(np.unique(m)).issubset({0, 1}):
            raise ValueError(f"Invalid labels in {name}")
        x = torch.from_numpy(scale_rgb(a, self.prep))
        y = torch.from_numpy(m.astype(np.float32)).unsqueeze(0)
        x = F.interpolate(x[None], size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=True)[0]
        y = F.interpolate(y[None], size=(self.size, self.size), mode="nearest")[0]
        if self.augment:
            x, y = paired_geometry(x, y, random.random() < .5, random.random() < .5, random.randrange(4))
            # Brightness, contrast and saturation; mask never receives photometric transforms.
            x = x * random.uniform(.9, 1.1)
            center = x.mean(dim=(-2, -1), keepdim=True)
            x = (x - center) * random.uniform(.9, 1.1) + center
            gray = (x * x.new_tensor([.299, .587, .114])[:, None, None]).sum(0, keepdim=True)
            x = ((x - gray) * random.uniform(.9, 1.1) + gray).clamp(0, 1)
        return dict(image=normalize(x, self.prep), mask=y, filename=name)
