from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("data_root", "split_dir", "preprocessing", "pretrained", "run_dir"):
        cfg[key] = str(project_path(cfg[key]))
    for key in ("image_size", "batch_size", "accumulation_steps", "epochs"):
        if cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    if cfg["image_size"] < 32 or cfg["image_size"] % 32:
        raise ValueError("image_size must be a positive multiple of 32 (>=32)")
    for key in ("max_train_batches", "max_val_batches"):
        if cfg.get(key) is not None and cfg[key] < 1:
            raise ValueError(f"{key} must be positive or null")
    if cfg["boundary_weight"] < 0:
        raise ValueError("boundary_weight must be nonnegative")
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install CUDA PyTorch or choose cpu")
    return torch.device(name)
