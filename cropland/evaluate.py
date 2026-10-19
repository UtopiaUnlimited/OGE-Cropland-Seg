"""Evaluate a held-out split at original resolution using deployment inference."""
import argparse
import warnings
from pathlib import Path

import rasterio
from rasterio.errors import NotGeoreferencedWarning
import torch
from tqdm import tqdm

from .common import get_device, project_path, save_json
from .data import read_rows
from .inference import predict_file
from .metrics import BinaryMetrics
from .model import load_checkpoint
from .train import split_fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=project_path("data/farm"))
    parser.add_argument("--split-dir", type=Path, default=project_path("splits"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", required=True, type=Path, help="New output directory for masks and metrics")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--limit", type=int, help="Smoke test only; metrics will be marked partial")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    if split_fingerprints(args.split_dir) != ckpt["split_fingerprints"]:
        raise ValueError("Evaluation splits differ from the training checkpoint")
    rows = read_rows(args.split_dir / f"{args.split}.csv")
    selected = rows[:args.limit] if args.limit else rows
    size, prep = ckpt["config"]["image_size"], ckpt["preprocessing"]
    del ckpt
    metrics, per_image = BinaryMetrics(), []
    for row in tqdm(selected, desc=f"Evaluate {args.split}"):
        name = row["filename"]
        output = args.output / "masks" / name
        predict_file(model, prep, args.data_root / "train/images" / name, output, device, size)
        with rasterio.open(output) as src, rasterio.open(args.data_root / "train/masks" / name) as lab:
            p, y = torch.from_numpy(src.read(1)), torch.from_numpy(lab.read(1))
        metrics.update(p, y)
        item = BinaryMetrics()
        item.update(p, y)
        per_image.append(dict(filename=name, **item.compute()))
    report = dict(split=args.split, count=len(selected), partial=len(selected) < len(rows),
                  metrics=metrics.compute(), per_image=per_image)
    save_json(args.output / "metrics.json", report)
    print({k: v for k, v in report.items() if k != "per_image"})


if __name__ == "__main__":
    main()
