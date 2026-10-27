"""Evaluate a held-out split at original resolution using deployment inference."""
import argparse
import csv
import hashlib
import time
import warnings
from datetime import datetime, timezone
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
from .visualization import save_comparison, write_gallery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=project_path("data/farm"))
    parser.add_argument("--split-dir", type=Path, default=project_path("splits"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", required=True, type=Path, help="New output directory for masks and metrics")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--limit", type=int, help="Smoke test only; metrics will be marked partial")
    parser.add_argument("--save-png", action="store_true", help="Save comparison PNGs and an offline gallery")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    started = time.perf_counter()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    if split_fingerprints(args.split_dir) != ckpt["split_fingerprints"]:
        raise ValueError("Evaluation splits differ from the training checkpoint")
    rows = read_rows(args.split_dir / f"{args.split}.csv")
    selected = rows[:args.limit] if args.limit else rows
    size, prep = ckpt["config"]["image_size"], ckpt["preprocessing"]
    with args.checkpoint.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    protocol = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha256,
                    checkpoint_epoch=ckpt["epoch"] + 1, tile_size=size, overlap=0.25, threshold=0.5,
                    preprocessing=prep, split_fingerprints=ckpt["split_fingerprints"],
                    device=str(device), torch_version=str(torch.__version__),
                    gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    aggregation="dataset-wide confusion counts", background_included=True,
                    inference_autocast=device.type == "cuda", tta=False, postprocessing=False)
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
        result = dict(filename=name, **item.compute())
        if args.save_png:
            relative = Path("comparisons") / (Path(name).stem + ".png")
            with rasterio.open(args.data_root / "train/images" / name) as src:
                rgb = src.read()
            save_comparison(rgb, y.numpy(), p.numpy(), prep, args.output / relative, name, result)
            result["comparison_png"] = relative.as_posix()
        per_image.append(result)
    report = dict(split=args.split, count=len(selected), partial=len(selected) < len(rows),
                  metrics=metrics.compute(), per_image=per_image, protocol=protocol,
                  evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.perf_counter() - started)
    save_json(args.output / "metrics.json", report)
    with (args.output / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    if args.save_png:
        write_gallery(args.output, report)
    print({k: v for k, v in report.items() if k != "per_image"})


if __name__ == "__main__":
    main()
