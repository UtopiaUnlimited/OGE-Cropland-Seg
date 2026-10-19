"""Audit all rasters, group duplicates, split, then fit scaling on TRAIN ONLY."""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

from .common import project_path, save_json


def raster_files(directory):
    return {p.name: p for p in Path(directory).iterdir() if p.suffix.lower() in {".tif", ".tiff"}}


def split_groups(rows, seed=42, fractions=(0.8, 0.1, 0.1)):
    """Keep identical images together and stratify by foreground proportion."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_sha256"]].append(row)
    strata = defaultdict(list)
    for key, group in sorted(groups.items()):
        rate = float(np.mean([x["foreground_ratio"] for x in group]))
        stratum = int(np.digitize(rate, [0.001, 0.1, 0.3, 0.6, 0.9, 0.999]))
        strata[stratum].append(key)
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for keys in strata.values():
        rng.shuffle(keys)
        # Largest-remainder allocation avoids losing small strata to rounding.
        targets = np.array(fractions) * len(keys)
        sizes = np.floor(targets).astype(int)
        for i in np.argsort(-(targets - sizes), kind="stable")[:len(keys) - sizes.sum()]:
            sizes[i] += 1
        offset = 0
        for name, size in zip(splits, sizes):
            for key in keys[offset:offset + size]:
                splits[name].extend(groups[key])
            offset += size
    if any(not items for items in splits.values()):
        raise ValueError("Too few independent groups for train/val/test")
    return {k: sorted(v, key=lambda x: x["filename"]) for k, v in splits.items()}


def prepare(root, out, seed=42):
    root, out = Path(root), Path(out)
    if (out / "train.csv").exists():
        raise FileExistsError(f"Split already exists: {out}. Choose a new --output to avoid replacing an experiment split.")
    images = raster_files(root / "train/images")
    masks = raster_files(root / "train/masks")
    if images.keys() != masks.keys() or not images:
        raise ValueError(f"Image/mask filename mismatch: {images.keys() ^ masks.keys()}")
    rows = []
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    for name, p in tqdm(sorted(images.items()), desc="Audit rasters"):
        with rasterio.open(p) as src, rasterio.open(masks[name]) as lab:
            a, m = src.read(), lab.read(1)  # nodata=0 in supplied masks IS a valid background label.
            if src.count != 3 or lab.count != 1 or a.shape[1:] != m.shape:
                raise ValueError(f"Invalid dimensions/channels: {name}")
            if src.crs != lab.crs or src.transform != lab.transform:
                raise ValueError(f"Image/mask alignment mismatch: {name}")
            if a.dtype not in (np.uint8, np.uint16):
                raise ValueError(f"Expected uint8/uint16 image: {name}: {a.dtype}")
            if not set(np.unique(m)).issubset({0, 1}):
                raise ValueError(f"Labels outside 0/1: {name}")
            rows.append(dict(filename=name, foreground_ratio=float(m.mean()),
                             height=src.height, width=src.width, dtype=str(a.dtype),
                             minimum=int(a.min()), maximum=int(a.max()),
                             crs=str(src.crs), image_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                             mask_sha256=hashlib.sha256(m.tobytes()).hexdigest()))
    groups = defaultdict(set)
    for row in rows:
        groups[row["image_sha256"]].add(row["mask_sha256"])
    if any(len(labels) > 1 for labels in groups.values()):
        raise ValueError("Duplicate images have conflicting masks; resolve annotations before splitting")
    splits = split_groups(rows, seed)
    histogram = np.zeros(65536, dtype=np.int64)
    for row in tqdm(splits["train"], desc="Fit training radiometry"):
        with rasterio.open(images[row["filename"]]) as src:
            histogram += np.bincount(src.read().ravel(), minlength=65536)
    cumulative = histogram.cumsum()
    low, high = [int(np.searchsorted(cumulative, q * cumulative[-1])) for q in (0.005, 0.995)]
    if high <= low:
        raise ValueError("Training radiometry has no usable dynamic range")
    preprocessing = dict(version=1, method="train_global_percentile_linear", lower=low, upper=high,
                         quantiles=[0.005, 0.995], channels=[1, 2, 3],
                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                         fit_split="train", do_reduce_labels=False)
    out.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        with (out / f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(items)
    save_json(out / "preprocessing.json", preprocessing)
    report = dict(seed=seed, strategy="foreground-stratified image split; identical images grouped",
                  spatial_warning="No source-region metadata supplied. Random image splitting cannot rule out spatial adjacency leakage.",
                  counts={k: len(v) for k, v in splits.items()},
                  foreground_ratio={k: float(np.mean([r["foreground_ratio"] for r in v])) for k, v in splits.items()},
                  total=len(rows), global_range=[min(r["minimum"] for r in rows), max(r["maximum"] for r in rows)],
                  crs_values=sorted({r["crs"] for r in rows}),
                  duplicate_images=len(rows) - len(groups),
                  mask_nodata_note="Stored nodata=0 is background and is included in loss/metrics.",
                  preprocessing=preprocessing)
    save_json(out / "audit.json", report)
    print(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project_path("data/farm"))
    parser.add_argument("--output", type=Path, default=project_path("splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.data_root, args.output, args.seed)


if __name__ == "__main__":
    main()
