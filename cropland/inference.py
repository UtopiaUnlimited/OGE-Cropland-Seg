"""Offline OGE inference: RGB images under test/images -> same-name uint8 0/1 GeoTIFFs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window
import torch

from .common import get_device
from .data import normalize, scale_rgb
from .model import load_checkpoint


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def discover_images(dataset):
    """Follow the official image boundary; never search sibling masks/train folders."""
    dataset = Path(dataset)
    image_dir = dataset / "test" / "images"
    if not image_dir.is_dir():
        entries = sorted(p.name for p in dataset.iterdir())[:20] if dataset.is_dir() else []
        raise FileNotFoundError(f"Expected image directory: {image_dir}; dataset entries (up to 20): {entries}")
    paths, samples, extensions = [], [], Counter()
    def walk_error(error):
        raise error
    for directory, folders, files in os.walk(image_dir, onerror=walk_error):
        folders.sort()
        for name in sorted(files):
            path = Path(directory) / name
            extensions[path.suffix.lower() or "<no extension>"] += 1
            if len(samples) < 20:
                samples.append(path.relative_to(image_dir).as_posix())
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                paths.append(path)
    print("INPUT_DISCOVERY " + json.dumps(dict(
        version="input-compat-v2", image_dir=str(image_dir), recursive=True,
        supported_extensions=sorted(IMAGE_SUFFIXES), extension_counts=dict(extensions),
        image_count=len(paths), sample_files=samples), ensure_ascii=False), flush=True)
    if not paths:
        raise ValueError(f"No supported images under {image_dir}; "
                         f"extensions={dict(extensions)}; sample_files={samples}. "
                         "Expected TIFF/PNG/JPG RGB images, including subdirectories.")
    names = {}
    for path in sorted(paths):
        key = path.stem.casefold()
        if key in names:
            raise ValueError(f"Duplicate output name {path.stem}.tif: {names[key]} and {path}. "
                             "OGE requires flat outputs; input stems must be unique.")
        names[key] = path
    return sorted(paths)


def starts(length, tile, stride):
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


@torch.inference_mode()
def predict_file(model, prep, source, destination, device, tile_size=512, overlap=.25, threshold=.5):
    if tile_size < 32 or tile_size % 32:
        raise ValueError("tile_size must be a multiple of 32 and >=32")
    if not 0 <= overlap < 1 or not 0 <= threshold <= 1:
        raise ValueError("Require 0 <= overlap < 1 and 0 <= threshold <= 1")
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite prediction: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(tile_size * (1 - overlap)))
    model.eval()
    # Disk-backed sums bound RAM usage for large rasters. The temporary directory is removed after each image.
    with rasterio.open(source) as src, tempfile.TemporaryDirectory(prefix="oge-infer-", dir=destination.parent) as tmp:
        if src.count != 3:
            raise ValueError(f"Expected 3-band RGB: {source}; "
                             f"driver={src.driver}, bands={src.count}, dtypes={src.dtypes}")
        shape = (src.height, src.width)
        sums = np.memmap(Path(tmp) / "sum.dat", dtype="float32", mode="w+", shape=shape)
        counts = np.memmap(Path(tmp) / "count.dat", dtype="float32", mode="w+", shape=shape)
        sums[:] = 0
        counts[:] = 0
        try:
            for top in starts(src.height, tile_size, stride):
                for left in starts(src.width, tile_size, stride):
                    height, width = min(tile_size, src.height - top), min(tile_size, src.width - left)
                    a = src.read(window=Window(left, top, width, height))
                    a = np.pad(a, ((0, 0), (0, tile_size - height), (0, tile_size - width)), mode="edge")
                    x = normalize(torch.from_numpy(scale_rgb(a, prep)), prep)[None].to(device)
                    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        p = model(x)["logits"].float().sigmoid()[0, 0, :height, :width].cpu().numpy()
                    if not np.isfinite(p).all():
                        raise FloatingPointError(f"Non-finite prediction in {source}")
                    sums[top:top + height, left:left + width] += p
                    counts[top:top + height, left:left + width] += 1
            profile = dict(driver="GTiff", height=src.height, width=src.width, count=1,
                           dtype="uint8", crs=src.crs, transform=src.transform,
                           compress="lzw", nodata=None)
            partial = Path(tmp) / destination.name
            with rasterio.open(partial, "w", **profile) as dst:
                for top in range(0, src.height, 256):
                    height = min(256, src.height - top)
                    n = np.asarray(counts[top:top + height])
                    if (n == 0).any():
                        raise RuntimeError("Uncovered pixels in sliding-window inference")
                    mask = (sums[top:top + height] / n >= threshold).astype(np.uint8)
                    dst.write(mask, 1, window=Window(0, top, src.width, height))
                dst.update_tags(class_0="background", class_1="cropland")
            partial.replace(destination)
        finally:
            # Windows refuses to remove open mmap files.
            sums._mmap.close()
            counts._mmap.close()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", required=True, type=Path, help="Trained best.pt, includes configuration and preprocessing")
    parser.add_argument("--input_dataset", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=float, default=.25)
    parser.add_argument("--threshold", type=float, default=.5)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    paths = discover_images(args.input_dataset)
    device = get_device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    model, checkpoint = load_checkpoint(args.pretrained, device)
    prep, size = checkpoint["preprocessing"], args.tile_size or checkpoint["config"]["image_size"]
    del checkpoint
    print("MODEL " + json.dumps(dict(tile_size=size, overlap=args.overlap,
          threshold=args.threshold, preprocessing=prep, device=str(device))), flush=True)
    warned_uint8 = False
    for index, source in enumerate(paths, 1):
        with rasterio.open(source) as src:
            sample = src.read(window=Window(0, 0, min(64, src.width), min(64, src.height)))
            print("INPUT_IMAGE " + json.dumps(dict(index=index, total=len(paths),
                  filename=source.relative_to(args.input_dataset).as_posix(),
                  driver=src.driver, width=src.width, height=src.height, bands=src.count,
                  dtypes=src.dtypes, sample_min=float(sample.min()),
                  sample_max=float(sample.max())), ensure_ascii=False), flush=True)
            if "uint8" in src.dtypes and prep["upper"] > 255 and not warned_uint8:
                print("INPUT_NOTE: uint8 input differs from the uint16 training data. "
                      "Keeping checkpoint preprocessing unchanged; the platform's "
                      "radiometric conversion and official accuracy are unverified.", flush=True)
                warned_uint8 = True
        predict_file(model, prep, source, args.output_dir / (source.stem + ".tif"),
                     device, size, args.overlap, args.threshold)
    print(f"INFERENCE_COMPLETE images={len(paths)} output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
