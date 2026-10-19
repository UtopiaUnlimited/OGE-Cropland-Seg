"""Offline OGE inference: test/images/*.tif -> same-name uint8 0/1 GeoTIFFs."""
from __future__ import annotations

import argparse
import tempfile
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window
import torch
from tqdm import tqdm

from .common import get_device
from .data import normalize, scale_rgb
from .model import load_checkpoint


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
            raise ValueError(f"Expected 3-band RGB: {source}")
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
    image_dir = args.input_dataset / "test/images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Expected official dataset structure: {image_dir}")
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff"})
    if not paths:
        raise ValueError("No TIFF images found")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Duplicate TIFF stems would overwrite output files")
    device = get_device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    model, checkpoint = load_checkpoint(args.pretrained, device)
    prep, size = checkpoint["preprocessing"], args.tile_size or checkpoint["config"]["image_size"]
    del checkpoint
    for source in tqdm(paths, desc="Offline inference"):
        predict_file(model, prep, source, args.output_dir / (source.stem + ".tif"),
                     device, size, args.overlap, args.threshold)


if __name__ == "__main__":
    main()
