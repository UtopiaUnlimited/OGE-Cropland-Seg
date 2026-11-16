"""Export E4 raw foreground probabilities, aligned to all 91 validation images."""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import torch
from tqdm import tqdm

from .common import get_device, project_path, save_json
from .data import normalize, read_rows, scale_rgb
from .metrics import BinaryMetrics
from .model import load_checkpoint
from .splits import split_fingerprints


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dependency_versions():
    packages = {}
    for name in ("torch", "transformers", "numpy", "rasterio", "PyYAML", "Pillow",
                 "tqdm", "safetensors", "huggingface-hub", "tensorboard", "setuptools"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            # TensorBoard is a training UI, not required by offline export.
            if name != "tensorboard":
                raise
    return packages


@torch.inference_mode()
def probability(model, array, prep, device):
    """No resampling, flipping, thresholding or postprocessing of probabilities."""
    x = normalize(torch.from_numpy(scale_rgb(array, prep)), prep)[None].to(device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = model(x)["logits"].float()
    if logits.shape[1] == 1:
        p = logits.sigmoid()[0, 0]
    elif logits.shape[1] == 2:
        p = logits.softmax(1)[0, 1]
    else:
        raise ValueError("Expected binary single- or two-channel logits")
    a = p.cpu().numpy().astype(np.float32)
    if a.shape != array.shape[1:] or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError("Invalid probability shape/range")
    return a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=project_path("data/farm"))
    parser.add_argument("--split-dir", type=Path, default=project_path("splits"))
    parser.add_argument("--output", type=Path, required=True, help="New directory containing val_probs and validation.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    device = get_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint.get("model_type") != "upernet_r50":
        raise ValueError("E4 export requires a UPerNet-R50 checkpoint")
    if checkpoint["split_fingerprints"] != split_fingerprints(args.split_dir):
        raise ValueError("Validation split differs from training")
    rows = read_rows(args.split_dir / "val.csv")
    if len(rows) != 91 or checkpoint["config"]["image_size"] != 512:
        raise ValueError("E4 requires exactly 91 validation images and 512 input")
    if len({Path(r["filename"]).stem.casefold() for r in rows}) != 91:
        raise ValueError("Non-unique output stems")
    args.output.mkdir(parents=True, exist_ok=False)
    folder = args.output / "val_probs"
    folder.mkdir()
    model.eval()
    metrics, files = BinaryMetrics(), {}
    for row in tqdm(rows, desc="Export E4 validation probabilities"):
        name = row["filename"]
        with rasterio.open(args.data_root / "train/images" / name) as src:
            image = src.read()
        with rasterio.open(args.data_root / "train/masks" / name) as src:
            truth = src.read(1)
        if image.shape != (3, 512, 512) or truth.shape != (512, 512):
            raise ValueError(f"Invalid validation image dimensions: {name}")
        for a, key in ((image, "image_sha256"), (truth, "mask_sha256")):
            if hashlib.sha256(a.tobytes()).hexdigest() != row[key]:
                raise ValueError(f"Data changed: {name} {key}")
        p = probability(model, image, checkpoint["preprocessing"], device)
        path = folder / (Path(name).stem + ".npy")
        np.save(path, p, allow_pickle=False)
        metrics.update(torch.from_numpy(p >= .5), torch.from_numpy(truth))
        files[path.name] = dict(source=name, sha256=digest(path), shape=list(p.shape), dtype=str(p.dtype))
    report = dict(model="E4 UPerNet + ResNet-50", checkpoint_sha256=digest(args.checkpoint),
        best_epoch=checkpoint["epoch"] + 1, selected_validation_iou=checkpoint["best_iou"],
        exported_validation_metrics=metrics.compute(), threshold=.5,
        probability="sigmoid(single logit), foreground=1; no binarization/TTA/postprocessing",
        input_size=512, count=91, split_fingerprints=checkpoint["split_fingerprints"],
        preprocessing=checkpoint["preprocessing"], pretrained_source=checkpoint["pretrained_source"],
        model_config=checkpoint["model_config"], config=checkpoint["config"],
        dependencies=dependency_versions(),
        device=str(device), files=files)
    report["iou_difference_from_training"] = metrics.compute()["iou"] - checkpoint["best_iou"]
    # Batch-1 export versus batch-2 validation can differ slightly in CUDA AMP.
    if abs(report["iou_difference_from_training"]) > 1e-4:
        raise ValueError("Exported probability IoU differs from selected checkpoint; inspect before delivery")
    save_json(args.output / "validation.json", report)
    print(report["exported_validation_metrics"])


if __name__ == "__main__":
    main()
