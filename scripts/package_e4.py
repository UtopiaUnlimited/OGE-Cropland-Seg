"""Package trained E4 checkpoint, verified validation probabilities and offline code."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/e4_upernet_r50/best.pt")
    parser.add_argument("--export", type=Path, default=ROOT / "outputs/e4_handoff")
    parser.add_argument("--output", type=Path, default=ROOT / "E4_UPerNet.zip")
    args = parser.parse_args()
    report = json.loads((args.export / "validation.json").read_text(encoding="utf-8"))
    if digest(args.checkpoint) != report["checkpoint_sha256"] or report["count"] != 91:
        raise ValueError("Probability export does not match the supplied best.pt")
    expected = set(report["files"])
    if {p.name for p in (args.export / "val_probs").iterdir()} != expected:
        raise ValueError("Missing or extra probability files")
    for name, info in report["files"].items():
        path = args.export / "val_probs" / name
        a = np.load(path, allow_pickle=False)
        if digest(path) != info["sha256"] or a.shape != (512, 512) or a.dtype != np.float32:
            raise ValueError(f"Invalid probability: {name}")
        if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
            raise ValueError(f"Invalid range: {name}")
    # No data, last.pt, virtualenv, pretrained source weights or local absolute-path logs.
    members = {"best.pt": args.checkpoint, "validation.json": args.export / "validation.json"}
    members.update({"val_probs/" + n: args.export / "val_probs" / n for n in sorted(expected)})
    for pattern in ("cropland/*.py", "splits/*.csv", "splits/preprocessing.json"):
        for path in ROOT.glob(pattern):
            members[path.relative_to(ROOT).as_posix()] = path
    members["config.yaml"] = ROOT / "configs/e4_upernet_r50.yaml"
    members["E4_GUIDE.md"] = ROOT / "docs/e4-upernet.md"
    result_doc = ROOT / "docs/result-e4-upernet-20261116.md"
    if result_doc.is_file():
        members["RESULTS.md"] = result_doc
        members[result_doc.name] = result_doc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, path in members.items():
            archive.write(path, name)
        archive.writestr("requirements.txt", "\n".join(f"{k}=={v}" for k, v in report["dependencies"].items()) + "\n")
        archive.writestr("resolved_config.json", json.dumps(report["config"], indent=2))
        archive.writestr("SOURCE_SHA256.json", json.dumps(
            {name: digest(path) for name, path in members.items()}, indent=2))
        archive.writestr("SCORES.md", "# E4 UPerNet + ResNet-50\n\n"
            f"Best epoch: {report['best_epoch']}\n\n"
            f"Selection val foreground IoU: {report['selected_validation_iou']:.10f}\n\n"
            f"Exported val foreground IoU: {report['exported_validation_metrics']['iou']:.10f}\n\n"
            "Threshold used for IoU: 0.5. Validation: all 91 images, global pixel confusion counts.\n\n"
            "val_probs/*.npy: raw sigmoid foreground probabilities; float32, 512x512.\n\n"
            "Pretrained source: " + json.dumps(report["pretrained_source"], indent=2) + "\n")
        archive.writestr("README.md", "E4 UPerNet + ResNet-50\n\n"
            "See E4_GUIDE.md for architecture and preprocessing. best.pt contains all weights.\n"
            "Install requirements.txt with a PyTorch build matching the target OS/GPU. No download is needed during inference.\n"
            "Regenerate all 91 probabilities from this extracted ZIP root (data supplied separately):\n\n"
            "python -m cropland.e4_export --checkpoint best.pt --data-root /path/to/farm --split-dir splits --output regenerated --device cuda\n\n"
            "Offline batch TIFF masks for OGE input_dataset/test/images:\n\n"
            "python -m cropland.inference --pretrained best.pt --input_dataset /input --output_dir /output --device cuda\n\n"
            f"Best epoch: {report['best_epoch']}; val foreground IoU: {report['selected_validation_iou']:.10f}; threshold: 0.5.\n"
            "Raw float32 foreground probabilities are in val_probs/. Provenance and file hashes: validation.json.\n")
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
    args.output.with_suffix(".zip.sha256").write_text(digest(args.output) + "  " + args.output.name + "\n", encoding="ascii")
    print(args.output)


if __name__ == "__main__":
    main()
