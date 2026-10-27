"""Build, test offline with a GPU, and export the OGE submission image.

Run from an Internet-connected machine with Docker and the oge-seg environment:
    python scripts/package_submission.py
The public Git repository never contains the generated weights or archive.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cropland  # Select the Rasterio wheel's own GDAL/PROJ resources.
import numpy as np
import rasterio
import torch

from cropland.common import save_json
from cropland.metrics import BinaryMetrics
from cropland.train import split_fingerprints


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command, log=None):
    print("+", subprocess.list2cmdline(list(map(str, command))), flush=True)
    if log:
        with Path(log).open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print(Path(log).read_text(encoding="utf-8", errors="replace")[-12000:])
            raise subprocess.CalledProcessError(result.returncode, command)
        return ""
    return subprocess.check_output(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace")



def finalize_archive(archive, image_info, validation):
    """Verify saved runtime/layers for both classic and containerd image stores."""
    archive = Path(archive)
    if image_info["Id"] != validation["image_id"]:
        raise ValueError("The current image is not the validated image")
    if archive.stat().st_size >= 20_000_000_000:
        raise ValueError(f"Archive exceeds 20 GB: {archive}")
    with tarfile.open(archive, "r:") as tar:
        saved = json.load(tar.extractfile("manifest.json"))
        matching = [item for item in saved if validation["image_tag"] in (item.get("RepoTags") or [])]
        if len(matching) != 1:
            raise ValueError("Unexpected saved image manifest")
        config = json.load(tar.extractfile(matching[0]["Config"]))
        if config.get("os") != "linux" or config.get("architecture") != "amd64":
            raise ValueError("Saved image platform differs")
        # containerd uses a manifest/index digest as image Id; it is NOT the config digest.
        if config["rootfs"]["diff_ids"] != image_info["RootFS"]["Layers"]:
            raise ValueError("Saved filesystem layers differ from the tested image")
        for key in ("Entrypoint", "Cmd", "Env", "WorkingDir", "User", "Labels"):
            # Docker inspect may omit empty defaults that the saved OCI config retains.
            default = {} if key == "Labels" else ([] if key in ("Entrypoint", "Cmd", "Env") else "")
            if (config["config"].get(key) or default) != (image_info["Config"].get(key) or default):
                raise ValueError(f"Saved runtime configuration differs: {key}")
    validation["archive"] = dict(filename=archive.name, bytes=archive.stat().st_size, sha256=digest(archive))
    save_json(archive.with_suffix(".manifest.json"), validation)
    archive.with_suffix(".sha256").write_text(
        validation["archive"]["sha256"] + "  " + archive.name + "\n", encoding="ascii")
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="oge-cropland-seg:b5-20261027")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/b5_server/best.pt")
    parser.add_argument("--reference", type=Path, help="Optional local TIFF predictions for pixel agreement")
    parser.add_argument("--base-image", help="Optional pinned official base image tag/digest")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._:/-]*", args.image):
        parser.error("Use a lowercase Docker image reference")
    submission = ROOT / "submission"
    archive = submission / (re.sub(r"[^a-zA-Z0-9_.-]", "_", args.image) + ".tar")
    if archive.exists():
        raise FileExistsError(f"Archive already exists; choose a new --image tag: {archive}")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Copy the trained best.pt to {checkpoint} first")
    print(run(["docker", "version", "--format", "{{.Server.Os}}/{{.Server.Arch}}"]))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    fingerprints = split_fingerprints(ROOT / "splits")
    if state["split_fingerprints"] != fingerprints:
        raise ValueError("Checkpoint and local split manifests do not match")
    manifest = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint_sha256=digest(checkpoint),
        checkpoint_epoch=state["epoch"] + 1,
        validation_iou=state["best_iou"],
        input_size=state["config"]["image_size"],
        preprocessing=state["preprocessing"],
        split_fingerprints=fingerprints,
        source_files={p.relative_to(ROOT).as_posix(): digest(p)
                      for p in sorted((ROOT / "cropland").glob("*.py"))},
        data_source="OGE platform; copyright belongs to the original authors",
    )
    del state
    build = submission / "build"
    build.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, build / "best.pt")
    save_json(build / "manifest.json", manifest)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = submission / ("validation_" + stamp)
    work.mkdir(parents=True, exist_ok=False)
    images = work / "input/test/images"
    images.mkdir(parents=True)
    output = work / "predictions"
    output.mkdir()
    rows = list(csv.DictReader((ROOT / "splits/test.csv").open(encoding="utf-8")))
    if not rows or len({r["filename"] for r in rows}) != len(rows):
        raise ValueError("Empty test split or duplicate filenames")
    for row in rows:
        name = row["filename"]
        if Path(name).name != name:
            raise ValueError("Invalid filename in manifest")
        source = ROOT / "data/farm/train/images" / name
        with rasterio.open(source) as src:
            if hashlib.sha256(src.read().tobytes()).hexdigest() != row["image_sha256"]:
                raise ValueError(f"Test source changed: {name}")
        shutil.copy2(source, images / name)
    build_command = ["docker", "build", "--platform", "linux/amd64", "--tag", args.image]
    if args.base_image:
        build_command += ["--build-arg", "BASE_IMAGE=" + args.base_image]
    run(build_command + ["."], work / "build.log")
    image_info = json.loads(run(["docker", "image", "inspect", args.image]))[0]
    expected_entrypoint = ["python", "-m", "cropland.inference", "--pretrained", "/opt/oge/best.pt", "--device", "cuda"]
    if image_info["Os"] != "linux" or image_info["Architecture"] != "amd64":
        raise ValueError("Submission must be linux/amd64")
    if image_info["Config"]["Entrypoint"] != expected_entrypoint:
        raise ValueError("Unexpected ENTRYPOINT")
    if image_info["Size"] >= 20_000_000_000:
        raise ValueError("Image exceeds the conservative 20 GB limit")
    runtime_probe = (
        "import hashlib,json,torch; "
        "m=json.load(open('/opt/oge/manifest.json')); "
        "f=open('/opt/oge/best.pt','rb'); "
        "assert hashlib.file_digest(f,'sha256').hexdigest()==m['checkpoint_sha256']; "
        "assert torch.cuda.is_available(); "
        "x=torch.ones(16,device='cuda'); assert x.sum().item()==16; "
        "print(json.dumps({'torch':str(torch.__version__),'cuda':torch.version.cuda,"
        "'gpu':torch.cuda.get_device_name(0),'checkpoint_sha256':m['checkpoint_sha256']}))"
    )
    runtime = json.loads(run(["docker", "run", "--rm", "--network", "none", "--gpus", "all",
                              "--entrypoint", "python", args.image, "-c", runtime_probe]))
    if runtime["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
        raise ValueError("Image contains a different checkpoint")
    # The only model arguments below are precisely the two provided by the OGE platform.
    name = "oge-validation-" + stamp.replace("_", "-")
    command = ["docker", "run", "--name", name, "--network", "none", "--gpus", "all",
               "--cpus", "8", "--memory", "32g", "--memory-swap", "32g", "--shm-size", "16g",
               "--read-only", "--tmpfs", "/tmp:rw,size=256m",
               "--mount", f"type=bind,source={images.parents[1]},target=/input,readonly",
               "--mount", f"type=bind,source={output},target=/output",
               args.image, "--input_dataset", "/input", "--output_dir", "/output"]
    run(command, work / "inference.log")
    container_info = json.loads(run(["docker", "container", "inspect", name]))[0]
    if container_info["State"]["ExitCode"] != 0 or container_info["HostConfig"]["NetworkMode"] != "none":
        raise ValueError("Offline container validation failed")
    freeze = run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "python",
                  args.image, "-m", "pip", "freeze"])
    (work / "requirements-linux-runtime.txt").write_text(freeze, encoding="utf-8")
    expected = {r["filename"] for r in rows}
    actual = {p.name for p in output.iterdir()}
    if expected != actual:
        raise ValueError(f"Output filenames differ: missing={expected-actual}, extra={actual-expected}")
    metrics = BinaryMetrics()
    comparison_differences = 0
    pixels = 0
    for row in rows:
        filename = row["filename"]
        with rasterio.open(images / filename) as src, rasterio.open(output / filename) as pred, rasterio.open(ROOT / "data/farm/train/masks" / filename) as lab:
            p, y = pred.read(1), lab.read(1)
            if hashlib.sha256(y.tobytes()).hexdigest() != row["mask_sha256"]:
                raise ValueError(f"Test label changed: {filename}")
            if pred.count != 1 or pred.dtypes != ("uint8",) or pred.nodata is not None:
                raise ValueError(f"Invalid output format: {filename}")
            if pred.shape != src.shape or pred.crs != src.crs or pred.transform != src.transform:
                raise ValueError(f"Output geometry mismatch: {filename}")
            if not np.isin(p, [0, 1]).all():
                raise ValueError(f"Invalid class encoding: {filename}")
            metrics.update(torch.from_numpy(p), torch.from_numpy(y))
            pixels += p.size
            if args.reference:
                with rasterio.open(args.reference / filename) as ref:
                    a = ref.read(1)
                    if a.shape != p.shape:
                        raise ValueError("Reference dimensions differ")
                    comparison_differences += int(np.count_nonzero(a != p))
    validation = dict(image_id=image_info["Id"], image_tag=args.image, image_bytes=image_info["Size"],
                      checkpoint=manifest, count=len(rows), partial=False, metrics=metrics.compute(),
                      runtime=runtime, network="none", read_only_root=True, cpus=8,
                      configured_memory_bytes=32*1024**3, shm_bytes=16*1024**3,
                      container_exit_code=0, output_format_checked=True,
                      reference_pixel_differences=comparison_differences if args.reference else None,
                      reference_pixel_agreement=1-comparison_differences/pixels if args.reference else None,
                      docker_run_arguments=command,
                      tested_at_utc=datetime.now(timezone.utc).isoformat())
    save_json(work / "validation.json", validation)
    print(json.dumps(validation["metrics"], indent=2), flush=True)
    run(["docker", "container", "rm", name])
    run(["docker", "save", "--output", str(archive), args.image], work / "export.log")
    finalize_archive(archive, image_info, validation)
    run(["docker", "load", "--input", str(archive)], work / "load.log")
    loaded = json.loads(run(["docker", "image", "inspect", args.image]))[0]
    if loaded["RootFS"] != image_info["RootFS"]:
        raise ValueError("Reloaded image layers differ")
    for key in ("Entrypoint", "Cmd", "Env", "WorkingDir", "User", "Labels"):
        if loaded["Config"].get(key) != image_info["Config"].get(key):
            raise ValueError(f"Reloaded image runtime differs: {key}")
    validation["archive"]["load_verified"] = True
    validation["archive"]["loaded_image_id"] = loaded["Id"]
    save_json(archive.with_suffix(".manifest.json"), validation)
    print("READY TO UPLOAD:", archive, flush=True)
    print("Archive bytes:", archive.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
