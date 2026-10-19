"""Explicit network step; training and inference load local artifacts only."""
import argparse
import hashlib
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
import requests

from .common import project_path, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=project_path("models/mit-b5"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--http", action="store_true", help="Use direct HTTPS streaming if the Hub transfer backend stalls")
    args = parser.parse_args()
    repo = "nvidia/mit-b5"
    info = HfApi().model_info(repo, revision=args.revision, files_metadata=True)
    names = {f.rfilename for f in info.siblings}
    weights = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
    selected = [n for n in ["config.json", "preprocessor_config.json", weights, "README.md"] if n in names]
    if args.http:
        args.output.mkdir(parents=True, exist_ok=True)
        for name in selected:
            temporary = args.output / (name + ".partial")
            digest = hashlib.sha256()
            with requests.get(f"https://huggingface.co/{repo}/resolve/{info.sha}/{name}?download=true",
                              stream=True, timeout=(15, 30)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                        digest.update(chunk)
            metadata = next(f for f in info.siblings if f.rfilename == name)
            if metadata.lfs and digest.hexdigest() != metadata.lfs.sha256:
                raise RuntimeError(f"SHA256 mismatch: {name}")
            temporary.replace(args.output / name)
            print(f"Downloaded {name}", flush=True)
    else:
        snapshot_download(repo, revision=info.sha, local_dir=args.output, allow_patterns=selected)
    save_json(args.output / "source.json", dict(repository=repo, revision=info.sha, weights=weights))
    print(f"Downloaded {repo}@{info.sha} to {args.output}")


if __name__ == "__main__":
    main()
