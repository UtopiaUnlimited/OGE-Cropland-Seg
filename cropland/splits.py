"""Validate split identities without importing training/TensorBoard dependencies."""
import hashlib
from pathlib import Path

from .data import read_rows


def split_fingerprints(directory):
    result, names, hashes = {}, set(), set()
    for name in ("train", "val", "test"):
        path = Path(directory) / f"{name}.csv"
        rows = read_rows(path)
        current = {r["filename"] for r in rows}
        current_hashes = {r["image_sha256"] for r in rows}
        if current & names or current_hashes & hashes:
            raise ValueError(f"Data leakage: overlapping {name} split")
        names |= current
        hashes |= current_hashes
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
