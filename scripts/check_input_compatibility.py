"""Exercise real image ENTRYPOINT with official formats and nested image folders.

Synthetic uint8 conversions only test I/O, not accuracy on official hidden data.
"""
from pathlib import Path
import json
import shutil
import subprocess

import cropland
import numpy as np
from PIL import Image
import rasterio


def validate_input_compatibility(image, work, source, reference):
    work = Path(work) / "input_compatibility"
    images = work / "input/test/images"
    images.mkdir(parents=True)
    output = work / "predictions"
    output.mkdir()
    shutil.copy2(source, images / "flat.TIF")
    nested = images / "nested"
    nested.mkdir()
    shutil.copy2(source, nested / "nested_tiff.TIFF")
    with rasterio.open(source) as src:
        array = src.read()
    height, width = array.shape[1:]

    def write_image(path, pixels, driver):
        with rasterio.open(path, "w", driver=driver, height=height, width=width,
                           count=3, dtype=pixels.dtype) as dst:
            dst.write(pixels)

    write_image(nested / "png16.png", array, "PNG")
    # Cover genuine RGB uint8 codecs too; compare to TIFF with identical decoded pixels.
    rgb8 = np.rint(np.clip(array.astype(np.float32) / 1634, 0, 1) * 255).astype(np.uint8)
    pairs = []
    for stem, suffix in (("png8", ".PNG"), ("jpg8", ".jpg"), ("jpeg8", ".JPEG")):
        path = nested / (stem + suffix)
        Image.fromarray(rgb8.transpose(1, 2, 0)).save(path)
        with rasterio.open(path) as src:
            decoded = src.read()
        baseline = stem + "_decoded"
        write_image(nested / (baseline + ".tif"), decoded, "GTiff")
        pairs.append((stem, baseline))
    # A label-like image outside test/images must never be selected.
    masks = work / "input/test/masks"
    masks.mkdir()
    shutil.copy2(source, masks / "must_not_predict.tif")
    command = ["docker", "run", "--rm", "--network", "none", "--gpus", "all",
               "--cpus", "8", "--memory", "32g", "--memory-swap", "32g", "--shm-size", "16g",
               "--read-only", "--tmpfs", "/tmp:rw,size=256m",
               "--mount", f"type=bind,source={work / 'input'},target=/input,readonly",
               "--mount", f"type=bind,source={output},target=/output",
               image, "--input_dataset", "/input", "--output_dir", "/output"]
    with (work / "inference.log").open("w", encoding="utf-8") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    expected = {"flat", "nested_tiff", "png16"} | {s for pair in pairs for s in pair}
    assert {p.name for p in output.iterdir()} == {s + ".tif" for s in expected}
    predictions = {}
    for stem in sorted(expected):
        with rasterio.open(output / (stem + ".tif")) as src:
            assert src.shape == (height, width) and src.count == 1
            assert src.dtypes == ("uint8",) and src.nodata is None
            predictions[stem] = src.read(1)
            assert np.isin(predictions[stem], [0, 1]).all()
    with rasterio.open(reference) as src:
        original = src.read(1)
    for stem in ("flat", "nested_tiff", "png16"):
        assert np.array_equal(predictions[stem], original), stem
    for a, b in pairs:
        assert np.array_equal(predictions[a], predictions[b]), (a, b)
    result = dict(count=len(expected), network="none", read_only_root=True,
                  real_checkpoint=True, tile_size=512, container_exit_code=0,
                  formats=["TIF", "TIFF", "PNG uint16", "PNG uint8", "JPG", "JPEG"],
                  recursive_discovery=True, sibling_masks_excluded=True,
                  uint16_reference_pixel_differences=0, decoded_codec_pixel_differences=0,
                  uint8_accuracy_claim=False, docker_run_arguments=command)
    (work / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("INPUT COMPATIBILITY PASSED: " + str(work), flush=True)
    return result
