import numpy as np
import pytest
import rasterio
from PIL import Image
import torch

from cropland.inference import discover_images, predict_file


def test_discovery_official_formats_nested_and_excludes_labels(tmp_path):
    image_dir = tmp_path / "test/images"
    names = ["a.TIF", "b.tiff", "nested/c.PNG", "nested/d.jpg", "e.JPEG"]
    for name in names:
        path = image_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (image_dir / "metadata.json").write_text("{}")
    (image_dir / "not_an_image.tif").mkdir()
    for folder in ("test/masks", "train/images"):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "label.tif").touch()
    assert {p.relative_to(image_dir).as_posix() for p in discover_images(tmp_path)} == set(names)


@pytest.mark.parametrize("names", [["same.png", "sub/same.tif"], ["A.JPG", "sub/a.TIFF"]])
def test_duplicate_output_names_fail_before_inference(tmp_path, names):
    for name in names:
        path = tmp_path / "test/images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    with pytest.raises(ValueError, match="Duplicate output name"):
        discover_images(tmp_path)


def test_unsupported_and_missing_inputs_have_diagnostics(tmp_path, capsys):
    with pytest.raises(FileNotFoundError, match="Expected image directory"):
        discover_images(tmp_path)
    image_dir = tmp_path / "test/images"
    image_dir.mkdir(parents=True)
    (image_dir / "bundle.zip").touch()
    with pytest.raises(ValueError, match="bundle.zip"):
        discover_images(tmp_path)
    assert '".zip": 1' in capsys.readouterr().out


class PixelModel(torch.nn.Module):
    def forward(self, x):
        return {"logits": (x[:, :1] - .5) * 20}


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg"])
def test_rgb_formats_produce_original_size_binary_tiff(tmp_path, suffix):
    array = np.zeros((39, 47, 3), np.uint8)
    array[8:30, 12:36] = 255
    source = tmp_path / ("source" + suffix)
    Image.fromarray(array).save(source)
    target = tmp_path / "out.tif"
    prep = dict(lower=0, upper=255, mean=[0, 0, 0], std=[1, 1, 1])
    predict_file(PixelModel(), prep, source, target, torch.device("cpu"), tile_size=32)
    with rasterio.open(source) as src, rasterio.open(target) as dst:
        assert dst.shape == src.shape == (39, 47)
        assert dst.count == 1 and dst.dtypes == ("uint8",)
        assert dst.nodata is None
        assert np.array_equal(dst.read(1), (src.read(1) >= 127.5).astype(np.uint8))


def test_uint16_png_keeps_training_pixel_values(tmp_path):
    array = np.zeros((3, 39, 47), np.uint16)
    array[:, 8:30, 12:36] = 4095
    source, target = tmp_path / "source.png", tmp_path / "out.tif"
    with rasterio.open(source, "w", driver="PNG", height=39, width=47,
                       count=3, dtype="uint16") as dst:
        dst.write(array)
    prep = dict(lower=0, upper=4095, mean=[0, 0, 0], std=[1, 1, 1])
    predict_file(PixelModel(), prep, source, target, torch.device("cpu"), tile_size=32)
    with rasterio.open(source) as src, rasterio.open(target) as dst:
        assert np.array_equal(src.read(), array)
        assert np.array_equal(dst.read(1), (array[0] > 0).astype(np.uint8))
