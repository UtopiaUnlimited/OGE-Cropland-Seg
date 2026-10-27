from pathlib import Path

import numpy as np
from PIL import Image

from cropland.visualization import save_comparison, write_gallery


def test_comparison_pixels_match_source_truth_prediction_and_error_colors(tmp_path):
    source = np.full((3, 32, 32), 2048, dtype=np.uint16)
    truth = np.zeros((32, 32), dtype=np.uint8)
    prediction = truth.copy()
    truth[0, :2] = 1
    prediction[0, 1:3] = 1
    original_truth, original_prediction = truth.copy(), prediction.copy()
    metrics = dict(iou=1 / 3, f1=.5, precision=.5, recall=.5)
    target = tmp_path / "comparisons/sample.png"
    save_comparison(source, truth, prediction, dict(lower=0, upper=4095),
                    target, "sample.tif", metrics)
    with Image.open(target) as image:
        pixels = np.asarray(image)
        assert image.size == (32 * 4 + 12 * 5, 32 + 96 + 58)
        assert np.all(pixels[96:128, 12:44] == 128)
        assert np.array_equal(pixels[96:128, 56:88, 0], truth * 255)
        assert np.array_equal(pixels[96:128, 100:132, 0], prediction * 255)
        assert pixels[96, 144:148].tolist() == [[78, 153, 244], [48, 194, 121],
                                               [240, 91, 100], [0, 0, 0]]
    assert np.array_equal(truth, original_truth)
    assert np.array_equal(prediction, original_prediction)
    report = dict(split="test", count=1, metrics=metrics,
                  per_image=[dict(filename="sample.tif", comparison_png="comparisons/sample.png", **metrics)])
    write_gallery(tmp_path, report)
    assert 'src="comparisons/sample.png"' in (tmp_path / "gallery.html").read_text(encoding="utf-8")
