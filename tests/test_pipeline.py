from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch
from transformers import SegformerConfig

from cropland.data import paired_geometry, scale_rgb
from cropland.inference import predict_file, starts
from cropland.losses import bce_dice, boundary_target
from cropland.metrics import BinaryMetrics
from cropland.model import CroplandModel, load_checkpoint, save_checkpoint
from cropland.prepare import split_groups
from cropland.train import accumulation_weight


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(4, previous))
    yield
    torch.set_num_threads(previous)


PREP = dict(lower=0, upper=4095, mean=[0, 0, 0], std=[1, 1, 1])


def test_gradient_accumulation_with_partial_group_matches_full_batch():
    inputs = torch.tensor([1., 2., 3., 4., 5., 6., 7.])
    a = torch.tensor(0.3, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    ((a * inputs - 2) ** 2).mean().backward()
    for i, chunk in enumerate(inputs.split(2)):
        weight = accumulation_weight(i, 4, 2, 7, 4, len(chunk))
        (((b * chunk - 2) ** 2).mean() * weight).backward()
    assert torch.allclose(a.grad, b.grad)
    assert accumulation_weight(4, 5, 2, 9, 4, 1) == 1.0


def test_synchronized_geometry():
    mask = torch.arange(35).reshape(1, 5, 7)
    image = mask.repeat(3, 1, 1)
    for h in (True, False):
        for v in (True, False):
            for k in range(4):
                x, y = paired_geometry(image, mask, h, v, k)
                assert torch.equal(x[0], y[0])


def test_uint16_scaling():
    a = np.full((3, 2, 2), 2048, dtype=np.uint16)
    scaled = scale_rgb(a, PREP)
    assert np.allclose(scaled, 2048 / 4095)
    assert scaled.dtype == np.float32


def test_splits_cover_and_group_duplicates():
    rows = [dict(filename=f"{i}.tif", image_sha256=str(i // 2), foreground_ratio=.4) for i in range(40)]
    s = split_groups(rows)
    assert s == split_groups(rows)
    assert sum(map(len, s.values())) == 40
    a, b, c = [{x["image_sha256"] for x in s[k]} for k in ("train", "val", "test")]
    assert not (a & b or a & c or b & c)


def test_metrics_background_and_empty_foreground():
    m = BinaryMetrics()
    m.update(torch.tensor([1, 1, 0, 0]), torch.tensor([1, 0, 1, 0]))
    assert m.compute()["iou"] == pytest.approx(1 / 3)
    assert m.compute()["f1"] == .5
    m = BinaryMetrics()
    m.update(torch.zeros(10), torch.zeros(10))
    assert m.compute()["iou"] == 1
    assert m.compute()["tn"] == 10


def test_loss_empty_and_full_labels():
    for value in (0., 1.):
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        target = torch.full_like(logits, value)
        loss = bce_dice(logits, target)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
        assert boundary_target(target).sum() == 0


class PixelModel(torch.nn.Module):
    def forward(self, x):
        return {"logits": (x[:, :1] - .5) * 20}


def test_tiled_inference_preserves_pixels_georeference_and_background(tmp_path):
    h, w = 73, 107
    array = np.zeros((3, h, w), dtype=np.uint16)
    array[:, 20:60, 31:90] = 4095
    transform = from_origin(320000, 3400000, .5, .5)
    source, target = tmp_path / "source.tif", tmp_path / "result.tif"
    with rasterio.open(source, "w", driver="GTiff", height=h, width=w, count=3,
                       dtype="uint16", crs="EPSG:32650", transform=transform, nodata=0) as dst:
        dst.write(array)
    predict_file(PixelModel(), PREP, source, target, torch.device("cpu"), tile_size=32)
    with rasterio.open(target) as src:
        assert src.shape == (h, w) and src.count == 1
        assert src.transform == transform and src.crs.to_epsg() == 32650
        assert src.nodata is None
        assert np.array_equal(src.read(1), (array[0] > 0).astype(np.uint8))
    assert not list(tmp_path.glob("oge-infer-*"))
    with pytest.raises(FileExistsError):
        predict_file(PixelModel(), PREP, source, target, torch.device("cpu"), 32)


def test_small_image_padding(tmp_path):
    source, target = tmp_path / "small.tif", tmp_path / "out.tif"
    with rasterio.open(source, "w", driver="GTiff", height=7, width=11, count=3,
                       dtype="uint16", transform=from_origin(3, 4, 1, 1)) as dst:
        dst.write(np.full((3, 7, 11), 4095, np.uint16))
    predict_file(PixelModel(), PREP, source, target, torch.device("cpu"), 32)
    with rasterio.open(target) as src:
        assert src.read(1).sum() == 77


def test_checkpoint_is_self_contained_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = SegformerConfig(depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 32, 64],
                          num_attention_heads=[1, 2, 4, 8], decoder_hidden_size=32, num_labels=1)
    model = CroplandModel(cfg, boundary=True).eval()
    x = torch.randn(1, 3, 32, 32)
    path = tmp_path / "model.pt"
    with torch.no_grad():
        expected = model(x)
    save_checkpoint(path, model, PREP, {"image_size": 32})
    restored, state = load_checkpoint(path)
    restored.eval()
    with torch.no_grad():
        actual = restored(x)
    assert torch.equal(expected["logits"], actual["logits"])
    assert actual["boundary"].shape == (1, 1, 32, 32)
    assert state["preprocessing"] == PREP


def test_epoch_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    import csv
    import cropland.train as training
    from cropland.common import save_json

    root = tmp_path / "farm"
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for folder in ("images", "masks"):
        (root / "train" / folder).mkdir(parents=True)
    for split, ids in {"train": [0, 1, 2], "val": [3], "test": [4]}.items():
        with (split_dir / f"{split}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "image_sha256"])
            writer.writeheader()
            for i in ids:
                name = f"img{i}.tif"
                writer.writerow(dict(filename=name, image_sha256=str(i)))
                array = np.random.default_rng(i).integers(0, 4096, (3, 32, 32), dtype=np.uint16)
                for folder, a in (("images", array), ("masks", (array[:1] > 2000).astype(np.uint8))):
                    with rasterio.open(root / "train" / folder / name, "w", driver="GTiff",
                                       width=32, height=32, count=len(a), dtype=a.dtype,
                                       transform=from_origin(3, 4, 1, 1)) as dst:
                        dst.write(a)
    save_json(split_dir / "preprocessing.json", PREP)
    small = SegformerConfig(depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 32, 64],
                            num_attention_heads=[1, 2, 4, 8], decoder_hidden_size=32, num_labels=1)
    monkeypatch.setattr(training, "build_model", lambda cfg: CroplandModel(small, boundary=True))
    cfg = dict(seed=42, data_root=str(root), split_dir=str(split_dir),
               preprocessing=str(split_dir / "preprocessing.json"), pretrained="unused", random_init=True,
               run_dir=str(tmp_path / "continuous"), image_size=32, batch_size=2,
               accumulation_steps=2, epochs=2, learning_rate=.0001, weight_decay=.01,
               warmup_epochs=0, patience=5, num_workers=0, amp=False, device="cpu",
               freeze_encoder=False, boundary_weight=.2, max_train_batches=None, max_val_batches=None)
    training.train(cfg)
    cfg["run_dir"] = str(tmp_path / "interrupted")
    original_save = training.save_checkpoint
    def interrupt_after_epoch(path, *args, **kwargs):
        original_save(path, *args, **kwargs)
        if Path(path).name == "last.pt" and kwargs["epoch"] == 0:
            raise InterruptedError("Simulated interruption after persisted epoch")
    monkeypatch.setattr(training, "save_checkpoint", interrupt_after_epoch)
    with pytest.raises(InterruptedError):
        training.train(cfg)
    monkeypatch.setattr(training, "save_checkpoint", original_save)
    training.train(cfg, Path(cfg["run_dir"]) / "last.pt")
    a = torch.load(tmp_path / "continuous/last.pt", weights_only=True)
    b = torch.load(tmp_path / "interrupted/last.pt", weights_only=True)
    assert a["global_step"] == b["global_step"] == 2
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name]), name
