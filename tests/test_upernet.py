import numpy as np
import torch
from transformers import ResNetConfig, UperNetConfig

from cropland.e4_export import probability
from cropland.losses import combined_loss
from cropland.model import load_checkpoint, save_checkpoint
from cropland.upernet import UPerNetR50


PREP = dict(lower=0, upper=4095, mean=[0, 0, 0], std=[1, 1, 1])


def tiny_model():
    backbone = ResNetConfig(depths=[1, 1, 1, 1], hidden_sizes=[32, 64, 128, 256],
        embedding_size=16, out_features=["stage1", "stage2", "stage3", "stage4"])
    return UPerNetR50(UperNetConfig(backbone_config=backbone, hidden_size=32,
        auxiliary_in_channels=128, auxiliary_channels=16, num_labels=1))


def test_upernet_auxiliary_gradients_batch_one_and_offline_checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(4)
    model = tiny_model().train()
    x = torch.randn(1, 3, 64, 64)
    output = model(x)
    assert output["logits"].shape == (1, 1, 64, 64)
    loss = combined_loss(output, torch.zeros_like(output["logits"]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.semantic.auxiliary_head.classifier.weight.grad is not None
    assert not any(m.training for m in model.semantic.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d))
    model.eval()
    with torch.no_grad():
        expected = model(x)["logits"]
    save_checkpoint(tmp_path / "best.pt", model, PREP, dict(model_type="upernet_r50", image_size=64))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    restored, state = load_checkpoint(tmp_path / "best.pt")
    restored.eval()
    with torch.no_grad():
        assert torch.equal(restored(x)["logits"], expected)
    assert state["model_type"] == "upernet_r50"


def test_probability_is_unthresholded_float32_and_preserves_pixel_alignment():
    class Pixel(torch.nn.Module):
        def forward(self, x):
            return {"logits": x[:, :1] * 2 - 1}
    array = np.zeros((3, 512, 512), dtype=np.uint16)
    array[0, 100:200, 200:300] = 4095
    p = probability(Pixel(), array, PREP, torch.device("cpu"))
    assert p.shape == (512, 512) and p.dtype == np.float32
    assert np.allclose(p[100:200, 200:300], torch.tensor(1.).sigmoid().item())
    assert p[0, 0] == torch.tensor(-1.).sigmoid().item()
    assert ((p > 0) & (p < 1)).all()


def test_export_dependency_record_does_not_require_tensorboard(monkeypatch):
    from importlib.metadata import PackageNotFoundError
    import cropland.e4_export as export
    original = export.version
    def without_tensorboard(name):
        if name == "tensorboard":
            raise PackageNotFoundError(name)
        return original(name)
    monkeypatch.setattr(export, "version", without_tensorboard)
    dependencies = export.dependency_versions()
    assert "torch" in dependencies and "tensorboard" not in dependencies
