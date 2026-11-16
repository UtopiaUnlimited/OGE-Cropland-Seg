import numpy as np
import torch

from cropland.farseg import FarSegModel
from cropland.model import foreground_probability, load_checkpoint


def test_farseg_offline_checkpoint_roundtrip_and_two_class_probability(tmp_path, monkeypatch):
    torch.set_num_threads(4)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = FarSegModel().eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        expected = model(x)
    assert expected["class_logits"].shape == (1, 2, 64, 64)
    assert torch.equal(foreground_probability(expected), expected["class_logits"].float().softmax(1)[:, 1:2])
    checkpoint = tmp_path / "farseg.pt"
    torch.save(dict(format_version=2, model_type="farseg", model_config=model.model_config,
                    model=model.state_dict()), checkpoint)
    restored, _ = load_checkpoint(checkpoint)
    restored.eval()
    with torch.inference_mode():
        actual = restored(x)
    assert torch.equal(actual["class_logits"], expected["class_logits"])


def test_foreground_probability_keeps_float32_softmax_before_half_subtraction():
    classes = torch.tensor([[[[1000.]], [[1000.5]]]], dtype=torch.float16)
    result = foreground_probability({"class_logits": classes, "logits": torch.zeros(1, 1, 1, 1)})
    assert result.dtype == torch.float32
    assert np.isclose(result.item(), torch.tensor(.5).sigmoid().item())
    assert torch.equal(foreground_probability({"logits": torch.zeros(1, 1, 1, 1)}), torch.full((1, 1, 1, 1), .5))
