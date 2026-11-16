from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel

from .common import load_json


def b5_config():
    return SegformerConfig(num_channels=3, num_labels=1,
                           depths=[3, 6, 40, 3], hidden_sizes=[64, 128, 320, 512],
                           num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1],
                           decoder_hidden_size=768, classifier_dropout_prob=0.1)


class CroplandModel(nn.Module):
    def __init__(self, config=None, boundary=False):
        super().__init__()
        self.semantic = SegformerForSemanticSegmentation(config or b5_config())
        self.has_boundary = boundary
        if boundary:
            c = self.semantic.config.hidden_sizes
            self.boundary = nn.Sequential(nn.Conv2d(c[0] + c[1], 64, 3, padding=1),
                                          nn.GroupNorm(8, 64), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        output = self.semantic(pixel_values=x, output_hidden_states=self.has_boundary)
        result = {"logits": F.interpolate(output.logits, x.shape[-2:], mode="bilinear", align_corners=False)}
        if self.has_boundary:
            shallow, middle = output.hidden_states[:2]
            middle = F.interpolate(middle, shallow.shape[-2:], mode="bilinear", align_corners=False)
            edge = self.boundary(torch.cat([shallow, middle], dim=1))
            result["boundary"] = F.interpolate(edge, x.shape[-2:], mode="bilinear", align_corners=False)
        return result


def build_model(cfg):
    if cfg.get("model_type") == "upernet_r50":
        from .upernet import build_upernet
        return build_upernet(cfg)
    if cfg.get("model_type", "segformer_b5") != "segformer_b5":
        raise ValueError("This shared training entrypoint supports SegFormer/E4. E5 training remains in the supplied E5 ZIP.")
    model = CroplandModel(boundary=cfg["boundary_weight"] > 0)
    if not cfg.get("random_init", False):
        source = Path(cfg["pretrained"])
        if not source.is_dir():
            raise FileNotFoundError(f"Local MiT-B5 weights missing: {source}. Run python -m cropland.download first.")
        encoder, info = SegformerModel.from_pretrained(source, local_files_only=True, output_loading_info=True)
        if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"):
            raise RuntimeError(f"Pretrained encoder did not load completely: {info}")
        model.semantic.segformer.load_state_dict(encoder.state_dict(), strict=True)
        del encoder
    if cfg.get("freeze_encoder", False):
        model.semantic.segformer.requires_grad_(False)
    return model


def save_checkpoint(path, model, preprocessing, cfg, **state):
    source_file = Path(cfg.get("pretrained", ".")) / "source.json"
    payload = dict(format_version=1, model_type=cfg.get("model_type", "segformer_b5"), model=model.state_dict(),
                   model_config=model.semantic.config.to_dict(), boundary=model.has_boundary,
                   pretrained_source=load_json(source_file) if source_file.is_file() else None,
                   preprocessing=preprocessing, config=cfg, **state)
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model_type = checkpoint.get("model_type", "segformer_b5")
    version = checkpoint.get("format_version")
    if version == 2 and model_type == "farseg":
        from .farseg import FarSegModel
        model = FarSegModel(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"], strict=True)
        return model.to(device), checkpoint
    if version != 1 or model_type not in {"segformer_b5", "upernet_r50"}:
        raise ValueError("Unsupported checkpoint format")
    if checkpoint.get("model_type", "segformer_b5") == "upernet_r50":
        from transformers import UperNetConfig
        from .upernet import UPerNetR50
        model = UPerNetR50(UperNetConfig.from_dict(checkpoint["model_config"]))
    else:
        model = CroplandModel(SegformerConfig.from_dict(checkpoint["model_config"]), checkpoint["boundary"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device), checkpoint


def foreground_probability(output):
    """Preserve E5's float32 two-class softmax; E2/E4 use one-logit sigmoid."""
    if "class_logits" in output:
        logits = output["class_logits"].float()
        if logits.shape[1] != 2:
            raise ValueError("Expected background/foreground class logits")
        return logits.softmax(dim=1)[:, 1:2]
    return output["logits"].float().sigmoid()
