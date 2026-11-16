"""E4: Transformers UPerNet PPM/FPN decoder with an ImageNet ResNet-50 backbone."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import ResNetConfig, ResNetBackbone, UperNetConfig, UperNetForSemanticSegmentation


def group_norm(module):
    """Decoder GN supports small batches and the PPM 1x1 pooled feature."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            groups = min(32, max(1, child.num_features // 2))
            while child.num_features % groups:
                groups -= 1
            replacement = nn.GroupNorm(groups, child.num_features, eps=child.eps)
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            group_norm(child)


class UPerNetR50(nn.Module):
    has_boundary = False

    def __init__(self, config=None):
        super().__init__()
        config = config or UperNetConfig(
            backbone_config=ResNetConfig(depths=[3, 4, 6, 3],
                hidden_sizes=[256, 512, 1024, 2048],
                out_features=["stage1", "stage2", "stage3", "stage4"]),
            num_labels=1, hidden_size=512, pool_scales=[1, 2, 3, 6],
            auxiliary_in_channels=1024, auxiliary_channels=256,
            use_auxiliary_head=True, auxiliary_loss_weight=.4)
        self.semantic = UperNetForSemanticSegmentation(config)
        group_norm(self.semantic.decode_head)
        if self.semantic.auxiliary_head is not None:
            group_norm(self.semantic.auxiliary_head)

    def train(self, mode=True):
        super().train(mode)
        # Keep pretrained backbone BN statistics; affine weights still learn.
        for layer in self.semantic.backbone.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self, x):
        features = self.semantic.backbone(x).feature_maps
        result = {"logits": F.interpolate(self.semantic.decode_head(features),
                  x.shape[-2:], mode="bilinear", align_corners=False)}
        if self.training and self.semantic.auxiliary_head is not None:
            result["auxiliary_logits"] = F.interpolate(self.semantic.auxiliary_head(features),
                x.shape[-2:], mode="bilinear", align_corners=False)
            result["auxiliary_weight"] = self.semantic.config.auxiliary_loss_weight
        return result


def build_upernet(cfg):
    source = Path(cfg["pretrained"])
    if cfg.get("random_init", False):
        return UPerNetR50()
    if not source.is_dir():
        raise FileNotFoundError(f"Run python -m cropland.download --repo microsoft/resnet-50 --http first: {source}")
    backbone, info = ResNetBackbone.from_pretrained(source, local_files_only=True,
        out_features=["stage1", "stage2", "stage3", "stage4"], output_loading_info=True)
    if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Incomplete pretrained backbone: {info}")
    if backbone.config.depths != [3, 4, 6, 3] or backbone.config.hidden_sizes != [256, 512, 1024, 2048]:
        raise ValueError("E4 requires a ResNet-50 backbone")
    config = UperNetConfig(backbone_config=backbone.config, num_labels=1, hidden_size=512,
                          auxiliary_in_channels=1024, auxiliary_channels=256,
                          use_auxiliary_head=True, auxiliary_loss_weight=.4)
    model = UPerNetR50(config)
    model.semantic.backbone.load_state_dict(backbone.state_dict(), strict=True)
    return model
