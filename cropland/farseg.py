"""FarSeg (CVPR 2020), adapted to binary cropland segmentation.

Architecture follows Z-Zheng/FarSeg@35ba29fc and SimpleCV@4fa67581.
The native PyTorch ResNet-50 avoids a torchvision/SimpleCV runtime dependency.
See third_party/NOTICE.md for attribution and the retained upstream licenses.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(channels * 4),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        return self.relu(self.bn3(self.conv3(y)) + identity)


class ResNet50Encoder(nn.Module):
    """ImageNet ResNet-50, stride in the 3x3 bottleneck convolution."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        in_channels = 64
        for level, (channels, count) in enumerate(zip((64, 128, 256, 512), (3, 4, 6, 3)), 1):
            stride = 1 if level == 1 else 2
            layers = [Bottleneck(in_channels, channels, stride)]
            layers.extend(Bottleneck(channels * 4, channels) for _ in range(count - 1))
            setattr(self, f"layer{level}", nn.Sequential(*layers))
            in_channels = channels * 4
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        features = []
        for level in range(1, 5):
            x = getattr(self, f"layer{level}")(x)
            features.append(x)
        return features

    def load_pretrained(self, path):
        from .download import RESNET50_SHA256

        with Path(path).open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != RESNET50_SHA256:
            raise ValueError(f"ImageNet ResNet-50 SHA256 mismatch: {checksum}")
        # The original, hash-pinned 2018 PyTorch asset uses the legacy tar format,
        # which PyTorch 2.6's weights-only reader cannot decode. Never use this
        # loader for any bytes other than the exact official artifact above.
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        # The original ImageNet classifier is the only part omitted from the encoder.
        state = {key: value for key, value in state.items() if key not in {"fc.weight", "fc.bias"}}
        self.load_state_dict(state, strict=True)


class FeaturePyramid(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        for level, in_channels in enumerate((256, 512, 1024, 2048), 1):
            for name, source, kernel in ((f"fpn_inner{level}", in_channels, 1),
                                         (f"fpn_layer{level}", channels, 3)):
                conv = nn.Conv2d(source, channels, kernel, padding=kernel // 2)
                nn.init.kaiming_uniform_(conv.weight, a=1)
                nn.init.zeros_(conv.bias)
                self.add_module(name, conv)

    def forward(self, features):
        inner = self.fpn_inner4(features[-1])
        result = [self.fpn_layer4(inner)]
        for level in (3, 2, 1):
            lateral = getattr(self, f"fpn_inner{level}")(features[level - 1])
            inner = lateral + F.interpolate(inner, size=lateral.shape[-2:], mode="nearest")
            result.insert(0, getattr(self, f"fpn_layer{level}")(inner))
        return result


class SceneRelation(nn.Module):
    """Independent scene projections gate the four FPN scales."""

    def __init__(self, channels=256):
        super().__init__()
        self.scene_encoder = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2048, channels, 1), nn.ReLU(inplace=True),
                          nn.Conv2d(channels, channels, 1)) for _ in range(4)
        ])
        self.content_encoders = nn.ModuleList([self.projection(channels) for _ in range(4)])
        self.feature_reencoders = nn.ModuleList([self.projection(channels) for _ in range(4)])

    @staticmethod
    def projection(channels):
        return nn.Sequential(nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels),
                             nn.ReLU(inplace=True))

    def forward(self, scene, features):
        refined = []
        for feature, scene_encoder, content_encoder, reencoder in zip(
                features, self.scene_encoder, self.content_encoders, self.feature_reencoders):
            relation = (scene_encoder(scene) * content_encoder(feature)).sum(1, keepdim=True).sigmoid()
            refined.append(relation * reencoder(feature))
        return refined


class AsymmetricDecoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=128):
        super().__init__()
        self.blocks = nn.ModuleList()
        for upsample_count in (0, 1, 2, 3):
            layers = []
            for index in range(max(1, upsample_count)):
                layers.append(nn.Sequential(
                    nn.Conv2d(in_channels if index == 0 else out_channels, out_channels, 3, padding=1,
                              bias=False),
                    nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2) if upsample_count else nn.Identity(),
                ))
            self.blocks.append(nn.Sequential(*layers))

    def forward(self, features):
        decoded = [block(feature) for block, feature in zip(self.blocks, features)]
        return sum(decoded) / 4.0


class FarSegModel(nn.Module):
    model_type = "farseg"
    has_boundary = False

    def __init__(self, fpn_channels=256, decoder_channels=128):
        super().__init__()
        self.model_config = dict(fpn_channels=fpn_channels, decoder_channels=decoder_channels)
        self.encoder = ResNet50Encoder()
        self.fpn = FeaturePyramid(fpn_channels)
        self.scene_relation = SceneRelation(fpn_channels)
        self.decoder = AsymmetricDecoder(fpn_channels, decoder_channels)
        self.classifier = nn.Conv2d(decoder_channels, 2, 1)

    def forward(self, x):
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError("FarSeg input height and width must be multiples of 32")
        features = self.encoder(x)
        scene = F.adaptive_avg_pool2d(features[-1], 1)
        refined = self.scene_relation(scene, self.fpn(features))
        classes = F.interpolate(self.classifier(self.decoder(refined)), size=x.shape[-2:],
                                mode="bilinear", align_corners=True)
        # sigmoid(foreground - background) is exactly the two-class softmax probability.
        return {"class_logits": classes, "logits": classes[:, 1:2] - classes[:, 0:1]}
