# FarSeg implementation sources

`cropland/farseg.py` and the foreground-aware loss in `cropland/losses.py`
adapt the following upstream implementations. The adaptation removes the
SimpleCV/Apex dependency, provides a native PyTorch ResNet-50 encoder, changes
the classifier to two classes, exposes binary logits for the existing OGE
interface, and handles degenerate focal-loss normalization. It does not use
FarSeg++ or the iSAID-trained segmentation classifier.

- FarSeg, Zhuo Zheng et al., Apache-2.0:
  https://github.com/Z-Zheng/FarSeg/tree/35ba29fc00fda1ea59b22687fb7c118f676ae32a
  (`module/farseg.py`, `module/loss.py`, `configs/isaid/farseg50.py`).
  License retained as `FarSeg-LICENSE`.
- SimpleCV, copyright (c) 2018 Zhuo Zheng, MIT:
  https://github.com/Z-Zheng/SimpleCV/tree/4fa67581441ad150e82b3aa2c394a921f74e4ecd
  (`simplecv/module/fpn.py`, `simplecv/module/_resnets.py`).
  The FPN source also credits Facebook, Inc. and its affiliates.
  License retained as `SimpleCV-LICENSE`.
- ResNet implementation lineage: PyTorch/torchvision, BSD-3-Clause.
  https://github.com/pytorch/vision/tree/v0.21.0
  License retained as `torchvision-LICENSE`.

Paper: Zheng Z., Zhong Y., Wang J., Ma A. Foreground-Aware Relation Network
for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing
Imagery. CVPR 2020, pp. 4096–4105.

The ResNet-50 initialization comes from the official PyTorch ImageNet-1K
weight file `resnet50-19c8e357.pth`, also used by the original SimpleCV
implementation. Its source URL and SHA256 are recorded locally in
`models/resnet50/source.json`; weights are excluded from Git.
