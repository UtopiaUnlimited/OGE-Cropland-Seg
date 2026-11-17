# E4 Lovasz 损失实验

2026-11-17：正式训练已由用户完成，45轮早停，第30轮最佳，验证耕地IoU 79.7081%，原 E4 为79.7554%。全部91张最佳权重概率已独立导出复核。本次未提高验证IoU，继续保留原 E4 交付 ZIP 和网站镜像。详见 [实验结果与对比](result-e4-lovasz-20261117.md)。

## 改动与比较方法

学长提到的“罗瓦兹”对应 Lovász。本项目 E4 是单通道二值分割，因此使用原始 logits 上的 **Lovasz hinge**，作为耕地前景 Jaccard/IoU 的可优化替代目标；不把单通道输出误用成多类 Lovasz-Softmax，也不先二值化概率。

本轮固定采用 `BCEWithLogits + Dice + 1.0 × LovaszHinge`。训练总损失为主头损失加 `0.4 × 辅助头相同损失`；验证损失只用主头。Lovasz 按当前 mini-batch 的全部像素统一排序，计算使用 float32，支持 AMP。背景标签 0 仍是有效像素；无耕地图也参与训练，不作为 ignore 类。

Lovasz 每次作用于实际 batch=2 的像素；累积 4 步会累积梯度，并不等价于一次对 8 张图统一排序，更不等价于直接计算整个数据集的不可微 IoU。

配置为 `configs/e4_upernet_r50_lovasz.yaml`。相对于原 E4，只有新增 `lovasz_weight: 1.0` 和独立 `run_dir` 两项差异。仍是 512×512 RGB、0=背景/1=耕地、固定 train=719/val=91、原预处理、ResNet-50 ImageNet 预训练、batch=2、累积4、AdamW 1e-4、5轮 warmup+cosine、最多100轮、早停15轮、seed=42。重新从同一 ImageNet 骨干初始化，分割头随机初始化，便于与原 E4 对照；不是从旧 best.pt 接着训练。

仍按全部 91 张验证图、阈值 0.5、累计像素 TP/FP/FN 后的耕地 IoU 选 `best.pt`。原 E4 最佳验证 IoU 为 **79.7554%**（第69轮）。新损失数值包含额外一项，不能拿其绝对大小直接和旧 loss 比；主要看 `val/iou`。Lovasz 不能保证官网成绩提高，也不能直接消除本地与官网数据分布差异。训练后先比较同一验证集，再用官网相同样本集/设置比较，验证改善后再决定是否采用。

原 `runs/e4_upernet_r50`、E4_UPerNet.zip、已导出网站 tar 均继续保留；E2/E5 配置未开启此项。旧配置省略 `lovasz_weight` 时等价于 0，继续使用原损失。新训练权重、完整配置、最佳轮次和 IoU 已保存于 `runs/e4_upernet_r50_lovasz/`，概率保存在 `outputs/e4_lovasz_handoff/val_probs/`，对比图在 `outputs/e4_lovasz_review/gallery.html`；本轮不替换正式交付包。

## 手动开始训练（PowerShell）

以下为本轮使用过的命令。已有训练结果，勿再次从头启动或为了本次结果继续训练；完成状态与分数见上方结果报告。

```powershell
conda activate oge-seg
cd F:\Class_7\遥感比赛\OGE-Cropland-Seg
$env:PYTHONNOUSERSITE="1"
python -m cropland.train --config configs/e4_upernet_r50_lovasz.yaml
```

不需要新增 Python 依赖；沿用现有 oge-seg 环境及本地 models/resnet-50。训练终端保持运行。已有 last.pt 时会拒绝从头覆盖；中断后从完整 epoch 末恢复：

```powershell
python -m cropland.train --config configs/e4_upernet_r50_lovasz.yaml --resume runs/e4_upernet_r50_lovasz/last.pt
```

恢复需要相同损失及训练计划。旧损失 last.pt 不能用来恢复新损失实验；best.pt 也不是含优化器状态的恢复点。

## 查看 TensorBoard（另开 PowerShell）

```powershell
conda activate oge-seg
cd F:\Class_7\遥感比赛\OGE-Cropland-Seg
$env:PYTHONNOUSERSITE="1"
python -m tensorboard.main --logdir_spec baseline:runs/e4_upernet_r50/tensorboard,lovasz:runs/e4_upernet_r50_lovasz/tensorboard --host 127.0.0.1 --port 6009
```

打开 http://127.0.0.1:6009 ，同时对比 baseline 与 lovasz 的 `val/iou`，以及 `val/f1`、`loss/train`、`val/loss`、`gpu/peak_memory_mb`。横坐标默认从0开始记录，所以第69轮在 Step=68。新训练第一次完成验证后才有第一个 epoch 的曲线。网页不会启动或维持训练；它与训练分别在两个终端运行。

## 来源与许可

- Berman 等，CVPR 2018：[论文](https://arxiv.org/abs/1705.08790)。其目标是 IoU 的可优化替代损失。
- [作者 PyTorch 实现](https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py)，本项目适配了二值 hinge/Jaccard 梯度计算，MIT 许可保存在 `third_party/lovasz/`。不需要联网下载额外损失库。
