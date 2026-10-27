# OGE-Cropland-Seg

Transformers SegFormer-B5 耕地二值分割初版，支持 Windows 本地验证与 Linux 服务器训练。Docker 留到最终提交阶段。

本机已通过 10 项测试和完整 B5 的 256×256 GPU 小规模训练/离线推理验证。具体记录见 [验证记录](docs/validation.md)。当前检查点只用于流程检查，尚未正式训练。

本研究所使用的语义分割样本数据来源于 OGE 平台（http://openge.org.cn/），版权归原作者所有。数据及衍生成果的使用遵守官方数据包 readme.md；本仓库不分发数据或权重。

## 环境

开发环境名为 `oge-seg`。在 Anaconda Prompt 或初始化 Conda 的终端中：

```bash
conda activate oge-seg
cd <项目目录>/OGE-Cropland-Seg
```

新 Windows/Linux 机器重建环境，不能直接复制 Windows Conda 目录到 Linux：

```bash
conda env create -f environment.yml
conda activate oge-seg
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[test]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CUDA 轮子自带运行库，需 NVIDIA 驱动支持 CUDA 12.4。CPU 环境将索引换成 https://download.pytorch.org/whl/cpu 。服务器配置明确要求 CUDA，无 GPU 时会报错。

## 文件布局

```text
configs/                 本地、服务器和快速验证配置
cropland/                数据、模型、训练、评估和推理代码
data/farm/train/images/  官方影像副本（Git 忽略）
data/farm/train/masks/   官方掩膜副本（Git 忽略）
splits/                  固定划分清单、哈希和预处理统计
models/mit-b5/           预训练编码器（Git 忽略）
runs/                    检查点、指标和 TensorBoard 日志（Git 忽略）
outputs/                 预测结果（Git 忽略）
tests/                   数据、推理和离线检查点测试
docs/                    设计决策与验证记录
```

配置里的相对路径相对于项目根目录，迁移不用改盘符；命令中 --config 文件路径相对于当前目录。

## 数据和预训练

已有 splits 时直接使用。首次准备新数据：

```bash
python -m cropland.prepare --data-root data/farm --output splits
python -m cropland.download
```

若 Hub 默认传输后端停滞，可用 `python -m cropland.download --http` 切换到官方 HTTPS 流式下载，仍固定 commit 并校验权重 SHA256。

数据准备严格按同名 TIFF 配对，全量检查标签，按耕地比例近似 80/10/10 划分，重复影像不能跨集合。缺少来源区域信息，当前不能保证空间独立，见 docs/design-decisions.md。程序拒绝覆盖既有划分，变更实验请输出到新目录并修改配置。

预处理只在训练集拟合固定范围，保留 UInt16 精度；掩膜 0 是有效背景。splits/audit.json 记录实际数量和范围。下载是单独联网步骤，记录固定 commit 到 models/mit-b5/source.json；训练和推理只读本地。若规则禁止公开预训练，可配置 random_init: true，但会改变训练成本与实验设置。

## 训练

```bash
python -m pytest -q

# 完整 B5 + 真实样本，仅 2 个训练批次、2 个验证批次
python -m cropland.train --config configs/b5_smoke.yaml

# 本地 8GB GPU：256×256、batch=1、梯度累积 4 批
python -m cropland.train --config configs/b5_local.yaml

# 服务器：512×512、batch=2、累积 4 批；需在目标 GPU 实测显存
python -m cropland.train --config configs/b5_server.yaml

# 从完整 epoch 末尾继续，关键配置保持一致
python -m cropland.train --config configs/b5_server.yaml --resume runs/b5_server/last.pt
```

smoke 的指标只验证工程流程，不代表模型精度。输出 best.pt（验证 IoU 最佳）、last.pt（含恢复状态）、history.json、配置和 TensorBoard 日志。已有检查点时拒绝从头覆盖，请修改 run_dir。恢复时不允许改变 epochs，因为会改变 cosine 学习率计划；新的训练计划使用新目录。

默认 BCE + Dice，仅语义分支。边界消融可将 boundary_weight 改为 0.2，并使用新 run_dir。尚未证明边界分支收益。2026-10-27 已在本地 RTX 4060 Laptop 8GB 完成原版 512×512 配置训练与内部测试，详见 [完整结果报告](docs/result-b5-server-20261027.md)。

## 网页训练曲线

```bash
python -m tensorboard.main --logdir runs --host 127.0.0.1 --port 6006
```

浏览器访问 http://127.0.0.1:6006 。也可运行 scripts/tensorboard.ps1 或 bash scripts/tensorboard.sh。

服务器运行同一命令，本地用 SSH 转发后打开同一网址：

```bash
ssh -L 6006:127.0.0.1:6006 <用户名>@<服务器地址>
```

## 原尺寸评估与官方接口

```bash
python -m cropland.evaluate --checkpoint runs/b5_local/best.pt --split val --output outputs/local-val
```

输出全局混淆矩阵指标、逐图 CSV 和预测掩膜。添加 --save-png 可输出原图/真值/预测/误差四栏 PNG 及离线 gallery.html；原图显示采用训练集确定的固定拉伸范围，不更改原始 TIFF。metrics.json 同时记录模型 SHA256、最佳轮次和推理协议。内部 test 仅在方案固定后使用 --split test，不用于调参。--limit 2 是快速检查，结果标记 partial。

官方输入必须是 `<输入目录>/test/images/*.tif`：

```bash
python -m cropland.inference --pretrained runs/b5_local/best.pt --input_dataset <输入目录> --output_dir outputs/predictions --device auto
```

输出同名、原尺寸、单波段 uint8 的 0/1 TIFF，保留空间参考。窗口尺寸默认与训练配置一致，重叠 25%，概率平均再阈值化；拒绝覆盖已有预测。best.pt 内含架构、权重和预处理，推理无需联网或下载 HF 文件。

Docker 提交已提供 Dockerfile、构建白名单和一键验证导出脚本。运行 python scripts/package_submission.py 可打包最佳权重，进行断网 GPU 推理验证，再导出小于20GB的镜像tar。详见 [提交指南](docs/submission-guide.md) 与 [实际验证记录](docs/submission-validation.md)。

## GitHub 与服务器迁移

代码仓库：[UtopiaUnlimited/OGE-Cropland-Seg](https://github.com/UtopiaUnlimited/OGE-Cropland-Seg)。公开仓库只保存源代码、配置、测试、文档及小型划分清单；gitignore 排除数据、权重、日志、预测与模型归档。上传前检查：

```bash
git status --short
git check-ignore data/farm/train/images/img0.tif
```

迁移时复制整个项目，包括 Git 忽略的 data 和 models，再重建 Linux 环境。只克隆仓库时需另行传输数据和权重。splits 是小型清单和统计，随代码保留以复现实验。

### 完整测试与 PNG 对比

```bash
python -m cropland.evaluate --checkpoint runs/b5_server/best.pt --split test --output outputs/b5_server_test --device cuda --save-png
```

已有输出时请指定新目录。2026-10-27 的结果见 [训练测试报告](docs/result-b5-server-20261027.md)；本地可直接打开 outputs/b5_server_test/report.html 或 gallery.html。PNG/预测仍在 outputs 忽略目录中，分享本地报告时需一并复制图片。