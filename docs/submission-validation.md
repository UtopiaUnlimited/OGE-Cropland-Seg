# Docker 提交验证记录

更新日期：2026-10-27。状态：**本机镜像构建、断网 GPU 推理、镜像导出及重新导入验证均已通过。**

## 可上传文件

- 文件：submission/oge-cropland-seg_b5-20261027.tar
- 大小：3,480,882,176 字节，约 3.481 GB / 3.242 GiB，低于官方20GB限制。
- SHA256：bbf85309e2a8b8c26f4b34f596780e61d82c8a2eb6e372c2fff15ea4a19ba7c8
- 镜像标签：oge-cropland-seg:b5-20261027
- 测试镜像ID：sha256:afb5bee30704d07ae8af9a64dd0753bc5283bdc0bf8f4d0894ffc96f1cb6b85b
- 最佳模型：第31轮，SHA256：dd57a22ed643815ee9b7b74b9a11066570f147ad100af7e79c238507a5833b01

.tar、校验和及完整运行清单均仅保存在本地 submission/，不上传公开GitHub。上传步骤见 [提交指南](submission-guide.md)。

## 验证内容

| 项目 | 结果 |
| --- | --- |
| 平台 | linux/amd64 |
| 环境 | 官方Python 3.11镜像 + 官方PyTorch 2.6.0+cu124 / CUDA 12.4 包 |
| 测试GPU | NVIDIA GeForce RTX 4060 Laptop GPU，本机8GB显存 |
| 网络 | --network none，实际容器检查确认 |
| 入口 | 镜像内固定入口及权重，只追加官方两个路径参数 |
| 资源配置 | 8 CPU，32GB内存上限，16GB共享内存 |
| 根文件系统 | --read-only；仅输出挂载目录及临时目录可写 |
| 数据范围 | 固定内部保留测试集90张，未把标签挂载进容器 |
| 产物 | 90张同名原尺寸TIFF，uint8单波段、0/1类别、空间参考正确 |
| 容器退出码 | 0 |
| 权重一致性 | 容器内SHA256与原best.pt一致 |
| Windows/Linux像素比较 | 差异0像素，一致率100.00000000% |
| 导出验证 | tar中的运行配置和文件系统层与已测试镜像一致 |
| 重新导入 | docker load成功，导入后运行配置与文件系统层一致 |
| 代码测试 | 原有与PNG相关11项通过；新增归档兼容性2项通过 |

本机Docker Desktop的WSL可用物理内存约16GB；32GB是配置的容器上限，并不意味着本机实际提供了32GB物理内存。本次未在A30实机或OGE线上环境运行，因此尚未验证官方A30驱动兼容性和线上评分；实际官方隐藏测试需上传后运行。

只读根文件系统验证了推理不依赖容器可写层。临时栅格文件在输出挂载目录中逐图清理；本次没有模拟官方存储驱动的20G配额，也没有对未知尺寸的官方影像作容量保证。

## 镜像内测试指标

| 指标 | 数值 |
| --- | --- |
| 耕地IoU | 78.16684219% |
| 背景IoU | 89.58754274% |
| mIoU | 83.87719247% |
| F1 | 87.74566719% |
| Precision | 86.56525179% |
| Recall | 88.95872024% |
| Accuracy | 92.41508908% |

这些是本地内部测试成绩，不能代替官方隐藏测试或比赛排名。没有根据测试集调整模型、阈值或后处理。

## 复现与证据

- 一键入口：python scripts/package_submission.py --reference outputs/b5_server_test/masks
- 已有同名tar时，使用新的 --image 标签，避免覆盖原成品。
- 本次构建/推理/导出/导入日志及90张预测在 submission/validation_20261027_135611/。
- 完整环境快照：[Linux运行依赖](../docker/requirements-lock-linux.txt)。这是本次实际安装记录；构建仍以Dockerfile和requirements-inference.txt为入口。
- 完整模型来源、内部指标、像素比较、镜像ID及tar哈希在 .manifest.json 中。

原先尝试下载完整PyTorch基础镜像时网络较慢，最终改用固定摘要的官方轻量Python基础镜像和官方CUDA wheel，核心模型及PyTorch/CUDA版本保持一致。较新的Docker containerd存储使用manifest/index摘要作镜像ID，归档检查已兼容此行为，并通过回归测试及真实docker load验证。

本研究所使用的语义分割样本数据来源于 OGE 平台（http://openge.org.cn/），版权归原作者所有。
