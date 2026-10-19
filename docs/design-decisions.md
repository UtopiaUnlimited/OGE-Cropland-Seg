# 初版设计决策

本版先建立 SegFormer-B5 基线，覆盖训练、验证、断点恢复、原始尺寸评估与官方接口推理。完整设计中的其他骨干、Distance/Region、困难样本补标与集成留待基线稳定后开展。

## 模型

- Transformers 原生 SegformerForSemanticSegmentation；B5 深度 [3,6,40,3]，通道 [64,128,320,512]，MLP 解码通道 768。
- NVIDIA nvidia/mit-b5 的 ImageNet 预训练编码器，单通道语义头随机初始化；加载时检查编码器参数完整性。
- 单 logit、BCEWithLogits + Dice，sigmoid 后以 0.5 阈值分类。不将背景 0 转为忽略标签。
- F1/F2 浅层边界分支已实现，boundary_weight 默认 0；可设 0.2 做消融。二值标签形态学梯度不能恢复不存在的实例内部边界。
- DECUR 的 B5 RGB 来自 SUN RGB-D 自监督路线，不是标准 SegFormer-B5 分割检查点，本版不采用其自定义代码。
- 提供的方案将公开预训练权重使用列为规则允许时的选项；本地基线先采用公开权重，最终提交前仍需核对正式规则，不能把假设当成官方确认。

## 数据

- 900 对 RGB UInt16 TIFF 与单通道 UInt8 掩膜，512×512，像素 0–4095，类别 0/1。
- 所有影像无 CRS，具有相同默认仿射矩阵，没有明确来源区域信息。暂按耕地占比分层、固定种子 42、约 8:1:1 划分。
- 完全相同影像以内容哈希归组，不能跨集合，冲突标注立即报错。无法排除空间近邻泄漏，获得区域映射后应重建分组划分。
- 只用训练集像素直方图拟合全局 0.5%–99.5% 范围；三个波段共用同一线性变换，随后 ImageNet 标准化。验证、测试和推理使用 checkpoint 中的固定参数，不逐图拉伸。
- 官方掩膜 nodata=0 与背景类别冲突，读取原始值，0 参与损失和指标。输出不设置 nodata，背景仍是有效类别。
- 几何增强图像与标签同步；亮度、对比度、饱和度仅作用于图像。本版未加 hue 扰动。
- 内部 test 用于方案固定后的本地检验，不等于官方隐藏测试集，不用于调参。

## 训练与推理

- AdamW、warmup + cosine、CUDA AMP、梯度累积、梯度裁剪、早停。
- last.pt 保存优化器、调度器、AMP 和随机状态，从 epoch 边界恢复；best.pt 用于独立离线推理。恢复要求关键配置、划分和预处理相同，跨硬件不保证逐位一致。
- TensorBoard 记录损失、分类指标、学习率、显存和验证样例。
- 训练验证在配置分辨率运行；evaluate 在原始分辨率分块评估，与提交推理一致。使用本地低分辨率配置时，两套指标不能混为同一评估协议。
- 重叠窗口概率平均；磁盘映射缓冲控制大图 RAM 占用，临时磁盘约每像素 8 字节。输出同名、原尺寸、单通道 uint8 0/1 TIFF，保留空间参考。
- Windows/Linux 路径与入口已兼容设计；Linux 和目标服务器显存配置仍需实测。
- 本机存在 PostgreSQL/PostGIS 的全局 PROJ 路径。项目启动时优先使用 Rasterio wheel 自带的 PROJ/GDAL 数据，仅调整当前 Python 进程，不修改系统环境变量；Conda 包无内置目录时保持其默认配置。
- Docker 留到提交阶段，只打包代码、推理依赖与 best.pt，之后在断网和官方资源约束下测试。

## 来源

- [论文与原作者实现](https://github.com/NVlabs/SegFormer)
- [Transformers SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer)
- [NVIDIA MiT-B5](https://huggingface.co/nvidia/mit-b5)
- [DECUR 模型说明](https://huggingface.co/BiliSakura/DECUR-transformers)
- 原始研发方案、架构示意图、官方 personal_demo。含团队联系方式的原始文档未复制进代码仓库。
