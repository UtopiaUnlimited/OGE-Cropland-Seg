# E4 UPerNet + ResNet-50 训练及交付记录

更新日期：2026-11-16。训练由用户手动完成。只交付E4，不训练E5/FarSeg、SAM，不执行困难样本流程或融合权重搜索。

## 训练结果

- 完成轮次：84，最多100轮，连续15轮验证IoU未提升后早停。
- 最佳轮次：69；权重：runs/e4_upernet_r50/best.pt。
- 最佳验证耕地IoU：79.75541328%。
- 验证划分：原有91张；训练719张；阈值0.5，全数据集累计像素混淆计数后计算IoU。
- 没有使用内部test选择模型或调参，本次未评估test，也没有将旧模型的test成绩与E4验证成绩混比。
- 权重SHA256：e39a2b459da13c3d45ae4f99b9300a5d37f19287144ff6b75239db4c6aba4369。
- 预训练：microsoft/resnet-50，ImageNet-1k，revision 34c2154c194f829b11125337b98c8f5f9965ff19。
- ResNet-50骨干、UPerNet PPM+FPN，主解码512通道，辅助头loss权重0.4；骨干BN统计固定，解码GN。完整配置见configs/e4_upernet_r50.yaml。
- AdamW，lr=1e-4，weight_decay=0.01；5轮warmup+cosine，512输入，batch2、累积4、AMP；标签0背景/1耕地。
- 训练实际峰值显存约2672.7MB，本机RTX4060 Laptop 8GB。

## 91张验证概率

- 位置：outputs/e4_handoff/val_probs/，每张img123.tif对应img123.npy。
- float32、512×512、[0,1]、前景概率；不二值化、不重采样、不做TTA或后处理，保留原图像素坐标。
- 文件名、源影像及标签像素哈希与split清单核对，输出逐文件SHA256写入validation.json。

| 由交付概率以0.5阈值重算的指标 | 数值 |
| --- | --- |
| 耕地IoU | 79.75516635% |
| 背景IoU | 90.34678522% |
| mIoU | 85.05097578% |
| F1 | 88.73755116% |
| Precision | 89.46886173% |
| Recall | 88.01809900% |
| Accuracy | 93.00639394% |

训练批量2、导出批量1的CUDA AMP数值差异导致IoU差值-0.0000024693（约0.000247个百分点），在1e-4核验限值内。交付保留实际原始概率，没有为了凑训练分数调整任何概率或阈值。

这些是固定划分上的内部验证成绩，不代表官方分数。没有必须超过SegFormer的门槛；是否与其他模型互补需待组员提供同一91图的概率后再比较。

## 组员交付与复现

- 最终ZIP：项目根目录E4_UPerNet.zip。
- 包含best.pt、91张val_probs、模型及推理代码、完整配置、依赖版本、SCORES.md、来源和校验清单、预处理说明。
- 排除原始数据、last.pt、虚拟环境、独立ResNet预训练权重。GitHub只同步代码、配置、测试及说明。
- 解压后从ZIP根目录重新生成验证概率（数据单独提供）：

    python -m cropland.e4_export --checkpoint best.pt --data-root /path/to/farm --split-dir splits --output regenerated --device cuda

- OGE批量断网掩膜推理（不需要源数据标签）：

    python -m cropland.inference --pretrained best.pt --input_dataset /input --output_dir /output --device cuda

best.pt自包含模型结构、权重、预处理和来源，加载时不下载HF文件。概率再生成使用ZIP内splits及单独提供的原始farm数据。正式融合时，每个模型按自身预处理读取同一影像，概率必须保持相同像素坐标。

## 交付检查

26项代码回归测试通过。实际导出91张概率，所有shape/dtype/范围与文件哈希均校验；ZIP经CRC与源码哈希检查，排除了数据、last.pt和环境。

从交付ZIP解压的代码和best.pt，在Linux GPU容器（--network none、只读根文件系统）中重新生成了全部91张概率；与Windows输出的最大/平均绝对差均为0，0.5阈值下差异0像素。纯推理无需TensorBoard或另行下载预训练权重。

校验记录与日志：本地outputs/e4_delivery_check_v2/verification.json、offline.log；最终归档的记录见同目录final_verification.json。Windows交付ZIP约325MB；具体字节和SHA256见旁边的E4_UPerNet.zip.sha256。GitHub不分发ZIP、权重或概率。

本研究所使用的语义分割样本数据来源于OGE平台（http://openge.org.cn/），版权归原作者所有。
