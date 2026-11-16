# SegFormer-B5 E2边界辅助模型结果与提交记录

更新日期：2026-11-16。来源：同学提供的 OGE-Cropland-Seg-E2-changes.zip，已合并四个文件。旧基线模型及其报告保留作历史记录。

## 模型与合并内容

- configs/b5_boundary.yaml：512窗口，batch=2，累积4步，boundary_weight=0.2。
- cropland/metrics.py：保留BinaryMetrics，增加BoundaryMetrics。
- cropland/evaluate.py：逐图及总体增加边界Precision、Recall、F1。
- runs/b5_boundary/best.pt：启用边界分支，第26轮最佳权重，验证耕地IoU=81.87841849%。
- 权重SHA256：ff3dba54725394c1487729ad47523dfd23e0f04d3ce5cd4ec1f4255c9ee5e7b0。

解压时仅规范化文本换行符；权重与ZIP内文件逐字节一致。检查点的数据划分指纹、固定预处理均与本地匹配。边界分支使用浅层特征辅助训练；提交掩膜仍来自语义分支，不将边界结果用于后处理。

## 内部测试对比

统一使用90张固定保留测试图、原尺寸、512窗口、25%重叠、0.5阈值、无TTA与后处理。边界指标由最终分割掩膜提取，与辅助分支直接输出的准确率不同。边界匹配采用2像素方形邻域，全数据集计数汇总。旧基线的边界指标由原预测文件重新计算，未重新训练或调参。

| 指标 | 旧基线 | E2 | 变化（百分点） |
| --- | --- | --- | --- |
| 耕地IoU | 78.1668% | 79.0788% | +0.9120 |
| 背景IoU | 89.5875% | 90.2348% | +0.6473 |
| mIoU | 83.8772% | 84.6568% | +0.7796 |
| F1 | 87.7457% | 88.3173% | +0.5717 |
| Precision | 86.5653% | 88.3170% | +1.7517 |
| Recall | 88.9587% | 88.3177% | -0.6410 |
| Accuracy | 92.4151% | 92.8675% | +0.4524 |
| 边界Precision | 57.8243% | 57.2178% | -0.6065 |
| 边界Recall | 55.9092% | 55.6433% | -0.2660 |
| 边界F1 | 56.8506% | 56.4195% | -0.4311 |

这些是一次固定划分上的内部结果；数据缺少来源区域信息，不能排除空间邻近影响。E2训练次数、搜索范围和早停轮次无法从这个四文件包完整恢复，因此不能将差异完全归因于边界损失，也不能声称训练严格可复现。没有收到history.json、last.pt或官方新模型分数。

## 本地产物与复现

- 90张预测：outputs/b5_boundary_e2_test_20261116/masks/。
- 90张原图/真值/预测/误差对比：outputs/b5_boundary_e2_test_20261116/comparisons/。
- 离线浏览：outputs/b5_boundary_e2_test_20261116/gallery.html。
- 逐图CSV和完整指标：同目录 per_image.csv、metrics.json。
- Git不包含上述图片、数据、权重及Docker归档；分享可视化时请另传对应目录。

    python -m cropland.evaluate --checkpoint runs/b5_boundary/best.pt --split test --output outputs/b5_boundary_e2_test --device cuda --save-png

    python scripts/package_submission.py --checkpoint runs/b5_boundary/best.pt --image oge-cropland-seg:b5-boundary-e2-20261116

训练配置用于新实验；已有best.pt不包含优化器等恢复状态，不能将它当成last.pt恢复完整训练。

## 验证状态

23项代码测试通过，包括边界容差、空掩膜及全数据集计数汇总；本地CUDA评估完成90图。Docker断网GPU验证、90张TIFF输出检查、9张多格式兼容性检查、镜像导出及重新导入均通过；Windows与Linux容器的90张原始测试图预测差异0像素。

最终上传文件：submission/oge-cropland-seg_b5-boundary-e2-20261116.tar。

- 镜像：oge-cropland-seg:b5-boundary-e2-20261116。
- 文件大小：3,481,297,920字节，约3.481GB，小于20GB。
- tar SHA256：24fe2922a0e98a0921f8f0098805c802619fd85867ed362353c7a5874bf1b1dd。
- 镜像ID：sha256:a2acbd2596bdc47b1a9a349cb614656183093791fb10e8a463b0a52767b8a673。
- 运行：linux/amd64，network=none，8CPU，32GB内存配置上限，16GB共享内存，只读根文件系统。
- 本机GPU为RTX4060 Laptop 8GB，不等于已在平台A30上实测。
- 兼容性参考图由同一Linux容器解码，避免Windows/Linux JPEG解码舍入差异影响输入比较；8位样例仅检验格式可运行，不代表平台真实数据或精度。
- 证据目录：submission/validation_20261116_184116/。

在OGE新建模型，名称可填OGE-Cropland-Seg-B5-Boundary-E2-512，上传本次tar后再启动快速推理。不需要重新训练。只上传tar，sha256及manifest.json留作本地核验。

本研究所使用的语义分割样本数据来源于 OGE 平台（http://openge.org.cn/），版权归原作者所有。
