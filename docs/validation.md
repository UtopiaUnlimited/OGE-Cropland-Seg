# 初版验证记录

验证日期：2026-10-19。系统：Windows，RTX 4060 Laptop GPU（8GB）。本记录证明流程能运行，不代表经过正式训练的分割精度。

## 数据

- 原始数据已复制到 data/farm，逐文件 SHA256 校验 1,813 个文件，无差异。
- 900 对影像与掩膜，尺寸 512×512，3 波段 UInt16，实际范围 0–4095；掩膜 UInt8，仅 0/1。
- 未发现完全相同的影像；缺少有效 CRS 和来源区域信息。
- 固定分层划分：训练 719、验证 91、内部测试 90。各集合平均耕地比例分别约 30.75%、31.30%、30.53%。
- 训练集全像素直方图拟合得到全局线性映射范围 [0,1634]，超出范围截断。参数已保存到 splits/preprocessing.json 和模型检查点。

## 环境

- Conda 环境：oge-seg；启用 PYTHONNOUSERSITE=1，隔离用户级包。
- torch 2.6.0+cu124，CUDA 可用；transformers 4.51.3，numpy 2.2.6，rasterio 1.4.3，tensorboard 2.19.0。
- setuptools 固定 80.9.0，以提供 TensorBoard 使用的 pkg_resources。全部已安装包记录在 requirements-lock-windows.txt；pip check 通过。
- 解决了 PostgreSQL/PostGIS 全局 PROJ 路径与 Rasterio wheel 内置数据库的冲突，未修改系统环境变量。
- MiT-B5 权重来自 nvidia/mit-b5，固定 revision 为 40357155205b036cf11b61f132d53d2f8861f170，下载后按 HF LFS SHA256 校验通过。

## 自动测试

最终结果：10 passed。覆盖：

1. 最后不足一批时的梯度累积与整批梯度一致。
2. 所有翻转/旋转组合中图像与掩膜保持对齐。
3. UInt16 数值转换无 8 位截断。
4. 划分覆盖全部样本且重复影像不跨集合。
5. 全局混淆矩阵与空前景指标行为。
6. 全背景/全前景损失和梯度有限，边界监督不引入伪图框。
7. 重叠分块推理覆盖所有像素，保留 CRS/仿射矩阵，输出背景是有效像素。
8. 小于推理窗口的影像正确补齐并裁回原尺寸。
9. 自包含检查点离线加载结果完全一致，包括边界分支。
10. 使用小型测试网络验证：中断后从 epoch 检查点恢复，与连续训练参数逐项一致。

第三方 Rasterio/Affine 产生 PendingDeprecationWarning，不影响测试结果。

## 真实 B5 集成验证

- 模型共 84,594,113 个参数；编码器加载完整的 NVIDIA MiT-B5 预训练权重，未冻结编码器。
- 128×128 快速检查：2 个真实训练批次、2 个验证批次，AMP、1 次优化器更新，检查点保存成功。
- 256×256 本地检查：2 个真实训练批次、2 个验证批次，batch=1、梯度累积=2、Windows num_workers=2，完整训练/验证通过。
- 初始默认 AMP scale=65536 在 256 检查中跳过首步；现已将初始 scale 调为 1024，并记录 optimizer_steps 和 amp_scale，完整运行没有更新时会明确报错。
- 修正后的结果保存在 runs/b5_local_verified：确认 1 次优化器更新，峰值 PyTorch 分配显存 1640.06 MiB（约 1.60 GiB；不等于 nvidia-smi 的整进程显存）。
- 同一检查点对两张验证图执行原尺寸滑窗推理，结果存于 outputs/verified-val；指标明确标记 partial=true。
- 官方命令行入口使用 test/images 目录，阻断 Python socket 连接后成功生成两张 512×512、单波段 uint8、0/1 TIFF。该检查不是 Docker 网络隔离验证，Docker 留到后期。

这些检查只有极少量参数更新，所有 IoU/F1 仅作工程记录，不作为基线精度结论。内部 test 未用于训练或调参。

## TensorBoard

本机网页服务：http://127.0.0.1:6006 。HTTP 200，scalars 接口可读取实际训练损失及验证指标。选择 b5_local_verified/tensorboard（Windows 显示反斜杠）查看已验证的运行。

runs/b5_smoke 是早期 128 检查；runs/b5_local_check 是保留的 AMP 首步溢出诊断，不能作为有效训练结果。后续正式训练使用 b5_local 或 b5_server 的独立目录。

## 尚未执行

- 正式长周期训练、完整验证集精度评估与内部测试集最终评估。
- Linux 服务器实测及 A30 24GB 配置调优。
- 边界辅助分支的真实 B5 效果消融；仅其功能测试通过。
- Docker 打包与官方平台提交。

## 代码发布

源代码发布到 [UtopiaUnlimited/OGE-Cropland-Seg](https://github.com/UtopiaUnlimited/OGE-Cropland-Seg) 公开仓库。训练数据、模型权重、运行日志与预测结果不随仓库发布。
