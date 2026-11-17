# E4：UPerNet + ResNet-50

2026-11-17 独立 [E4 Lovasz 损失实验](e4-lovasz.md)已由用户完成训练：45轮早停、第30轮最佳，验证耕地IoU 79.7081%，未超过以下原 E4 的79.7554%。[对比报告](result-e4-lovasz-20261117.md)记录独立复核和91张PNG。本次继续保留原 E4 配置、权重、成绩及交付包；TensorBoard 6009可对比新旧曲线。

状态：正式训练已由用户完成，84轮早停，第69轮最佳。已完成E4_UPerNet.zip交付打包及91图Linux断网重现验证；训练及交付验证见 [E4结果报告](result-e4-upernet-20261116.md)。只推进E4，不训练E5/FarSeg、SAM或困难样本集，不提前拟合融合权重。SegFormer-B5/E2原配置和权重保留。

2026-11-16工程验证：25项测试通过；使用两张真实图片完成一次512×512、batch=2的CUDA AMP前向、反向和AdamW更新检查，显存峰值约1886MB，模型参数66,382,146。该工程测试当时没有保存正式权重；随后用户自行完成了正式训练。结果保存在本地outputs/e4_setup/smoke.json，不是训练成绩。

## 模型与训练协议

- Transformers 4.51.3 原生UPerNet PPM+FPN，ResNet-50 v1.5骨干，四阶段深度[3,4,6,3]、通道[256,512,1024,2048]，解码器512通道，PPM尺度[1,2,3,6]。
- ImageNet-1k预训练来源：microsoft/resnet-50，固定已下载版本34c2154c194f829b11125337b98c8f5f9965ff19；分割头随机初始化。训练只加载本地models/resnet-50，不联网。
- 骨干BN保留预训练运行统计，仿射参数可训练；解码器改用GroupNorm以支持小批量及PPM的1×1特征。这是明确记录的小批量训练变体。
- 单通道logits，主头BCEWithLogits+Dice；训练时另加0.4倍辅助语义头BCE+Dice，验证仅用主头。无Boundary辅助损失。推理sigmoid后得到耕地前景概率。
- 仅使用现有splits/train.csv的719张和val.csv的91张，检查划分指纹与交集；不使用内部test选权重或调参。
- RGB为原生512×512。沿用训练集拟合的[0,1634]共享三通道线性裁剪，再做ImageNet mean/std归一化，不使用PIL隐式转8位、中心裁剪或验证随机增强。标签0=背景、1=耕地，均参与损失和指标。
- AdamW，学习率1e-4，weight_decay=0.01，5轮warmup+cosine，batch=2，累积4步，AMP，梯度裁剪1.0，最多100轮，patience=15。配置：configs/e4_upernet_r50.yaml。
- 每轮对91张验证图，以0.5阈值累计全部像素TP/FP/FN/TN，IoU=TP/(TP+FP+FN)，最高者保存best.pt。无超过SegFormer的分数门槛。
- best.pt自带架构、预处理、预训练来源、配置、划分指纹和模型参数，断网推理不需要另发ResNet预训练权重。last.pt只供本地恢复，不放交付ZIP。

## 用户手动启动

在Anaconda Prompt或初始化Conda的PowerShell中：

    conda activate oge-seg
    cd F:\Class_7\遥感比赛\OGE-Cropland-Seg
    $env:PYTHONNOUSERSITE='1'
    python -m cropland.train --config configs/e4_upernet_r50.yaml

另开终端查看曲线：

    conda activate oge-seg
    cd F:\Class_7\遥感比赛\OGE-Cropland-Seg
    python -m tensorboard.main --logdir runs/e4_upernet_r50/tensorboard --host 127.0.0.1 --port 6008

浏览器访问 http://127.0.0.1:6008 。主要看val/iou、loss/train、val/loss、gpu/peak_memory_mb。val/loss只含主头，训练loss还含辅助头，二者不宜直接比较高低。TensorBoard端口与旧项目分开，需保持终端运行。

从完整epoch末尾恢复：

    python -m cropland.train --config configs/e4_upernet_r50.yaml --resume runs/e4_upernet_r50/last.pt

请勿用best.pt恢复训练或在已有检查点的目录中重新从头启动。若新机器缺少预训练文件，先联网运行：

    python -m cropland.download --repo microsoft/resnet-50 --revision 34c2154c194f829b11125337b98c8f5f9965ff19

## 训练完成后的交付流程

已完成训练后执行的交付命令如下；现有输出会拒绝覆盖，重新验证时请使用新的输出目录：

    python -m cropland.e4_export --checkpoint runs/e4_upernet_r50/best.pt --output outputs/e4_handoff --device cuda
    python scripts/package_e4.py

第一条产生全部91张outputs/e4_handoff/val_probs/*.npy及validation.json。每张数组为float32、512×512、范围[0,1]、与原图同一行列坐标，无二值化、TTA、裁剪或后处理。img123.tif对应img123.npy。按0.5阈值重算IoU核验，与训练批量大小造成的微小AMP数值差异记录在结果中；差异超过1e-4则拒绝交付。

第二条生成E4_UPerNet.zip和校验和，含best.pt、91张val_probs、模型代码、config.yaml、resolved_config.json、requirements.txt、划分清单、预处理说明与成绩记录；排除原始数据、last.pt、虚拟环境和独立预训练权重。验证概率文件的形状、dtype、数值范围及SHA256；已有输出拒绝覆盖。ZIP根目录附用best.pt重新生成概率及OGE批量离线推理的完整命令。只提前分享概率时可先发送val_probs与validation.json。

训练结束后同步代码和文档到GitHub；权重、npy和ZIP不上传公开仓库。后续集成必须让各模型用自身预处理读取同一原图，并在同一像素坐标上融合前景概率。

## 参考

- UPerNet官方实现：https://github.com/CSAILVision/unifiedparsing
- Transformers UPerNet：https://huggingface.co/docs/transformers/model_doc/upernet
- ResNet-50预训练说明：https://huggingface.co/microsoft/resnet-50

本研究所使用的语义分割样本数据来源于OGE平台（http://openge.org.cn/），版权归原作者所有。

E4单独网站测试镜像已打包并通过本地断网验证，上传文件与操作见 [E4网站提交说明](e4-website-submission.md)。组员融合交付ZIP继续保留。
