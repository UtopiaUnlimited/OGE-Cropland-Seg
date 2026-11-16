# E5 FarSeg交付检查及独立网站镜像

日期：2026-11-16。状态：原交付包校验、91图断网GPU推理、多格式兼容性、镜像导出及重新导入验证均通过；尚未在OGE线上运行本次E5镜像。

## 模型与收到的ZIP

- 原始文件：项目根目录E5_FarSeg.zip，保留未修改。
- ZIP SHA256：e170894552723372355d28014437d833c18f065b8991b60c759cd5e856e679d5。
- 按包内manifest核对133个声明文件哈希；附加README和manifest本身没有自哈希。未整包覆盖本项目E4文件。
- E5训练记录：75轮早停，第60轮最佳；验证耕地IoU=78.64589280%，F1=88.04668450%。
- 训练/验证CSV指纹与本项目一致（719/91），91张概率为float32、512×512、[0,1]；按0.5阈值累计混淆计数重算，结果与包内记录一致。
- best.pt SHA256：9bf973c4d1819c2c501ff69da00f22744679e9c1590f5fea2f12fcf4166e70cd；本地副本：runs/farseg_e5/best.pt。
- 原生PyTorch ResNet-50 + FPN + SceneRelation + AsymmetricDecoder，输出背景/耕地两个logit。保留同学的架构、align_corners设置和固定预处理，不转换为其他FarSeg实现。
- 预训练：ImageNet-1K，resnet50-19c8e357.pth；最终best.pt自包含所需权重，推理不再下载预训练文件。

只导入FarSeg推理架构和检查点读取，并保留原包配置作为参考；SGD/poly/focal等E5训练改动未并入本项目共享训练器。要复现E5训练请使用原ZIP中的代码，不能把其配置交给本项目E4/SegFormer训练入口。

## 网站上传文件

- 文件：submission/oge-cropland-seg_e5-farseg-20261116.tar。
- 大小：3,283,290,624字节，约3.283GB，低于20GB。
- tar SHA256：8360aae634a7e8f7a0dc96a0c9b4938b54fae491eebc93c90f9a52eeb3606f3d。
- 镜像标签：oge-cropland-seg:e5-farseg-20261116。
- 镜像ID：sha256:db15ce20bf37af3f580360db324fa9710fddbc13761efab04200087da52bef48。

在OGE平台进入AI中心→快速推理→新建个人模型，建议名称OGE-E5-FarSeg-512-v1。选择本页tar上传，等待成功后启动这个新E5模型的快速推理。不解压tar，不重新训练；E5_FarSeg.zip仍用于组内融合开发。

## 推理和验证

- 固定512窗口、25%重叠、阈值0.5；RGB数值按检查点[0,1634]裁剪后做ImageNet标准化。
- 耕地概率直接取softmax(class_logits.float(), dim=1)的通道1，避免半精度下先减logit引入额外舍入。
- 输出同名原尺寸单波段uint8 TIFF，0背景、1耕地，保留输入空间参考。
- 网络none、8CPU、32GB容器内存上限、16GB共享内存、只读根文件系统。本机GPU为RTX4060 Laptop 8GB，未实测平台A30及平台存储驱动20G配额。
- 91张val图容器IoU=78.64666960%，F1=88.04717130%；与原包概率以0.5阈值生成的参考掩膜差异1625像素，一致率99.99318804%。本次不使用内部test或改变阈值。
- 9张格式样例覆盖TIFF、uint16 PNG、uint8 PNG/JPG/JPEG及子目录；该兼容性测试不代表人工8位样例与真实平台的精度一致。
- 原包概率采用batch=8，网站入口逐图推理；少量CUDA AMP舍入差异在记录中保留，没有为匹配原包结果修改阈值或概率。
- 28项代码回归测试通过，包括E2/E4、FarSeg离线检查点重载及softmax路径。
- 归档完成docker save、tar层/入口核对和docker load重新导入验证。
- 验证日志及预测在submission/validation_20261116_235822/；ZIP原始解压内容在outputs/e5_received/，不上传Git。

## 复现

    python scripts/package_submission.py --checkpoint runs/farseg_e5/best.pt --image oge-cropland-seg:e5-farseg-20261116 --split val --reference outputs/e5_website_reference/masks

同名tar存在时请使用新镜像标签。平台只需追加--input_dataset和--output_dir，权重路径和CUDA设备已固定在镜像入口中。

## 来源

FarSeg架构按原交付包导入。FarSeg Apache-2.0、SimpleCV MIT及torchvision BSD许可证和原NOTICE保存在third_party/e5/，镜像内保存在/opt/oge/licenses/e5/。原NOTICE中的训练损失说明描述原E5包，不表示已合并到本项目训练器。

本研究所使用的语义分割样本数据来源于OGE平台（http://openge.org.cn/），版权归原作者所有。内部验证成绩不代表网站隐藏测试分数。
