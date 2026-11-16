# E4 UPerNet + ResNet-50 独立网站测试镜像

日期：2026-11-16。状态：镜像构建、断网GPU推理、多格式检查、归档导出及重新导入均通过。尚未在OGE线上运行本次E4镜像。

## 上传文件

- 文件：submission/oge-cropland-seg_e4-upernet-r50-20261116.tar。
- 大小：3,413,360,640字节，约3.413GB，小于20GB。
- tar SHA256：ecfdfbbf414561e80af7bc766ed9e539f6fef4a3b4744c2c90492219f7bded34。
- 镜像标签：oge-cropland-seg:e4-upernet-r50-20261116。
- 镜像ID：sha256:075f6203448d5b285079424ea1f599175351ed87a16f33256d8391d50bea8a2f。
- 权重：E4第69轮best.pt，SHA256为e39a2b459da13c3d45ae4f99b9300a5d37f19287144ff6b75239db4c6aba4369。

本次仅使用E4 UPerNet-R50，没有混入SegFormer、FarSeg或SAM。给组员的E4_UPerNet.zip保留，用于融合开发；网站上传本页的tar。

## 验证与输出协议

- linux/amd64；PyTorch2.6.0+cu124、CUDA12.4；本机RTX4060 Laptop 8GB，尚未在平台A30实机测试。
- --network none、8CPU、32GB配置内存上限、16GB共享内存，只读容器根文件系统。宿主机实际可用内存不等于所配上限，未模拟平台存储驱动20G配额。
- 本机使用全部91张val图作接口验证，不动内部test，不训练或调整权重/阈值。
- 原尺寸、512窗口、25%重叠、阈值0.5；输出同名、单波段uint8、0=背景/1=耕地的TIFF，保留原空间参考。
- 输出与已交付的val_probs按0.5阈值生成的掩膜差异0像素；验证耕地IoU为79.75516635%，F1为88.73755116%。内部成绩不代表官方成绩。
- 9张兼容性样例通过，覆盖TIFF、16位PNG、8位PNG/JPG/JPEG及嵌套目录。8位人工样例只验证格式与解码一致性，不代表官方实际输入或精度。
- 权重和完整运行依赖在镜像内，运行不从GitHub/HuggingFace/PyPI下载文件。
- 导出tar后已使用docker load重新导入并检查镜像层及ENTRYPOINT配置。
- 运行日志和预测：submission/validation_20261116_232604/；完整清单为与tar同名的.manifest.json。

## 上传步骤

1. OGE平台进入AI中心→快速推理→新建个人模型。
2. 建议模型名：OGE-E4-UPerNet-R50-512-v1。
3. 选择本页tar文件，等待模型上传成功。
4. 点击这个新E4模型的快速推理，查看网站成绩。

不用重新训练，不解压tar。每次模型包独立上传，点击旧E2模型不会自动切换为E4。

## 复现打包

    python scripts/package_submission.py --checkpoint runs/e4_upernet_r50/best.pt --image oge-cropland-seg:e4-upernet-r50-20261116 --split val --reference outputs/e4_website_reference/masks

脚本拒绝覆盖同名tar，重跑需改用新镜像标签。参考掩膜由已校验的91张E4 val_probs以固定0.5阈值生成，仅用于跨环境一致性检查。

平台只追加两个参数，镜像入口固定加载内部E4权重：

    docker run [平台GPU及挂载参数] oge-cropland-seg:e4-upernet-r50-20261116 --input_dataset /input --output_dir /output

本研究所使用的语义分割样本数据来源于OGE平台（http://openge.org.cn/），版权归原作者所有。
