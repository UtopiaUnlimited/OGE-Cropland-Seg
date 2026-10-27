# OGE 赛道三 Docker 提交指南

本项目按随赛事数据提供的 personal_demo/README.md（2026-09-21版本）、Dockerfile 和 2026-09-29 官方培训材料实现。本文说明的是模型上传和精度评估；赛事如另行要求方案书或其他材料，仍需按报名通知提交。

## 最终要交什么

提交由 docker save 生成的 Linux/amd64 Docker 镜像 .tar 文件。不是源代码 ZIP、不是单独的 best.pt、不是 Dockerfile，也不是预测 PNG/PDF 报告。

镜像包含：

- Python、PyTorch 2.6.0 / CUDA 12.4 运行库和推理依赖；
- SegFormer-B5 推理代码；
- 训练后的 best.pt，以及其中保存的模型结构和固定预处理；
- ENTRYPOINT，平台只需追加 --input_dataset 和 --output_dir 即可启动。

训练图片、验证/测试标签、训练日志、last.pt、Git 历史和凭据不进入镜像。镜像不需要从 GitHub、Hugging Face 或 PyPI 在线下载任何内容。

数据声明：本研究所使用的语义分割样本数据来源于 OGE 平台（http://openge.org.cn/），版权归原作者所有。

## 官方接口与资源约束

| 项目 | 官方要求 / 本项目实现 |
| --- | --- |
| 输入 | 平台传入 DATASET，影像位于 DATASET/test/images/*.tif |
| 输出 | 直接写到 OUTPUT/img1.tif 等，不再嵌套 test/images |
| 类型与类别 | 单波段 uint8；0=背景，1=耕地；原尺寸、原坐标参考 |
| 启动 | 必须有 ENTRYPOINT，权重路径由镜像内部固定 |
| 运行网络 | none，断网推理 |
| GPU | NVIDIA A30 单卡，24GB显存 |
| CPU / RAM | 8核 / 32GB |
| 共享内存 | 16GB |
| 可写层 | 20G |
| 镜像 / tar | 官方限制20GB以内；脚本按严格小于20,000,000,000字节检查 |

项目默认 ENTRYPOINT：

    python -m cropland.inference --pretrained /opt/oge/best.pt --device cuda

平台执行形式：

    docker run [平台的GPU及挂载等参数] your_image:tag --input_dataset /input --output_dir /output

--input_dataset 指的是包含 test/images 的父目录，不是 images 目录本身。固定预测阈值0.5，窗口512，默认重叠0.25，不使用TTA或后处理。

## 一键构建、断网验证与导出

在能够访问互联网、有 NVIDIA GPU 和 Docker GPU 支持的电脑上操作。Windows 使用 Docker Desktop 的 Linux/WSL2 引擎；Linux 需要 Docker Engine 和 NVIDIA Container Toolkit。激活已安装本项目依赖的 oge-seg 环境，在项目根目录运行：

    python scripts/package_submission.py --reference outputs/b5_server_test/masks

若没有之前的 Windows 预测结果，省略 --reference 即可。该参数只用于比较，不影响模型预测或选取。

脚本会：

1. 读取 runs/b5_server/best.pt，核对划分指纹，将权重和来源清单放到 submission/build/。
2. 复制内部测试集图片到本地验证目录；标签不会挂载进容器。
3. 构建 linux/amd64 镜像。首次需要联网下载基础镜像和 Python 包。
4. 验证镜像中的权重SHA256、CUDA实际运算，以及规定的 ENTRYPOINT。
5. 使用 --network none、8 CPU、32GB内存上限、16GB共享内存运行全部90张内部测试图。本地WSL宿主机可用内存可能低于配置上限；这不等于实际提供了32GB物理内存。
6. 额外使用只读容器根文件系统，只将输出目录和临时目录设为可写。该设置比官方普通可写层更严格，但不等于实测了A30或平台20G存储驱动配额。
7. 校验输出文件名、尺寸、空间参考、编码，计算内部测试指标；指定参考目录时记录跨环境像素一致率。
8. docker save 导出 .tar，核对归档的运行配置和文件系统层，再用 docker load 重新导入核验，生成SHA256校验和与说明清单。

默认输出：

    submission/oge-cropland-seg_b5-20261027.tar
    submission/oge-cropland-seg_b5-20261027.sha256
    submission/oge-cropland-seg_b5-20261027.manifest.json

其中只有 .tar 需要上传模型页面；.sha256 和 .manifest.json 留作核验记录。每次验证的构建日志、推理日志、输出TIFF和Linux依赖列表保存在 submission/validation_日期时间/。

脚本拒绝覆盖已有同名tar。更换权重后请使用新镜像标签，例如：

    python scripts/package_submission.py --checkpoint runs/new_run/best.pt --image oge-cropland-seg:b5-v2

默认 .dockerignore 只允许代码、依赖清单和 submission/build 中的权重/清单进入构建上下文。不要改成 COPY . .，否则可能打入训练数据或其他大文件。

## 上传 OGE 平台

以下路径依据提供的官方培训材料，具体名称以当前平台界面为准：

1. 登录开放地球引擎，进入 AI中心 → 快速推理。
2. 点击“新建个人模型”。
3. 模型名称可填写 OGE-Cropland-Seg-B5-512-v1。
4. 描述可填写：“基于SegFormer-B5的耕地二值分割，输入512像素窗口，输出原尺寸0/1 TIFF，离线GPU推理。内部保留测试集成绩不代表官方测试成绩。”
5. 上传已经通过本地验证的 .tar 文件，点击“创建个人模型”，等待显示上传成功。
6. 点击模型卡片右下角“快速推理”，填写任务名称，例如 B5-512-v1-first-test，确认执行。官方测试样本由平台固定提供，不需要上传本地数据集。
7. 在个人中心 → 模型训练 → 快速推理结果管理中查看状态、日志和官方评估结果。

2026-09-29培训提到每天3次推理额度，当前次数以平台提示为准。应先完成本地断网验证，再使用平台额度。

无需把GitHub设为私有或向平台提供GitHub地址；官方培训明确采用tar镜像上传。公开GitHub只用于保存源代码和文档。

## 只克隆 GitHub / ZJUGit 时还缺什么

Git不包含 data/、models/、runs/、outputs/、submission/。要重新构建本次模型，至少还需要真实训练产生的 runs/b5_server/best.pt；若使用一键脚本做全量验证，还需要原始 data/farm 和仓库中的 splits。不要用随机初始化权重替代丢失的 best.pt。

如果服务器只能访问浙大内网：

- 推荐在可联网电脑完成构建及验证，再通过SCP等方式传输完整镜像tar。
- 在服务器执行 docker load -i 镜像.tar，导入的是已经装好依赖和权重的环境，不需要联网安装Python包。
- 服务器需要已有可用Docker和NVIDIA Container Toolkit；这两项属于宿主机配置，tar不能替代显卡驱动。
- 不要尝试将Windows的Conda环境直接拷入Linux镜像。

## 手动检查与故障处理

基础检查：

    docker version
    docker info
    docker image inspect oge-cropland-seg:b5-20261027

快速查看入口帮助（无需GPU）：

    docker run --rm --network none oge-cropland-seg:b5-20261027 --help

若镜像已构建，但自动化脚本因路径或主机工具问题中止，可按以下步骤排查。只有实际通过断网推理后再上传：

    docker run --rm --network none --gpus all --cpus 8 --memory 32g --shm-size 16g --mount type=bind,source=输入目录绝对路径,target=/input,readonly --mount type=bind,source=输出目录绝对路径,target=/output oge-cropland-seg:b5-20261027 --input_dataset /input --output_dir /output

Windows路径包含空格时，将每个完整的 --mount 参数用双引号包住；输入目录应包含 test/images，输出目录需事先创建且为空。

手动导出（不要使用 docker export，它导出容器文件系统，不能替代包含ENTRYPOINT等配置的镜像归档）：

    docker save --output submission/oge-cropland-seg_b5-20261027.tar oge-cropland-seg:b5-20261027

| 问题 | 处理 |
| --- | --- |
| Docker Engine无法连接 | 启动Docker Desktop，确认Linux引擎可用；Linux联系管理员检查Docker服务 |
| 拉取基础镜像超时 | 在能访问官方镜像源的联网电脑构建；不要把失败的下载文件当成提交tar |
| CUDA不可用 / GPU capability错误 | 检查宿主机NVIDIA驱动、WSL2 GPU支持或Linux NVIDIA Container Toolkit |
| CUDA驱动版本不足 | 由管理员升级驱动，或重新构建兼容运行库并完整验证；镜像本身不能升级宿主机驱动 |
| 找不到 best.pt | 从训练电脑复制真实最佳检查点，Git克隆不会包含权重 |
| 输出已存在 | 使用新的空输出目录，推理程序拒绝覆盖已有TIFF |
| 内部图像分数与官方不同 | 数据划分不同；以平台隐藏测试为准，不能用内部指标代替 |
| 上传失败或超过20GB | 检查实际tar大小和上传状态，不要上传代码压缩包或未完成的tar |

相关Docker命令依据官方文档：
- https://docs.docker.com/reference/cli/docker/image/save/
- https://docs.docker.com/engine/containers/resource_constraints/

本地构建/验证的实际状态另见 docs/submission-validation.md。只有该记录明确写明验证通过且对应tar存在，才表示本机已产出可上传文件。
