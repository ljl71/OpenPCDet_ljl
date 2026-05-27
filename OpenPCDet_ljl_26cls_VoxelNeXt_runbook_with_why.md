# OpenPCDet_ljl 使用公司正式数据训练 26 类 VoxelNeXt 跑通教程

> 适用时间：2026-05-26  
> 适用工程：`/home/ubuntu/WXY/OpenPCDet_ljl`  
> 容器内路径：`/workspace/OpenPCDet`  
> 目标：使用公司正式 `nuScenes v1.0-trainval` 格式数据训练 26 类 `VoxelNeXt`  
> 推荐模型配置：`tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml`  
> 本教程根据实际服务器跑通过程整理，命令尽量写成单行，避免 ToDesk 粘贴时只粘第一行。

---

## 0. 本次最终跑通状态

本次已经完成并验证：

- 代码版本对齐到 `2d144e7 feat: train formal company data with 26-class VoxelNeXt`
- 正式数据目录 `data/nuscenes` 可读
- `sample_data.json` 中 24142 条 `LIDAR_TOP` 记录全部能找到对应 `.bin`
- 成功生成训练/验证 info
- info 严格检查通过
- dataloader smoke test 通过
- VoxelNeXt GPU smoke test 通过
- 正式训练已经成功启动到第 1 个 iteration，出现 `loss=303`，说明数据、模型、CUDA、spconv 主链路已通
- 训练过程中最后遇到的是 `ckpt_save_time_interval=None` 的保存逻辑 bug，需要按本文第 12 节修复后继续训练

---


---

## 0.1 整体逻辑：为什么要按这个顺序跑

这套流程不是简单地“直接运行 `train.py`”，而是按风险从低到高逐层验证：

```text
代码版本对
→ 容器挂载对
→ GPU 权限对
→ 数据目录对
→ JSON 元数据完整
→ 点云路径完整
→ train/val 划分合理
→ info 能生成
→ info 内容健康
→ dataloader 能出 batch
→ GPU 上能 forward/loss/backward
→ 正式训练能启动
→ 修复训练保存逻辑
```

这样做的原因是：3D 检测训练链路很长，任何一个环节出错都会导致后面训练失败。如果不分层验证，直接开训，一旦报错就很难判断到底是代码版本错、容器挂载错、数据路径错、类别映射错、CUDA 没挂上，还是模型本身有问题。

所以本教程采用“先验证轻量步骤，再进入重计算步骤”的方式。前面的检查大多不占 GPU，能提前排除路径、元数据、类别、info 文件等问题；只有这些都通过后，才进入 GPU smoke 和正式训练。

---

## 0.2 为什么要先确认代码版本

本次不是跑原版 OpenPCDet，而是跑针对公司正式 26 类 nuScenes 数据适配过的分支。这个分支里包含：

- 公司正式数据配置
- 26 类 VoxelNeXt 配置
- `CompanyNuScenesDataset`
- 公司类别映射逻辑
- `create_company_infos.py`
- `check_company_infos.py`
- `preview_formal_split.py`
- dataloader smoke test
- VoxelNeXt GPU smoke test
- `voxelnext_head.py` 的多 head 标签修复

如果代码不是 `2d144e7 feat: train formal company data with 26-class VoxelNeXt`，后续流程可能根本没有正式配置文件或 smoke test 脚本。

本次实际就遇到过一次：宿主机项目是新版本，但容器里还显示旧版本 `21b7dfc main`。如果不先检查版本，后面很容易把“代码旧”误判成“数据错”或“配置错”。

因此必须先检查：

```bash
git log -1 --oneline
git branch
find . -name "company_voxelnext_26cls_trainval.yaml"
find . -name "smoke_test_formal_voxelnext.py"
```

这一步的目的：确认当前运行环境确实是正式 26 类训练版本。

---

## 0.3 为什么要确认容器挂载

训练是在 Docker 容器里跑的，但代码和数据实际存放在宿主机上。容器通过 bind mount 看到它们：

```text
/home/ubuntu/WXY/OpenPCDet_ljl -> /workspace/OpenPCDet
/home/ubuntu/WXY/data          -> /workspace/OpenPCDet/data
```

如果挂载错了，容器里看到的可能是旧项目；如果数据挂载错了，代码再正确也读不到正式数据。

本次就遇到过：宿主机 `/home/ubuntu/WXY/OpenPCDet_ljl` 已经是正确版本，但容器里一开始看不到新增的正式训练脚本。重启容器后，容器才重新看到正确目录。

所以要检查：

```bash
sudo docker inspect detection3d_v5 --format='{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

这一步的目的：确认容器里跑的 `/workspace/OpenPCDet` 就是宿主机上的正确工程目录，`/workspace/OpenPCDet/data` 就是正确数据目录。

---

## 0.4 为什么要创建带 GPU 的容器

Docker 容器是否能访问 GPU，是创建容器时决定的。如果创建时没有加：

```bash
--gpus all
```

那么容器内部即使安装了 PyTorch，也看不到 NVIDIA 驱动。此时运行 GPU 相关代码会报：

```text
RuntimeError: Found no NVIDIA driver on your system.
```

这不是模型问题，也不是数据问题，而是容器没有 GPU 权限。

所以需要删除旧容器并重新创建：

```bash
sudo docker run -it -d --gpus all --name detection3d_v5 ...
```

删除容器不会删除代码和数据，因为代码、数据都在宿主机挂载目录中：

```text
/home/ubuntu/WXY/OpenPCDet_ljl
/home/ubuntu/WXY/data
```

这一步的目的：确保 VoxelNeXt 能在 CUDA 上完成前向、loss 和反向传播。

---

## 0.5 为什么要检查正式数据目录

公司数据是 nuScenes 风格，但服务器上可能同时存在多个相似目录，例如：

```text
data/nuscenes
data/v1.0-trainval
```

这两个不能混用。本次正式训练要求使用：

```text
/workspace/OpenPCDet/data/nuscenes
```

其结构应该是：

```text
data/nuscenes/
├── samples/
│   └── LIDAR_TOP/
│       └── *.bin
└── v1.0-trainval/
    ├── category.json
    ├── instance.json
    ├── sample.json
    ├── sample_data.json
    ├── sample_annotation.json
    ├── scene.json
    ├── calibrated_sensor.json
    ├── ego_pose.json
    └── ...
```

不要使用：

```text
/workspace/OpenPCDet/data/v1.0-trainval
```

因为该目录中的元数据曾经引用 `.pcd`，不一定和正式 `.bin` 点云目录组成可训练闭环。

这一步的目的：确认配置文件中的 `DATA_PATH: ../data/nuscenes` 能真正对应到正确数据根目录。

---

## 0.6 为什么要检查 JSON 数量

nuScenes 数据不是只有点云文件，还依赖一组 JSON 表共同描述数据关系：

- `scene.json`：场景列表
- `sample.json`：每一帧样本
- `sample_data.json`：每一帧对应的传感器文件路径
- `sample_annotation.json`：3D 标注框
- `instance.json`：实例与类别关系
- `category.json`：类别定义

这些表之间互相关联。如果某个 JSON 缺失或数量异常，后面生成 info 时就会出错，或者训练时找不到样本、类别和点云路径。

本次检查到：

```text
category            : 26
scene               : 412
sample              : 24142
sample_data         : 241420
sample_annotation   : 1222229
instance            : 53246
```

这一步的目的：确认正式数据的元信息完整，能够支持后续样本索引、类别映射和 3D box 解析。

---

## 0.7 为什么不能只用 `find *.bin | wc -l` 判断点云完整

一开始用普通目录计数：

```bash
find data/nuscenes/samples/LIDAR_TOP -maxdepth 1 -type f -name "*.bin" | wc -l
```

曾经得到 `24083`，看起来比 `sample.json` 的 `24142` 少。但后来按 `sample_data.json` 真实引用路径检查，得到：

```text
LIDAR_TOP records: 24142
Missing LIDAR_TOP files: 0
suffix: Counter({'.bin': 24142})
parents:
samples/LIDAR_TOP 24142
```

训练代码不是根据“目录里有多少个 bin”来决定样本，而是根据 `sample_data.json` 中每条 `LIDAR_TOP` 记录的 `filename` 去读取点云。因此更准确的检查方式是：

```text
遍历 sample_data.json 中的 LIDAR_TOP 记录
→ 拼接 data/nuscenes + filename
→ 检查每个文件是否存在
```

这一步的目的：确认“标注引用到的点云文件”全部存在，而不是只粗略统计目录文件数量。

---

## 0.8 为什么要先预览 train/val split

正式数据没有现成的 `train.txt` 和 `val.txt`，需要代码自己划分。划分不能随便随机按帧切，因为同一个 scene 内的连续帧高度相似。如果同一 scene 的帧同时出现在训练集和验证集，会造成场景泄漏，使验证结果不可靠。

所以划分要按 scene 进行：

```text
329 个 scene 用于训练
83 个 scene 用于验证
```

同时还要保证所有可学习类别至少进入训练集。如果某个类别只在验证集出现，训练时模型完全没见过这个类别，就不可能学会检测它。

预览结果中最关键的是：

```text
annotated_but_absent_from_train: []
PASS: all learnable annotated classes are present in training.
```

`vehicle.emergency.other` 没有进入验证集是合理的，因为它有效样本太少，必须优先保证训练集有它。

这一步的目的：在正式生成 info 前，确认 train/val 划分既没有场景泄漏，又保证训练集覆盖所有可学习类别。

---

## 0.9 为什么要生成 info

OpenPCDet 训练时通常不会在每个 iteration 动态解析原始 nuScenes JSON。原始 JSON 表之间关系复杂，例如：

```text
sample
→ sample_data
→ lidar 文件路径

sample
→ sample_annotation
→ instance
→ category
```

如果每次训练都实时查这些表，会很慢，也容易在训练过程中才暴露路径或类别错误。

`create_company_infos.py` 的作用是提前把这些信息整理成 OpenPCDet 可直接读取的缓存文件：

```text
company_nuscenes_infos_train.pkl
company_nuscenes_infos_val.pkl
```

info 文件里通常包含：

- 样本 id
- 点云文件路径
- 当前样本属于 train 还是 val
- 3D 标注框
- 类别名称与类别 id
- `num_lidar_pts` 过滤结果
- 训练时 dataloader 需要快速读取的元信息

所以 info 不是额外文件，而是 OpenPCDet 训练数据入口的一部分。没有 info，训练脚本不知道有哪些样本、每个样本点云在哪里、有哪些框、每个框属于哪个类别。

这一步的目的：把公司原始 nuScenes 格式数据转换成 OpenPCDet 训练可直接消费的索引文件。

---

## 0.10 为什么生成 info 后还要检查 info

info 生成成功不代表内容一定健康。可能出现：

- 点云路径缺失
- 有空 GT 样本
- 类别名超出配置
- 某些类别没有进入训练集
- 某些类别经过 `min_lidar_points` 过滤后没有可学习框
- `num_lidar_pts` 字段非法
- `null` 点数处理不符合预期

所以要运行：

```bash
python tools/company_nuscenes/check_company_infos.py --root data/nuscenes/v1.0-trainval --data_root data/nuscenes --strict --min_lidar_points 1
```

本次关键结果：

```text
missing_lidar_paths: 0
invalid_num_lidar_pts_samples: 0
outside_config: []
annotated_but_absent_from_train: []
annotated_but_unlearnable_after_train_filter: []
```

这一步的目的：确保 info 文件不仅存在，而且可以安全用于训练。

---

## 0.11 为什么要修复 `av2` 缺包

本次 dataloader smoke 一开始报：

```text
ModuleNotFoundError: No module named 'av2'
```

这是因为 `pcdet/datasets/__init__.py` 强制导入了 Argoverse2 数据集：

```python
from .argo2.argo2_dataset import Argo2Dataset
```

虽然本次不用 Argo2，但 Python 导入 `pcdet.datasets` 时会加载这个文件，导致缺少 `av2` 时直接失败。

解决方式是改成可选导入：

```python
try:
    from .argo2.argo2_dataset import Argo2Dataset
except ModuleNotFoundError:
    Argo2Dataset = None
```

这一步的目的：避免无关数据集依赖阻断公司 nuScenes 训练链路。

---

## 0.12 为什么要跑 dataloader smoke test

info 检查通过只能说明索引文件健康，但还不能说明训练 batch 能构造出来。dataloader smoke test 会验证：

- 配置文件能读取
- `CompanyNuScenesDataset` 能加载 train info
- 数据增强能执行
- voxelization 能执行
- batch 能 collate
- 点云维度符合模型输入
- GT box 维度符合 loss 计算

本次输出：

```text
dataset_len: 19268
points_shape: (201843, 5)
gt_boxes_shape: (1, 78, 8)
```

其中：

- `points_shape` 第二维为 5，表示 `batch_index + x + y + z + intensity占位`
- `gt_boxes_shape` 最后一维为 8，表示 `x,y,z,dx,dy,dz,yaw,class_id`

这一步的目的：确认数据能以模型需要的格式进入训练 pipeline。

---

## 0.13 为什么要跑 VoxelNeXt GPU smoke test

dataloader smoke test 不会完整证明模型训练能跑。GPU smoke test 会真正执行：

```text
构建 VoxelNeXt
→ 放到 CUDA
→ 读取一个 batch
→ forward
→ loss
→ backward
```

本次结果：

```text
formal_voxelnext_smoke: PASS
classes: 26
dataset_len: 19268
points_shape: (194422, 5)
gt_boxes_shape: (1, 48, 8)
loss: 466.34649658203125
```

这说明：

- 26 类 VoxelNeXt 构建成功
- GPU 可用
- spconv 可用
- forward 正常
- loss 正常
- backward 正常

这一步的目的：在正式长时间训练前，用一个 batch 低成本验证模型训练主链路。

---

## 0.14 为什么正式训练要从 `tools` 目录启动

OpenPCDet 的很多配置路径是按 `tools` 目录下运行设计的，例如：

```bash
--cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
```

如果在 `/workspace/OpenPCDet` 根目录运行，某些相对路径可能找不到。之前 dataloader smoke 就出现过相对配置路径找不到的问题，所以正式训练建议：

```bash
cd /workspace/OpenPCDet/tools
```

再运行：

```bash
python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml ...
```

这一步的目的：让配置路径、数据路径和 OpenPCDet 原始运行方式保持一致。

---

## 0.15 为什么训练参数这样设置

本次训练命令：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --batch_size 1 --epochs 20 --workers 4 --extra_tag formal_company_26cls --ckpt_save_interval 1 --max_ckpt_save_num 20
```

参数含义：

| 参数 | 值 | 为什么这样设置 |
|---|---:|---|
| `CUDA_VISIBLE_DEVICES` | `1` | GPU 0 被占用较多，GPU 1 基本空闲 |
| `cfg_file` | `company_voxelnext_26cls_trainval.yaml` | 正式 26 类 VoxelNeXt 配置 |
| `batch_size` | `1` | 先保证跑通，降低显存风险 |
| `epochs` | `20` | 先进行一轮完整训练验证 |
| `workers` | `4` | 数据加载速度和稳定性折中 |
| `extra_tag` | `formal_company_26cls` | 让输出目录清晰可追踪 |
| `ckpt_save_interval` | `1` | 每个 epoch 保存一次 |
| `max_ckpt_save_num` | `20` | 最多保留 20 个 checkpoint |

这一步的目的：在显存和稳定性可控的前提下，启动正式 26 类训练，并将输出清晰归档到独立实验目录。

---

## 0.16 为什么会有 `ckpt_save_time_interval=None` 问题

训练已经成功启动到第 1 个 iteration：

```text
epochs: 0%| ... loss=303, lr=0.0003 ...
```

这说明数据、模型、CUDA、loss 主链路都已经通了。之后报错：

```text
TypeError: unsupported operand type(s) for //: 'float' and 'NoneType'
```

原因是代码中执行：

```python
time_past_this_epoch // ckpt_save_time_interval
```

但 `ckpt_save_time_interval` 是 `None`。你的 `train.py` 又没有开放 `--ckpt_save_time_interval` 参数，所以不能直接通过命令行传值。

补丁：

```python
time_past_this_epoch // (ckpt_save_time_interval or 1e18)
```

含义是：如果没有设置按时间保存 checkpoint，就使用一个极大的数，相当于禁用按时间保存，只保留按 epoch 保存。

这一步的目的：修复 checkpoint 保存逻辑的小 bug，不改变模型、数据和训练目标。

---

## 0.17 为什么 checkpoint 输出在这个目录

训练日志中：

```text
cfg.TAG: company_voxelnext_26cls_trainval
cfg.EXP_GROUP_PATH: nuscenes_models
extra_tag: formal_company_26cls
```

所以输出目录是：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/
```

checkpoint 目录是：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/
```

这一步的目的：明确模型文件保存位置，方便后续测试、恢复训练和交付。

---

## 0.18 为什么测试 checkpoint 不等于模型效果合格

当前流程能训练出 26 类 checkpoint，但这只说明训练链路跑通，不代表模型效果已经达标。

原因有三点：

1. 当前 `CompanyNuScenesDataset.evaluation()` 只报告各类 GT/预测数量，不计算正式 AP/NDS。
2. `human_pedestrian_personal_mobility` 和 `vehicle_bus_bendy` 当前没有标注框，模型虽然有 26 类输出，但这两类无法学到有效检测能力。
3. 当前点云第 4 维是零占位值，不是真实 intensity，点特征信息弱于真实强度数据。

所以最终结论应是：

```text
公司正式数据的 26 类 VoxelNeXt 训练链路已经跑通；模型效果评估还需要后续补正式指标或人工抽检。
```

这一步的目的：区分“工程链路跑通”和“模型效果合格”，避免对训练结果过度解读。

---

## 1. 服务器目录与容器挂载关系

宿主机工作目录：

```bash
/home/ubuntu/WXY
```

工程目录：

```bash
/home/ubuntu/WXY/OpenPCDet_ljl
```

数据目录：

```bash
/home/ubuntu/WXY/data
```

容器内对应路径：

```text
/home/ubuntu/WXY/OpenPCDet_ljl -> /workspace/OpenPCDet
/home/ubuntu/WXY/data          -> /workspace/OpenPCDet/data
```

检查容器挂载：

```bash
sudo docker inspect detection3d_v5 --format='{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

期望输出：

```text
/home/ubuntu/WXY/data -> /workspace/OpenPCDet/data
/home/ubuntu/WXY/OpenPCDet_ljl -> /workspace/OpenPCDet
```

---

## 2. 正确代码版本检查

进入宿主机项目目录：

```bash
cd /home/ubuntu/WXY/OpenPCDet_ljl
```

检查版本：

```bash
git log -1 --oneline
```

期望看到：

```text
2d144e7 feat: train formal company data with 26-class VoxelNeXt
```

检查分支：

```bash
git branch
```

期望看到：

```text
* codex/formal-company-voxelnext-26cls
  main
```

检查关键脚本：

```bash
ls -lh tools/company_nuscenes
```

至少应包含：

```text
check_company_infos.py
check_lidar_format.py
create_company_infos.py
preview_formal_split.py
smoke_test_company_dataloader.py
smoke_test_formal_voxelnext.py
```

检查关键配置：

```bash
find . -name "company_voxelnext_26cls_trainval.yaml"
find . -name "company_nuscenes_trainval_dataset.yaml"
find . -name "voxelnext_head.py"
```

期望能找到：

```text
./tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
./tools/cfgs/dataset_configs/company_nuscenes_trainval_dataset.yaml
./pcdet/models/dense_heads/voxelnext_head.py
```

---

## 3. 如果容器里看不到新版代码怎么办

如果宿主机代码是 `2d144e7`，但容器里显示旧版本，例如：

```text
21b7dfc main
```

说明容器还没看到新的挂载目录。先重启容器：

```bash
sudo docker restart detection3d_v5
```

然后进入：

```bash
sudo docker exec -u root -it detection3d_v5 /bin/bash
```

容器内检查：

```bash
cd /workspace/OpenPCDet
git log -1 --oneline
find . -name "company_voxelnext_26cls_trainval.yaml"
find . -name "smoke_test_formal_voxelnext.py"
```

如果重启后仍然不对，再重建容器。注意：如果要训练，容器必须带 GPU 参数创建，详见第 4 节。

---

## 4. 创建带 GPU 的训练容器

一开始创建的 `detection3d_v5` 如果没有加 `--gpus all`，容器内会出现：

```text
RuntimeError: Found no NVIDIA driver on your system.
```

这说明容器没有拿到 GPU，不是代码问题。

先在宿主机检查 GPU：

```bash
nvidia-smi
```

本次服务器有两张 RTX 3090：

```text
GPU 0: NVIDIA GeForce RTX 3090
GPU 1: NVIDIA GeForce RTX 3090
```

其中 GPU 0 已被占用较多，GPU 1 基本空闲，所以后续训练建议使用：

```bash
CUDA_VISIBLE_DEVICES=1
```

删除旧容器：

```bash
sudo docker stop detection3d_v5
sudo docker rm detection3d_v5
```

重新创建带 GPU 的容器：

```bash
sudo docker run -it -d --gpus all --name detection3d_v5 -v /home/ubuntu/WXY/OpenPCDet_ljl:/workspace/OpenPCDet -v /home/ubuntu/WXY/data:/workspace/OpenPCDet/data pcdet_back_3dv2:20260520 /bin/bash
```

进入容器：

```bash
sudo docker exec -u root -it detection3d_v5 /bin/bash
```

检查 GPU：

```bash
cd /workspace/OpenPCDet
nvidia-smi
```

如果容器里也能看到两张 3090，说明 GPU 挂载成功。

---

## 5. 正式数据目录要求

本次必须使用：

```text
/workspace/OpenPCDet/data/nuscenes/
```

不要使用：

```text
/workspace/OpenPCDet/data/v1.0-trainval
```

正式目录结构应类似：

```text
data/nuscenes/
├── samples/
│   └── LIDAR_TOP/
│       └── *.bin
└── v1.0-trainval/
    ├── category.json
    ├── instance.json
    ├── sample.json
    ├── sample_data.json
    ├── sample_annotation.json
    ├── scene.json
    ├── calibrated_sensor.json
    ├── ego_pose.json
    └── ...
```

进入容器工程目录：

```bash
cd /workspace/OpenPCDet
```

检查目录：

```bash
test -d data/nuscenes/v1.0-trainval && echo "v1.0-trainval OK"
test -d data/nuscenes/samples/LIDAR_TOP && echo "LIDAR_TOP OK"
```

检查 JSON 数量：

```bash
python - <<'PY'
from pathlib import Path
import json
root = Path('data/nuscenes/v1.0-trainval')
for name in ['category', 'scene', 'sample', 'sample_data', 'sample_annotation', 'instance']:
    p = root / f'{name}.json'
    with open(p, 'r', encoding='utf-8') as f:
        print(f'{name:20s}: {len(json.load(f))}')
PY
```

本次实际结果：

```text
category            : 26
scene               : 412
sample              : 24142
sample_data         : 241420
sample_annotation   : 1222229
instance            : 53246
```

---

## 6. 检查 LIDAR_TOP 点云是否完整

一开始用普通 `find` 检查时出现过：

```text
24083
```

但这不够准确。最终应以 `sample_data.json` 实际引用路径为准。

执行：

```bash
python - <<'PY'
from pathlib import Path
import json
from collections import Counter

data_root = Path('data/nuscenes')
meta_root = data_root / 'v1.0-trainval'

sample_data = json.load(open(meta_root / 'sample_data.json', 'r', encoding='utf-8'))

lidar_top = [
    sd for sd in sample_data
    if sd.get('channel') == 'LIDAR_TOP' or 'LIDAR_TOP' in sd.get('filename', '')
]

missing = []
suffix_counter = Counter()
parent_counter = Counter()

for sd in lidar_top:
    filename = sd['filename']
    p = data_root / filename
    suffix_counter[p.suffix.lower() or 'NO_SUFFIX'] += 1
    parent_counter[str(Path(filename).parent)] += 1
    if not p.exists():
        missing.append(filename)

print('LIDAR_TOP records:', len(lidar_top))
print('Missing LIDAR_TOP files:', len(missing))
print('suffix:', suffix_counter)
print('parents:')
for k, v in parent_counter.most_common():
    print(k, v)
PY
```

本次最终结果：

```text
LIDAR_TOP records: 24142
Missing LIDAR_TOP files: 0
suffix: Counter({'.bin': 24142})
parents:
samples/LIDAR_TOP 24142
```

这说明点云路径完整，可以继续。

---

## 7. 预览正式 train/val 切分

由于 ToDesk 粘贴多行命令容易出问题，下面都给单行命令。

执行：

```bash
python tools/company_nuscenes/preview_formal_split.py --data_path data/nuscenes --version v1.0-trainval --train_ratio 0.8 --seed 0 --min_lidar_points 1
```

本次实际输出：

```text
train_scenes: 329
val_scenes: 83
train_samples: 19268
val_samples: 4874
min_lidar_points: 1
annotated_but_absent_from_train: []
annotated_but_absent_from_val: ['vehicle.emergency.other']
PASS: all learnable annotated classes are present in training.
```

解释：

- `annotated_but_absent_from_train: []` 是关键，说明所有可学习类别都进入训练集。
- `vehicle.emergency.other` 没进验证集是合理的，因为它有效样本极少，优先保证进入训练集。
- 这一步是只读预览，不生成文件。

---

## 8. 生成 info 文件

执行：

```bash
python tools/company_nuscenes/create_company_infos.py --data_path data/nuscenes --save_path data/nuscenes --version v1.0-trainval --max_sweeps 1 --train_ratio 0.8 --seed 0 --min_lidar_points 1
```

本次实际输出：

```text
Company nuScenes train infos: 19268
Company nuScenes val infos: 4874
Saved: data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl
Saved: data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
```

生成文件位置：

```text
/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl
/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
```

本次文件大小：

```text
company_nuscenes_infos_train.pkl  332M
company_nuscenes_infos_val.pkl     86M
```

检查命令：

```bash
ls -lh data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
```

注意：本次 `ImageSets/train.txt` 和 `ImageSets/val.txt` 的行数是 scene 数，不是 sample 数：

```bash
wc -l data/nuscenes/v1.0-trainval/ImageSets/train.txt data/nuscenes/v1.0-trainval/ImageSets/val.txt
```

实际结果：

```text
329 data/nuscenes/v1.0-trainval/ImageSets/train.txt
 83 data/nuscenes/v1.0-trainval/ImageSets/val.txt
412 total
```

而 info 内实际 sample 数是：

```text
train infos: 19268
val infos: 4874
```

---

## 9. 严格检查 info

执行：

```bash
python tools/company_nuscenes/check_company_infos.py --root data/nuscenes/v1.0-trainval --data_root data/nuscenes --strict --min_lidar_points 1
```

本次关键结果：

```text
company_nuscenes_infos_train.pkl
  samples: 19268
  empty_gt_samples: 0
  empty_gt_after_min_points_1: 0
  invalid_num_lidar_pts_samples: 0
  boxes_with_unknown_lidar_pts_kept: 485
  missing_lidar_paths: 0
  classes: 24
  retained_boxes_min_points_1: 711959

company_nuscenes_infos_val.pkl
  samples: 4874
  empty_gt_samples: 0
  empty_gt_after_min_points_1: 0
  invalid_num_lidar_pts_samples: 0
  boxes_with_unknown_lidar_pts_kept: 126
  missing_lidar_paths: 0
  classes: 23
  retained_boxes_min_points_1: 183358

Class-name check
  outside_config: []
  absent_from_infos: ['human_pedestrian_personal_mobility', 'vehicle_bus_bendy']
  annotated_but_absent_from_train: []
  annotated_but_absent_from_val: ['vehicle_emergency_other']
  annotated_but_unlearnable_after_train_filter: []
```

重点判断：

- `missing_lidar_paths: 0`：点云路径无缺失
- `invalid_num_lidar_pts_samples: 0`：点数统计无非法样本
- `outside_config: []`：类别没有超出配置
- `annotated_but_absent_from_train: []`：训练集覆盖所有可学习类别
- `annotated_but_unlearnable_after_train_filter: []`：过滤后没有不可学习类别
- `human_pedestrian_personal_mobility` 与 `vehicle_bus_bendy` 缺失是允许的，因为当前正式标注中没有这两类框

---

## 10. 修复 `av2` 缺包问题

第一次跑 smoke test 报错：

```text
ModuleNotFoundError: No module named 'av2'
```

原因是：

```python
from .argo2.argo2_dataset import Argo2Dataset
```

会强制导入 Argo2 数据集，而本次训练不用 Argo2，容器里也没有 `av2` 包。

解决方法：把 Argo2 导入改成可选导入。

备份：

```bash
cp pcdet/datasets/__init__.py pcdet/datasets/__init__.py.bak_av2
```

一行修复：

```bash
python -c "from pathlib import Path; p=Path('pcdet/datasets/__init__.py'); s=p.read_text(); old='from .argo2.argo2_dataset import Argo2Dataset'; new='try:\n    from .argo2.argo2_dataset import Argo2Dataset\nexcept ModuleNotFoundError:\n    Argo2Dataset = None'; assert old in s, 'target import line not found'; p.write_text(s.replace(old,new))"
```

检查：

```bash
grep -n "Argo2Dataset\|ModuleNotFoundError" pcdet/datasets/__init__.py
```

期望看到：

```text
from .argo2.argo2_dataset import Argo2Dataset
except ModuleNotFoundError:
    Argo2Dataset = None
```

---

## 11. dataloader smoke test

注意：本次相对路径执行失败过：

```text
FileNotFoundError: No such file or directory: 'tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml'
```

因此建议使用绝对路径。

执行：

```bash
python /workspace/OpenPCDet/tools/company_nuscenes/smoke_test_company_dataloader.py --cfg_file /workspace/OpenPCDet/tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
```

本次实际结果：

```text
Loading CompanyNuScenes dataset
Total samples for CompanyNuScenes dataset: 19268
dataset_len: 19268
batch_keys: ['batch_size', 'flip_x', 'flip_y', 'frame_id', 'gt_boxes', 'metadata', 'noise_rot', 'noise_scale', 'points', 'use_lead_xyz', 'voxel_coords', 'voxel_num_points', 'voxels']
points_shape: (201843, 5)
gt_boxes_shape: (1, 78, 8)
first_gt_box: [-2.443835973739624, -13.712844848632812, 0.24773462116718292, 4.907037258148193, 2.0718600749969482, 1.5334491729736328, -1.8094402551651, 6.0]
```

解释：

- `dataset_len: 19268` 与 train infos 对齐
- `points_shape` 第二维为 5，表示 `batch_index + x + y + z + intensity占位`
- `gt_boxes_shape` 最后一维为 8，表示 `x,y,z,dx,dy,dz,yaw,class_id`

---

## 12. VoxelNeXt GPU smoke test

确认容器带 GPU：

```bash
nvidia-smi
```

如果 GPU 0 占用较多，优先使用 GPU 1：

```bash
CUDA_VISIBLE_DEVICES=1 python /workspace/OpenPCDet/tools/company_nuscenes/smoke_test_formal_voxelnext.py --cfg_file /workspace/OpenPCDet/tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --workers 0
```

本次实际结果：

```text
formal_voxelnext_smoke: PASS
classes: 26
dataset_len: 19268
points_shape: (194422, 5)
gt_boxes_shape: (1, 48, 8)
loss: 466.34649658203125
loss_terms: {'hm_loss_head_0': 274.2383728027344, 'loc_loss_head_0': 0.0, 'hm_loss_head_1': 8.182193756103516, 'loc_loss_head_1': 1.729399561882019, 'hm_loss_head_2': 19.335277557373047, 'loc_loss_head_2': 2.768141508102417, 'hm_loss_head_3': 69.55618286132812, 'loc_loss_head_3': 0.0, 'hm_loss_head_4': 90.53694152832031, 'loc_loss_head_4': 0.0, 'rpn_loss': 466.34649658203125}
```

看到 `formal_voxelnext_smoke: PASS` 后，说明：

- 数据加载正常
- 模型能构建
- CUDA 可用
- spconv 可用
- forward 正常
- loss 正常
- backward 正常

---

## 13. 启动正式训练

进入 tools 目录：

```bash
cd /workspace/OpenPCDet/tools
```

启动训练：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --batch_size 1 --epochs 20 --workers 4 --extra_tag formal_company_26cls --ckpt_save_interval 1 --max_ckpt_save_num 20
```

本次训练参数：

| 参数 | 值 | 说明 |
|---|---:|---|
| `CUDA_VISIBLE_DEVICES` | `1` | 使用第二张 3090 |
| `cfg_file` | `cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml` | 26 类正式 VoxelNeXt 配置 |
| `batch_size` | `1` | 单卡 batch size |
| `epochs` | `20` | 训练 20 个 epoch |
| `workers` | `4` | dataloader worker 数 |
| `extra_tag` | `formal_company_26cls` | 输出目录标签 |
| `ckpt_save_interval` | `1` | 每个 epoch 保存一次 |
| `max_ckpt_save_num` | `20` | 最多保留 20 个 checkpoint |

训练日志中应确认：

```text
CUDA_VISIBLE_DEVICES=1
cfg.CLASS_NAMES: 26 类
cfg.DATA_CONFIG.DATASET: CompanyNuScenesDataset
cfg.DATA_CONFIG.DATA_PATH: ../data/nuscenes
cfg.DATA_CONFIG.VERSION: v1.0-trainval
cfg.DATA_CONFIG.MAX_SWEEPS: 1
cfg.DATA_CONFIG.LIDAR_POINT_DIM: 4
cfg.DATA_CONFIG.PRED_VELOCITY: False
cfg.DATA_CONFIG.FILTER_MIN_POINTS_IN_GT: 1
cfg.MODEL.NAME: VoxelNeXt
cfg.MODEL.DENSE_HEAD.NAME: VoxelNeXtHead
cfg.DATA_CONFIG.INFO_PATH.train: ['company_nuscenes_infos_train.pkl']
cfg.DATA_CONFIG.INFO_PATH.test: ['company_nuscenes_infos_val.pkl']
```

本次训练已经成功进入第 1 个 iteration：

```text
epochs: 0%| ... loss=303, lr=0.0003 ...
```

这说明训练主链路已经跑起来。

---

## 14. 修复训练中 `ckpt_save_time_interval=None` 报错

训练启动后遇到报错：

```text
TypeError: unsupported operand type(s) for //: 'float' and 'NoneType'
```

报错位置：

```text
tools/train_utils/train_utils.py
if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
```

原因是 `ckpt_save_time_interval` 默认为 `None`，而代码尝试对它做整除。

本项目的 `train.py` 不支持命令行参数：

```text
--ckpt_save_time_interval
```

所以需要改代码。

进入目录：

```bash
cd /workspace/OpenPCDet/tools
```

备份：

```bash
cp train_utils/train_utils.py train_utils/train_utils.py.bak_ckpt_interval
```

打一行补丁：

```bash
python -c "from pathlib import Path; p=Path('train_utils/train_utils.py'); s=p.read_text(); old='time_past_this_epoch // ckpt_save_time_interval'; new='time_past_this_epoch // (ckpt_save_time_interval or 1e18)'; assert old in s, 'target not found'; p.write_text(s.replace(old,new))"
```

检查：

```bash
grep -n "ckpt_save_time_interval" train_utils/train_utils.py | head -20
```

期望看到：

```text
time_past_this_epoch // (ckpt_save_time_interval or 1e18)
```

然后重新启动训练：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --batch_size 1 --epochs 20 --workers 4 --extra_tag formal_company_26cls --ckpt_save_interval 1 --max_ckpt_save_num 20
```

---

## 15. 模型、日志和 checkpoint 输出位置

训练输出目录由配置名和 `extra_tag` 共同决定。

本次配置：

```text
cfg.TAG: company_voxelnext_26cls_trainval
cfg.EXP_GROUP_PATH: nuscenes_models
extra_tag: formal_company_26cls
```

因此输出根目录为：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/
```

checkpoint 目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/
```

查看 checkpoint：

```bash
ls -lh /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/
```

常见文件：

```text
checkpoint_epoch_1.pth
checkpoint_epoch_2.pth
...
checkpoint_epoch_20.pth
```

日志目录通常也在：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/
```

可以查看：

```bash
find /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls -maxdepth 2 -type f | sort
```

如果想保存终端日志，可以用：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --batch_size 1 --epochs 20 --workers 4 --extra_tag formal_company_26cls --ckpt_save_interval 1 --max_ckpt_save_num 20 2>&1 | tee train_formal_company_26cls.log
```

但注意：如果训练很久，ToDesk 断开可能影响交互式终端。更稳妥的方式是配合 `tmux` 或 `nohup`。

---

## 16. 断开远程连接时的训练建议

推荐使用 `tmux`：

```bash
tmux new -s voxelnext26
```

进入后：

```bash
cd /workspace/OpenPCDet/tools
```

启动训练：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --batch_size 1 --epochs 20 --workers 4 --extra_tag formal_company_26cls --ckpt_save_interval 1 --max_ckpt_save_num 20 2>&1 | tee train_formal_company_26cls.log
```

临时离开 tmux：

```text
Ctrl + B，然后按 D
```

重新进入：

```bash
tmux attach -t voxelnext26
```

如果没有 tmux，也可以用 `nohup`：

```bash
cd /workspace/OpenPCDet/tools
nohup bash -c "CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --batch_size 1 --epochs 20 --workers 4 --extra_tag formal_company_26cls --ckpt_save_interval 1 --max_ckpt_save_num 20" > train_formal_company_26cls.log 2>&1 &
```

查看日志：

```bash
tail -f train_formal_company_26cls.log
```

---

## 17. 训练完成后测试 checkpoint

假设训练得到：

```text
checkpoint_epoch_20.pth
```

测试命令：

```bash
cd /workspace/OpenPCDet/tools
```

```bash
CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --ckpt ../output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth --batch_size 1 --workers 4 --extra_tag formal_company_26cls --eval_tag epoch20
```

测试输出目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/
```

注意：当前 `CompanyNuScenesDataset.evaluation()` 只报告各类 GT/预测数量，不计算正式 AP/NDS。因此测试能跑通不等于模型效果已经达标，后续仍需要补正式评估指标或进行人工抽检。

---

## 18. 常见问题汇总

### 18.1 ToDesk 粘贴多行命令只执行第一行

现象：

```text
bash: --data_path: command not found
bash: --version: command not found
```

原因：参数被拆成单独命令执行。

解决：使用本文提供的单行命令。

---

### 18.2 容器内看不到新版代码

现象：

```text
git log -1
21b7dfc main
```

但宿主机是：

```text
2d144e7 codex/formal-company-voxelnext-26cls
```

解决：

```bash
sudo docker restart detection3d_v5
```

如果仍然不行，重建容器。

---

### 18.3 容器内没有 GPU

现象：

```text
RuntimeError: Found no NVIDIA driver on your system.
```

原因：创建容器时没加 `--gpus all`。

解决：删除旧容器，重新创建：

```bash
sudo docker stop detection3d_v5
sudo docker rm detection3d_v5
sudo docker run -it -d --gpus all --name detection3d_v5 -v /home/ubuntu/WXY/OpenPCDet_ljl:/workspace/OpenPCDet -v /home/ubuntu/WXY/data:/workspace/OpenPCDet/data pcdet_back_3dv2:20260520 /bin/bash
```

---

### 18.4 `av2` 缺包

现象：

```text
ModuleNotFoundError: No module named 'av2'
```

原因：`pcdet/datasets/__init__.py` 强制导入 Argo2 数据集，而本次不用 Argo2。

解决：将 Argo2 导入改为可选导入，见第 10 节。

---

### 18.5 相对路径找不到配置文件

现象：

```text
FileNotFoundError: No such file or directory: 'tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml'
```

解决：使用绝对路径：

```bash
python /workspace/OpenPCDet/tools/company_nuscenes/smoke_test_company_dataloader.py --cfg_file /workspace/OpenPCDet/tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
```

---

### 18.6 训练保存逻辑报 `NoneType`

现象：

```text
TypeError: unsupported operand type(s) for //: 'float' and 'NoneType'
```

解决：补丁见第 14 节。

---

### 18.7 退出容器时提示 `There are stopped jobs`

现象：

```text
There are stopped jobs.
```

查看：

```bash
jobs -l
```

如果看到：

```text
Stopped python tools/company_nuscenes/create_company_infos.py
```

杀掉：

```bash
kill %1
```

确认无任务后退出：

```bash
jobs -l
exit
```

---

## 19. 最终从零复现命令清单

下面是假设代码和数据已经放好后的最短复现流程。

宿主机创建 GPU 容器：

```bash
sudo docker stop detection3d_v5
sudo docker rm detection3d_v5
sudo docker run -it -d --gpus all --name detection3d_v5 -v /home/ubuntu/WXY/OpenPCDet_ljl:/workspace/OpenPCDet -v /home/ubuntu/WXY/data:/workspace/OpenPCDet/data pcdet_back_3dv2:20260520 /bin/bash
```

进入容器：

```bash
sudo docker exec -u root -it detection3d_v5 /bin/bash
```

进入工程：

```bash
cd /workspace/OpenPCDet
```

检查版本：

```bash
git log -1 --oneline
```

检查 GPU：

```bash
nvidia-smi
```

生成 info：

```bash
python tools/company_nuscenes/create_company_infos.py --data_path data/nuscenes --save_path data/nuscenes --version v1.0-trainval --max_sweeps 1 --train_ratio 0.8 --seed 0 --min_lidar_points 1
```

检查 info：

```bash
python tools/company_nuscenes/check_company_infos.py --root data/nuscenes/v1.0-trainval --data_root data/nuscenes --strict --min_lidar_points 1
```

修复 `av2` 可选导入：

```bash
cp pcdet/datasets/__init__.py pcdet/datasets/__init__.py.bak_av2
python -c "from pathlib import Path; p=Path('pcdet/datasets/__init__.py'); s=p.read_text(); old='from .argo2.argo2_dataset import Argo2Dataset'; new='try:\n    from .argo2.argo2_dataset import Argo2Dataset\nexcept ModuleNotFoundError:\n    Argo2Dataset = None'; assert old in s, 'target import line not found'; p.write_text(s.replace(old,new))"
```

dataloader smoke：

```bash
python /workspace/OpenPCDet/tools/company_nuscenes/smoke_test_company_dataloader.py --cfg_file /workspace/OpenPCDet/tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
```

GPU smoke：

```bash
CUDA_VISIBLE_DEVICES=1 python /workspace/OpenPCDet/tools/company_nuscenes/smoke_test_formal_voxelnext.py --cfg_file /workspace/OpenPCDet/tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --workers 0
```

修复训练保存逻辑：

```bash
cd /workspace/OpenPCDet/tools
cp train_utils/train_utils.py train_utils/train_utils.py.bak_ckpt_interval
python -c "from pathlib import Path; p=Path('train_utils/train_utils.py'); s=p.read_text(); old='time_past_this_epoch // ckpt_save_time_interval'; new='time_past_this_epoch // (ckpt_save_time_interval or 1e18)'; assert old in s, 'target not found'; p.write_text(s.replace(old,new))"
```

正式训练：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --batch_size 1 --epochs 20 --workers 4 --extra_tag formal_company_26cls --ckpt_save_interval 1 --max_ckpt_save_num 20
```

测试 epoch 20：

```bash
CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --ckpt ../output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth --batch_size 1 --workers 4 --extra_tag formal_company_26cls --eval_tag epoch20
```

---

## 20. 当前能力边界

当前流程能完成：

- 读取公司正式 nuScenes 格式数据
- 生成 train/val info
- 构建 26 类 VoxelNeXt
- 执行 dataloader smoke
- 执行 GPU forward/loss/backward smoke
- 启动正式训练并输出 checkpoint

当前流程还不能直接证明：

- 26 类全部有有效检测能力，因为 `human_pedestrian_personal_mobility` 与 `vehicle_bus_bendy` 当前无标注框
- 模型性能达标，因为当前 evaluation 不计算正式 AP/NDS
- 第 4 维点特征有真实强度，因为当前第 4 维是零占位值

因此，当前结论应表述为：

> 公司正式数据的 26 类 VoxelNeXt 训练链路已经跑通；模型效果评估还需要后续补正式指标或人工抽检。
