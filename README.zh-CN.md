<img src="docs/open_mmlab.png" align="right" width="30%">

# OpenPCDet

`OpenPCDet` 是一个清晰、简洁、自包含的开源项目，用于基于 LiDAR 的 3D 目标检测。

它也是 [`[PointRCNN]`](https://arxiv.org/abs/1812.04244)、[`[Part-A2-Net]`](https://arxiv.org/abs/1907.03670)、[`[PV-RCNN]`](https://arxiv.org/abs/1912.13192)、[`[Voxel R-CNN]`](https://arxiv.org/abs/2012.15712)、[`[PV-RCNN++]`](https://arxiv.org/abs/2102.00463) 和 [`[MPPNet]`](https://arxiv.org/abs/2205.05979) 的官方代码发布仓库。

**亮点**：
* `OpenPCDet` 已更新至 `v0.6.0`（2022 年 9 月）。
* 已支持 PV-RCNN++ 的代码。
* 已支持 MPPNet 的代码。

## 概览
- [更新日志](#更新日志)
- [设计模式](#openpcdet-设计模式)
- [模型库](#模型库)
- [安装](docs/INSTALL.md)
- [快速演示](docs/DEMO.md)
- [入门指南](docs/GETTING_STARTED.md)
- [引用](#引用)


## 更新日志
[2022-09-02] **新增：** 将 `OpenPCDet` 更新至 v0.6.0：
* 正式发布用于时序 3D 目标检测的 [MPPNet](https://arxiv.org/abs/2205.05979) 官方代码。该方法支持长期多帧 3D 目标检测，并于 2022 年 9 月 2 日在 Waymo Open Dataset 的 [3D detection learderboard](https://waymo.com/open/challenges/2020/3d-detection) 上排名第一。对于验证集，MPPNet 在车辆、行人和骑行者类别上的 mAPH@Level_2 分别达到 74.96%、75.06% 和 74.52%。（关于如何使用 MPPNet 进行训练/测试，请参见[指南](docs/guidelines_of_approaches/mppnet.md)）。
* 支持在 Waymo Open Dataset 上进行多帧训练/测试（关于如何处理数据的更多细节，请参见[更新日志](docs/changelog.md)）。
* 支持将训练过程中变化的细节（例如 loss、iter、epoch）保存到文件中（仍可通过 `--use_tqdm_to_record` 使用之前的 tqdm 进度条）。如果你还想记录 GPU 相关信息，请使用 `pip install gpustat`。
* 支持每 5 分钟保存一次最新模型，因此可以从最新状态恢复模型训练，而不必从上一个 epoch 恢复。

[2022-08-22] 增加了对[自定义数据集教程和模板](docs/CUSTOM_DATASET_TUTORIAL.md)的支持。

[2022-07-05] 增加了对 3D 目标检测骨干网络 [`Focals Conv`](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Focal_Sparse_Convolutional_Networks_for_3D_Object_Detection_CVPR_2022_paper.pdf) 的支持。

[2022-02-12] 增加了对使用 Docker 的支持。请参考 [./docker](./docker) 中的指南。

[2022-02-07] 增加了对 NuScenes 数据集上 CenterPoint 模型的支持。

[2022-01-14] 增加了对动态 pillar 体素化的支持，遵循 [H^23D R-CNN](https://arxiv.org/abs/2107.14391) 中提出的实现，并使用 unique 操作和 [`torch_scatter`](https://github.com/rusty1s/pytorch_scatter) 包。

[2022-01-05] **新增：** 将 `OpenPCDet` 更新至 v0.5.2：
* [PV-RCNN++](https://arxiv.org/abs/2102.00463) 的代码已发布到该仓库中；与 PV-RCNN 相比，它具有更高性能、更快的训练/推理速度以及更低的内存消耗。
* 增加了若干模型在 [Waymo Open Dataset](#waymo-open-dataset-baselines) 完整训练集上训练后的性能结果。
* 支持 Lyft 数据集，参见[此 pull request](https://github.com/open-mmlab/OpenPCDet/pull/720)。


[2021-12-09] **新增：** 将 `OpenPCDet` 更新至 v0.5.1：
* 增加了 [Waymo Open Dataset](#waymo-open-dataset-baselines) 上 PointPillar 相关的基线配置/结果。
* 支持 Pandaset dataloader，参见[此 pull request](https://github.com/open-mmlab/OpenPCDet/pull/396)。
* 支持一组新的数据增强方法，参见[此 pull request](https://github.com/open-mmlab/OpenPCDet/pull/653)。

[2021-12-01] **新增：** `OpenPCDet` v0.5.0 发布，包含以下特性：
* 提升了 [Waymo Open Dataset](#waymo-open-dataset-baselines) 上所有模型的性能。注意，你需要重新准备 Waymo Open Dataset 的训练/验证数据以及 ground-truth 数据库（见 [GETTING_STARTED.md](docs/GETTING_STARTED.md)）。
* 支持 anchor-free 的 [CenterHead](pcdet/models/dense_heads/center_head.py)，并增加了 `CenterPoint` 和 `PV-RCNN with CenterHead` 的配置。
* 支持最新的 **PyTorch 1.1~1.10** 和 **spconv 1.0~2.x**，其中 **spconv 2.x** 应该可以通过 pip 轻松安装，并且比之前版本更快（参见 spconv 的官方更新[这里](https://github.com/traveller59/spconv)）。
* 支持配置 [`USE_SHARED_MEMORY`](tools/cfgs/dataset_configs/waymo_dataset.yaml) 以使用共享内存；当你遇到 IO 问题时，这可能加快训练过程。
* 支持更好、更快的[可视化脚本](tools/visual_utils/open3d_vis_utils.py)，需要先安装 [Open3D](https://github.com/isl-org/Open3D)。

[2021-06-08] 增加了对基于体素的 3D 目标检测模型 [`Voxel R-CNN`](#KITTI-3D-Object-Detection-Baselines) 的支持。

[2021-05-14] 增加了对单目 3D 目标检测模型 [`CaDDN`](#KITTI-3D-Object-Detection-Baselines) 的支持。

[2020-11-27] Bug 修复：如果你想使用我们提供的 Waymo 评测工具，请重新准备 Waymo 数据集（version 1.2）的验证 infos（见 [PR](https://github.com/open-mmlab/OpenPCDet/pull/383)）。注意，你不需要重新准备训练数据和 ground-truth 数据库。

[2020-11-10] 已支持 [Waymo Open Dataset](#waymo-open-dataset-baselines)，并取得了先进结果。目前我们提供了 Waymo Open Dataset 上 `SECOND`、`PartA2` 和 `PV-RCNN` 的配置与结果；更多模型可以通过修改它们的数据集配置轻松支持。

[2020-08-10] Bug 修复：已更新所提供的 NuScenes 模型，以修复加载问题。如果需要使用预训练 NuScenes 模型，请重新下载。

[2020-07-30] `OpenPCDet` v0.3.0 发布，包含以下特性：
   * 现已支持基于点和 Anchor-Free 的模型（[`PointRCNN`](#KITTI-3D-Object-Detection-Baselines)、[`PartA2-Free`](#KITTI-3D-Object-Detection-Baselines)）。
   * 支持 NuScenes 数据集，并提供强基线结果（[`SECOND-MultiHead (CBGS)`](#NuScenes-3D-Object-Detection-Baselines) 和 [`PointPillar-MultiHead`](#NuScenes-3D-Object-Detection-Baselines)）。
   * 相比上一版本效率更高，同时支持 **PyTorch 1.1~1.7** 和 **spconv 1.0~1.2**。

[2020-07-17] 增加了简单的可视化代码和快速演示，用于测试自定义数据。

[2020-06-24] `OpenPCDet` v0.2.0 发布，采用了全新的结构以支持更多模型和数据集。

[2020-03-16] `OpenPCDet` v0.1.0 发布。


## 介绍


### `OpenPCDet` 工具箱能做什么？

注意，我们已经将 `PCDet` 从 `v0.1` 升级到 `v0.2`，采用了全新的结构以支持各种数据集和模型。

`OpenPCDet` 是一个基于 PyTorch 的通用点云 3D 目标检测代码库。它目前支持多种先进的 3D 目标检测方法，并对单阶段和两阶段 3D 检测框架的代码进行了高度重构。

基于 `OpenPCDet` 工具箱，我们在 Waymo Open Dataset 挑战赛中，在所有仅使用 LiDAR 的方法中赢得了 [3D Detection](https://waymo.com/open/challenges/3d-detection/)、[3D Tracking](https://waymo.com/open/challenges/3d-tracking/) 和 [Domain Adaptation](https://waymo.com/open/challenges/domain-adaptation/) 三个赛道；与 Waymo 相关的模型将很快发布到 `OpenPCDet`。

我们目前正在积极更新该仓库，未来将支持更多数据集和模型。也欢迎大家贡献。

### `OpenPCDet` 设计模式

* 数据-模型分离，并使用统一的点云坐标系，便于扩展到自定义数据集：
<p align="center">
  <img src="docs/dataset_vs_model.png" width="95%" height="320">
</p>

* 统一的 3D 框定义：(x, y, z, dx, dy, dz, heading)。

* 灵活且清晰的模型结构，便于支持各种 3D 检测模型：
<p align="center">
  <img src="docs/model_framework.png" width="95%">
</p>

* 在一个框架内支持多种模型，如下所示：
<p align="center">
  <img src="docs/multiple_models_demo.png" width="95%">
</p>


### 当前支持的功能

- [x] 支持单阶段和两阶段 3D 目标检测框架
- [x] 支持多 GPU、多机器的分布式训练与测试
- [x] 支持在不同尺度上使用多个 head 来检测不同类别
- [x] 支持 stacked 版本的 set abstraction，以编码不同场景中数量各异的点
- [x] 支持用于目标分配的 Adaptive Training Sample Selection（ATSS）
- [x] 支持 RoI-aware 点云池化和 RoI-grid 点云池化
- [x] 支持 GPU 版本的 3D IoU 计算和旋转 NMS


## 模型库

<a name="KITTI-3D-Object-Detection-Baselines"></a>
### KITTI 3D 目标检测基线
下表展示了部分受支持的方法。结果为 KITTI 数据集 *val* 集中等难度下的 3D 检测性能。
* 所有基于 LiDAR 的模型均使用 8 块 GTX 1080Ti GPU 训练，并可供下载。
* 训练时间使用 8 块 TITAN XP GPU 和 PyTorch 1.5 测得。

|                                             | 训练时间 | Car@R11 | Pedestrian@R11 | Cyclist@R11  | 下载 | 
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:---------:|
| [PointPillar](tools/cfgs/kitti_models/pointpillar.yaml) |~1.2 小时| 77.28 | 52.29 | 62.68 | [model-18M](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing) | 
| [SECOND](tools/cfgs/kitti_models/second.yaml)       |  ~1.7 小时  | 78.62 | 52.98 | 67.15 | [model-20M](https://drive.google.com/file/d/1-01zsPOsqanZQqIIyy7FpNXStL3y4jdR/view?usp=sharing) |
| [SECOND-IoU](tools/cfgs/kitti_models/second_iou.yaml)       | -  | 79.09 | 55.74 | 71.31 | [model-46M](https://drive.google.com/file/d/1AQkeNs4bxhvhDQ-5sEo_yvQUlfo73lsW/view?usp=sharing) |
| [PointRCNN](tools/cfgs/kitti_models/pointrcnn.yaml) | ~3 小时 | 78.70 | 54.41 | 72.11 | [model-16M](https://drive.google.com/file/d/1BCX9wMn-GYAfSOPpyxf6Iv6fc0qKLSiU/view?usp=sharing)| 
| [PointRCNN-IoU](tools/cfgs/kitti_models/pointrcnn_iou.yaml) | ~3 小时 | 78.75 | 58.32 | 71.34 | [model-16M](https://drive.google.com/file/d/1V0vNZ3lAHpEEt0MlT80eL2f41K2tHm_D/view?usp=sharing)|
| [Part-A2-Free](tools/cfgs/kitti_models/PartA2_free.yaml)   | ~3.8 小时| 78.72 | 65.99 | 74.29 | [model-226M](https://drive.google.com/file/d/1lcUUxF8mJgZ_e-tZhP1XNQtTBuC-R0zr/view?usp=sharing) |
| [Part-A2-Anchor](tools/cfgs/kitti_models/PartA2.yaml)    | ~4.3 小时| 79.40 | 60.05 | 69.90 | [model-244M](https://drive.google.com/file/d/10GK1aCkLqxGNeX3lVu8cLZyE0G8002hY/view?usp=sharing) |
| [PV-RCNN](tools/cfgs/kitti_models/pv_rcnn.yaml) | ~5 小时| 83.61 | 57.90 | 70.47 | [model-50M](https://drive.google.com/file/d/1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-/view?usp=sharing) |
| [Voxel R-CNN (Car)](tools/cfgs/kitti_models/voxel_rcnn_car.yaml) | ~2.2 小时| 84.54 | - | - | [model-28M](https://drive.google.com/file/d/19_jiAeGLz7V0wNjSJw4cKmMjdm5EW5By/view?usp=sharing) |
| [Focals Conv - F](tools/cfgs/kitti_models/voxel_rcnn_car_focal_multimodal.yaml) | ~4 小时| 85.66 | - | - | [model-30M](https://drive.google.com/file/d/1u2Vcg7gZPOI-EqrHy7_6fqaibvRt2IjQ/view?usp=sharing) |
||
| [CaDDN (Mono)](tools/cfgs/kitti_models/CaDDN.yaml) |~15 小时| 21.38 | 13.02 | 9.76 | [model-774M](https://drive.google.com/file/d/1OQTO2PtXT8GGr35W9m2GZGuqgb6fyU1V/view?usp=sharing) |

<a name="Waymo-Open-Dataset-Baselines"></a>
### Waymo Open Dataset 基线
我们在 Waymo Open Dataset（WOD）上提供了 [`DATA_CONFIG.SAMPLED_INTERVAL`](tools/cfgs/dataset_configs/waymo_dataset.yaml) 设置，用于对部分样本进行子采样以训练和评估；因此，即便 GPU 资源有限，你也可以通过设置较小的 `DATA_CONFIG.SAMPLED_INTERVAL` 来尝试使用 WOD。

默认情况下，所有模型均使用全部训练样本中 **单帧** 的 **20% 数据（约 32k 帧）**，在 8 块 GTX 1080Ti GPU 上训练；表中每个单元格的结果是在**完整**验证集（version 1.2）上通过 Waymo 官方评测指标计算得到的 mAP/mAPH。

|    性能@（使用 20\% 数据训练）            | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |  
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| [SECOND](tools/cfgs/waymo_models/second.yaml) | 70.96/70.34|62.58/62.02|65.23/54.24	|57.22/47.49|	57.13/55.62 |	54.97/53.53 | 
| [PointPillar](tools/cfgs/waymo_models/pointpillar_1x.yaml) | 70.43/69.83 |	62.18/61.64 | 66.21/46.32|58.18/40.64|55.26/51.75|53.18/49.80 |
[CenterPoint-Pillar](tools/cfgs/waymo_models/centerpoint_pillar_1x.yaml)| 70.50/69.96|62.18/61.69|73.11/61.97|65.06/55.00|65.44/63.85|62.98/61.46| 
[CenterPoint-Dynamic-Pillar](tools/cfgs/waymo_models/centerpoint_dyn_pillar_1x.yaml)| 70.46/69.93|62.06/61.58|73.92/63.35|65.91/56.33|66.24/64.69|63.73/62.24| 
[CenterPoint](tools/cfgs/waymo_models/centerpoint_without_resnet.yaml)| 71.33/70.76|63.16/62.65|	72.09/65.49	|64.27/58.23|	68.68/67.39	|66.11/64.87|
| [CenterPoint (ResNet)](tools/cfgs/waymo_models/centerpoint.yaml)|72.76/72.23|64.91/64.42	|74.19/67.96	|66.03/60.34|	71.04/69.79	|68.49/67.28 |
| [Part-A2-Anchor](tools/cfgs/waymo_models/PartA2.yaml) | 74.66/74.12	|65.82/65.32	|71.71/62.24	|62.46/54.06	|66.53/65.18	|64.05/62.75 |
| [PV-RCNN (AnchorHead)](tools/cfgs/waymo_models/pv_rcnn.yaml) | 75.41/74.74	|67.44/66.80	|71.98/61.24	|63.70/53.95	|65.88/64.25	|63.39/61.82 | 
| [PV-RCNN (CenterHead)](tools/cfgs/waymo_models/pv_rcnn_with_centerhead_rpn.yaml) | 75.95/75.43	|68.02/67.54	|75.94/69.40	|67.66/61.62	|70.18/68.98	|67.73/66.57|
| [Voxel R-CNN (CenterHead)-Dynamic-Voxel](tools/cfgs/waymo_models/voxel_rcnn_with_centerhead_dyn_voxel.yaml) | 76.13/75.66	|68.18/67.74	|78.20/71.98	|69.29/63.59	| 70.75/69.68	|68.25/67.21|
| [PV-RCNN++](tools/cfgs/waymo_models/pv_rcnn_plusplus.yaml) | 77.82/77.32|	69.07/68.62|	77.99/71.36|	69.92/63.74|	71.80/70.71|	69.31/68.26|
| [PV-RCNN++ (ResNet)](tools/cfgs/waymo_models/pv_rcnn_plusplus_resnet.yaml) |77.61/77.14|	69.18/68.75|	79.42/73.31|	70.88/65.21|	72.50/71.39|	69.84/68.77|


这里我们还提供了若干模型在完整训练集上训练后的性能（参考 [PV-RCNN++](https://arxiv.org/abs/2102.00463) 论文）：

|    性能@（使用 100\% 数据训练）            | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |  
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| [SECOND](tools/cfgs/waymo_models/second.yaml) | 72.27/71.69 | 63.85/63.33 | 68.70/58.18 | 60.72/51.31 | 60.62/59.28 | 58.34/57.05 | 
| [CenterPoint-Pillar](tools/cfgs/waymo_models/centerpoint_pillar_1x.yaml)| 73.37/72.86 | 65.09/64.62 | 75.35/65.11 | 67.61/58.25 | 67.76/66.22 | 65.25/63.77 | 
| [Part-A2-Anchor](tools/cfgs/waymo_models/PartA2.yaml) | 77.05/76.51 | 68.47/67.97 | 75.24/66.87 | 66.18/58.62 | 68.60/67.36 | 66.13/64.93 |
| [PV-RCNN (CenterHead)](tools/cfgs/waymo_models/pv_rcnn_with_centerhead_rpn.yaml) | 78.00/77.50 | 69.43/68.98 | 79.21/73.03 | 70.42/64.72 | 71.46/70.27 | 68.95/67.79 |
| [PV-RCNN++](tools/cfgs/waymo_models/pv_rcnn_plusplus.yaml) | 79.10/78.63 | 70.34/69.91 | 80.62/74.62 | 71.86/66.30 | 73.49/72.38 | 70.70/69.62 |
| [PV-RCNN++ (ResNet)](tools/cfgs/waymo_models/pv_rcnn_plusplus_resnet.yaml) | 79.25/78.78 | 70.61/70.18 | 81.83/76.28 | 73.17/68.00 | 73.72/72.66 | 71.21/70.19 |
| [PV-RCNN++ (ResNet, 2 frames)](tools/cfgs/waymo_models/pv_rcnn_plusplus_resnet_2frames.yaml) | 80.17/79.70 | 72.14/71.70 | 83.48/80.42 | 75.54/72.61 | 74.63/73.75 | 72.35/71.50 |
| [MPPNet (4 frames)](docs/guidelines_of_approaches/mppnet.md) | 81.54/81.06 | 74.07/73.61 | 84.56/81.94 | 77.20/74.67 | 77.15/76.50 | 75.01/74.38 |
| [MPPNet (16 frames)](docs/guidelines_of_approaches/mppnet.md) | 82.74/82.28 | 75.41/74.96 | 84.69/82.25 | 77.43/75.06 | 77.28/76.66 | 75.13/74.52 |






由于 [Waymo Dataset License Agreement](https://waymo.com/open/terms/) 的限制，我们无法提供上述预训练模型；但你可以使用默认配置进行训练，从而轻松达到相近的性能。

<a name="NuScenes-3D-Object-Detection-Baselines"></a>
### NuScenes 3D 目标检测基线
所有模型均使用 8 块 GTX 1080Ti GPU 训练，并可供下载。

|                                             | mATE | mASE | mAOE | mAVE | mAAE | mAP | NDS | 下载 | 
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:---------:|:-------:|:-------:|:---------:|
| [PointPillar-MultiHead](tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml) | 33.87	| 26.00 | 32.07	| 28.74 | 20.15 | 44.63 | 58.23	 | [model-23M](https://drive.google.com/file/d/1p-501mTWsq0G9RzroTWSXreIMyTUUpBM/view?usp=sharing) | 
| [SECOND-MultiHead (CBGS)](tools/cfgs/nuscenes_models/cbgs_second_multihead.yaml) | 31.15 |	25.51 |	26.64 | 26.26 | 20.46 | 50.59 | 62.29 | [model-35M](https://drive.google.com/file/d/1bNzcOnE3u9iooBFMk2xK7HqhdeQ_nwTq/view?usp=sharing) |
| [CenterPoint-PointPillar](tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml) | 31.13 |	26.04 |	42.92 | 23.90 | 19.14 | 50.03 | 60.70 | [model-23M](https://drive.google.com/file/d/1UvGm6mROMyJzeSRu7OD1leU_YWoAZG7v/view?usp=sharing) |
| [CenterPoint (voxel_size=0.1)](tools/cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml) | 30.11 |	25.55 |	38.28 | 21.94 | 18.87 | 56.03 | 64.54 | [model-34M](https://drive.google.com/file/d/1Cz-J1c3dw7JAWc25KRG1XQj8yCaOlexQ/view?usp=sharing) |
| [CenterPoint (voxel_size=0.075)](tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml) | 28.80 |	25.43 |	37.27 | 21.55 | 18.24 | 59.22 | 66.48 | [model-34M](https://drive.google.com/file/d/1XOHAWm1MPkCKr1gqmc3TWi5AYZgPsgxU/view?usp=sharing) |


### 其他数据集
欢迎通过提交 pull request 来支持其他数据集。

## 安装

请参考 [INSTALL.md](docs/INSTALL.md) 来安装 `OpenPCDet`。


## 快速演示
请参考 [DEMO.md](docs/DEMO.md)，以了解如何使用预训练模型进行快速演示，并在你的自定义数据或原始 KITTI 数据上可视化预测结果。

## 入门指南

请参考 [GETTING_STARTED.md](docs/GETTING_STARTED.md)，进一步了解本项目的用法。


## 许可证

`OpenPCDet` 基于 [Apache 2.0 许可证](LICENSE) 发布。

## 致谢
`OpenPCDet` 是一个面向基于 LiDAR 的 3D 场景感知的开源项目，支持上文所示的多种基于 LiDAR 的感知模型。`PCDet` 的部分内容借鉴了上述受支持方法的官方发布代码。我们感谢这些方法的提出者及其官方实现。

我们希望该仓库能够作为一个强大且灵活的代码库，通过加速复现已有工作和/或开发新方法的过程，使研究社区受益。


## 引用
如果你觉得本项目对你的研究有帮助，请考虑引用：


```
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```

## 贡献
欢迎通过为本仓库做贡献来成为 OpenPCDet 开发团队的一员，也欢迎就任何潜在贡献与我们联系。

