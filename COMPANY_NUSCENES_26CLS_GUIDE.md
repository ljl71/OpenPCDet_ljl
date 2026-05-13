# OpenPCDet_ljl 公司 26 类 nuScenes 数据训练与自动标注文档

这份文档是给第一次接触本项目的人看的。它不假设你熟悉 OpenPCDet、nuScenes、CenterPoint、VoxelNeXt 或自动标注流程。

本项目的目标是：在 OpenPCDet 基础上，支持公司自定义 26 类、nuScenes 风格数据集的 LiDAR 3D 检测训练，并让训练出来的模型可以接到后续自动标注推理流程里。

---

## 1. 项目现在能做什么

当前 `OpenPCDet_ljl` 已经做了一个公司数据适配层，核心能力如下：

1. 使用公司 nuScenes 风格数据，而不是官方 nuScenes 10 类。
2. 支持公司 26 类标签。
3. 支持 LiDAR-only 训练闭环。
4. 支持单帧点云训练，默认 `MAX_SWEEPS: 1`。
5. 支持 `.bin` 点云文件，默认每个点为 4 维：`x, y, z, intensity`。
6. 生成 OpenPCDet 训练需要的 `infos_train.pkl` 和 `infos_val.pkl`。
7. 提供 26 类 CenterPoint 配置。
8. 提供 26 类 VoxelNeXt 配置。
9. 自动标注推理入口支持切换到公司 26 类配置和对应 checkpoint。

一句话理解：

> 公司数据虽然长得像 nuScenes，但类别不是官方 10 类，所以本项目新增了 `CompanyNuScenesDataset`，把公司数据转换成 OpenPCDet 内部统一需要的 `points + gt_boxes + gt_names` 格式。

---

## 2. 推荐先记住的项目目录

项目根目录：

```text
G:\DFAC\WXY\OpenPCDet_ljl
```

最重要的目录和文件：

```text
OpenPCDet_ljl/
├── pcdet/
│   └── datasets/
│       ├── __init__.py
│       └── company_nuscenes/
│           ├── __init__.py
│           ├── company_nuscenes_dataset.py
│           └── company_nuscenes_utils.py
│
├── tools/
│   ├── train.py
│   ├── test.py
│   ├── cfgs/
│   │   ├── dataset_configs/
│   │   │   └── company_nuscenes_dataset.yaml
│   │   └── nuscenes_models/
│   │       ├── company_centerpoint_12cls.yaml
│   │       ├── company_centerpoint_26cls.yaml
│   │       └── company_voxelnext_26cls.yaml
│   │
│   ├── company_nuscenes/
│   │   ├── prepare_company_mini.py
│   │   ├── create_company_infos.py
│   │   ├── check_company_infos.py
│   │   ├── smoke_test_company_dataloader.py
│   │   └── README.md
│   │
│   └── inference/
│       ├── DataSet.py
│       └── inference_nms.py
│
└── data/
    └── company_nuscenes/
        └── v1.0-mini/
```

---

## 3. OpenPCDet 的运行逻辑

OpenPCDet 的训练不是直接读取原始标注文件训练的。它通常分成几个阶段：

```text
原始数据
  ↓
数据适配层 Dataset
  ↓
生成 infos_train.pkl / infos_val.pkl
  ↓
Dataloader 读取点云和标注
  ↓
模型配置构建网络
  ↓
训练 train.py
  ↓
保存 checkpoint
  ↓
测试 test.py 或自动标注 inference
```

对本项目来说，具体是：

```text
公司 nuScenes json + LIDAR_TOP/*.bin
  ↓
CompanyNuScenesDataset / company_nuscenes_utils.py
  ↓
company_nuscenes_infos_train.pkl
company_nuscenes_infos_val.pkl
  ↓
company_centerpoint_26cls.yaml
或 company_voxelnext_26cls.yaml
  ↓
tools/train.py
  ↓
checkpoint_epoch_xx.pth
  ↓
tools/inference/inference_nms.py 自动标注
```

---

## 4. 为什么不能直接用官方 NuScenesDataset

官方 nuScenes 只有 10 个检测类：

```text
car
truck
bus
trailer
construction_vehicle
pedestrian
motorcycle
bicycle
traffic_cone
barrier
```

公司数据是 26 类，而且类别名类似：

```text
human.pedestrian.adult
vehicle.emergency.ambulance
group.vehicle.bicycle
movable_object.pushable_pullable
```

这会影响这些地方：

1. `CLASS_NAMES` 必须改成 26 类。
2. 类别映射必须从公司原始名映射到训练名。
3. 生成 info 时不能丢掉非官方类别。
4. 检测头必须知道一共有 26 类。
5. multi-head 分组必须覆盖全部 26 类。
6. db sampling 如果开启，也必须按 26 类配置。
7. 官方 nuScenes 评估默认不认识这 26 类。

所以本项目没有直接污染官方 `NuScenesDataset`，而是新增了：

```text
pcdet/datasets/company_nuscenes/
```

---

## 5. 公司 26 类标签

### 5.1 原始类别名

这是公司数据 json 里的原始类别体系：

| 序号 | 原始类别名 |
|---:|---|
| 1 | `human.pedestrian.adult` |
| 2 | `human.pedestrian.child` |
| 3 | `human.pedestrian.wheelchair` |
| 4 | `human.pedestrian.stroller` |
| 5 | `human.pedestrian.personal_mobility` |
| 6 | `vehicle.car` |
| 7 | `vehicle.bus.bendy` |
| 8 | `vehicle.bus.rigid` |
| 9 | `vehicle.truck` |
| 10 | `vehicle.construction` |
| 11 | `vehicle.emergency.ambulance` |
| 12 | `vehicle.emergency.police` |
| 13 | `vehicle.trailer` |
| 14 | `movable_object.barrier` |
| 15 | `movable_object.trafficcone` |
| 16 | `movable_object.pushable_pullable` |
| 17 | `movable_object.debris` |
| 18 | `vehicle.emergency.other` |
| 19 | `vehicle.motorcycle` |
| 20 | `vehicle.bicycle` |
| 21 | `group.human.pedestrian` |
| 22 | `group.vehicle.bicycle` |
| 23 | `other` |
| 24 | `animal` |
| 25 | `vehicle.tricycle` |
| 26 | `bicycle` |

### 5.2 训练使用的类别名

OpenPCDet 配置里建议使用更稳定的 snake_case 类别名，所以项目里映射成下面这 26 类：

| 序号 | 原始类别名 | 训练类别名 |
|---:|---|---|
| 1 | `human.pedestrian.adult` | `human_pedestrian_adult` |
| 2 | `human.pedestrian.child` | `human_pedestrian_child` |
| 3 | `human.pedestrian.wheelchair` | `human_pedestrian_wheelchair` |
| 4 | `human.pedestrian.stroller` | `human_pedestrian_stroller` |
| 5 | `human.pedestrian.personal_mobility` | `human_pedestrian_personal_mobility` |
| 6 | `vehicle.car` | `vehicle_car` |
| 7 | `vehicle.bus.bendy` | `vehicle_bus_bendy` |
| 8 | `vehicle.bus.rigid` | `vehicle_bus_rigid` |
| 9 | `vehicle.truck` | `vehicle_truck` |
| 10 | `vehicle.construction` | `vehicle_construction` |
| 11 | `vehicle.emergency.ambulance` | `vehicle_emergency_ambulance` |
| 12 | `vehicle.emergency.police` | `vehicle_emergency_police` |
| 13 | `vehicle.trailer` | `vehicle_trailer` |
| 14 | `movable_object.barrier` | `movable_object_barrier` |
| 15 | `movable_object.trafficcone` | `movable_object_trafficcone` |
| 16 | `movable_object.pushable_pullable` | `movable_object_pushable_pullable` |
| 17 | `movable_object.debris` | `movable_object_debris` |
| 18 | `vehicle.emergency.other` | `vehicle_emergency_other` |
| 19 | `vehicle.motorcycle` | `vehicle_motorcycle` |
| 20 | `vehicle.bicycle` | `vehicle_bicycle` |
| 21 | `group.human.pedestrian` | `group_human_pedestrian` |
| 22 | `group.vehicle.bicycle` | `group_vehicle_bicycle` |
| 23 | `other` | `other` |
| 24 | `animal` | `animal` |
| 25 | `vehicle.tricycle` | `vehicle_tricycle` |
| 26 | `bicycle` | `bicycle` |

### 5.3 类别定义位置

类别映射代码在：

```text
pcdet/datasets/company_nuscenes/company_nuscenes_utils.py
```

重点看这些变量：

```python
COMPANY_RAW_CLASS_NAMES
COMPANY_26_CLASS_NAMES
COMPANY_NAME_MAPPING
```

模型配置里的类别在：

```text
tools/cfgs/nuscenes_models/company_centerpoint_26cls.yaml
tools/cfgs/nuscenes_models/company_voxelnext_26cls.yaml
```

重点看：

```yaml
CLASS_NAMES:
```

以及：

```yaml
MODEL:
  DENSE_HEAD:
    CLASS_NAMES_EACH_HEAD:
```

---

## 6. 数据目录应该长什么样

推荐数据目录：

```text
OpenPCDet_ljl/
└── data/
    └── company_nuscenes/
        └── v1.0-mini/
            ├── sample.json
            ├── sample_data.json
            ├── sample_annotation.json
            ├── calibrated_sensor.json
            ├── ego_pose.json
            ├── scene.json
            ├── log.json
            ├── category.json
            ├── instance.json
            ├── sensor.json
            ├── ImageSets/
            │   ├── train.txt
            │   └── val.txt
            └── samples/
                └── LIDAR_TOP/
                    ├── xxx.bin
                    ├── yyy.bin
                    └── ...
```

如果是完整数据，可以用类似：

```text
data/company_nuscenes/v1.0-trainval/
```

对应地，配置里的版本也要改。

数据集配置文件是：

```text
tools/cfgs/dataset_configs/company_nuscenes_dataset.yaml
```

关键字段：

```yaml
DATASET: CompanyNuScenesDataset
DATA_PATH: ../data/company_nuscenes
VERSION: v1.0-mini
MAX_SWEEPS: 1
LIDAR_POINT_DIM: 4
```

如果你的实际版本目录不是 `v1.0-mini`，需要改：

```yaml
VERSION:
```

或者运行生成 info 时传入对应版本。

---

## 7. 点云格式要求

当前项目默认点云是 `.bin` 文件，每个点 4 个 `float32`：

```text
x y z intensity
```

也就是：

```python
np.fromfile(path, dtype=np.float32).reshape(-1, 4)
```

配置文件中对应：

```yaml
POINT_FEATURE_ENCODING:
  encoding_type: absolute_coordinates_encoding
  used_feature_list: ['x', 'y', 'z', 'intensity']
  src_feature_list: ['x', 'y', 'z', 'intensity']
```

如果你的点云不是 4 维，例如：

```text
x y z
```

或者：

```text
x y z intensity ring timestamp
```

就必须同步修改：

```yaml
LIDAR_POINT_DIM
POINT_FEATURE_ENCODING
```

否则最常见的错误是：

```text
cannot reshape array of size ... into shape (-1,4)
```

---

## 8. 配置文件说明

### 8.1 数据配置

路径：

```text
tools/cfgs/dataset_configs/company_nuscenes_dataset.yaml
```

作用：

1. 指定数据集类：`CompanyNuScenesDataset`
2. 指定数据目录：`../data/company_nuscenes`
3. 指定版本：`v1.0-mini`
4. 指定点云范围。
5. 指定体素大小。
6. 指定点云特征维度。
7. 指定数据增强。
8. 指定 train/val infos 路径。

### 8.2 12 类调试配置

路径：

```text
tools/cfgs/nuscenes_models/company_centerpoint_12cls.yaml
```

用途：

1. mini 数据快速调试。
2. 只训练 mini 里实际常见的一部分类别。
3. 用来验证数据闭环，而不是最终 26 类业务模型。

如果你只是想确认数据能不能读、loss 能不能跑，可以先用这个。

### 8.3 26 类 CenterPoint 配置

路径：

```text
tools/cfgs/nuscenes_models/company_centerpoint_26cls.yaml
```

用途：

1. 推荐的 26 类入门训练配置。
2. 相比 anchor-based SECOND，CenterPoint 对 26 类更友好。
3. 不需要给每个类别手写 anchor size。
4. 更适合先跑通完整 26 类闭环。

训练入口：

```powershell
cd G:\DFAC\WXY\OpenPCDet_ljl\tools

python train.py --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml --batch_size 1 --epochs 20
```

### 8.4 26 类 VoxelNeXt 配置

路径：

```text
tools/cfgs/nuscenes_models/company_voxelnext_26cls.yaml
```

用途：

1. 用于后续更贴近平台自动标注流程。
2. 适合拿训练好的 26 类模型接 `tools/inference/inference_nms.py`。
3. 如果平台原本使用 VoxelNeXt，这个配置更容易接入自动标注。

训练入口：

```powershell
cd G:\DFAC\WXY\OpenPCDet_ljl\tools

python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls.yaml --batch_size 1 --epochs 20
```

---

## 9. 26 类检测头分组

26 类配置里使用了 multi-head 分组。这样做是为了让相近类别共享一个检测头，减少不同类别之间的学习冲突。

当前分组如下。

### 9.1 行人相关

```text
human_pedestrian_adult
human_pedestrian_child
human_pedestrian_wheelchair
human_pedestrian_stroller
human_pedestrian_personal_mobility
group_human_pedestrian
```

### 9.2 机动车相关

```text
vehicle_car
vehicle_bus_bendy
vehicle_bus_rigid
vehicle_truck
vehicle_construction
vehicle_emergency_ambulance
vehicle_emergency_police
vehicle_trailer
vehicle_emergency_other
vehicle_tricycle
```

### 9.3 两轮车/骑行相关

```text
vehicle_motorcycle
vehicle_bicycle
group_vehicle_bicycle
bicycle
```

### 9.4 可移动障碍物

```text
movable_object_barrier
movable_object_trafficcone
movable_object_pushable_pullable
movable_object_debris
```

### 9.5 其他

```text
other
animal
```

注意：

1. 26 个类别必须全部出现。
2. 不能有重复类别。
3. 不能漏类别。
4. `CLASS_NAMES_EACH_HEAD` 里的名字必须和 `CLASS_NAMES` 完全一致。

---

## 10. 从零开始怎么运行

下面按实际操作顺序写。

### 10.1 进入项目

```powershell
cd G:\DFAC\WXY\OpenPCDet_ljl
```

### 10.2 准备环境

建议使用 Linux、WSL2 或 Docker。原生 Windows 跑 OpenPCDet 通常会卡在 CUDA、spconv、iou3d_nms 编译上。

推荐环境大致如下：

```text
Python: 3.8 或 3.9
PyTorch: 与 CUDA 匹配的版本
CUDA: 11.x 或项目环境要求的版本
GPU: NVIDIA GPU
```

训练 3D 检测模型基本需要 GPU。CPU 可以做少量脚本检查，但正式训练不现实。

### 10.3 安装 PyTorch

根据服务器 CUDA 版本安装对应 PyTorch。

示例，仅供参考：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

实际要以服务器 CUDA 版本为准。

### 10.4 安装 OpenPCDet 依赖

常见依赖包括：

```text
numpy
scipy
numba
pyyaml
easydict
tqdm
tensorboardX
opencv-python
scikit-image
SharedArray
nuscenes-devkit
spconv
```

如果项目根目录有 `requirements.txt`，优先执行：

```bash
pip install -r requirements.txt
```

然后安装 OpenPCDet 本体：

```bash
python setup.py develop
```

或者：

```bash
pip install -e .
```

如果这里失败，通常不是代码逻辑问题，而是 CUDA / PyTorch / spconv / 编译器版本不匹配。

---

## 11. 准备公司数据

把数据放到：

```text
OpenPCDet_ljl/data/company_nuscenes/v1.0-mini/
```

至少应包含：

```text
sample.json
sample_data.json
sample_annotation.json
calibrated_sensor.json
ego_pose.json
scene.json
category.json
instance.json
sensor.json
samples/LIDAR_TOP/*.bin
```

如果你的 `sample_data.json` 里写的是 `.pcd`，但真实文件是 `.bin`，需要先修正。项目提供了准备脚本：

```powershell
cd G:\DFAC\WXY\OpenPCDet_ljl

python tools/company_nuscenes/prepare_company_mini.py
```

这个脚本主要做几类事情：

1. 整理 mini 数据。
2. 修正 LiDAR 文件路径。
3. 生成或修正 `ImageSets/train.txt` 和 `ImageSets/val.txt`。
4. 确保后续 info 生成能找到点云文件。

---

## 12. 生成 infos

OpenPCDet 训练前通常要先生成 infos。

命令：

```powershell
cd G:\DFAC\WXY\OpenPCDet_ljl

python tools/company_nuscenes/create_company_infos.py --cfg_file tools/cfgs/dataset_configs/company_nuscenes_dataset.yaml --version v1.0-mini
```

生成后应看到类似文件：

```text
data/company_nuscenes/company_nuscenes_infos_train.pkl
data/company_nuscenes/company_nuscenes_infos_val.pkl
```

这些 pkl 文件里存的是训练需要的索引信息，不是模型权重。

每个 info 里大致包括：

```text
lidar_path
token
sweeps
gt_boxes
gt_names
num_lidar_pts
```

其中：

```text
gt_boxes: x, y, z, dx, dy, dz, heading
gt_names: 26 类里的类别名
```

---

## 13. 检查 infos 是否正确

生成 infos 后，不要立刻训练。先检查。

命令：

```powershell
cd G:\DFAC\WXY\OpenPCDet_ljl

python tools/company_nuscenes/check_company_infos.py
```

它默认按公司 26 类检查。

你希望看到：

```text
num infos > 0
num boxes > 0
lidar_path missing = 0
unknown classes = []
```

如果要严格检查：

```powershell
python tools/company_nuscenes/check_company_infos.py --strict
```

如果这里失败，不要训练。先修数据。

---

## 14. Dataloader 冒烟测试

这个测试用来确认训练前的最后一公里。

命令：

```powershell
cd G:\DFAC\WXY\OpenPCDet_ljl

python tools/company_nuscenes/smoke_test_company_dataloader.py
```

它会尝试构造 dataset 和 dataloader，然后读取一个 batch。

你希望看到：

```text
points shape 正常
gt_boxes shape 正常
gt_boxes 不是空
class_id 在合法范围内
```

如果 dataloader 都过不了，训练一定会失败。

---

## 15. 开始训练

### 15.1 训练 26 类 CenterPoint

```powershell
cd G:\DFAC\WXY\OpenPCDet_ljl\tools

python train.py --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml --batch_size 1 --epochs 20
```

### 15.2 训练 26 类 VoxelNeXt

```powershell
cd G:\DFAC\WXY\OpenPCDet_ljl\tools

python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls.yaml --batch_size 1 --epochs 20
```

### 15.3 训练 12 类调试模型

如果你只想快速确认 mini 闭环：

```powershell
cd G:\DFAC\WXY\OpenPCDet_ljl\tools

python train.py --cfg_file cfgs/nuscenes_models/company_centerpoint_12cls.yaml --batch_size 1 --epochs 5
```

### 15.4 多卡训练

如果服务器有多张 GPU，可以用分布式脚本：

```bash
cd /path/to/OpenPCDet_ljl/tools

bash scripts/dist_train.sh 4 --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml --batch_size 4 --epochs 20
```

这里的 `4` 表示 4 张 GPU。

注意：总 batch size 和显存有关。如果显存不够，先降：

```text
--batch_size
```

再考虑调小：

```text
VOXEL_SIZE
POINT_CLOUD_RANGE
MAX_SWEEPS
```

---

## 16. 训练输出在哪里

OpenPCDet 通常会把输出放到：

```text
output/
```

类似：

```text
output/company_nuscenes_models/company_centerpoint_26cls/default/
├── ckpt/
│   ├── checkpoint_epoch_1.pth
│   ├── checkpoint_epoch_2.pth
│   └── ...
├── tensorboard/
└── log_train_*.txt
```

真正用于推理和自动标注的是：

```text
checkpoint_epoch_xx.pth
```

---

## 17. 测试模型

训练完成后可以用 `test.py`。

示例：

```powershell
cd G:\DFAC\WXY\OpenPCDet_ljl\tools

python test.py --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml --ckpt ../output/xxx/ckpt/checkpoint_epoch_20.pth
```

或者 VoxelNeXt：

```powershell
python test.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls.yaml --ckpt ../output/xxx/ckpt/checkpoint_epoch_20.pth
```

注意：

当前公司 26 类不适合直接套官方 nuScenes 10 类评估。第一阶段建议关注：

1. 是否能正常推理。
2. 是否能输出预测框。
3. 每类预测数量是否合理。
4. 预测框坐标是否在点云范围内。
5. 可视化结果是否大体正确。

如果后续需要正式指标，建议做公司 26 类自定义 AP/Recall 评估。

---

## 18. 自动标注怎么接

自动标注入口相关文件：

```text
tools/inference/DataSet.py
tools/inference/inference_nms.py
```

### 18.1 配置模型路径

`inference_nms.py` 支持通过环境变量指定配置和权重：

```powershell
$env:PCDET_CFG_FILE="G:\DFAC\WXY\OpenPCDet_ljl\tools\cfgs\nuscenes_models\company_voxelnext_26cls.yaml"
$env:PCDET_CKPT_PATH="G:\DFAC\WXY\OpenPCDet_ljl\output\xxx\ckpt\checkpoint_epoch_20.pth"
```

然后再启动你的自动标注推理流程。

### 18.2 为什么推荐自动标注用 VoxelNeXt 配置

因为你平台里原先的自动标注流程更接近 VoxelNeXt 的使用方式。项目里新增的：

```text
company_voxelnext_26cls.yaml
```

就是为了让训练后的 26 类模型更容易接到平台推理入口。

### 18.3 点云维度兼容

以前推理数据读取逻辑会固定往点云里插入两列，导致原始 intensity 没有被正确使用。

现在 `DataSet.py` 里会根据配置自动处理点云维度：

1. 如果点云维度少于配置要求，会补 0。
2. 如果点云维度多于配置要求，会截断。
3. 如果点云本来有 intensity，会尽量保留。

这可以减少训练配置和自动标注输入格式不一致导致的问题。

---

## 19. 重要配置项解释

### 19.1 `CLASS_NAMES`

位置：

```text
tools/cfgs/nuscenes_models/company_centerpoint_26cls.yaml
tools/cfgs/nuscenes_models/company_voxelnext_26cls.yaml
```

含义：

模型要训练和预测的类别列表。

要求：

1. 必须是 26 类。
2. 顺序要稳定。
3. 名字要和 `gt_names` 完全一致。

### 19.2 `CLASS_NAMES_EACH_HEAD`

位置：

```yaml
MODEL:
  DENSE_HEAD:
    CLASS_NAMES_EACH_HEAD:
```

含义：

把 26 类分成多个检测头。

要求：

1. 每个类别必须出现一次。
2. 不能漏。
3. 不能重复。

### 19.3 `POINT_CLOUD_RANGE`

含义：

模型只处理这个空间范围内的点和框。

格式：

```text
[x_min, y_min, z_min, x_max, y_max, z_max]
```

如果范围太小，会丢掉远处目标。

如果范围太大，显存和计算量会上升。

### 19.4 `VOXEL_SIZE`

含义：

点云体素化分辨率。

体素越小，细节越多，但显存和计算量越大。

### 19.5 `MAX_SWEEPS`

当前建议：

```yaml
MAX_SWEEPS: 1
```

原因：

1. 单帧最容易跑通。
2. 多帧 sweep 需要 ego pose、timestamp、坐标变换完全正确。
3. 第一阶段先不要把多帧问题和类别问题混在一起。

后续确认单帧闭环稳定后，可以考虑扩展到：

```yaml
MAX_SWEEPS: 5
```

或：

```yaml
MAX_SWEEPS: 10
```

---

## 20. 当前项目的核心代码职责

### 20.1 `company_nuscenes_utils.py`

路径：

```text
pcdet/datasets/company_nuscenes/company_nuscenes_utils.py
```

职责：

1. 定义公司类别映射。
2. 读取 nuScenes 风格 json。
3. 根据 split 生成 train/val 样本。
4. 解析 LiDAR 路径。
5. 解析标注框。
6. 输出 OpenPCDet 需要的 info 结构。

### 20.2 `company_nuscenes_dataset.py`

路径：

```text
pcdet/datasets/company_nuscenes/company_nuscenes_dataset.py
```

职责：

1. 实现 `CompanyNuScenesDataset`。
2. 读取 pkl infos。
3. 加载点云。
4. 加载 gt_boxes / gt_names。
5. 把数据交给 OpenPCDet 的数据增强、体素化、batch collate 流程。

### 20.3 `pcdet/datasets/__init__.py`

职责：

注册数据集类。

里面应该能找到：

```python
CompanyNuScenesDataset
```

如果没有注册，配置里的：

```yaml
DATASET: CompanyNuScenesDataset
```

会找不到类。

### 20.4 `create_company_infos.py`

路径：

```text
tools/company_nuscenes/create_company_infos.py
```

职责：

调用公司数据适配逻辑生成训练用 infos。

### 20.5 `check_company_infos.py`

路径：

```text
tools/company_nuscenes/check_company_infos.py
```

职责：

检查 infos 是否非空、类别是否合法、点云路径是否存在。

### 20.6 `smoke_test_company_dataloader.py`

路径：

```text
tools/company_nuscenes/smoke_test_company_dataloader.py
```

职责：

真正构造 dataloader，读取一个 batch，确认训练入口前的数据闭环可用。

---

## 21. 推荐工作流

### 21.1 第一次跑 mini

```text
准备数据
  ↓
prepare_company_mini.py
  ↓
create_company_infos.py
  ↓
check_company_infos.py
  ↓
smoke_test_company_dataloader.py
  ↓
company_centerpoint_12cls.yaml 训练 5 epoch
  ↓
确认 loss 能跑
```

### 21.2 跑 26 类训练

```text
确认 26 类数据都在
  ↓
create_company_infos.py
  ↓
check_company_infos.py --strict
  ↓
smoke_test_company_dataloader.py
  ↓
company_centerpoint_26cls.yaml 或 company_voxelnext_26cls.yaml
  ↓
训练
  ↓
test.py
  ↓
可视化/自动标注
```

### 21.3 接自动标注

```text
训练 VoxelNeXt 26 类模型
  ↓
拿到 checkpoint
  ↓
设置 PCDET_CFG_FILE
  ↓
设置 PCDET_CKPT_PATH
  ↓
运行 inference_nms.py 所在的平台流程
  ↓
输出自动标注结果
```

---

## 22. 依赖 GPU 吗

### 22.1 训练

训练基本必须用 NVIDIA GPU。

原因：

1. OpenPCDet 3D 检测模型计算量大。
2. spconv 稀疏卷积主要依赖 CUDA。
3. iou3d_nms 等算子通常需要 CUDA 编译。

CPU 理论上可以跑极少数 Python 脚本，但不适合训练。

### 22.2 可以不用 GPU 做什么

一般可以不用 GPU 做：

1. 检查 json。
2. 统计类别。
3. 生成 split。
4. 部分 info 生成逻辑。
5. 检查 pkl。
6. 静态检查配置。

但只要进入模型训练、模型推理、NMS CUDA 算子，就大概率需要 GPU。

---

## 23. 常见问题

### 23.1 `DATASET CompanyNuScenesDataset not found`

检查：

```text
pcdet/datasets/__init__.py
```

里面是否注册了：

```python
CompanyNuScenesDataset
```

### 23.2 `infos_train.pkl` 是空的

常见原因：

1. split 没匹配上 scene。
2. `ImageSets/train.txt` 里的 scene name 和 `scene.json` 不一致。
3. `VERSION` 配错。
4. 数据目录放错。
5. `sample.json` 和 `sample_data.json` 关联关系异常。

先运行：

```powershell
python tools/company_nuscenes/check_company_infos.py
```

### 23.3 点云路径不存在

常见原因：

1. `sample_data.json` 里是 `.pcd`，实际是 `.bin`。
2. 路径写的是旧目录。
3. `samples/LIDAR_TOP` 大小写不一致。
4. 数据没有复制完整。

### 23.4 类别被过滤没了

常见原因：

1. `sample_annotation.json` 里的 `category_name` 没有写进映射表。
2. 映射后的名字不在 `CLASS_NAMES` 里。
3. 26 类配置和数据层类别不一致。

检查：

```text
pcdet/datasets/company_nuscenes/company_nuscenes_utils.py
tools/cfgs/nuscenes_models/company_centerpoint_26cls.yaml
tools/cfgs/nuscenes_models/company_voxelnext_26cls.yaml
```

### 23.5 `cannot reshape array`

说明点云维度和配置不一致。

检查：

```yaml
LIDAR_POINT_DIM
POINT_FEATURE_ENCODING
```

如果 `.bin` 是 4 维，就应该是：

```yaml
LIDAR_POINT_DIM: 4
src_feature_list: ['x', 'y', 'z', 'intensity']
```

### 23.6 CUDA / spconv 报错

这通常是环境问题，不是数据代码问题。

重点检查：

1. PyTorch CUDA 版本。
2. 本机 CUDA toolkit。
3. GPU 驱动。
4. spconv 版本。
5. OpenPCDet 编译是否成功。

建议在 Linux/WSL2/Docker 里跑。

### 23.7 官方 nuScenes eval 报类别错误

这是正常风险。

官方 nuScenes eval 默认只认 10 类，公司是 26 类。

解决方式：

1. 临时只做预测输出和可视化。
2. 把 26 类映射回官方 10 类再评估。
3. 写公司自定义 26 类评估。

正式业务建议使用第 3 种。

---

## 24. 修改或新增类别时应该改哪里

如果以后 26 类又变了，需要同步改这些地方：

1. 数据层类别映射：

```text
pcdet/datasets/company_nuscenes/company_nuscenes_utils.py
```

2. CenterPoint 训练配置：

```text
tools/cfgs/nuscenes_models/company_centerpoint_26cls.yaml
```

3. VoxelNeXt 训练配置：

```text
tools/cfgs/nuscenes_models/company_voxelnext_26cls.yaml
```

4. 检查脚本默认类别：

```text
tools/company_nuscenes/check_company_infos.py
```

目前检查脚本会从 `company_nuscenes_utils.py` 读取默认 26 类，所以一般只要改映射源头即可。

---

## 25. `.gitignore` 注意事项

当前项目可能忽略了：

```text
*.yaml
data/
```

这意味着：

1. 新增的 yaml 配置文件本地存在，但 `git status` 可能不显示。
2. 如果要提交到 git，可能需要强制添加：

```bash
git add -f tools/cfgs/nuscenes_models/company_centerpoint_26cls.yaml
git add -f tools/cfgs/nuscenes_models/company_voxelnext_26cls.yaml
git add -f tools/cfgs/dataset_configs/company_nuscenes_dataset.yaml
```

3. 数据目录通常不建议提交到 git。

---

## 26. 最小验收标准

在正式训练前，至少满足下面这些条件：

```text
1. data/company_nuscenes/v1.0-mini/ 存在
2. sample.json 等 nuScenes json 存在
3. samples/LIDAR_TOP/*.bin 存在
4. ImageSets/train.txt 和 val.txt 存在
5. company_nuscenes_infos_train.pkl 非空
6. company_nuscenes_infos_val.pkl 非空
7. check_company_infos.py 无 unknown class
8. check_company_infos.py 无 missing lidar path
9. smoke_test_company_dataloader.py 能读出 batch
10. CLASS_NAMES 是 26 类
11. CLASS_NAMES_EACH_HEAD 覆盖全部 26 类
```

只要这些都满足，再开始训练才有意义。

---

## 27. 推荐命令速查

### 准备数据

```powershell
cd G:\DFAC\WXY\OpenPCDet_ljl

python tools/company_nuscenes/prepare_company_mini.py
```

### 生成 infos

```powershell
python tools/company_nuscenes/create_company_infos.py --cfg_file tools/cfgs/dataset_configs/company_nuscenes_dataset.yaml --version v1.0-mini
```

### 检查 infos

```powershell
python tools/company_nuscenes/check_company_infos.py --strict
```

### Dataloader 测试

```powershell
python tools/company_nuscenes/smoke_test_company_dataloader.py
```

### 训练 CenterPoint 26 类

```powershell
cd G:\DFAC\WXY\OpenPCDet_ljl\tools

python train.py --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml --batch_size 1 --epochs 20
```

### 训练 VoxelNeXt 26 类

```powershell
cd G:\DFAC\WXY\OpenPCDet_ljl\tools

python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls.yaml --batch_size 1 --epochs 20
```

### 测试模型

```powershell
python test.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls.yaml --ckpt ../output/xxx/ckpt/checkpoint_epoch_20.pth
```

### 自动标注配置

```powershell
$env:PCDET_CFG_FILE="G:\DFAC\WXY\OpenPCDet_ljl\tools\cfgs\nuscenes_models\company_voxelnext_26cls.yaml"
$env:PCDET_CKPT_PATH="G:\DFAC\WXY\OpenPCDet_ljl\output\xxx\ckpt\checkpoint_epoch_20.pth"
```

---

## 28. 建议的学习顺序

如果你对项目完全不了解，建议按这个顺序看：

1. 先看本文档第 3 节，理解数据流。
2. 再看第 5 节，确认 26 类。
3. 再看第 6 节，确认数据目录。
4. 再看第 8 节，理解配置文件。
5. 然后跑第 12、13、14 节，不要直接训练。
6. 最后再跑第 15 节训练。
7. 训练完成后再看第 18 节接自动标注。

最重要的原则：

> 先确认数据闭环，再确认类别闭环，最后才训练模型。

