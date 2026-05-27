# OpenPCDet_ljl 公司正式 nuScenes 数据 26 类 VoxelNeXt 适配、部署、训练、测试与可视化说明文档

> 项目仓库：`https://github.com/ljl71/OpenPCDet_ljl`  
> 目标分支：`codex/formal-company-voxelnext-26cls`  
> 关键提交：`2d144e7 feat: train formal company data with 26-class VoxelNeXt`  
> 服务器工程目录：`/home/ubuntu/WXY/OpenPCDet_ljl`  
> 容器内工程目录：`/workspace/OpenPCDet`  
> 数据挂载目录：`/home/ubuntu/WXY/data -> /workspace/OpenPCDet/data`  
> 容器名：`detection3d_v5`  
> 本次实验名：`formal_company_26cls`  
> 模型配置：`tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml`  
> 数据配置：`tools/cfgs/dataset_configs/company_nuscenes_trainval_dataset.yaml`  
> 最终模型：`checkpoint_epoch_20.pth`  
> 推荐测试阈值：`SCORE_THRESH=0.2`

---

## 1. 文档目的

这份文档用于说明 `OpenPCDet_ljl` 如何适配公司正式 nuScenes 风格 26 类数据集，并完整记录从代码部署、Docker 容器连接、数据检查、info 生成、训练、测试、阈值分析到 BEV 可视化的全流程。

它适合以下场景：

1. 新同学第一次接手该项目；
2. 需要在服务器重新复现 26 类 VoxelNeXt 训练；
3. 需要知道训练生成了哪些文件、作用是什么、位置在哪里；
4. 需要判断当前模型测试结果是否正常；
5. 需要用 `SCORE_THRESH=0.2` 的结果进行可视化检查；
6. 后续需要把现场修复同步回 GitHub。

---

## 2. 项目目标与整体思路

### 2.1 项目目标

公司数据是 nuScenes 风格数据，但类别体系不是官方 nuScenes 10 类，而是公司自定义 26 类。原版 OpenPCDet 的官方 `NuScenesDataset` 不能直接满足以下需求：

- 识别公司 26 个类别；
- 保留公司原始类别名并映射到训练类别名；
- 读取公司正式目录结构；
- 读取 `samples/LIDAR_TOP/*.bin` 点云；
- 支持公司数据的 scene-level train/val 划分；
- 生成 OpenPCDet 训练所需的 `infos_train.pkl` 和 `infos_val.pkl`；
- 构建 26 类 VoxelNeXt 检测头；
- 在公司正式数据上完成训练、测试和可视化。

因此，本项目在 OpenPCDet 基础上新增了公司数据适配层，使公司正式 nuScenes 风格数据能够进入 OpenPCDet 的训练闭环。

---

### 2.2 整体数据流

整体流程可以理解为：

```text
公司 nuScenes JSON + samples/LIDAR_TOP/*.bin
        ↓
CompanyNuScenesDataset / company_nuscenes_utils.py
        ↓
company_nuscenes_infos_train.pkl
company_nuscenes_infos_val.pkl
        ↓
Dataloader 读取 points + gt_boxes + gt_names
        ↓
VoxelNeXt 26 类模型
        ↓
train.py 训练
        ↓
checkpoint_epoch_*.pth
        ↓
test.py 验证集推理
        ↓
result.pkl + log_eval_*.txt
        ↓
阈值分析 / BEV 可视化 / 后续正式 AP/NDS 评估
```

---

## 3. 公司 26 类数据适配介绍

### 3.1 公司原始 26 类

公司标注中的原始类别名如下：

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

---

### 3.2 训练使用的 26 类名称

OpenPCDet 配置中使用更稳定的 snake_case 类别名：

| 序号 | 训练类别名 |
|---:|---|
| 1 | `human_pedestrian_adult` |
| 2 | `human_pedestrian_child` |
| 3 | `human_pedestrian_wheelchair` |
| 4 | `human_pedestrian_stroller` |
| 5 | `human_pedestrian_personal_mobility` |
| 6 | `vehicle_car` |
| 7 | `vehicle_bus_bendy` |
| 8 | `vehicle_bus_rigid` |
| 9 | `vehicle_truck` |
| 10 | `vehicle_construction` |
| 11 | `vehicle_emergency_ambulance` |
| 12 | `vehicle_emergency_police` |
| 13 | `vehicle_trailer` |
| 14 | `movable_object_barrier` |
| 15 | `movable_object_trafficcone` |
| 16 | `movable_object_pushable_pullable` |
| 17 | `movable_object_debris` |
| 18 | `vehicle_emergency_other` |
| 19 | `vehicle_motorcycle` |
| 20 | `vehicle_bicycle` |
| 21 | `group_human_pedestrian` |
| 22 | `group_vehicle_bicycle` |
| 23 | `other` |
| 24 | `animal` |
| 25 | `vehicle_tricycle` |
| 26 | `bicycle` |

类别映射主要在：

```text
pcdet/datasets/company_nuscenes/company_nuscenes_utils.py
```

相关变量通常包括：

```python
COMPANY_RAW_CLASS_NAMES
COMPANY_26_CLASS_NAMES
COMPANY_NAME_MAPPING
```

模型配置中的类别位于：

```text
tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
```

---

## 4. 26 类适配是怎么实现的

### 4.1 新增公司数据集 Dataset

新增或使用的关键目录：

```text
pcdet/datasets/company_nuscenes/
```

核心文件：

```text
pcdet/datasets/company_nuscenes/company_nuscenes_dataset.py
pcdet/datasets/company_nuscenes/company_nuscenes_utils.py
pcdet/datasets/company_nuscenes/point_io.py
```

作用：

| 文件 | 作用 |
|---|---|
| `company_nuscenes_dataset.py` | 实现 `CompanyNuScenesDataset`，负责读取 info、点云和 GT |
| `company_nuscenes_utils.py` | 负责类别映射、JSON 表解析、scene split、annotation 转换等 |
| `point_io.py` | 负责点云读取逻辑，支持公司正式 `.bin` 点云格式 |

---

### 4.2 正式数据目录结构

本次正式数据使用的目录是：

```text
/workspace/OpenPCDet/data/nuscenes/
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

原因是这个早期目录可能仍引用 `.pcd`，不能和当前正式 `.bin` 点云形成训练闭环。

---

### 4.3 点云格式适配

本次正式数据中，`samples/LIDAR_TOP/*.bin` 每个点按 4 个 `float32` 读取：

```text
x, y, z, intensity
```

但需要注意：第 4 维是转换生成的零占位值，并不是真实 intensity、ring 或 timestamp。

对应配置：

```yaml
LIDAR_POINT_DIM: 4
LIDAR_POINT_FORMAT: float32
LIDAR_POINT_FIELDS: ['x', 'y', 'z', 'intensity']
```

训练后 batch 中的点云形状会变成：

```text
points_shape: (N, 5)
```

这里的 5 维不是原始点云 5 维，而是：

```text
batch_index + x + y + z + intensity占位
```

---

### 4.4 26 类 VoxelNeXt 配置

模型配置文件：

```text
tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
```

数据配置文件：

```text
tools/cfgs/dataset_configs/company_nuscenes_trainval_dataset.yaml
```

关键配置包括：

```yaml
DATASET: CompanyNuScenesDataset
DATA_PATH: ../data/nuscenes
VERSION: v1.0-trainval
MAX_SWEEPS: 1
FILTER_MIN_POINTS_IN_GT: 1
```

模型部分：

```yaml
MODEL:
  NAME: VoxelNeXt
  DENSE_HEAD:
    NAME: VoxelNeXtHead
```

检测头使用 5 个 head 分组：

```text
head 0: 行人和行人组
head 1: 车辆大类
head 2: 摩托车 / 自行车相关
head 3: 可移动障碍物
head 4: other / animal
```

---

### 4.5 多 head 标签修复

为了适配 VoxelNeXt 的 26 类 multi-head 检测，本项目修复了 VoxelNeXt 多 head 标签被原地修改的问题，相关文件：

```text
pcdet/models/dense_heads/voxelnext_head.py
```

该修复保证 26 类在不同 head 中能够正确参与训练，而不是因为 label 被原地改动导致后续 head 标签异常。

---

## 5. 服务器部署代码

### 5.1 宿主机目录

宿主机工作目录：

```text
/home/ubuntu/WXY
```

当前目录内容曾确认包括：

```text
data
OpenPCDet
OpenPCDet_ljl
OpenPCDet_ljl_bak_20260526_152808
OpenPCDetv4
v1.0-trainval
yuan
```

本次实际使用的是：

```text
/home/ubuntu/WXY/OpenPCDet_ljl
```

数据目录：

```text
/home/ubuntu/WXY/data
```

---

### 5.2 上传代码后检查版本

进入项目目录：

```bash
cd /home/ubuntu/WXY/OpenPCDet_ljl
```

检查提交：

```bash
git log -1 --oneline
```

本次正确结果：

```text
2d144e7 (HEAD -> codex/formal-company-voxelnext-26cls, origin/codex/formal-company-voxelnext-26cls) feat: train formal company data with 26-class VoxelNeXt
```

检查分支：

```bash
git branch
```

正确结果：

```text
* codex/formal-company-voxelnext-26cls
  main
```

检查关键文件：

```bash
find /home/ubuntu/WXY/OpenPCDet_ljl -name "company_voxelnext_26cls_trainval.yaml"
find /home/ubuntu/WXY/OpenPCDet_ljl -name "company_nuscenes_trainval_dataset.yaml"
find /home/ubuntu/WXY/OpenPCDet_ljl -name "preview_formal_split.py"
find /home/ubuntu/WXY/OpenPCDet_ljl -name "smoke_test_formal_voxelnext.py"
```

期望存在：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
/home/ubuntu/WXY/OpenPCDet_ljl/tools/cfgs/dataset_configs/company_nuscenes_trainval_dataset.yaml
/home/ubuntu/WXY/OpenPCDet_ljl/tools/company_nuscenes/preview_formal_split.py
/home/ubuntu/WXY/OpenPCDet_ljl/tools/company_nuscenes/smoke_test_formal_voxelnext.py
```

---

## 6. Docker 容器连接与 GPU 部署

### 6.1 容器挂载关系

本次容器名：

```text
detection3d_v5
```

检查挂载：

```bash
sudo docker inspect detection3d_v5 --format='{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

正确挂载关系：

```text
/home/ubuntu/WXY/data -> /workspace/OpenPCDet/data
/home/ubuntu/WXY/OpenPCDet_ljl -> /workspace/OpenPCDet
```

含义：

| 宿主机 | 容器内 |
|---|---|
| `/home/ubuntu/WXY/OpenPCDet_ljl` | `/workspace/OpenPCDet` |
| `/home/ubuntu/WXY/data` | `/workspace/OpenPCDet/data` |

---

### 6.2 为什么要重建带 GPU 的容器

最开始旧容器没有加：

```bash
--gpus all
```

导致 GPU smoke test 报：

```text
RuntimeError: Found no NVIDIA driver on your system.
```

这说明容器内部没有 GPU 权限。

删除旧容器不会删除代码和数据，因为代码和数据都是宿主机目录挂载进去的。

---

### 6.3 宿主机检查 GPU

退出容器后，在宿主机执行：

```bash
nvidia-smi
```

本次服务器有两张 RTX 3090：

```text
GPU 0: NVIDIA GeForce RTX 3090
GPU 1: NVIDIA GeForce RTX 3090
```

因为 GPU 0 占用较多，后续训练和测试主要使用：

```bash
CUDA_VISIBLE_DEVICES=1
```

---

### 6.4 重建带 GPU 的容器

停掉旧容器：

```bash
sudo docker stop detection3d_v5
```

删除旧容器：

```bash
sudo docker rm detection3d_v5
```

重新创建：

```bash
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

容器内确认 GPU：

```bash
nvidia-smi
```

确认代码版本：

```bash
git log -1 --oneline
```

应看到：

```text
2d144e7 feat: train formal company data with 26-class VoxelNeXt
```

---

## 7. 正式数据检查

### 7.1 检查目录

进入容器工程根目录：

```bash
cd /workspace/OpenPCDet
```

检查正式数据目录：

```bash
test -d data/nuscenes/v1.0-trainval && echo "v1.0-trainval OK"
test -d data/nuscenes/samples/LIDAR_TOP && echo "LIDAR_TOP OK"
```

本次结果：

```text
v1.0-trainval OK
LIDAR_TOP OK
```

---

### 7.2 检查 JSON 数量

执行：

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

本次结果：

```text
category            : 26
scene               : 412
sample              : 24142
sample_data         : 241420
sample_annotation   : 1222229
instance            : 53246
```

说明正式数据元信息完整。

---

### 7.3 检查 LIDAR_TOP 点云是否完整

按 `sample_data.json` 中真实引用路径检查：

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

本次结果：

```text
LIDAR_TOP records: 24142
Missing LIDAR_TOP files: 0
suffix: Counter({'.bin': 24142})
parents:
samples/LIDAR_TOP 24142
```

说明标注引用的点云文件全部存在。

---

## 8. train/val split 预览

执行：

```bash
python tools/company_nuscenes/preview_formal_split.py --data_path data/nuscenes --version v1.0-trainval --train_ratio 0.8 --seed 0 --min_lidar_points 1
```

本次结果：

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

- 按 scene 做 80%/20% 划分；
- 训练集 329 个 scene；
- 验证集 83 个 scene；
- 训练样本 19268；
- 验证样本 4874；
- 所有可学习类别都进入训练集；
- `vehicle.emergency.other` 没进入验证集是合理的，因为该类有效样本只出现在单个 scene，优先保留在训练集。

---

## 9. 生成 info 文件

### 9.1 生成命令

执行：

```bash
python tools/company_nuscenes/create_company_infos.py --data_path data/nuscenes --save_path data/nuscenes --version v1.0-trainval --max_sweeps 1 --train_ratio 0.8 --seed 0 --min_lidar_points 1
```

本次结果：

```text
Company nuScenes train infos: 19268
Company nuScenes val infos: 4874
Saved: data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl
Saved: data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
```

---

### 9.2 生成文件位置

容器内：

```text
/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl
/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
```

宿主机：

```text
/home/ubuntu/WXY/data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl
/home/ubuntu/WXY/data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
```

文件大小：

```text
company_nuscenes_infos_train.pkl  332M
company_nuscenes_infos_val.pkl     86M
```

---

### 9.3 info 的作用

info 文件是 OpenPCDet 的训练索引缓存，里面包含：

- 样本 token；
- 点云路径；
- train/val 归属；
- GT 3D box；
- GT 类别名；
- 类别映射结果；
- `num_lidar_pts`；
- dataloader 读取所需元信息。

如果没有这两个 info 文件，训练脚本不知道有哪些样本、每个样本点云在哪里、有哪些框、每个框属于哪个类别。

---

## 10. 检查 info 文件

执行：

```bash
python tools/company_nuscenes/check_company_infos.py --root data/nuscenes/v1.0-trainval --data_root data/nuscenes --strict --min_lidar_points 1
```

本次关键结果：

```text
company_nuscenes_infos_train.pkl
  samples: 19268
  empty_gt_samples: 0
  invalid_num_lidar_pts_samples: 0
  boxes_with_unknown_lidar_pts_kept: 485
  missing_lidar_paths: 0
  classes: 24
  retained_boxes_min_points_1: 711959

company_nuscenes_infos_val.pkl
  samples: 4874
  empty_gt_samples: 0
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

判断：

- `missing_lidar_paths: 0`：点云路径没有缺失；
- `outside_config: []`：没有配置外类别；
- `annotated_but_absent_from_train: []`：训练集覆盖所有可学习类别；
- `annotated_but_unlearnable_after_train_filter: []`：过滤后没有不可学习类别；
- `human_pedestrian_personal_mobility` 和 `vehicle_bus_bendy` 当前没有标注框，属于数据本身缺失，不是代码错误。

---

## 11. 现场修复项

### 11.1 修复 `av2` 缺包

问题：

```text
ModuleNotFoundError: No module named 'av2'
```

原因：`pcdet/datasets/__init__.py` 强制导入 Argo2 数据集，但本次不用 Argo2。

备份：

```bash
cp pcdet/datasets/__init__.py pcdet/datasets/__init__.py.bak_av2
```

修复：

```bash
python -c "from pathlib import Path; p=Path('pcdet/datasets/__init__.py'); s=p.read_text(); old='from .argo2.argo2_dataset import Argo2Dataset'; new='try:\n    from .argo2.argo2_dataset import Argo2Dataset\nexcept ModuleNotFoundError:\n    Argo2Dataset = None'; assert old in s, 'target import line not found'; p.write_text(s.replace(old,new))"
```

作用：避免无关数据集依赖阻断公司 nuScenes 训练。

---

### 11.2 修复训练保存逻辑

问题：

```text
TypeError: unsupported operand type(s) for //: 'float' and 'NoneType'
```

原因：`ckpt_save_time_interval` 是 `None`，但代码直接做整除。

备份：

```bash
cd /workspace/OpenPCDet/tools
cp train_utils/train_utils.py train_utils/train_utils.py.bak_ckpt_interval
```

修复：

```bash
python -c "from pathlib import Path; p=Path('train_utils/train_utils.py'); s=p.read_text(); old='time_past_this_epoch // ckpt_save_time_interval'; new='time_past_this_epoch // (ckpt_save_time_interval or 1e18)'; assert old in s, 'target not found'; p.write_text(s.replace(old,new))"
```

作用：如果未设置按时间保存 checkpoint，就禁用按时间保存，仅保留按 epoch 保存。

---

### 11.3 修复 `test.py` 与 `eval_utils.py` 接口不匹配

第一次 evaluation 报错：

```text
TypeError: eval_one_epoch() got an unexpected keyword argument 'save_to_file'
```

删除 `save_to_file=args.save_to_file`：

```bash
cp test.py test.py.bak_save_to_file
python -c "from pathlib import Path; p=Path('test.py'); s=p.read_text(); old=', save_to_file=args.save_to_file'; assert old in s, 'target not found'; p.write_text(s.replace(old,''))"
```

随后确认 `eval_utils.py` 函数签名：

```bash
grep -n "def eval_one_epoch" eval_utils/eval_utils.py
```

结果：

```text
def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
```

因此 `test.py` 中两处调用需要补 `args`：

```bash
cp test.py test.py.bak_eval_args
python -c "from pathlib import Path; p=Path('test.py'); s=p.read_text(); s=s.replace('cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,', 'cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,'); s=s.replace('cfg, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,', 'cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,'); p.write_text(s)"
```

作用：使 `test.py` 与当前 `eval_utils.eval_one_epoch()` 的接口对齐。

---

## 12. Dataloader smoke test

执行：

```bash
python /workspace/OpenPCDet/tools/company_nuscenes/smoke_test_company_dataloader.py --cfg_file /workspace/OpenPCDet/tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
```

本次结果：

```text
Loading CompanyNuScenes dataset
Total samples for CompanyNuScenes dataset: 19268
dataset_len: 19268
batch_keys: ['batch_size', 'flip_x', 'flip_y', 'frame_id', 'gt_boxes', 'metadata', 'noise_rot', 'noise_scale', 'points', 'use_lead_xyz', 'voxel_coords', 'voxel_num_points', 'voxels']
points_shape: (201843, 5)
gt_boxes_shape: (1, 78, 8)
```

判断：

- `dataset_len=19268`：训练集读取正确；
- `points_shape` 第二维为 5：`batch_index + x + y + z + intensity占位`；
- `gt_boxes_shape` 最后一维为 8：`x,y,z,dx,dy,dz,yaw,class_id`；
- dataloader 链路正常。

---

## 13. GPU smoke test

执行：

```bash
CUDA_VISIBLE_DEVICES=1 python /workspace/OpenPCDet/tools/company_nuscenes/smoke_test_formal_voxelnext.py --cfg_file /workspace/OpenPCDet/tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --workers 0
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

判断：

- 26 类 VoxelNeXt 构建成功；
- CUDA 可用；
- spconv 可用；
- forward 正常；
- loss 正常；
- backward 正常。

---

## 14. 正式训练

进入 `tools` 目录：

```bash
cd /workspace/OpenPCDet/tools
```

---

### 14.1 首次跑通时使用的保守训练命令

本项目第一次正式跑通时，为了最大程度降低显存风险，使用的是较保守的 `batch_size=1`：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --batch_size 1 --epochs 20 --workers 4 --extra_tag formal_company_26cls --ckpt_save_interval 1 --max_ckpt_save_num 20
```

该命令的作用是先验证完整训练链路，包括：

```text
正式数据读取
→ dataloader
→ 26 类 VoxelNeXt
→ forward / backward
→ loss 下降
→ checkpoint 保存
→ epoch20 模型生成
```

本次 `batch_size=1` 的训练已经完整完成，训练耗时约：

```text
20 epochs: 11:17:45
单 epoch: 约 33～34 分钟
```

最终训练日志：

```text
epochs: 100%|████████| 20/20 [11:17:45<00:00, 2033.27s/it, loss=1.53, lr=3e-8]
End training nuscenes_models/company_voxelnext_26cls_trainval(formal_company_26cls)
```

说明 `batch_size=1` 版本已经成功跑通，并生成了完整的 20 个 checkpoint。

---

### 14.2 为什么后续推荐改用 `batch_size=8`

服务器使用的是 RTX 3090，单卡显存为 24GB。本次主要使用 GPU 1 进行训练：

```text
CUDA_VISIBLE_DEVICES=1
```

在第一次使用 `batch_size=1` 跑通后，又对更大的 batch size 做了测试。实际观察到：

```text
batch_size=4 时，GPU 1 显存约 5.1GB / 24GB
batch_size=8 时，GPU 1 显存约 8.6GB / 24GB
```

这说明 `batch_size=1` 对当前服务器来说明显偏保守，`batch_size=8` 在显存上仍然比较安全。

`batch_size=8, workers=4` 的训练速度大约为：

```text
2409 it/epoch
约 2.27 it/s
单 epoch 约 18 分钟
```

虽然 `it/s` 比 `batch_size=1` 低，但每个 iteration 处理 8 帧数据，因此实际样本吞吐更高：

```text
batch_size=1: 约 9.46 samples/s
batch_size=8: 约 2.27 × 8 ≈ 18.16 samples/s
```

因此，`batch_size=8` 的实际训练吞吐大约是 `batch_size=1` 的接近 2 倍。

此外，`batch_size=8, workers=4` 时日志中：

```text
d_time=0.00(0.01)
f_time≈0.43~0.53
b_time≈0.43~0.53
```

其中：

| 字段 | 含义 | 当前判断 |
|---|---|---|
| `d_time` | 数据加载耗时 | 基本为 0，说明 dataloader 不是主要瓶颈 |
| `f_time` | forward 前向耗时 | 主要计算耗时之一 |
| `b_time` | backward 反向耗时 | 主要计算耗时之一 |

这说明当前主要耗时在模型计算，而不是数据加载。因此 `workers=4` 已经够用，暂时没有必要强行提高到 `workers=8`。

---

### 14.3 当前推荐正式训练命令

在当前服务器和数据配置下，推荐后续正式训练使用：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --batch_size 8 --epochs 20 --workers 4 --extra_tag formal_company_26cls_bs8 --ckpt_save_interval 1 --max_ckpt_save_num 20
```

参数说明：

| 参数 | 推荐值 | 说明 |
|---|---:|---|
| `CUDA_VISIBLE_DEVICES` | `1` | 使用第二张 RTX 3090，避开 GPU 0 上的桌面和其他进程 |
| `batch_size` | `8` | 当前服务器上显存安全，吞吐明显高于 `batch_size=1` |
| `epochs` | `20` | 与首次跑通实验保持一致，方便对比 |
| `workers` | `4` | 当前 `d_time` 很低，数据加载不是瓶颈，暂不必增加 |
| `extra_tag` | `formal_company_26cls_bs8` | 新实验单独保存，避免覆盖首次跑通结果 |
| `ckpt_save_interval` | `1` | 每个 epoch 保存一次 checkpoint |
| `max_ckpt_save_num` | `20` | 最多保留 20 个 checkpoint |

---

### 14.4 新训练结果会保存在哪里

使用：

```text
--extra_tag formal_company_26cls_bs8
```

后，新的训练输出目录会变成：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_bs8/
```

宿主机对应路径：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_bs8/
```

checkpoint 会保存在：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_bs8/ckpt/
```

不会覆盖之前 `batch_size=1` 首次跑通的实验目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/
```

因此可以同时保留：

```text
formal_company_26cls      # 首次 batch_size=1 跑通结果
formal_company_26cls_bs8  # 推荐 batch_size=8 新训练结果
```

---

### 14.5 是否需要使用 `workers=8`

当前不强制推荐。

原因是 `batch_size=8, workers=4` 时：

```text
d_time=0.00(0.01)
```

说明数据加载时间非常低，dataloader 基本没有拖慢训练。此时把 `workers` 从 4 提到 8，收益可能不明显，甚至可能增加 CPU 调度开销。

如果后续想进一步比较，也可以单独跑一个 1 epoch 的测试：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --batch_size 8 --epochs 1 --workers 8 --extra_tag formal_company_26cls_bs8_w8_test --ckpt_save_interval 1 --max_ckpt_save_num 2
```

比较重点：

```text
1. it/s 是否明显提升；
2. 单 epoch 时间是否缩短；
3. d_time 是否下降；
4. CPU 是否明显卡顿；
5. 训练是否稳定。
```

如果 `workers=8` 没有明显提升，就继续使用：

```text
batch_size=8, workers=4
```

---

### 14.6 是否需要尝试 `batch_size=16`

可以尝试，但不建议直接作为默认正式训练配置。

原因是 3D 点云训练的显存占用会随着不同 batch 中点数、voxel 数和 GT 数量波动。虽然当前 `batch_size=8` 只占用约 8.6GB 显存，但 `batch_size=16` 在某些点云密集 batch 上可能出现更高峰值。

如果要测试，可以先跑 1 个 epoch：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --batch_size 16 --epochs 1 --workers 4 --extra_tag formal_company_26cls_bs16_test --ckpt_save_interval 1 --max_ckpt_save_num 2
```

判断标准：

```text
显存 < 18GB：可以考虑
显存 18GB～21GB：能跑但需要谨慎
显存 > 21GB：不建议长期训练，容易在个别 batch OOM
```

当前综合稳定性和训练速度，默认推荐仍然是：

```text
batch_size=8, workers=4
```

---

### 14.7 学习率是否需要跟着 batch size 修改

暂时不建议立刻修改学习率。

虽然从理论上讲，batch size 从 1 增大到 8 后，可以进一步研究学习率缩放，但当前阶段主要目标是获得一个稳定、可复现的公司 26 类 baseline。为了避免同时改变多个变量，建议：

```text
先只修改 batch_size，不改学习率；
确认 batch_size=8 能稳定完整训练；
再根据 loss 曲线和验证结果决定是否调学习率。
```

因此当前推荐命令仍然保持原配置中的学习率，只修改：

```text
batch_size: 1 -> 8
extra_tag: formal_company_26cls -> formal_company_26cls_bs8
```

---

### 14.8 当前训练配置结论

当前阶段推荐结论：

```text
首次跑通记录：batch_size=1, workers=4, extra_tag=formal_company_26cls
后续正式推荐：batch_size=8, workers=4, extra_tag=formal_company_26cls_bs8
暂不强制使用：workers=8
暂不默认使用：batch_size=16
暂不修改：学习率
```

一句话总结：

> `batch_size=1` 适合首次跑通链路，但对 RTX 3090 24GB 来说偏保守；实测 `batch_size=8` 显存仍然安全，训练吞吐明显更高，因此后续正式训练建议使用 `batch_size=8, workers=4`。



## 15. 训练生成的文件

### 15.1 checkpoint 目录

容器内：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/
```

宿主机：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/
```

本次生成：

```text
checkpoint_epoch_1.pth
checkpoint_epoch_2.pth
...
checkpoint_epoch_20.pth
```

最终模型：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth
```

每个 checkpoint 约：

```text
90M
```

总计约：

```text
1.8G
```

---

### 15.2 训练输出根目录

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/
```

该目录由：

```text
cfg.EXP_GROUP_PATH: nuscenes_models
cfg.TAG: company_voxelnext_26cls_trainval
extra_tag: formal_company_26cls
```

共同决定。

---

## 16. 测试 checkpoint

测试第 20 轮模型：

```bash
cd /workspace/OpenPCDet/tools
CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --ckpt ../output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth --batch_size 1 --workers 4 --extra_tag formal_company_26cls --eval_tag epoch20
```

测试输出目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/
```

生成文件：

```text
log_eval_20260527-004119.txt
log_eval_20260527-004719.txt
result.pkl
```

其中：

| 文件 | 作用 |
|---|---|
| `log_eval_20260527-004119.txt` | 第一次测试日志，包含接口报错记录 |
| `log_eval_20260527-004719.txt` | 最终成功测试日志 |
| `result.pkl` | 验证集预测结果 |

---

## 17. 原始测试结果：`SCORE_THRESH=0.1`

配置默认：

```text
SCORE_THRESH: 0.1
```

测试结果：

```text
recall_rcnn_0.3: 0.724822
recall_rcnn_0.5: 0.517463
recall_rcnn_0.7: 0.222161
Average predicted number of objects(4874 samples): 103.799
```

主要类别：

```text
human_pedestrian_adult: gt=33740, pred=102300
vehicle_car: gt=136394, pred=236430
vehicle_motorcycle: gt=33999, pred=90677
vehicle_bicycle: gt=13635, pred=6126
vehicle_tricycle: gt=8661, pred=23255
bicycle: gt=2003, pred=1495
```

判断：

- recall 最高；
- 但平均每帧预测框 103.799，明显偏多；
- 存在较大误检风险；
- 不建议作为最终默认测试阈值。

---

## 18. 后处理阈值分析

### 18.1 `SCORE_THRESH=0.2`

测试命令：

```bash
CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --ckpt ../output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth --batch_size 1 --workers 4 --extra_tag formal_company_26cls --eval_tag epoch20_score020 --set MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH 0.2
```

输出目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/
```

结果：

```text
recall_rcnn_0.3: 0.657681
recall_rcnn_0.5: 0.495375
recall_rcnn_0.7: 0.220149
Average predicted number of objects(4874 samples): 45.709
```

主要类别：

```text
human_pedestrian_adult: gt=33740, pred=36289
vehicle_car: gt=136394, pred=120137
vehicle_motorcycle: gt=33999, pred=37300
vehicle_tricycle: gt=8661, pred=8513
vehicle_bicycle: gt=13635, pred=3154
bicycle: gt=2003, pred=560
```

判断：

- 平均预测框从 103.799/frame 降到 45.709/frame；
- recall@0.5 从 0.517463 小幅下降到 0.495375；
- recall@0.7 基本不变；
- 整体更平衡。

---

### 18.2 `SCORE_THRESH=0.25`

测试命令：

```bash
CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --ckpt ../output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth --batch_size 1 --workers 4 --extra_tag formal_company_26cls --eval_tag epoch20_score025 --set MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH 0.25
```

输出目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score025/
```

结果：

```text
recall_rcnn_0.3: 0.624331
recall_rcnn_0.5: 0.481173
recall_rcnn_0.7: 0.218387
Average predicted number of objects(4874 samples): 36.444
```

主要类别：

```text
human_pedestrian_adult: gt=33740, pred=26925
vehicle_car: gt=136394, pred=100697
vehicle_motorcycle: gt=33999, pred=28645
vehicle_tricycle: gt=8661, pred=6138
vehicle_bicycle: gt=13635, pred=2537
bicycle: gt=2003, pred=394
```

判断：

- 平均预测数最接近验证集 GT 平均数；
- 框更干净；
- 但弱类和长尾类被压得更明显；
- 更适合作为“偏保守展示阈值”。

---

### 18.3 阈值对比表

| SCORE_THRESH | recall@0.3 | recall@0.5 | recall@0.7 | 平均预测数/帧 | 结果倾向 |
|---:|---:|---:|---:|---:|---|
| 0.1 | 0.724822 | 0.517463 | 0.222161 | 103.799 | 召回高，但预测框明显过多 |
| 0.2 | 0.657681 | 0.495375 | 0.220149 | 45.709 | 比较均衡 |
| 0.25 | 0.624331 | 0.481173 | 0.218387 | 36.444 | 框数最干净，但偏保守 |

验证集 GT 总数：

```text
183358
```

验证样本数：

```text
4874
```

GT 平均数量：

```text
183358 / 4874 ≈ 37.6/frame
```

推荐：

```text
默认推荐值：SCORE_THRESH=0.2
偏保守展示：SCORE_THRESH=0.25
不建议继续使用：SCORE_THRESH=0.1
```

理由：

1. `0.2` 将预测框数量大幅降到合理范围；
2. `0.2` 的 recall@0.5 仍接近 0.5；
3. `0.2` 的 recall@0.7 基本保持；
4. `0.25` 对弱类压制明显；
5. 当前评估不是正式 AP/NDS，不能只看平均预测数是否接近 GT。

---

## 19. 当前测试指标的含义和限制

日志明确说明：

```text
CompanyNuScenes smoke evaluation
This first-stage evaluation only reports GT/pred counts; AP is not implemented yet.
```

因此当前测试能说明：

- 模型能输出预测；
- 验证集推理能完整跑完；
- recall 有一定参考价值；
- 每类 GT/pred 数量可用于判断是否爆框或空输出。

但不能说明：

- mAP；
- NDS；
- 每类 AP；
- precision；
- 真实部署效果。

当前结论应该是：

```text
公司正式数据的 26 类 VoxelNeXt 训练、测试、可视化链路已经跑通；模型在验证集上不是空输出，SCORE_THRESH=0.2 时预测框数量和 recall 相对均衡。但当前 evaluation 还不是正式 AP/NDS，需要后续补充正式评估或人工可视化抽检。
```

---

## 20. BEV 可视化

### 20.1 使用哪个结果文件

推荐使用 `SCORE_THRESH=0.2` 的预测结果：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl
```

该文件大小约：

```text
40M
```

---

### 20.2 检查字段

字段检查结果：

```text
pred type: <class 'list'> len: 4874
info type: <class 'list'> len: 4874
```

预测字段：

```text
name
score
boxes_lidar
pred_labels
frame_id
metadata
```

GT/info 字段：

```text
lidar_path
token
scene_token
timestamp
sweeps
gt_boxes
gt_names
gt_raw_names
gt_boxes_token
num_lidar_pts
num_radar_pts
lidar_path_exists
```

第一帧：

```text
pred boxes_lidar: (44, 7)
gt_boxes: (52, 7)
lidar_path: samples/LIDAR_TOP/1767948279422776259.bin
```

说明可视化脚本可以直接读取预测框、GT 框和点云。

---

### 20.3 可视化脚本

脚本位置：

```text
/workspace/OpenPCDet/tools/company_nuscenes/visualize_score020_bev.py
```

作用：

- 读取 `epoch20_score020/result.pkl`；
- 读取 `company_nuscenes_infos_val.pkl`；
- 读取点云 `samples/LIDAR_TOP/*.bin`；
- 生成 BEV PNG；
- 绿色框表示 GT；
- 红色框表示预测；
- 灰色点表示点云；
- 黄色短线表示预测框朝向；
- 青色短线表示 GT 框朝向。

---

### 20.4 生成前 20 帧 BEV 图

进入项目根目录：

```bash
cd /workspace/OpenPCDet
```

执行：

```bash
python tools/company_nuscenes/visualize_score020_bev.py --result output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl --infos data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl --data_root data/nuscenes --out_dir output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev --score_thresh 0.2 --num 20
```

本次实际结果：

```text
Saved 20 BEV visualizations to: output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev
Summary: output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev/summary.txt
```

---

### 20.5 可视化输出目录

容器内：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev/
```

宿主机：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev/
```

目录下包含：

```text
bev_idx_*.png
summary.txt
```

查看图片：

```bash
find output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev -name "*.png" | sort | head -20
```

查看汇总：

```bash
cat output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev/summary.txt
```

---

### 20.6 指定样本可视化

例如可视化第 0、10、100、500 帧：

```bash
python tools/company_nuscenes/visualize_score020_bev.py --result output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl --infos data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl --data_root data/nuscenes --out_dir output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev_selected --score_thresh 0.2 --indices 0 10 100 500
```

---

### 20.7 打包可视化图片

```bash
tar -czf output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev.tar.gz -C output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis epoch20_score020_bev
```

压缩包：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev.tar.gz
```

---

## 21. 最终产物总览

### 21.1 数据准备产物

```text
/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl
/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/ImageSets/train.txt
/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/ImageSets/val.txt
```

### 21.2 模型产物

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_1.pth
...
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth
```

最终模型：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth
```

### 21.3 测试产物

原始阈值：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/
```

推荐阈值 0.2：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/
```

保守阈值 0.25：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score025/
```

### 21.4 可视化产物

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev/
```

包含：

```text
bev_idx_*.png
summary.txt
```

### 21.5 现场修复备份文件

```text
/workspace/OpenPCDet/pcdet/datasets/__init__.py.bak_av2
/workspace/OpenPCDet/tools/train_utils/train_utils.py.bak_ckpt_interval
/workspace/OpenPCDet/tools/test.py.bak_save_to_file
/workspace/OpenPCDet/tools/test.py.bak_logger
/workspace/OpenPCDet/tools/test.py.bak_eval_args
```

---

## 22. 哪些文件建议提交 GitHub

建议提交源码修复：

```text
pcdet/datasets/__init__.py
tools/train_utils/train_utils.py
tools/test.py
tools/company_nuscenes/visualize_score020_bev.py
```

对应修复：

1. Argo2 的 `av2` 缺包导致无关导入失败；
2. `ckpt_save_time_interval=None` 导致训练保存逻辑报错；
3. `test.py` 与 `eval_utils.py` 接口不匹配；
4. 新增 `SCORE_THRESH=0.2` BEV 可视化工具脚本。

不建议提交：

```text
*.pkl
*.pth
output/
__pycache__/
*.pyc
*.bak_*
missing_lidar_top_files.txt
vis/*.png
```

---

## 23. 一键检查关键文件

执行：

```bash
echo "===== info files ====="; ls -lh /workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl /workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl; echo "===== checkpoints ====="; ls -lh /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/; echo "===== eval score010 ====="; ls -lh /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/; echo "===== eval score020 ====="; ls -lh /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/; echo "===== eval score025 ====="; ls -lh /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score025/; echo "===== vis score020 ====="; ls -lh /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev/
```

---

## 24. 当前结论

本次项目已经完成：

```text
公司正式 nuScenes 数据读取
→ 26 类类别映射
→ scene-level train/val split
→ info 生成
→ info 检查
→ dataloader smoke
→ VoxelNeXt GPU smoke
→ 20 epoch 正式训练
→ checkpoint 保存
→ epoch20 验证集测试
→ SCORE_THRESH 阈值分析
→ SCORE_THRESH=0.2 BEV 可视化
```

工程层面已经跑通。

模型层面当前结论：

```text
1. 模型不是空输出，验证集推理能完整完成；
2. SCORE_THRESH=0.1 预测框太多；
3. SCORE_THRESH=0.2 是当前更均衡的默认阈值；
4. SCORE_THRESH=0.25 更干净但偏保守；
5. 当前 evaluation 不是正式 AP/NDS，只能作为 smoke evaluation；
6. 后续应补 AP/NDS 或公司内部指标，并结合 BEV 可视化人工抽检。
```

推荐使用：

```text
默认测试阈值：SCORE_THRESH=0.2
最终模型：checkpoint_epoch_20.pth
重点查看：epoch20_score020/result.pkl
重点可视化：vis/epoch20_score020_bev/
```

---

## 25. 后续建议

1. 将现场源码修复提交回 GitHub；
2. 将 `SCORE_THRESH=0.2` 写入实验说明或配置注释；
3. 抽样查看 BEV 可视化图片，判断是否存在明显空框、重复框、方向错误；
4. 补充正式 AP/NDS 或公司内部评估指标；
5. 后续考虑分类别阈值：
   - 高频类如 `vehicle_car`、`human_pedestrian_adult`、`vehicle_motorcycle` 可用 `0.2~0.25`；
   - 弱类如 `vehicle_bicycle`、`bicycle`、`animal` 可保留 `0.15~0.2`；
6. 如果未来获得真实 intensity、ring 或 timestamp，应重新检查点云特征配置，避免继续使用零占位第 4 维作为强度。
