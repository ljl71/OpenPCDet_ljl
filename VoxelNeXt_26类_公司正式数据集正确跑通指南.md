# OpenPCDet_ljl 使用公司正式数据训练 26 类 VoxelNeXt 指南

> 更新时间：2026-05-26  
> 适用工程：`/home/ubuntu/WXY/OpenPCDet_ljl` 挂载为容器内 `/workspace/OpenPCDet`  
> 唯一目标：使用公司正式 `v1.0-trainval` 数据训练 26 类 `VoxelNeXt`。本文不使用调试数据流程。

## 1. 当前代码能否实现目标

修改后的 `OpenPCDet_ljl` 可以建立并训练一个 **26 输出类别的 VoxelNeXt 模型**，正式训练入口为：

```text
tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
```

这条链路已经具备：

1. `CompanyNuScenesDataset` 读取正式数据的 sibling 目录结构。
2. 正式点云 `.bin` 按每点 4 个 `float32` 读取。
3. 原始 26 类名称到训练类别名称的固定映射。
4. VoxelNeXt 的 26 类检测 head。
5. 单批次前向、loss 与反向传播 smoke 入口。
6. info 文件、点云存在性、类别合法性和过滤后有效 GT 检查。

但必须准确理解两个数据限制：

1. 正式标注 `category.json` 定义 26 类，当前 `sample_annotation.json` 实际有正样本的只有 24 类。  
   `human_pedestrian_personal_mobility` 与 `vehicle_bus_bendy` 没有标注框。模型仍会产生 26 类输出，但这两个类别无法从当前数据学到检测能力。
2. 已核验的公司转换 `.bin` 每点为 4 列；第 4 列是零占位值，不是真实 LiDAR intensity。  
   因而模型能够训练，但点特征信息弱于包含真实反射强度的数据。

另外，当前 `CompanyNuScenesDataset.evaluation()` 只报告各类 GT/预测数量，不计算正式 AP/NDS。训练是否真正有效，需要后续增加公司 26 类指标或使用人工抽检。

## 2. 已适配的关键文件

正式训练必须包含下列版本的代码与配置：

```text
pcdet/datasets/company_nuscenes/company_nuscenes_dataset.py
pcdet/datasets/company_nuscenes/company_nuscenes_utils.py
pcdet/models/dense_heads/voxelnext_head.py
tools/cfgs/dataset_configs/company_nuscenes_trainval_dataset.yaml
tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
tools/company_nuscenes/create_company_infos.py
tools/company_nuscenes/check_company_infos.py
tools/company_nuscenes/smoke_test_company_dataloader.py
tools/company_nuscenes/smoke_test_formal_voxelnext.py
```

其中 `voxelnext_head.py` 的修正很重要：每个检测 head 现在只修改克隆后的局部类别编号，不会覆盖原始 26 类 GT id。没有该修正时，多 head 训练会出现类别污染。

## 3. 正式数据目录要求

容器内使用且仅使用以下正式数据根目录：

```text
/workspace/OpenPCDet/data/nuscenes/
|-- samples/
|   `-- LIDAR_TOP/
|       `-- *.bin
`-- v1.0-trainval/
    |-- category.json
    |-- instance.json
    |-- sample.json
    |-- sample_data.json
    |-- sample_annotation.json
    |-- scene.json
    |-- calibrated_sensor.json
    |-- ego_pose.json
    `-- ...
```

根据已经查看到的数据内容，正式集应满足：

```text
scene.json               : 412
sample.json              : 24142
sample_data.json         : 241420
sample_annotation.json   : 1222229
samples/LIDAR_TOP/*.bin  : 24142 个
```

不要使用：

```text
/workspace/OpenPCDet/data/v1.0-trainval
```

该目录中的元数据引用 `.pcd`，但它不与实际样本目录组成可读取的正式训练根目录。

## 4. 正式数据读取约定

正式配置文件为：

```text
tools/cfgs/dataset_configs/company_nuscenes_trainval_dataset.yaml
```

其关键约定是：

```yaml
DATA_PATH: '../data/nuscenes'
VERSION: 'v1.0-trainval'
MAX_SWEEPS: 1
LIDAR_POINT_DIM: 4
LIDAR_POINT_FORMAT: float32
LIDAR_POINT_FIELDS: ['x', 'y', 'z', 'intensity']
PRED_VELOCITY: False
FILTER_MIN_POINTS_IN_GT: 1
```

说明：

1. `MAX_SWEEPS: 1` 表示本轮训练为单帧训练，不拼接历史帧。
2. `PRED_VELOCITY: False` 表示检测框不预测速度，因此 GT box 输入为 `x,y,z,dx,dy,dz,yaw,class_id`。
3. 字段名中保留 `intensity` 是为了兼容 OpenPCDet 四列输入接口；当前转换数据的第 4 列并没有真实强度含义。

## 5. 第一步：进入正确容器与工程

在宿主机执行：

```bash
sudo docker inspect detection3d_v5 --format='{{json .Mounts}}' | jq
sudo docker exec -u root -it detection3d_v5 /bin/bash
```

确认挂载包含：

```text
/home/ubuntu/WXY/OpenPCDet_ljl -> /workspace/OpenPCDet
/home/ubuntu/WXY/data          -> /workspace/OpenPCDet/data
```

容器内执行：

```bash
cd /workspace/OpenPCDet
pwd
test -f tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
test -f tools/company_nuscenes/smoke_test_formal_voxelnext.py
```

所有后续命令均从 `/workspace/OpenPCDet` 或文中明确标出的 `tools/` 目录执行。

## 6. 第二步：检查正式数据完整性

从 `/workspace/OpenPCDet` 执行：

```bash
test -d data/nuscenes/v1.0-trainval
test -d data/nuscenes/samples/LIDAR_TOP

find data/nuscenes/samples/LIDAR_TOP -maxdepth 1 -type f -name '*.bin' | wc -l

python - <<'PY'
from pathlib import Path
import json

root = Path('data/nuscenes/v1.0-trainval')
for name in ['category', 'scene', 'sample', 'sample_data', 'sample_annotation', 'instance']:
    with open(root / f'{name}.json', 'r', encoding='utf-8') as f:
        print(f'{name:20s}: {len(json.load(f))}')
PY
```

预期点云数量为 `24142`；关键 JSON 数量应与第 3 节一致。数量不一致时不要继续训练。

## 7. 第三步：预览并固定正式 train/val 切分

正式数据原始目录没有提供 `ImageSets/train.txt` 或 `val.txt`。代码会按 scene 进行 `80%/20%` 切分，不会将同一 scene 的帧分散到两侧。修正后的切分算法具有以下约束：

1. 已知 `num_lidar_pts < 1` 的框不作为可学习覆盖；`num_lidar_pts: null` 表示统计未知，代码会先保留该框并在检查结果中报告。
2. 所有存在可学习标注框的类别都必须在训练集中出现。
3. 对可以两侧覆盖的类别，优先让验证集也包含标注框。
4. 只有所有可用 scene 都已被训练覆盖约束占用的类别，才允许验证集中缺失。

先执行只读预览，它不会生成文件：

```bash
python tools/company_nuscenes/preview_formal_split.py \
  --data_path data/nuscenes \
  --version v1.0-trainval \
  --train_ratio 0.8 \
  --seed 0 \
  --min_lidar_points 1
```

必须看到：

```text
annotated_but_absent_from_train: []
PASS: all learnable annotated classes are present in training.
```

随后从 `/workspace/OpenPCDet` 执行正式落盘：

```bash
python tools/company_nuscenes/create_company_infos.py \
  --data_path data/nuscenes \
  --save_path data/nuscenes \
  --version v1.0-trainval \
  --max_sweeps 1 \
  --train_ratio 0.8 \
  --seed 0 \
  --min_lidar_points 1
```

该命令会按场景切分训练集与验证集，并生成：

```text
data/nuscenes/v1.0-trainval/ImageSets/train.txt
data/nuscenes/v1.0-trainval/ImageSets/val.txt
data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl
data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
```

首次生成后应保留 `ImageSets/train.txt` 与 `val.txt`，以保证后续复现实验时切分不改变。

## 8. 第四步：严格检查 info 与有效 GT

从 `/workspace/OpenPCDet` 执行：

```bash
python tools/company_nuscenes/check_company_infos.py \
  --root data/nuscenes/v1.0-trainval \
  --data_root data/nuscenes \
  --strict \
  --min_lidar_points 1
```

必须确认输出满足：

```text
missing_lidar_paths: 0
invalid_num_lidar_pts_samples: 0
retained_boxes_min_points_1: 大于 0
boxes_with_unknown_lidar_pts_kept: 记录实际数量
outside_config: []
annotated_but_absent_from_train: []
annotated_but_unlearnable_after_train_filter: []
```

`absent_from_infos` 中允许出现：

```text
human_pedestrian_personal_mobility
vehicle_bus_bendy
```

因为这两个类别在当前正式 annotation 中没有框。不要在当前数据上增加 `--require_all_classes`，否则它会按预期失败。

如果 `boxes_with_unknown_lidar_pts_kept` 数量较大，训练仍能执行，但在评估效果前应进一步决定是否从 `.bin` 与 3D box 重新计算点数统计，以减少空框或低质量框进入训练的风险。

本次在服务器正式数据上的预览结果已经通过验收：保留框共 `895317` 个，其中点数未知框 `611` 个；切分为训练 `329` 个 scene / `19268` 帧，验证 `83` 个 scene / `4874` 帧；所有 24 个有保留标注的类别均进入训练集。`vehicle.emergency.other` 仅有一个有效 scene，故仅进入训练集而不进入验证集。

## 9. 第五步：检查数据加载结果

从 `/workspace/OpenPCDet` 执行：

```bash
python tools/company_nuscenes/smoke_test_company_dataloader.py \
  --cfg_file tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
```

应重点检查：

```text
dataset_len: 大于 0
points_shape: 第二维应为 5
gt_boxes_shape: 最后一维应为 8
```

解释：

1. `points_shape` 中 5 列由 `batch_index + x + y + z + 占位第4列` 组成。
2. `gt_boxes` 的 8 列由 7 个框参数加 1 个 26 类类别 id 组成。

## 10. 第六步：执行 VoxelNeXt 单批次训练 smoke

这一步会真正创建 VoxelNeXt、执行一批训练前向计算和反向传播，需要容器 CUDA/OpenPCDet 算子环境可用。

从 `/workspace/OpenPCDet` 执行：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/company_nuscenes/smoke_test_formal_voxelnext.py \
  --cfg_file tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml \
  --workers 0
```

成功标志：

```text
formal_voxelnext_smoke: PASS
classes: 26
loss: 一个有限数值
```

如果本步骤没有通过，不要启动完整训练。常见问题分别处理：

| 报错类型 | 处理方向 |
|---|---|
| 找不到 info 文件 | 重新执行第 7 节生成 info |
| 找不到 `.bin` | 检查是否使用了 `data/nuscenes` 数据根目录 |
| CUDA/spconv 算子错误 | 使用已跑通过 VoxelNeXt 的容器环境，重新编译相应算子 |
| CUDA out of memory | 保持 `batch_size=1`，再酌情减少配置中 `MAX_NUMBER_OF_VOXELS['train']` |
| loss 非有限值 | 停止正式训练，检查 box 坐标、尺寸以及点云转换 |

## 11. 第七步：启动正式 26 类 VoxelNeXt 训练

从 `/workspace/OpenPCDet/tools` 执行：

```bash
cd /workspace/OpenPCDet/tools

CUDA_VISIBLE_DEVICES=0 python train.py \
  --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml \
  --batch_size 1 \
  --epochs 20 \
  --workers 4 \
  --extra_tag formal_company_26cls \
  --ckpt_save_interval 1 \
  --max_ckpt_save_num 20
```

checkpoint 输出目录为：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/
```

先观察训练日志中的以下内容：

```text
cfg.DATA_CONFIG.DATA_PATH: ../data/nuscenes
cfg.DATA_CONFIG.VERSION: v1.0-trainval
cfg.DATA_CONFIG.LIDAR_POINT_DIM: 4
cfg.MODEL.NAME: VoxelNeXt
cfg.CLASS_NAMES: 包含 26 类
```

训练过程中至少检查：

1. `loss` 不是 `nan` 或 `inf`。
2. checkpoint 每个 epoch 能正常保存。
3. 显存稳定，没有持续增长至 OOM。
4. 训练日志所列数据集长度与 info 切分结果一致。

## 12. 第八步：运行 checkpoint 测试

当前测试会正常跑模型推理并输出 recall 与每类 GT/预测数量，但不是正式 AP 指标。

从 `/workspace/OpenPCDet/tools` 执行：

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml \
  --ckpt ../output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth \
  --batch_size 1 \
  --workers 4 \
  --extra_tag formal_company_26cls \
  --eval_tag epoch20
```

测试日志和预测结果位于：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/
```

## 13. 当前能力边界与后续必须补齐的内容

### 已经可以完成

1. 读取正式公司数据集中的 LiDAR 与 3D box。
2. 生成正式训练/验证 info。
3. 构建 26 类 VoxelNeXt 检测头。
4. 正确进行多 head 类别目标分配。
5. 训练并输出 26 类 checkpoint。
6. 执行推理及训练链路 smoke 检查。

### 不能从当前数据或代码直接得到

1. 两个零样本类别的有效检测能力。
2. 真实 intensity 带来的模型性能，因为当前第 4 点特征是零占位值。
3. 公司 26 类正式 AP/NDS 结果，因为当前 evaluation 还没有实现该指标。

因此，“代码能训练 26 类 VoxelNeXt”与“26 类模型效果已经合格”是两件不同的事。前者在本次适配后已具备；后者需要真实覆盖 26 类的标注数据和正式评估实现。

## 14. 最终执行清单

正式开训前逐项打勾：

- [ ] 容器挂载的工程是修改后的 `OpenPCDet_ljl`
- [ ] 数据根目录使用 `data/nuscenes`
- [ ] `samples/LIDAR_TOP/*.bin` 数量为 `24142`
- [ ] 已生成 `company_nuscenes_infos_train.pkl` 和 `company_nuscenes_infos_val.pkl`
- [ ] info 检查中 `missing_lidar_paths: 0`
- [ ] info 检查中 `invalid_num_lidar_pts_samples: 0`
- [ ] info 检查中训练集 `retained_boxes_min_points_1` 大于 `0`
- [ ] dataloader smoke 通过
- [ ] `formal_voxelnext_smoke: PASS`
- [ ] 正式训练使用 `company_voxelnext_26cls_trainval.yaml`
