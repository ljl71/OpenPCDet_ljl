# OpenPCDet_ljl 26类 VoxelNeXt 训练与测试产物整理

> 工程：`OpenPCDet_ljl`  
> 任务：公司正式 nuScenes 格式数据训练 26 类 VoxelNeXt  
> 实验标识：`formal_company_26cls`  
> 配置文件：`company_voxelnext_26cls_trainval.yaml`  
> 训练轮数：20 epoch  
> 最终 checkpoint：`checkpoint_epoch_20.pth`  
> 测试结果目录：`eval/epoch_20/val/epoch20/`

---

## 1. 路径映射关系

本次训练是在 Docker 容器内完成的，但代码和数据实际挂载自宿主机。

### 1.1 工程路径映射

容器内路径：

```text
/workspace/OpenPCDet
```

对应宿主机路径：

```text
/home/ubuntu/WXY/OpenPCDet_ljl
```

因此，容器内的训练输出目录：

```text
/workspace/OpenPCDet/output/...
```

对应宿主机上的实际目录：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/...
```

### 1.2 数据路径映射

容器内路径：

```text
/workspace/OpenPCDet/data
```

对应宿主机路径：

```text
/home/ubuntu/WXY/data
```

因此，容器内的 info 文件路径：

```text
/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/
```

对应宿主机上的实际路径：

```text
/home/ubuntu/WXY/data/nuscenes/v1.0-trainval/
```

---

## 2. 数据准备阶段生成的文件

这些文件是在训练前运行 `create_company_infos.py` 生成的。它们不是模型权重，但训练和测试都依赖这些索引文件。

### 2.1 train/val info 文件

容器内路径：

```text
/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl
/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
```

宿主机路径：

```text
/home/ubuntu/WXY/data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl
/home/ubuntu/WXY/data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
```

本次实际生成结果：

```text
company_nuscenes_infos_train.pkl  332M
company_nuscenes_infos_val.pkl     86M
```

含义：

| 文件 | 作用 |
|---|---|
| `company_nuscenes_infos_train.pkl` | 训练集样本索引文件 |
| `company_nuscenes_infos_val.pkl` | 验证集样本索引文件 |

这两个文件中保存了训练/验证样本对应的点云路径、GT box、类别映射、样本元信息等。OpenPCDet 后续训练和测试读取公司正式数据时，主要依赖这两个 `.pkl` 文件。

### 2.2 train/val 场景划分文件

容器内路径：

```text
/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/ImageSets/train.txt
/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/ImageSets/val.txt
```

宿主机路径：

```text
/home/ubuntu/WXY/data/nuscenes/v1.0-trainval/ImageSets/train.txt
/home/ubuntu/WXY/data/nuscenes/v1.0-trainval/ImageSets/val.txt
```

本次划分结果：

```text
train.txt: 329 个 scene
val.txt:    83 个 scene
```

注意：这里的 `train.txt` 和 `val.txt` 保存的是 **scene 级划分**，不是 sample 数量。

实际 sample 数量是：

```text
train samples: 19268
val samples:    4874
```

---

## 3. 训练阶段生成的文件

训练输出根目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/
```

宿主机对应路径：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/
```

该路径由以下几个部分共同决定：

```text
cfg.EXP_GROUP_PATH: nuscenes_models
cfg.TAG: company_voxelnext_26cls_trainval
extra_tag: formal_company_26cls
```

### 3.1 checkpoint 文件，也就是训练好的模型

checkpoint 目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/
```

宿主机路径：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/
```

本次训练生成了 20 个 checkpoint：

```text
checkpoint_epoch_1.pth
checkpoint_epoch_2.pth
checkpoint_epoch_3.pth
checkpoint_epoch_4.pth
checkpoint_epoch_5.pth
checkpoint_epoch_6.pth
checkpoint_epoch_7.pth
checkpoint_epoch_8.pth
checkpoint_epoch_9.pth
checkpoint_epoch_10.pth
checkpoint_epoch_11.pth
checkpoint_epoch_12.pth
checkpoint_epoch_13.pth
checkpoint_epoch_14.pth
checkpoint_epoch_15.pth
checkpoint_epoch_16.pth
checkpoint_epoch_17.pth
checkpoint_epoch_18.pth
checkpoint_epoch_19.pth
checkpoint_epoch_20.pth
```

最终模型路径：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth
```

宿主机路径：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth
```

本次实际情况：

```text
每个 checkpoint 大约 90M
ckpt/ 目录总大小约 1.8G
```

说明：

- `checkpoint_epoch_20.pth` 是本次 20 epoch 训练后的最终模型。
- 测试阶段加载的就是这个模型。
- 日志显示测试时成功加载了 `443/443` 个参数，说明权重完整可用。

### 3.2 训练日志文件

训练日志通常也位于实验输出目录下：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/
```

可以使用以下命令查找：

```bash
find /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls -maxdepth 2 -type f | sort
```

常见文件类型：

```text
log_train_*.txt
events.out.tfevents.*
```

含义：

| 文件 | 作用 |
|---|---|
| `log_train_*.txt` | 训练终端日志 |
| `events.out.tfevents.*` | TensorBoard 日志，如果训练脚本启用了 TensorBoard |

训练日志中可以查看 loss、learning rate、每轮耗时、checkpoint 保存情况等。

---

## 4. 测试/评估阶段生成的文件

测试输出根目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/
```

本次 epoch20 的具体评估目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/
```

宿主机对应路径：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/
```

你实际查看到该目录下有：

```text
log_eval_20260527-004119.txt
log_eval_20260527-004719.txt
result.pkl
```

### 4.1 `log_eval_20260527-004119.txt`

路径：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/log_eval_20260527-004119.txt
```

说明：

这个文件大概率是第一次执行测试时生成的日志。那一次测试过程中遇到过 `test.py` 与 `eval_utils.py` 的接口不匹配问题，因此它更多是排错记录。

是否重要：

```text
可保留，但不是最终成功评估日志。
```

### 4.2 `log_eval_20260527-004719.txt`

路径：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/log_eval_20260527-004719.txt
```

说明：

这是最终成功完成测试的评估日志，最重要。

其中包含：

```text
recall_rcnn_0.3: 0.724822
recall_rcnn_0.5: 0.517463
recall_rcnn_0.7: 0.222161
Average predicted number of objects(4874 samples): 103.799
```

也包含每个类别的 GT 数量和预测数量，例如：

```text
vehicle_car: gt=136394, pred=236430
human_pedestrian_adult: gt=33740, pred=102300
vehicle_motorcycle: gt=33999, pred=90677
vehicle_bicycle: gt=13635, pred=6126
vehicle_tricycle: gt=8661, pred=23255
bicycle: gt=2003, pred=1495
```

该日志中还明确说明：

```text
CompanyNuScenes smoke evaluation
This first-stage evaluation only reports GT/pred counts; AP is not implemented yet.
```

因此，这份日志可以证明：

- 第 20 轮 checkpoint 成功加载；
- 验证集 4874 帧全部完成推理；
- 模型有实际预测输出；
- 当前评估只提供 recall 和 GT/pred 数量，不是正式 AP/NDS。

是否重要：

```text
非常重要，建议保留。
```

### 4.3 `result.pkl`

路径：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/result.pkl
```

宿主机路径：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/result.pkl
```

说明：

这是验证集 4874 帧的模型预测结果文件，是测试阶段最关键的输出之一。

它通常包含每一帧的预测信息，例如：

```text
frame_id
name
score
boxes_lidar / pred_boxes
pred_scores
pred_labels
```

具体字段取决于当前 `CompanyNuScenesDataset` 的输出格式。

后续如果要做以下工作，基本都要用到 `result.pkl`：

```text
可视化预测框
统计预测类别数量
分析预测分数分布
调整 score threshold
导出预测结果
补正式 AP/NDS 评估
人工抽检模型效果
```

是否重要：

```text
非常重要，建议保留。
```

---

## 5. 测试阶段不会生成新的模型

测试命令不会再生成新的 `.pth` 模型文件。

测试阶段只读取已有 checkpoint：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth
```

然后生成评估日志和预测结果：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/log_eval_*.txt
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/result.pkl
```

所以要区分：

| 类型 | 文件 |
|---|---|
| 模型权重 | `checkpoint_epoch_20.pth` |
| 测试日志 | `log_eval_20260527-004719.txt` |
| 预测结果 | `result.pkl` |

---

## 6. 运行过程中生成或修改过的临时文件

这些文件不是最终模型结果，但服务器上可能存在。

### 6.1 代码备份文件

为了修复运行过程中的兼容问题，修改前备份过一些源码文件，例如：

```text
/workspace/OpenPCDet/pcdet/datasets/__init__.py.bak_av2
/workspace/OpenPCDet/tools/train_utils/train_utils.py.bak_ckpt_interval
/workspace/OpenPCDet/tools/test.py.bak_save_to_file
/workspace/OpenPCDet/tools/test.py.bak_logger
/workspace/OpenPCDet/tools/test.py.bak_eval_args
```

说明：

这些文件只是现场修 bug 之前的备份，不参与训练和测试。

是否重要：

```text
可保留用于回滚，也可后续清理。
```

### 6.2 缺失点云检查文件

之前排查点云完整性时生成过：

```text
/workspace/OpenPCDet/missing_lidar_top_files.txt
```

后续检查结果显示：

```text
Missing LIDAR_TOP files: 0
```

因此该文件大概率是空文件，或没有实际缺失记录。

是否重要：

```text
不重要，可清理。
```

### 6.3 Python 缓存文件

运行 Python 脚本后会生成：

```text
__pycache__/
*.pyc
```

例如：

```text
/workspace/OpenPCDet/tools/company_nuscenes/__pycache__/
```

说明：

这些是 Python 缓存文件，不是训练结果，不需要提交 GitHub。

是否重要：

```text
不重要，可清理。
```

---

## 7. 最核心需要保存的文件

如果只保留最重要的结果，建议至少保存以下几类。

### 7.1 最终模型权重

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth
```

宿主机路径：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth
```

用途：

```text
后续测试、继续训练、部署、可视化推理都需要它。
```

### 7.2 完整训练 checkpoint

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_1.pth
...
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth
```

用途：

```text
可以回看不同 epoch 的效果，也可以选择更早 epoch 做对比测试。
```

如果空间紧张，可以只保留：

```text
checkpoint_epoch_10.pth
checkpoint_epoch_15.pth
checkpoint_epoch_20.pth
```

或者只保留最终：

```text
checkpoint_epoch_20.pth
```

### 7.3 成功测试日志

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/log_eval_20260527-004719.txt
```

用途：

```text
保存 epoch20 的验证 recall、预测数量和类别统计。
```

### 7.4 预测结果文件

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/result.pkl
```

用途：

```text
后续可视化、统计、阈值分析、补正式评估都依赖它。
```

### 7.5 info 文件

```text
/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl
/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
```

用途：

```text
再次训练或测试时不需要重新生成 info。
```

---

## 8. 一条命令查看所有关键输出

在容器中执行：

```bash
echo "===== info files ====="; ls -lh /workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl /workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl; echo "===== checkpoints ====="; ls -lh /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/; echo "===== eval result ====="; ls -lh /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/
```

---

## 9. 一条命令备份关键结果

如果想把关键结果单独备份到一个目录，可以执行：

```bash
mkdir -p /workspace/OpenPCDet/output_backup/formal_company_26cls_epoch20
```

复制最终模型：

```bash
cp /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth /workspace/OpenPCDet/output_backup/formal_company_26cls_epoch20/
```

复制成功测试日志：

```bash
cp /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/log_eval_20260527-004719.txt /workspace/OpenPCDet/output_backup/formal_company_26cls_epoch20/
```

复制预测结果：

```bash
cp /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/result.pkl /workspace/OpenPCDet/output_backup/formal_company_26cls_epoch20/
```

查看备份：

```bash
ls -lh /workspace/OpenPCDet/output_backup/formal_company_26cls_epoch20/
```

---

---

## 可视化阶段：使用 `SCORE_THRESH=0.2` 的结果生成 BEV 可视化图

本次已经基于 `SCORE_THRESH=0.2` 完成验证集测试，并生成了对应的预测结果文件：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl
```

该文件大小约为：

```text
40M
```

它是 `0.2` 阈值下的验证集预测结果，后续 BEV 可视化直接读取这个 `result.pkl`，不需要重新训练，也不需要重新测试。

---

### 1. 进入项目根目录

可视化命令建议在项目根目录执行：

```bash
cd /workspace/OpenPCDet
```

原因是后续命令中的路径都以项目根目录为基准，例如：

```text
output/...
data/...
tools/...
```

如果当前在 `/workspace/OpenPCDet/tools` 目录下，直接使用 `output/...` 会被解释为：

```text
/workspace/OpenPCDet/tools/output/...
```

从而导致找不到文件。

---

### 2. 确认 `SCORE_THRESH=0.2` 的预测结果文件存在

执行：

```bash
ls -lh output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl
```

本次实际结果：

```text
-rw-r--r-- 1 root root 40M May 27 01:56 output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl
```

说明 `0.2` 阈值下的测试预测结果已经存在，可以直接用于可视化。

---

### 3. 检查 `result.pkl` 和 `val info` 字段

执行：

```bash
python - <<'PY'
import pickle

result_path = 'output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl'
info_path = 'data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl'

preds = pickle.load(open(result_path, 'rb'))
infos = pickle.load(open(info_path, 'rb'))

print('pred type:', type(preds), 'len:', len(preds))
print('info type:', type(infos), 'len:', len(infos))

print('\nfirst pred keys:')
print(preds[0].keys())

print('\nfirst info keys:')
print(infos[0].keys())

print('\nfirst pred summary:')
for k, v in preds[0].items():
    if hasattr(v, 'shape'):
        print(k, v.shape)
    elif hasattr(v, '__len__') and not isinstance(v, str):
        print(k, len(v))
    else:
        print(k, type(v), v)

print('\nfirst info summary:')
for k, v in infos[0].items():
    if hasattr(v, 'shape'):
        print(k, v.shape)
    elif hasattr(v, '__len__') and not isinstance(v, str):
        print(k, type(v), len(v))
    else:
        print(k, type(v), v)
PY
```

本次实际检查结果：

```text
pred type: <class 'list'> len: 4874
info type: <class 'list'> len: 4874
```

预测结果字段：

```text
dict_keys(['name', 'score', 'boxes_lidar', 'pred_labels', 'frame_id', 'metadata'])
```

验证集 info 字段：

```text
dict_keys(['lidar_path', 'token', 'scene_token', 'timestamp', 'sweeps', 'gt_boxes', 'gt_names', 'gt_raw_names', 'gt_boxes_token', 'num_lidar_pts', 'num_radar_pts', 'lidar_path_exists'])
```

第一帧预测信息：

```text
name (44,)
score (44,)
boxes_lidar (44, 7)
pred_labels (44,)
frame_id ()
metadata 1
```

第一帧 GT 信息：

```text
lidar_path: samples/LIDAR_TOP/1767948279422776259.bin
gt_boxes (52, 7)
gt_names (52,)
gt_raw_names (52,)
num_lidar_pts (52,)
lidar_path_exists True
```

这说明当前可视化脚本需要读取的字段都存在：

| 内容 | 字段 |
|---|---|
| 预测框 | `boxes_lidar` |
| 预测分数 | `score` |
| 预测类别名 | `name` |
| GT 框 | `gt_boxes` |
| GT 类别名 | `gt_names` |
| 点云路径 | `lidar_path` |

因此，当前字段结构可以直接用于 BEV 可视化。

---

### 4. 创建 BEV 可视化脚本

脚本保存位置：

```text
/workspace/OpenPCDet/tools/company_nuscenes/visualize_score020_bev.py
```

创建脚本：

```bash
cat > tools/company_nuscenes/visualize_score020_bev.py <<'PY'
from pathlib import Path
import argparse
import pickle
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def get_pred_boxes(pred):
    for k in ["boxes_lidar", "pred_boxes", "box3d_lidar"]:
        if k in pred:
            return to_numpy(pred[k])
    return np.zeros((0, 7), dtype=np.float32)


def get_pred_scores(pred):
    for k in ["score", "scores", "pred_scores"]:
        if k in pred:
            return to_numpy(pred[k]).reshape(-1)
    boxes = get_pred_boxes(pred)
    return np.ones((len(boxes),), dtype=np.float32)


def get_gt_boxes(info):
    if "gt_boxes" in info:
        return to_numpy(info["gt_boxes"])
    if "annos" in info and isinstance(info["annos"], dict):
        annos = info["annos"]
        for k in ["gt_boxes_lidar", "boxes_lidar", "gt_boxes"]:
            if k in annos:
                return to_numpy(annos[k])
    return np.zeros((0, 7), dtype=np.float32)


def get_lidar_path(info, data_root):
    candidates = []
    for k in ["lidar_path", "lidar_file", "point_cloud_path"]:
        if k in info:
            candidates.append(info[k])

    if "point_cloud" in info and isinstance(info["point_cloud"], dict):
        pc = info["point_cloud"]
        for k in ["lidar_path", "lidar_file", "velodyne_path"]:
            if k in pc:
                candidates.append(pc[k])

    for c in candidates:
        if c is None:
            continue
        p = Path(str(c))
        if p.is_absolute() and p.exists():
            return p
        p1 = Path(data_root) / p
        if p1.exists():
            return p1
        p2 = Path("/workspace/OpenPCDet") / p
        if p2.exists():
            return p2

    return None


def read_points(path):
    if path is None or not Path(path).exists():
        return None
    arr = np.fromfile(str(path), dtype=np.float32)
    if arr.size % 4 == 0:
        return arr.reshape(-1, 4)
    if arr.size % 5 == 0:
        return arr.reshape(-1, 5)
    return None


def box_corners_bev(box):
    x, y, z, dx, dy, dz, yaw = box[:7]
    local = np.array([
        [ dx / 2,  dy / 2],
        [ dx / 2, -dy / 2],
        [-dx / 2, -dy / 2],
        [-dx / 2,  dy / 2],
        [ dx / 2,  dy / 2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def draw_boxes(ax, boxes, kind="pred"):
    if boxes is None or len(boxes) == 0:
        return

    for box in boxes:
        if len(box) < 7:
            continue
        corners = box_corners_bev(box)
        if kind == "gt":
            ax.plot(corners[:, 0], corners[:, 1], linewidth=1.2, color="lime")
        else:
            ax.plot(corners[:, 0], corners[:, 1], linewidth=1.0, color="red")

        x, y, _, dx, _, _, yaw = box[:7]
        ax.plot(
            [x, x + np.cos(yaw) * dx / 2],
            [y, y + np.sin(yaw) * dx / 2],
            linewidth=0.8,
            color="yellow" if kind == "pred" else "cyan",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--infos", required=True)
    parser.add_argument("--data_root", default="data/nuscenes")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--score_thresh", type=float, default=0.2)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--range", type=float, default=55.0)
    args = parser.parse_args()

    preds = pickle.load(open(args.result, "rb"))
    infos = pickle.load(open(args.infos, "rb"))

    if isinstance(infos, dict) and "infos" in infos:
        infos = infos["infos"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert len(preds) == len(infos), f"pred len {len(preds)} != info len {len(infos)}"

    if args.indices:
        indices = args.indices
    else:
        indices = list(range(min(args.num, len(preds))))

    lines = []
    for idx in indices:
        pred = preds[idx]
        info = infos[idx]

        pred_boxes = get_pred_boxes(pred)
        pred_scores = get_pred_scores(pred)
        keep = pred_scores >= args.score_thresh
        pred_boxes = pred_boxes[keep]

        gt_boxes = get_gt_boxes(info)

        lidar_path = get_lidar_path(info, args.data_root)
        points = read_points(lidar_path)

        fig, ax = plt.subplots(figsize=(10, 10))

        if points is not None and len(points) > 0:
            pts = points[:, :2]
            r = args.range
            mask = (
                (pts[:, 0] >= -r) & (pts[:, 0] <= r) &
                (pts[:, 1] >= -r) & (pts[:, 1] <= r)
            )
            pts = pts[mask]
            if len(pts) > 100000:
                sel = np.random.choice(len(pts), 100000, replace=False)
                pts = pts[sel]
            ax.scatter(pts[:, 0], pts[:, 1], s=0.1, c="gray", alpha=0.35)

        draw_boxes(ax, gt_boxes, kind="gt")
        draw_boxes(ax, pred_boxes, kind="pred")

        r = args.range
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3)
        ax.set_title(
            f"idx={idx} | GT={len(gt_boxes)} | Pred(score>={args.score_thresh})={len(pred_boxes)}\n"
            f"green=GT, red=Pred"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        out_path = out_dir / f"bev_idx_{idx:06d}_gt{len(gt_boxes)}_pred{len(pred_boxes)}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        frame_id = info.get("frame_id", info.get("token", idx))
        lines.append(f"{idx}\t{frame_id}\tgt={len(gt_boxes)}\tpred={len(pred_boxes)}\t{out_path}")

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {len(indices)} BEV visualizations to: {out_dir}")
    print(f"Summary: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
PY
```

该脚本会读取：

```text
result.pkl
company_nuscenes_infos_val.pkl
samples/LIDAR_TOP/*.bin
```

并生成 BEV 俯视图。

---

### 5. 使用 `SCORE_THRESH=0.2` 生成前 20 帧 BEV 可视化

执行：

```bash
python tools/company_nuscenes/visualize_score020_bev.py --result output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl --infos data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl --data_root data/nuscenes --out_dir output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev --score_thresh 0.2 --num 20
```

本次实际运行结果：

```text
Saved 20 BEV visualizations to: output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev
Summary: output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev/summary.txt
```

说明前 20 帧 BEV 可视化已经成功生成。

---

### 6. 可视化结果保存位置

容器内目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev/
```

宿主机对应目录：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev/
```

该目录下包含：

```text
bev_idx_*.png
summary.txt
```

其中：

| 文件 | 作用 |
|---|---|
| `bev_idx_*.png` | 每个验证样本对应的 BEV 可视化图 |
| `summary.txt` | 每张图对应的 index、frame/token、GT 数量、预测框数量和图片路径 |

---

### 7. 查看生成了哪些图片

执行：

```bash
find output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev -name "*.png" | sort | head -20
```

查看可视化汇总：

```bash
cat output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev/summary.txt
```

`summary.txt` 中每一行大致表示：

```text
idx    frame_id/token    gt=xx    pred=xx    image_path
```

可以优先挑选以下几类图片查看：

```text
1. pred 明显多于 gt 的样本；
2. pred 明显少于 gt 的样本；
3. pred 和 gt 数量接近的样本；
4. 点云目标密集的样本；
5. 含有车辆、行人、摩托车等主类的样本。
```

---

### 8. 可视化图的含义

BEV 可视化图中：

| 视觉元素 | 含义 |
|---|---|
| 灰色点 | LiDAR 点云俯视图 |
| 绿色框 | GT 标注框 |
| 红色框 | 模型预测框 |
| 青色短线 | GT 框朝向 |
| 黄色短线 | 预测框朝向 |

重点观察：

```text
1. 红色预测框是否大体落在点云目标上；
2. 是否存在大量漂浮在空白区域的红框；
3. 同一个目标周围是否有多个重复红框；
4. 车辆、行人、摩托车等主类是否大致能被检测到；
5. bicycle、vehicle_bicycle、animal 等弱类是否漏检较多；
6. 框的尺寸和朝向是否明显异常。
```

---

### 9. 可视化指定样本

如果只想看指定验证集 index，例如第 0、10、100、500 帧，可以执行：

```bash
python tools/company_nuscenes/visualize_score020_bev.py --result output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl --infos data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl --data_root data/nuscenes --out_dir output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev_selected --score_thresh 0.2 --indices 0 10 100 500
```

指定样本可视化保存目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev_selected/
```

---

### 10. 打包可视化图片

如果需要下载或备份生成的 BEV 图片，可以执行：

```bash
tar -czf output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev.tar.gz -C output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis epoch20_score020_bev
```

压缩包位置：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev.tar.gz
```

宿主机对应路径：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev.tar.gz
```

---

### 11. 可视化阶段生成的新文件

本次可视化阶段新增了以下文件或目录：

```text
/workspace/OpenPCDet/tools/company_nuscenes/visualize_score020_bev.py
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev/
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev/summary.txt
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev/*.png
```

其中最重要的是：

```text
vis/epoch20_score020_bev/*.png
vis/epoch20_score020_bev/summary.txt
```

这些文件用于人工检查 `SCORE_THRESH=0.2` 下预测框的位置、数量和方向是否合理。

如果后续要提交 GitHub：

- `visualize_score020_bev.py` 可以作为工具脚本提交；
- `vis/epoch20_score020_bev/*.png` 不建议提交；
- `summary.txt` 可作为本次实验记录保存，但一般也不提交到代码仓库。

## 10. 是否应该提交到 GitHub

### 10.1 不建议提交的文件

以下文件体积大或属于运行产物，不建议提交到 GitHub：

```text
data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl
data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/*.pth
output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/**/*.pkl
__pycache__/
*.pyc
*.bak_*
missing_lidar_top_files.txt
```

### 10.2 可以考虑提交的内容

如果要提交本次修复后的代码，可以考虑提交以下源码文件：

```text
pcdet/datasets/__init__.py
tools/train_utils/train_utils.py
tools/test.py
```

这些文件对应修复：

```text
av2 缺包导致无关 Argo2 导入失败
ckpt_save_time_interval=None 导致训练保存逻辑报错
test.py 与 eval_utils.py 的接口不匹配
```

---

## 11. 最终总结

本次训练和测试完成后，最重要的产物有三类：

```text
1. info 文件：
   data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl
   data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl

2. 模型文件：
   output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth

3. 测试结果：
   output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/log_eval_20260527-004719.txt
   output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/result.pkl
```

一句话：

> 最终模型看 `checkpoint_epoch_20.pth`，最终测试日志看 `log_eval_20260527-004719.txt`，最终预测结果看 `result.pkl`。这三类文件是本次训练测试最核心的产物。


---

## 阈值测试结果目录与后处理分析

在原始 `SCORE_THRESH=0.1` 测试完成后，又额外测试了两个后处理置信度阈值：

```text
SCORE_THRESH = 0.2
SCORE_THRESH = 0.25
```

这两个测试不会重新训练模型，只是读取同一个最终模型：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth
```

然后在不同后处理阈值下重新对验证集推理，并分别生成新的评估目录。

### 1. 原始阈值：`SCORE_THRESH=0.1`

容器内输出目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/
```

宿主机对应路径：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/
```

该目录下已经确认有：

```text
log_eval_20260527-004119.txt
log_eval_20260527-004719.txt
result.pkl
```

其中 `log_eval_20260527-004719.txt` 是成功完成测试的日志，`result.pkl` 是该阈值下的验证集预测结果。

关键结果：

```text
recall_rcnn_0.3: 0.724822
recall_rcnn_0.5: 0.517463
recall_rcnn_0.7: 0.222161
Average predicted number of objects(4874 samples): 103.799
```

判断：

```text
0.1 的召回最高，但平均每帧预测约 103.799 个目标，明显偏多，误检风险较大。
```

---

### 2. 阈值测试一：`SCORE_THRESH=0.2`

测试命令：

```bash
CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --ckpt ../output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth --batch_size 1 --workers 4 --extra_tag formal_company_26cls --eval_tag epoch20_score020 --set MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH 0.2
```

容器内输出目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/
```

宿主机对应路径：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/
```

该目录中通常包含：

```text
log_eval_*.txt
result.pkl
```

其中：

| 文件 | 作用 |
|---|---|
| `log_eval_*.txt` | `SCORE_THRESH=0.2` 的测试日志，包含 recall、平均预测框数量和每类 GT/pred 数量 |
| `result.pkl` | `SCORE_THRESH=0.2` 下的验证集预测结果 |

关键结果：

```text
recall_rcnn_0.3: 0.657681
recall_rcnn_0.5: 0.495375
recall_rcnn_0.7: 0.220149
Average predicted number of objects(4874 samples): 45.709
```

主要类别预测数量：

```text
human_pedestrian_adult: gt=33740, pred=36289
vehicle_car: gt=136394, pred=120137
vehicle_bus_rigid: gt=3171, pred=2423
vehicle_truck: gt=8307, pred=5251
movable_object_trafficcone: gt=4764, pred=6994
vehicle_motorcycle: gt=33999, pred=37300
vehicle_bicycle: gt=13635, pred=3154
vehicle_tricycle: gt=8661, pred=8513
bicycle: gt=2003, pred=560
```

判断：

```text
0.2 能明显减少低置信度框，同时保留较好的 recall。相比 0.1，平均预测框数量从 103.799/frame 降到 45.709/frame，预测框爆炸问题明显缓解。
```

---

### 3. 阈值测试二：`SCORE_THRESH=0.25`

测试命令：

```bash
CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --ckpt ../output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth --batch_size 1 --workers 4 --extra_tag formal_company_26cls --eval_tag epoch20_score025 --set MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH 0.25
```

容器内输出目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score025/
```

宿主机对应路径：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score025/
```

该目录中通常包含：

```text
log_eval_*.txt
result.pkl
```

其中：

| 文件 | 作用 |
|---|---|
| `log_eval_*.txt` | `SCORE_THRESH=0.25` 的测试日志 |
| `result.pkl` | `SCORE_THRESH=0.25` 下的验证集预测结果 |

关键结果：

```text
recall_rcnn_0.3: 0.624331
recall_rcnn_0.5: 0.481173
recall_rcnn_0.7: 0.218387
Average predicted number of objects(4874 samples): 36.444
```

主要类别预测数量：

```text
human_pedestrian_adult: gt=33740, pred=26925
vehicle_car: gt=136394, pred=100697
vehicle_bus_rigid: gt=3171, pred=1870
vehicle_truck: gt=8307, pred=4104
movable_object_trafficcone: gt=4764, pred=5263
vehicle_motorcycle: gt=33999, pred=28645
vehicle_bicycle: gt=13635, pred=2537
vehicle_tricycle: gt=8661, pred=6138
bicycle: gt=2003, pred=394
```

判断：

```text
0.25 的预测框数量最接近验证集 GT 平均数量，但对弱类和长尾类压制更明显。
```

验证集 GT 总数为 `183358`，验证样本数为 `4874`，平均每帧 GT 约为：

```text
183358 / 4874 ≈ 37.6 个/frame
```

`SCORE_THRESH=0.25` 时平均预测数为：

```text
36.444 个/frame
```

从数量上看非常接近 GT 平均值，但它会进一步降低部分难类的预测数量，例如 `vehicle_bicycle`、`bicycle`、`group_vehicle_bicycle` 等。

---

## 阈值对比与推荐结论

本次对同一个 `checkpoint_epoch_20.pth` 分别测试了 3 个置信度阈值：

```text
SCORE_THRESH=0.1
SCORE_THRESH=0.2
SCORE_THRESH=0.25
```

整体结果如下：

| SCORE_THRESH | recall@0.3 | recall@0.5 | recall@0.7 | 平均预测数/帧 | 结果倾向 |
|---:|---:|---:|---:|---:|---|
| 0.1 | 0.724822 | 0.517463 | 0.222161 | 103.799 | 召回高，但预测框明显过多 |
| 0.2 | 0.657681 | 0.495375 | 0.220149 | 45.709 | 比较均衡，预测数明显下降 |
| 0.25 | 0.624331 | 0.481173 | 0.218387 | 36.444 | 框数最干净，但偏保守 |

### 从预测框数量看

原始 `SCORE_THRESH=0.1` 时：

```text
Average predicted number: 103.799/frame
```

这个数量明显偏多。验证集 GT 平均数量约为：

```text
37.6/frame
```

因此 `0.1` 虽然 recall 较高，但会保留大量低置信度框，误检风险较大。

`SCORE_THRESH=0.2` 时：

```text
Average predicted number: 45.709/frame
```

预测框数量已经大幅降低，并且和 GT 平均数量接近。

`SCORE_THRESH=0.25` 时：

```text
Average predicted number: 36.444/frame
```

预测框数量最接近 GT 平均值，但部分弱类预测数量下降明显。

### 从 recall 看

从 `0.1` 到 `0.2`：

```text
recall@0.5: 0.517463 -> 0.495375
recall@0.7: 0.222161 -> 0.220149
```

`recall@0.5` 小幅下降，`recall@0.7` 基本不变，但平均预测数量从 `103.799` 降到 `45.709`，收益明显。

从 `0.2` 到 `0.25`：

```text
recall@0.5: 0.495375 -> 0.481173
recall@0.7: 0.220149 -> 0.218387
```

预测框进一步减少，但召回继续下降。尤其对低置信度类别、弱类和长尾类更不友好。

### 从类别预测数量看

`SCORE_THRESH=0.2` 对高频主类更平衡：

```text
vehicle_car: gt=136394, pred=120137
human_pedestrian_adult: gt=33740, pred=36289
vehicle_motorcycle: gt=33999, pred=37300
vehicle_tricycle: gt=8661, pred=8513
```

这些类别的预测数量和 GT 数量比较接近。

`SCORE_THRESH=0.25` 对整体框数控制更强，但弱类进一步减少：

```text
vehicle_bicycle: gt=13635, pred=2537
bicycle: gt=2003, pred=394
group_vehicle_bicycle: gt=2285, pred=141
animal: gt=241, pred=13
human_pedestrian_child: gt=946, pred=22
```

说明单一全局阈值无法同时兼顾所有类别。高频类别适合较高阈值，弱类和长尾类更适合较低阈值。

### 推荐阈值

当前推荐：

```text
默认推荐值：SCORE_THRESH = 0.2
```

理由：

1. 相比 `0.1`，`0.2` 将平均预测框数从 `103.799/frame` 降到 `45.709/frame`，明显缓解预测框过多问题。
2. `0.2` 的 `recall@0.5` 仍有 `0.495375`，相比 `0.1` 只小幅下降。
3. `0.2` 的 `recall@0.7` 为 `0.220149`，几乎和 `0.1` 的 `0.222161` 一致，说明高质量定位框基本被保留。
4. `0.25` 虽然预测框数量更接近 GT 平均值，但对 `vehicle_bicycle`、`bicycle`、`group_vehicle_bicycle`、`animal` 等弱类压制更明显。
5. 当前评估还没有正式 AP/NDS，不能仅凭预测数量最接近 GT 就选择最高阈值，需要保留一定召回。

因此，当前阶段建议：

```text
正式默认测试：SCORE_THRESH=0.2
偏保守展示：SCORE_THRESH=0.25
不建议继续使用：SCORE_THRESH=0.1
```

如果暂时不实现分类别阈值，建议先固定使用：

```text
MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH: 0.2
```

作为当前 baseline 的默认测试配置。

### 后续优化建议

当前最合理的下一步不是继续盲目提高全局阈值，而是做分类别阈值或补正式 AP/NDS 评估。

建议方向：

```text
1. 高频类别如 vehicle_car、human_pedestrian_adult、vehicle_motorcycle、vehicle_tricycle 可使用 0.2~0.25；
2. 弱类如 vehicle_bicycle、bicycle、group_vehicle_bicycle、animal 可保留 0.15~0.2；
3. 稀有类别需要单独分析，不宜简单用统一高阈值过滤；
4. 后续应补正式 AP/NDS 或公司内部评估指标；
5. 可抽样可视化 result.pkl，观察预测框空间位置是否合理。
```

---

## 更新后的核心产物补充

除原始 `epoch20/` 测试目录外，现在还应保存这两个阈值测试目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score025/
```

宿主机对应路径：

```text
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/
/home/ubuntu/WXY/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score025/
```

一条命令查看原始阈值、0.2 阈值和 0.25 阈值的结果目录：

```bash
echo "===== eval result score010 ====="; ls -lh /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20/; echo "===== eval result score020 ====="; ls -lh /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/; echo "===== eval result score025 ====="; ls -lh /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score025/
```
