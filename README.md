# OpenPCDet_ljl：公司 26 类 LiDAR 3D 检测项目

> 基于 [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) 的公司 nuScenes 风格数据适配工程。
> 当前推荐分支：`codex/company-26cls-evaluation-plus`
> 目标任务：使用公司正式数据训练和评估 26 类 `VoxelNeXt` LiDAR 3D 检测模型。

本仓库不再只是原版 OpenPCDet 示例工程。本项目已完成公司正式数据读取、26 类标签映射、scene-level 数据划分、VoxelNeXt 训练和测试、BEV 可视化，并在 `plus` 分支新增了可用于判断检测质量的 26 类评估指标。

## 项目概览

| 项目项 | 当前实现 |
|---|---|
| 检测模型 | `VoxelNeXt` |
| 数据集类 | `CompanyNuScenesDataset` |
| 类别数 | 公司定义的 26 类 |
| 数据格式 | nuScenes 风格 JSON + `samples/LIDAR_TOP/*.bin` |
| 输入方式 | LiDAR-only，单帧，`MAX_SWEEPS=1` |
| 点云输入 | 每点 4 个 `float32`：`x, y, z, intensity` |
| 正式模型配置 | `tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml` |
| 正式数据配置 | `tools/cfgs/dataset_configs/company_nuscenes_trainval_dataset.yaml` |
| 训练实验名 | `formal_company_26cls` |
| 已完成训练 | `20 epoch`，最终模型 `checkpoint_epoch_20.pth` |
| 推荐展示阈值 | `SCORE_THRESH=0.2` |
| 当前评估 | 自定义 26 类中心距离 `AP/mAP`、`precision/recall/F1`、匹配框误差、距离分段结果 |

说明：已经核验的正式 `.bin` 点云第 4 列是转换生成的零占位值，并不是真实反射强度、`ring` 或时间戳。当前标注表声明 26 类，但正式 annotation 中实际具有正样本的类别为 24 类。

## 服务器与容器约定

完整实验记录以 [OpenPCDet_ljl_company_26cls_project_readme.md](OpenPCDet_ljl_company_26cls_project_readme.md) 为主。记录中的部署环境为：

| 项目 | 路径或名称 |
|---|---|
| 宿主机工程目录 | `/home/ubuntu/WXY/OpenPCDet_ljl` |
| 容器工程目录 | `/workspace/OpenPCDet` |
| 数据挂载 | `/home/ubuntu/WXY/data -> /workspace/OpenPCDet/data` |
| 工程挂载 | `/home/ubuntu/WXY/OpenPCDet_ljl -> /workspace/OpenPCDet` |
| Docker 容器名 | `detection3d_v5` |
| 已验证 GPU | RTX 3090，训练时主要使用 `CUDA_VISIBLE_DEVICES=1` |

## 本分支新增内容

`codex/company-26cls-evaluation-plus` 在已跑通的正式训练版本上增加了检测质量评估能力：

- 每类 `AP@0.5m`、`AP@1.0m`、`AP@2.0m`、`AP@4.0m`。
- 对验证集中存在有效 GT 的类别计算整体 `mAP`。
- 在 `2.0m` 匹配条件下报告输出预测的 `precision`、`recall` 和 `F1`。
- 对匹配成功的框报告 `mATE`（中心误差）、`mASE`（尺寸误差）和 `mAOE`（方向误差）。
- 报告 `0-30m`、`30-50m`、`50m+` 三个距离区间的结果。
- 支持对已有 `result.pkl` 离线重算指标，无需重新运行模型推理。

这套指标适合比较当前公司 26 类模型的检测效果和阈值取舍，但**不是官方 nuScenes NDS**：本项目类别体系不是官方 10 类，并且当前模型不输出 NDS 所需要的速度与属性项。

## 数据与流程

正式数据使用以下目录结构：

```text
data/nuscenes/
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
    |-- ImageSets/
    |   |-- train.txt
    |   `-- val.txt
    |-- company_nuscenes_infos_train.pkl
    `-- company_nuscenes_infos_val.pkl
```

不要使用 `data/v1.0-trainval` 作为本轮正式训练入口；已有检查表明该早期目录仍可能引用无法闭环读取的 `.pcd` 路径。

整体处理流程如下：

```text
公司 JSON + LiDAR bin
        -> CompanyNuScenesDataset / company_nuscenes_utils.py
        -> scene-level train/val split + info pkl
        -> VoxelNeXt 26 类训练
        -> checkpoint_epoch_*.pth
        -> test.py 推理生成 result.pkl
        -> AP/mAP + P/R + 误差分析 + BEV 可视化
```

## 关键代码

| 路径 | 作用 |
|---|---|
| `pcdet/datasets/company_nuscenes/company_nuscenes_dataset.py` | 公司数据加载与评估入口 |
| `pcdet/datasets/company_nuscenes/company_nuscenes_utils.py` | 类别映射、JSON 解析、info 生成与划分 |
| `pcdet/datasets/company_nuscenes/company_nuscenes_eval.py` | `plus` 分支新增的 26 类评估实现 |
| `pcdet/datasets/company_nuscenes/point_io.py` | 正式点云格式读取 |
| `pcdet/models/dense_heads/voxelnext_head.py` | 26 类 multi-head 检测头相关逻辑 |
| `tools/company_nuscenes/create_company_infos.py` | 生成正式训练/验证 info |
| `tools/company_nuscenes/check_company_infos.py` | 检查 info、点云路径和类别覆盖 |
| `tools/company_nuscenes/evaluate_company_predictions.py` | 离线评估已有预测结果 |
| `tools/company_nuscenes/visualize_score020_bev.py` | BEV 可视化 |

## 快速开始

以下命令以服务器容器内工程目录 `/workspace/OpenPCDet` 为例。

### 1. 预览划分并生成 info

```bash
cd /workspace/OpenPCDet

python tools/company_nuscenes/preview_formal_split.py \
  --data_path data/nuscenes \
  --version v1.0-trainval \
  --train_ratio 0.8 \
  --seed 0 \
  --min_lidar_points 1

python tools/company_nuscenes/create_company_infos.py \
  --data_path data/nuscenes \
  --save_path data/nuscenes \
  --version v1.0-trainval \
  --max_sweeps 1 \
  --train_ratio 0.8 \
  --seed 0 \
  --min_lidar_points 1

python tools/company_nuscenes/check_company_infos.py \
  --root data/nuscenes/v1.0-trainval \
  --data_root data/nuscenes \
  --strict \
  --min_lidar_points 1
```

### 2. 训练前 smoke test

```bash
python tools/company_nuscenes/smoke_test_company_dataloader.py \
  --cfg_file tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml

CUDA_VISIBLE_DEVICES=0 python tools/company_nuscenes/smoke_test_formal_voxelnext.py \
  --cfg_file tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml \
  --workers 0

python tools/company_nuscenes/smoke_test_company_evaluation.py
```

最后一个命令为 `plus` 分支新增的 CPU 评估逻辑自检，成功时打印：

```text
company_evaluation_smoke: PASS
```

### 3. 训练 VoxelNeXt

已有的 `formal_company_26cls` 完整训练产物来自 `batch_size=1, workers=4`。首次确认新环境链路时仍可使用这一保守参数；在已验证的 RTX 3090 环境中，后续正式训练推荐使用 `batch_size=8, workers=4`，并以新的 experiment tag 保存结果。

```bash
cd /workspace/OpenPCDet/tools

CUDA_VISIBLE_DEVICES=1 python train.py \
  --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml \
  --batch_size 8 \
  --epochs 20 \
  --workers 4 \
  --extra_tag formal_company_26cls_bs8
```

该推荐重训命令的输出目录：

```text
output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_bs8/
```

### 4. 测试 checkpoint 并输出评估指标

下面命令测试文档中已经完成的 `batch_size=1` / `formal_company_26cls` 第 20 轮模型：

```bash
cd /workspace/OpenPCDet/tools

CUDA_VISIBLE_DEVICES=1 python test.py \
  --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml \
  --ckpt ../output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth \
  --batch_size 1 \
  --workers 4 \
  --extra_tag formal_company_26cls \
  --eval_tag epoch20_score020 \
  --set MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH 0.2
```

测试时 `CompanyNuScenesDataset.evaluation()` 会打印每类 AP、整体 mAP、P/R/F1 和距离分段指标，同时在结果目录写出：

```text
company_metrics_summary.json
result.pkl
```

### 5. 对已有预测离线计算指标

服务器上已存在 `result.pkl` 时，不需要重新推理：

```bash
cd /workspace/OpenPCDet

python tools/company_nuscenes/evaluate_company_predictions.py \
  --result output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl \
  --infos data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
```

比较模型本身的 AP 时，建议先以较低的推理 `SCORE_THRESH` 生成预测，保留更完整的 PR 曲线；`SCORE_THRESH=0.2` 更适合当前可视化和实际输出数量的平衡观察。

### 6. 生成 BEV 可视化

使用文档中推荐的 `SCORE_THRESH=0.2` 预测结果抽样查看 GT 与预测框：

```bash
cd /workspace/OpenPCDet

python tools/company_nuscenes/visualize_score020_bev.py \
  --result output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl \
  --infos data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl \
  --data_root data/nuscenes \
  --out_dir output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev \
  --score_thresh 0.2 \
  --num 20
```

可视化中绿色框为 GT，红色框为预测；输出目录包含 `bev_idx_*.png` 和 `summary.txt`。

## 已验证状态

2026 年 5 月 26 日的正式数据检查和训练记录：

| 项目 | 结果 |
|---|---:|
| scene 总数 | 412 |
| sample 总数 | 24,142 |
| LIDAR_TOP 文件 | 24,142 |
| train scenes / samples | 329 / 19,268 |
| val scenes / samples | 83 / 4,874 |
| 按 `num_lidar_pts >= 1` 或未知点数保留的框 | 895,317 |
| 实际有有效标注的类别 | 24 / 26 |
| 已完成训练 | 20 epoch |
| 最终 checkpoint | `checkpoint_epoch_20.pth` |

在尚未加入本分支评估模块前，已有模型测试记录如下。这些数值是 recall 和输出数量分析，不是 mAP；现在可使用离线评估工具对对应 `result.pkl` 补算本分支指标。

| `SCORE_THRESH` | `recall@0.3` | `recall@0.5` | `recall@0.7` | 平均预测数/帧 | 用途判断 |
|---:|---:|---:|---:|---:|---|
| `0.10` | 0.724822 | 0.517463 | 0.222161 | 103.799 | 召回较高，误检风险偏大 |
| `0.20` | 0.657681 | 0.495375 | 0.220149 | 45.709 | 当前推荐的平衡阈值 |
| `0.25` | 0.624331 | 0.481173 | 0.218387 | 36.444 | 更保守，弱类可能被压制 |

## 已有产物位置

文档记录的已完成实验产物位于容器内下列目录：

| 产物 | 路径 |
|---|---|
| 训练 info | `/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl` |
| 验证 info | `/workspace/OpenPCDet/data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl` |
| 最终 checkpoint | `/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth` |
| `SCORE_THRESH=0.2` 预测 | `/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl` |
| BEV 图片 | `/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/vis/epoch20_score020_bev/` |

这些训练产物、预测文件和图片不提交进 Git 仓库；本仓库保存的是产生与分析它们的代码、配置和说明文档。

## 文档导航

| 文档 | 内容 |
|---|---|
| [tools/company_nuscenes/README.md](tools/company_nuscenes/README.md) | 当前代码入口、正式 info、训练与 `plus` 评估说明 |
| [OpenPCDet_ljl_company_26cls_project_readme.md](OpenPCDet_ljl_company_26cls_project_readme.md) | 本首页主要依据：正式部署、训练、测试、阈值分析和可视化的完整实验记录 |
| [VoxelNeXt_26类_公司正式数据集正确跑通指南.md](VoxelNeXt_26类_公司正式数据集正确跑通指南.md) | 正式数据训练的逐步操作指南 |
| [OpenPCDet_ljl_26cls_VoxelNeXt_runbook_with_why.md](OpenPCDet_ljl_26cls_VoxelNeXt_runbook_with_why.md) | 带原因解释的运行手册与排障记录 |
| [OpenPCDet_ljl_26cls_training_outputs_summary_updated_with_vis.md](OpenPCDet_ljl_26cls_training_outputs_summary_updated_with_vis.md) | 训练产物、测试结果和 BEV 输出整理 |
| [COMPANY_NUSCENES_26CLS_GUIDE.md](COMPANY_NUSCENES_26CLS_GUIDE.md) | 公司 26 类数据、配置和自动标注总体说明 |
| [SERVER_RUN_GUIDE.md](SERVER_RUN_GUIDE.md) | 服务器环境部署与常见问题 |
| [README.zh-CN.md](README.zh-CN.md) | 上游 OpenPCDet 中文介绍 |

其中部分早期实验文档保留了“evaluation 仍为 smoke/count”的历史记录；在本 `plus` 分支中，以本首页和 `tools/company_nuscenes/README.md` 中的新增评估说明为准。

## 说明与致谢

本项目是在 OpenPCDet 的基础上为公司 26 类 nuScenes 风格 LiDAR 数据开发的适配与实验分支。OpenPCDet 原始模型、框架说明、安装文档及引用信息请参见：

- [OpenPCDet 官方仓库](https://github.com/open-mmlab/OpenPCDet)
- [上游中文 README](README.zh-CN.md)
- [安装说明](docs/INSTALL.md)

使用或发布基于本仓库的模型与代码时，请同时遵循 OpenPCDet 原项目的许可证与引用要求。
