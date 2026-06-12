# T23 2026 NuScenes-Like 数据集适配说明

本文档说明如何在 `FSHNet_ljl` 和 `OpenPCDet_ljl_plus` 中使用
`NuScenes-develop_t23_2026` 数据集训练。当前适配默认走 26 类
CompanyNuScenes 配置，也保留 FSHNet 的 10 类合并配置。

## 数据集格式

该数据集不是官方 nuScenes 原包，而是 nuScenes-like 自定义数据集：

- `v1.0-develop/`：nuScenes 风格 JSON 表。
- `samples/`：原始帧数据，按 `日期/场景/传感器` 组织。
- `sensor/`：全局标定模板。
- `can_bus/`：pose 与 steeranglefeedback。
- `gt_database_1sweeps_withvelo/`：OpenPCDet GT sampling 数据库。
- `infos_train_01sweeps_withvelo_filter_True.pkl`：包内自带训练索引，路径来自旧机器，适配层会自动重定向到当前 `DATA_PATH/samples/...`。

典型场景目录：

```text
NuScenes-develop_t23_2026/
  v1.0-develop/
  samples/<date>/<scene>/
    <scene>.json
    times.json
    sensor/*.json
    lidar_fusion/*.pcd
    camera_cam2_compressed/*.jpg
    camera_cam3_compressed/*.jpg
    camera_cam4_compressed/*.jpg
    camera_cam5_compressed/*.jpg
    camera_cam7_compressed/*.jpg
    camera_cam8_compressed/*.jpg
    aps_localization_fuse_position/*.yaml
```

点云是 binary PCD，字段为 `x y z intensity`，训练配置中按 4 维点读取。

## 推荐目录

tar 不能直接用于随机读取训练，建议先解压一次，再用目录链接给两个项目共用：

```powershell
tar -xf I:\NuScenes-develop_t23_2026.tar -C I:\

New-Item -ItemType Directory -Force I:\FSHNet_ljl\data
New-Item -ItemType Junction `
  -Path I:\FSHNet_ljl\data\NuScenes-develop_t23_2026 `
  -Target I:\NuScenes-develop_t23_2026

New-Item -ItemType Directory -Force I:\OpenPCDet_ljl_plus\data
New-Item -ItemType Junction `
  -Path I:\OpenPCDet_ljl_plus\data\NuScenes-develop_t23_2026 `
  -Target I:\NuScenes-develop_t23_2026
```

如果不建 junction，也可以在训练时用 `--set DATA_CONFIG.DATA_PATH I:/NuScenes-develop_t23_2026`
覆盖路径。

## 生成 26 类 infos

两个项目都已修复 `sample.json` 无 `data` 字段的问题，会从 `sample_data.json`
反查 `LIDAR_TOP`。

FSHNet:

```powershell
cd I:\FSHNet_ljl
python tools\company_nuscenes\create_company_infos.py `
  --data_path I:/NuScenes-develop_t23_2026 `
  --save_path I:/NuScenes-develop_t23_2026 `
  --version v1.0-develop `
  --max_sweeps 1 `
  --min_lidar_points 1
```

OpenPCDet_ljl_plus:

```powershell
cd I:\OpenPCDet_ljl_plus
python tools\company_nuscenes\create_company_infos.py `
  --data_path I:/NuScenes-develop_t23_2026 `
  --save_path I:/NuScenes-develop_t23_2026 `
  --version v1.0-develop `
  --max_sweeps 1 `
  --min_lidar_points 1
```

生成后应出现：

```text
I:\NuScenes-develop_t23_2026\v1.0-develop\company_nuscenes_infos_train.pkl
I:\NuScenes-develop_t23_2026\v1.0-develop\company_nuscenes_infos_val.pkl
```

## 检查与 smoke test

FSHNet:

```powershell
cd I:\FSHNet_ljl
python tools\company_nuscenes\check_company_infos.py `
  --root I:/NuScenes-develop_t23_2026/v1.0-develop `
  --data_root I:/NuScenes-develop_t23_2026 `
  --strict

python tools\company_nuscenes\smoke_test_company_dataloader.py `
  --cfg_file tools/cfgs/nuscenes_models/company_fshnet_26cls_trainval.yaml `
  --data_path I:/NuScenes-develop_t23_2026 `
  --version v1.0-develop
```

OpenPCDet_ljl_plus:

```powershell
cd I:\OpenPCDet_ljl_plus
python tools\company_nuscenes\check_company_infos.py `
  --root I:/NuScenes-develop_t23_2026/v1.0-develop `
  --data_root I:/NuScenes-develop_t23_2026 `
  --strict

python tools\company_nuscenes\smoke_test_company_dataloader.py `
  --cfg_file I:/OpenPCDet_ljl_plus/tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml `
  --data_path I:/NuScenes-develop_t23_2026 `
  --version v1.0-develop
```

## 训练命令

FSHNet 26 类：

```powershell
cd I:\FSHNet_ljl
python tools\train.py `
  --cfg_file tools/cfgs/nuscenes_models/company_fshnet_26cls_trainval.yaml `
  --batch_size 1 `
  --workers 4 `
  --extra_tag t23_2026_26cls `
  --set DATA_CONFIG.DATA_PATH I:/NuScenes-develop_t23_2026 DATA_CONFIG.VERSION v1.0-develop
```

OpenPCDet_ljl_plus 26 类 VoxelNeXt：

```powershell
cd I:\OpenPCDet_ljl_plus\tools
python train.py `
  --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml `
  --batch_size 1 `
  --workers 4 `
  --extra_tag t23_2026_26cls `
  --set DATA_CONFIG.DATA_PATH I:/NuScenes-develop_t23_2026 DATA_CONFIG.VERSION v1.0-develop
```

## 适配要点

- `CompanyNuScenesDataset` 支持 PCD 与 bin 两种点云，当前数据走 PCD。
- 如果 info 里是旧绝对路径，dataloader 会从路径中的 `samples/...` 后缀重定向到当前 `DATA_PATH`。
- `FILTER_MIN_POINTS_IN_GT` 在缺少 `num_lidar_pts` 的旧 PKL 上会自动跳过，不再报错。
- FSHNet 配置启用 `ZERO_VELOCITY_FOR_FSHNET`，用零速度补齐模型原有 velocity head 的 shape；CompanyNuScenes 自定义评测不评 velocity。
- `MAX_SWEEPS` 当前保持为 `1`，该数据没有官方 nuScenes sweeps/radar 结构。

