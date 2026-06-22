# Company NuScenes T23 2026 Adapter

This directory contains the 26-class CompanyNuScenes adapter for the T23 nuScenes-like dataset `NuScenes-develop_t23_2026`.

Scope:

- Dataset: `CompanyNuScenesDataset`
- Primary model config: `tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml`
- Default data root: `data/NuScenes-develop_t23_2026`
- Metadata: `data/NuScenes-develop_t23_2026/v1.0-develop/*.json`
- LiDAR: `data/NuScenes-develop_t23_2026/samples/<date>/<scene>/lidar_fusion/*.pcd`
- Classes: original 26 company classes, no 10-class merge
- LiDAR only, single frame: `MAX_SWEEPS: 1`
- Point format: `x, y, z, intensity`
- Evaluation: custom 26-class center-distance AP/mAP, not official nuScenes NDS

## Prepare 26-Class Infos

From the `OpenPCDet_ljl_plus` repository root:

```bash
python tools/company_nuscenes/preview_formal_split.py

python tools/company_nuscenes/create_company_infos.py \
  --data_path data/NuScenes-develop_t23_2026 \
  --save_path data/NuScenes-develop_t23_2026 \
  --version v1.0-develop \
  --max_sweeps 1 \
  --min_lidar_points 1

python tools/company_nuscenes/check_company_infos.py \
  --cfg_file tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml \
  --strict
```

The generator handles T23 samples whose `sample.json` records do not contain a `data` field by resolving the LiDAR sample from `sample_data.json`, `calibrated_sensor.json`, and `sensor.json`.

## Smoke Tests

```bash
python tools/company_nuscenes/smoke_test_company_dataloader.py \
  --cfg_file tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml

CUDA_VISIBLE_DEVICES=0 python tools/company_nuscenes/smoke_test_formal_voxelnext.py \
  --cfg_file tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml \
  --workers 0

python tools/company_nuscenes/smoke_test_company_evaluation.py
```

If the dataset is mounted elsewhere, pass `--data_path <path>` and `--version v1.0-develop` to the smoke scripts, or override training with `--set DATA_CONFIG.DATA_PATH <path> DATA_CONFIG.VERSION v1.0-develop`.

## Train VoxelNeXt

From `OpenPCDet_ljl_plus/tools`:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml \
  --batch_size 1 \
  --workers 4 \
  --extra_tag t23_2026_26cls
```

For an external data mount:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml \
  --batch_size 1 \
  --workers 4 \
  --extra_tag t23_2026_26cls \
  --set DATA_CONFIG.DATA_PATH /path/to/NuScenes-develop_t23_2026 DATA_CONFIG.VERSION v1.0-develop
```

`CompanyNuScenesDataset.evaluation()` writes `company_metrics_summary.json` next to `result.pkl` when `tools/test.py` supplies an output directory.
