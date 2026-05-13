# Company nuScenes Adapter

This adapter keeps the official nuScenes dataset code untouched and adds a first-stage LiDAR-only path for company data with nuScenes-like JSON tables.

## First-Stage Scope

- Dataset: `CompanyNuScenesDataset`
- Data root: `data/company_nuscenes/v1.0-mini`
- LiDAR only, no camera dependency
- Single frame only: `MAX_SWEEPS: 1`
- Raw LiDAR binary dimension: `LIDAR_POINT_DIM: 4`
- Model input features: `x, y, z, intensity`
- Debug config: 12 classes that actually appear in the current mini annotations
- Full config: 26 company classes from `category.json`
- DB sampling disabled
- Evaluation is a smoke count report, not AP/NDS

## Prepare Data

From the repo root:

```bash
python tools/company_nuscenes/prepare_company_mini.py
python tools/company_nuscenes/create_company_infos.py
python tools/company_nuscenes/check_company_infos.py
```

`prepare_company_mini.py` copies metadata and LiDAR `.bin` files into `data/company_nuscenes/v1.0-mini`, rewrites LiDAR rows in `sample_data.json` from `.pcd` to `.bin`, and creates `ImageSets/train.txt` and `ImageSets/val.txt`. The split tries to keep classes that appear in only one scene inside train.

## Smoke Test

After installing the OpenPCDet runtime dependencies:

```bash
python tools/company_nuscenes/smoke_test_company_dataloader.py
```

Expected checks:

- dataset length is non-zero
- `points` has 4 feature columns plus batch index after collation
- `gt_boxes` is non-empty
- the last `gt_boxes` column contains class ids

## Train

From `tools/`:

```bash
python train.py \
  --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml \
  --batch_size 1 \
  --epochs 20
```

For the mini-only 12-class debugging path, use:

```bash
python train.py \
  --cfg_file cfgs/nuscenes_models/company_centerpoint_12cls.yaml \
  --batch_size 1 \
  --epochs 5
```

If you want the VoxelNeXt style used by the platform vendor config, use:

```bash
python train.py \
  --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls.yaml \
  --batch_size 1 \
  --epochs 20
```

## Auto Labeling

`tools/inference/inference_nms.py` now supports environment-variable overrides:

```bash
export PCDET_CFG_FILE=/workspace/OpenPCDet/tools/cfgs/nuscenes_models/company_voxelnext_26cls.yaml
export PCDET_CKPT_PATH=/workspace/OpenPCDet/output/.../ckpt/checkpoint_epoch_20.pth
```

After training a 26-class checkpoint, point the platform to the matching config and checkpoint. The returned label names come from `cfg.CLASS_NAMES`.
