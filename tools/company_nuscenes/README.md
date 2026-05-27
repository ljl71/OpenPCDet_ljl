# Company nuScenes Formal VoxelNeXt Adapter

This adapter keeps the official nuScenes dataset implementation untouched and provides the formal company-data path for 26-class VoxelNeXt training.

## Scope

- Dataset: `CompanyNuScenesDataset`
- Model config: `tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml`
- Formal data root: `data/nuscenes`
- Metadata: `data/nuscenes/v1.0-trainval/*.json`
- LiDAR: `data/nuscenes/samples/LIDAR_TOP/*.bin`
- LiDAR only, one frame: `MAX_SWEEPS: 1`
- Input: four `float32` columns configured as `x, y, z, intensity`
- The inspected converted data has a zero-valued fourth column; it is not measured intensity or `ring/timestamp`
- Output contract: 26 company classes; the observed formal annotations contain positive boxes for 24 of them
- Evaluation: custom 26-class center-distance AP/mAP, emitted-prediction precision/recall,
  matched-box translation/scale/orientation error and distance-range breakdown
- The evaluation is not official nuScenes NDS because this model does not output velocity
  or attributes and the company taxonomy is not the official nuScenes detection taxonomy

Do not use `data/v1.0-trainval`: it references `.pcd` entries without the usable sibling sample directory.

## Prepare Formal Infos

From `/workspace/OpenPCDet`:

```bash
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

The split is scene-level and guarantees that every class with retained annotated boxes has training coverage; validation coverage is reserved whenever the scene constraints allow it. Known boxes below `--min_lidar_points` are filtered for coverage, while `num_lidar_pts: null` is treated as unknown and retained with an explicit report from the checker. `--strict` fails for missing LiDAR files or labels outside the configured 26-class contract. Do not use `--require_all_classes` with the currently observed annotations, because two declared classes contain no positive boxes.

Validated on the mounted formal annotations on 2026-05-26:

- `895317` boxes remain under the training retention rule, including `611` boxes with unknown `num_lidar_pts`.
- The revised scene split has `329` training scenes and `83` validation scenes.
- All 24 classes with retained boxes appear in training.
- `vehicle_emergency_other` is absent from validation because its retained boxes occur in one scene only, which is reserved for training.

## Smoke Test

First verify dataset loading:

```bash
python tools/company_nuscenes/smoke_test_company_dataloader.py \
  --cfg_file tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
```

Then run one VoxelNeXt forward and backward step in the CUDA container:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/company_nuscenes/smoke_test_formal_voxelnext.py \
  --cfg_file tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml \
  --workers 0
```

The second command must print `formal_voxelnext_smoke: PASS` before full training is started.

The metric implementation also has a CPU-only synthetic check:

```bash
python tools/company_nuscenes/smoke_test_company_evaluation.py
```

## Evaluate Detection Quality

`test.py` now invokes `CompanyNuScenesDataset.evaluation()` to report:

- Per-class `AP` at XY center-distance thresholds `0.5`, `1.0`, `2.0`, and `4.0` meters.
- `mAP` over validation classes that actually contain retained GT boxes.
- Precision, recall, and F1 at a `2.0 m` match threshold for the predictions emitted by post-processing.
- `mATE`, `mASE`, and `mAOE` on matched predictions, measuring center, size, and yaw error;
  traffic-cone yaw is ignored and barrier yaw is treated as `pi`-periodic.
- The same summary split into `0-30 m`, `30-50 m`, and `50 m+` distance ranges.

The validation split may legitimately have no GT for some declared classes. Their prediction
counts are still displayed, but they are excluded from `mAP` because AP is undefined without
positive examples. Their false positives are included in the emitted-prediction micro precision.

For a saved prediction file, compute metrics without rerunning model inference:

```bash
python tools/company_nuscenes/evaluate_company_predictions.py \
  --result output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl \
  --infos data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
```

This writes `company_metrics_summary.json` beside `result.pkl`. For useful AP comparison,
generate predictions with a low model `SCORE_THRESH` so the precision-recall curve is not
prematurely clipped; use `--min_score` on the saved predictions to inspect deployment
precision/recall tradeoffs afterward.

## Train

From `/workspace/OpenPCDet/tools`:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml \
  --batch_size 1 \
  --epochs 20 \
  --workers 4 \
  --extra_tag formal_company_26cls
```

For the complete server-side checklist and interpretation of the dataset limitations, see the workspace deliverable `VoxelNeXt_26类_公司正式数据集正确跑通指南.md`.
