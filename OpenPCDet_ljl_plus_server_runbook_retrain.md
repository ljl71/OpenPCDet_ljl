# OpenPCDet_ljl_plus 服务器操作说明与重新训练流程

> 项目目录（宿主机）：`/home/ubuntu/WXY/OpenPCDet_ljl_plus`  
> 项目目录（容器内）：`/workspace/OpenPCDet`  
> 数据目录（宿主机）：`/home/ubuntu/WXY/data`  
> 数据目录（容器内）：`/workspace/OpenPCDet/data`  
> 容器名：`detection3d_v5_plus`  
> 镜像：`pcdet_back_3dv2:20260520`  
> 推荐训练 GPU：`CUDA_VISIBLE_DEVICES=1`  
> 推荐训练配置：`batch_size=8, workers=4, epochs=20`  
> 推荐训练实验名：`formal_company_26cls_plus_bs8_e20`  
> 推荐 AP/mAP 评估阈值：`SCORE_THRESH=0.05`  
> 推荐可视化/实际输出阈值：`SCORE_THRESH=0.2`

---

## 1. 当前服务器项目结构

当前服务器 `/home/ubuntu/WXY` 下主要目录如下：

```text
/home/ubuntu/WXY
├── data
├── OpenPCDet_ljl
├── OpenPCDet_ljl_plus
├── OpenPCDet_ljl_bak_20260526_152808
├── OpenPCDet
├── OpenPCDetv4
├── v1.0-trainval
└── yuan
```

其中：

| 目录 | 作用 |
|---|---|
| `/home/ubuntu/WXY/OpenPCDet_ljl` | 之前已经跑通训练和测试的旧项目 |
| `/home/ubuntu/WXY/OpenPCDet_ljl_plus` | 当前新增评估指标后的 plus 项目 |
| `/home/ubuntu/WXY/data` | 公司正式 nuScenes 风格数据，两个项目共用 |
| `/home/ubuntu/WXY/OpenPCDet_ljl_plus/output` | 已复制旧项目训练产物，同时后续新训练也会写到这里 |

旧项目的 `output/` 已经复制到：

```text
/home/ubuntu/WXY/OpenPCDet_ljl_plus/output
```

复制后已经确认存在：

```text
/home/ubuntu/WXY/OpenPCDet_ljl_plus/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth

/home/ubuntu/WXY/OpenPCDet_ljl_plus/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl
```

---

## 2. plus 容器说明

当前新容器名为：

```text
detection3d_v5_plus
```

创建命令：

```bash
sudo docker run -it -d --gpus all --name detection3d_v5_plus -v /home/ubuntu/WXY/OpenPCDet_ljl_plus:/workspace/OpenPCDet -v /home/ubuntu/WXY/data:/workspace/OpenPCDet/data pcdet_back_3dv2:20260520 /bin/bash
```

挂载关系：

| 宿主机路径 | 容器内路径 | 说明 |
|---|---|---|
| `/home/ubuntu/WXY/OpenPCDet_ljl_plus` | `/workspace/OpenPCDet` | plus 项目代码和输出目录 |
| `/home/ubuntu/WXY/data` | `/workspace/OpenPCDet/data` | 公司正式数据 |

进入容器：

```bash
sudo docker exec -u root -it detection3d_v5_plus /bin/bash
```

进入项目目录：

```bash
cd /workspace/OpenPCDet
```

检查 GPU：

```bash
nvidia-smi
```

当前服务器有两张 RTX 3090，GPU 1 比较空，因此训练和测试建议使用：

```bash
CUDA_VISIBLE_DEVICES=1
```

---

## 3. plus 代码和评估功能检查

plus 项目中新增的评估相关文件：

```text
pcdet/datasets/company_nuscenes/company_nuscenes_eval.py
tools/company_nuscenes/evaluate_company_predictions.py
tools/company_nuscenes/smoke_test_company_evaluation.py
```

语法检查命令：

```bash
python -m py_compile pcdet/datasets/company_nuscenes/company_nuscenes_eval.py
python -m py_compile pcdet/datasets/company_nuscenes/company_nuscenes_dataset.py
python -m py_compile tools/company_nuscenes/evaluate_company_predictions.py
python -m py_compile tools/company_nuscenes/smoke_test_company_evaluation.py
```

这些命令无输出，说明语法检查通过。

新增评估冒烟测试：

```bash
python tools/company_nuscenes/smoke_test_company_evaluation.py
```

已通过：

```text
company_evaluation_smoke: PASS
```

说明 plus 分支新增的评估逻辑可以正常运行。

---

## 4. plus 分支新增了什么评估能力

旧版 evaluation 主要输出：

```text
recall_rcnn_0.3
recall_rcnn_0.5
recall_rcnn_0.7
Average predicted number
每类 GT / Pred 数量
```

plus 版本新增了公司 26 类自定义评估指标：

```text
AP@0.5m
AP@1.0m
AP@2.0m
AP@4.0m
mAP
precision / recall / F1
mATE
mASE
mAOE
距离分段 mAP
```

评估方式：

```text
class-matched XY-center distance AP
```

也就是：类别必须一致，并使用 BEV 平面中心点距离进行匹配。

注意：

```text
这不是官方 nuScenes AP/NDS。
```

因为公司数据是自定义 26 类，并且当前模型没有 velocity 和 attribute 输出，所以不能直接计算官方 nuScenes NDS。

---

## 5. 已有 checkpoint 的 plus 评估结果

### 5.1 `SCORE_THRESH=0.2` 结果

已有预测文件：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl
```

离线评估命令：

```bash
python tools/company_nuscenes/evaluate_company_predictions.py --result output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/result.pkl --infos data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl --output_json output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score020/company_metrics_summary.json
```

核心结果：

```text
mAP: 0.1369
P/R/F1 @2.00m: 0.5266 / 0.6399 / 0.5777
TP=117324
FP=105463
FN=66034
mATE=0.2943
mASE=0.1912
mAOE=0.7446 rad
Average predicted number: 45.709/frame
```

判断：

```text
SCORE_THRESH=0.2 的输出数量比较合理，precision 和 F1 明显更好，适合实际输出和可视化。
```

---

### 5.2 `SCORE_THRESH=0.05` 结果

重新测试命令：

```bash
cd /workspace/OpenPCDet/tools
CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --ckpt ../output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/ckpt/checkpoint_epoch_20.pth --batch_size 8 --workers 4 --extra_tag formal_company_26cls --eval_tag epoch20_score005_evalplus --set MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH 0.05
```

输出目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score005_evalplus/
```

指标文件：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls/eval/epoch_20/val/epoch20_score005_evalplus/final_result/data/company_metrics_summary.json
```

核心结果：

```text
mAP: 0.1500
P/R/F1 @2.00m: 0.1433 / 0.7429 / 0.2402
TP=136218
FP=814537
FN=47140
mATE=0.3400
mASE=0.2036
mAOE=0.8198 rad
Average predicted number: 195.067/frame
```

判断：

```text
SCORE_THRESH=0.05 保留了更多低分框，更适合计算 AP/mAP，但预测框数量过多，precision 很低，不适合作为最终可视化或部署阈值。
```

---

### 5.3 0.05 与 0.2 的用途区分

| 阈值 | mAP | Precision | Recall | F1 | 平均预测数/帧 | 建议用途 |
|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 0.1500 | 0.1433 | 0.7429 | 0.2402 | 195.067 | AP/mAP 评估 |
| 0.2 | 0.1369 | 0.5266 | 0.6399 | 0.5777 | 45.709 | 可视化、实际输出、展示 |

结论：

```text
0.05 用于看模型 AP/mAP 上限；
0.2 用于看实际输出效果和可视化效果。
```

---

## 6. 重新训练前检查

进入 plus 容器：

```bash
sudo docker exec -u root -it detection3d_v5_plus /bin/bash
```

进入项目目录：

```bash
cd /workspace/OpenPCDet
```

检查数据和 info 文件：

```bash
ls -lh data/nuscenes/v1.0-trainval/company_nuscenes_infos_train.pkl
ls -lh data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl
```

检查配置文件：

```bash
ls -lh tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
ls -lh tools/cfgs/dataset_configs/company_nuscenes_trainval_dataset.yaml
```

检查 GPU：

```bash
nvidia-smi
```

---

## 7. 重新训练推荐配置

### 7.1 推荐 `batch_size=8`

之前已经测试过：

```text
batch_size=4: GPU 1 显存约 5.1GB / 24GB
batch_size=8: GPU 1 显存约 8.6GB / 24GB
```

`batch_size=8, workers=4` 时训练速度约：

```text
2409 it/epoch
约 2.07~2.27 it/s
单 epoch 约 18~19 分钟
```

对比 `batch_size=1`：

```text
batch_size=1: 单 epoch 约 33~34 分钟
batch_size=8: 单 epoch 约 18~19 分钟
```

实际样本吞吐明显提升。

`workers=4` 时：

```text
d_time=0.00(0.01)
```

说明数据加载不是瓶颈，因此暂时没必要强制使用 `workers=8`。

当前推荐：

```text
batch_size=8
workers=4
epochs=20
```

---

### 7.2 为什么先训练 20 轮

20 轮不是最优值，而是第一版完整 baseline。

原因：

1. 之前 `batch_size=1, epochs=20` 已经完整跑通；
2. 使用 20 轮方便和旧实验比较；
3. `batch_size=8` 下每轮约 18~19 分钟，20 轮约 6~6.5 小时；
4. 比 1 轮 smoke test 更完整，又不会像 40/60 轮那样一开始成本过高。

注意：

```text
batch_size=8 的 20 epoch 和 batch_size=1 的 20 epoch 都是看 20 遍数据集，但参数更新步数不同。
```

`batch_size=8` 的更新步数约：

```text
2409 it/epoch × 20 epoch ≈ 48180 次参数更新
```

`batch_size=1` 的更新步数约：

```text
19268 it/epoch × 20 epoch = 385360 次参数更新
```

所以如果 `batch_size=8` 的 20 轮效果还没充分收敛，后续可以继续训练到 40 轮，或者再考虑学习率策略。

---

## 8. 正式重新训练命令

进入 `tools` 目录：

```bash
cd /workspace/OpenPCDet/tools
```

推荐使用新的实验名，避免覆盖旧实验：

```text
formal_company_26cls_plus_bs8_e20
```

训练命令：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --batch_size 8 --epochs 20 --workers 4 --extra_tag formal_company_26cls_plus_bs8_e20 --ckpt_save_interval 1 --max_ckpt_save_num 20
```

参数说明：

| 参数 | 值 | 说明 |
|---|---:|---|
| `CUDA_VISIBLE_DEVICES` | `1` | 使用第二张 RTX 3090 |
| `--batch_size` | `8` | 当前实测显存安全，吞吐较好 |
| `--epochs` | `20` | 第一版完整 baseline |
| `--workers` | `4` | 数据加载不是瓶颈，保持稳定 |
| `--extra_tag` | `formal_company_26cls_plus_bs8_e20` | 新实验名，避免覆盖旧结果 |
| `--ckpt_save_interval` | `1` | 每个 epoch 保存一次 checkpoint |
| `--max_ckpt_save_num` | `20` | 保留 20 个 checkpoint |

---

## 9. 训练时如何监控

另开一个宿主机终端，执行：

```bash
watch -n 1 nvidia-smi
```

重点观察：

```text
GPU 1 显存是否稳定
GPU-Util 是否有计算波动
是否出现 CUDA out of memory
```

GPU-Util 上下跳动是正常的，因为训练过程包含数据读取、voxelization、forward、backward、optimizer step 等步骤。只要显存稳定、loss 正常、没有 OOM，就可以继续训练。

训练日志中重点看：

```text
loss
lr
d_time
f_time
b_time
it/s
```

---

## 10. 训练输出位置

使用：

```text
--extra_tag formal_company_26cls_plus_bs8_e20
```

后，输出目录为：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/
```

宿主机对应目录：

```text
/home/ubuntu/WXY/OpenPCDet_ljl_plus/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/
```

checkpoint 目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/ckpt/
```

最终模型：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/ckpt/checkpoint_epoch_20.pth
```

---

## 11. 训练完成后评估

训练完成后，建议分别跑两个阈值：

```text
SCORE_THRESH=0.05：用于 AP/mAP 评估
SCORE_THRESH=0.2：用于可视化和实际输出效果
```

---

### 11.1 用 0.05 评估 AP/mAP

```bash
cd /workspace/OpenPCDet/tools

CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --ckpt ../output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/ckpt/checkpoint_epoch_20.pth --batch_size 8 --workers 4 --extra_tag formal_company_26cls_plus_bs8_e20 --eval_tag epoch20_score005_evalplus --set MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH 0.05
```

输出目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/eval/epoch_20/val/epoch20_score005_evalplus/
```

核心指标文件：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/eval/epoch_20/val/epoch20_score005_evalplus/final_result/data/company_metrics_summary.json
```

查看指标：

```bash
python -m json.tool ../output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/eval/epoch_20/val/epoch20_score005_evalplus/final_result/data/company_metrics_summary.json | head -160
```

---

### 11.2 用 0.2 评估实际输出效果

```bash
CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --ckpt ../output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/ckpt/checkpoint_epoch_20.pth --batch_size 8 --workers 4 --extra_tag formal_company_26cls_plus_bs8_e20 --eval_tag epoch20_score020_evalplus --set MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH 0.2
```

输出目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/eval/epoch_20/val/epoch20_score020_evalplus/
```

核心指标文件：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/eval/epoch_20/val/epoch20_score020_evalplus/final_result/data/company_metrics_summary.json
```

查看指标：

```bash
python -m json.tool ../output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/eval/epoch_20/val/epoch20_score020_evalplus/final_result/data/company_metrics_summary.json | head -160
```

---

## 12. 重新训练完成后的判断标准

### 12.1 看 0.05 结果

重点看：

```text
mAP
每类 AP
distance breakdown
```

与旧模型对比：

```text
旧模型 0.05 mAP: 0.1500
```

如果新模型 `0.05 mAP` 高于 0.1500，说明整体 AP/mAP 有提升。

---

### 12.2 看 0.2 结果

重点看：

```text
Precision
Recall
F1
Average predicted number
主类 AP
BEV 可视化效果
```

与旧模型对比：

```text
旧模型 0.2:
mAP = 0.1369
P/R/F1 = 0.5266 / 0.6399 / 0.5777
Average predicted number = 45.709/frame
```

如果新模型 `0.2 F1` 更高，且预测框数量合理，说明实际使用效果更好。

---

## 13. 可视化新模型 0.2 结果

如果可视化脚本存在：

```text
/workspace/OpenPCDet/tools/company_nuscenes/visualize_score020_bev.py
```

可视化命令：

```bash
cd /workspace/OpenPCDet

python tools/company_nuscenes/visualize_score020_bev.py --result output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/eval/epoch_20/val/epoch20_score020_evalplus/result.pkl --infos data/nuscenes/v1.0-trainval/company_nuscenes_infos_val.pkl --data_root data/nuscenes --out_dir output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/vis/epoch20_score020_bev --score_thresh 0.2 --num 20
```

输出目录：

```text
/workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/vis/epoch20_score020_bev/
```

查看：

```bash
find output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/vis/epoch20_score020_bev -name "*.png" | sort | head -20
cat output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/vis/epoch20_score020_bev/summary.txt
```

---

## 14. 如果训练中断怎么办

查看已有 checkpoint：

```bash
ls -lh /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/ckpt/
```

例如已经有：

```text
checkpoint_epoch_8.pth
```

可以从该 checkpoint 继续训练：

```bash
cd /workspace/OpenPCDet/tools

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --batch_size 8 --epochs 20 --workers 4 --extra_tag formal_company_26cls_plus_bs8_e20 --ckpt ../output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/ckpt/checkpoint_epoch_8.pth --start_epoch 8 --ckpt_save_interval 1 --max_ckpt_save_num 20
```

---

## 15. 推荐操作顺序

```text
1. 进入 plus 容器
2. 检查 GPU / info 文件 / 配置文件
3. 执行 batch_size=8、epochs=20 正式训练
4. 训练完成后用 SCORE_THRESH=0.05 测 AP/mAP
5. 再用 SCORE_THRESH=0.2 测实际输出效果
6. 对比旧模型 0.05 和 0.2 的指标
7. 生成新模型 0.2 的 BEV 可视化
8. 决定是否继续训练到 40 epoch 或调整学习率/类别策略
```

当前最推荐的正式训练命令是：

```bash
cd /workspace/OpenPCDet/tools
CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml --batch_size 8 --epochs 20 --workers 4 --extra_tag formal_company_26cls_plus_bs8_e20 --ckpt_save_interval 1 --max_ckpt_save_num 20
```

---

## 16. 结论

`OpenPCDet_ljl_plus` 在服务器上的代码、数据、旧模型结果和新增评估流程已经打通。当前已经验证：

```text
plus 容器创建成功
GPU 可见
新增评估文件存在
Python 语法检查通过
evaluation smoke test 通过
旧 result.pkl 可被 plus 评估
0.05 和 0.2 两套阈值评估均可用
```

下一步可以正式重新训练：

```text
batch_size=8
workers=4
epochs=20
extra_tag=formal_company_26cls_plus_bs8_e20
```

训练完成后重点比较：

```text
0.05 mAP 是否超过旧模型 0.1500
0.2 F1 是否超过旧模型 0.5777
0.2 平均预测数量是否仍然合理
主类 AP 是否稳定
长尾类是否有所改善
```
