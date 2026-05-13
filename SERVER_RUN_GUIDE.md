# OpenPCDet_ljl 公司服务器运行教程

本文档说明如何把当前电脑上的 `OpenPCDet_ljl` 拷贝到公司 Ubuntu 20.04 服务器上，并完成环境检查、依赖安装、CUDA 扩展编译、数据准备、训练、断点续训和常见问题排查。

默认假设服务器路径如下：

```bash
/workspace/OpenPCDet_ljl
```

如果你实际放在其他路径，请把下面命令中的 `/workspace/OpenPCDet_ljl` 替换成你的真实路径。

---

## 1. 先确认服务器基本信息

登录服务器后，先记录系统、磁盘、内存和当前目录：

```bash
hostname
whoami
pwd
uname -a
lsb_release -a
df -h
free -h
```

重点看：

1. 系统是否是 Ubuntu 20.04 或类似 Linux。
2. `/workspace`、`/data`、`/home` 哪个目录空间最大。
3. 数据集和训练输出不要放到空间很小的系统盘。

建议项目和输出放在大盘，例如：

```bash
/workspace/OpenPCDet_ljl
/data/company_nuscenes
```

---

## 2. 检查服务器显卡和 CUDA 配置

### 2.1 查看 GPU

```bash
nvidia-smi
```

如果正常，会看到类似信息：

```text
Driver Version: xxx.xx
CUDA Version: xx.x
GPU Name
Memory-Usage
```

进一步查看每张卡：

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
```

重点记录：

1. GPU 型号，例如 A10、A100、3090、4090、V100。
2. 显存大小，例如 24 GB、40 GB、80 GB。
3. Driver Version。
4. `nvidia-smi` 顶部显示的 CUDA Version。

注意：`nvidia-smi` 里显示的 CUDA Version 代表驱动最高支持的 CUDA 运行时版本，不等于当前环境里 PyTorch 实际使用的 CUDA 版本。

### 2.2 查看 CUDA toolkit / nvcc

OpenPCDet 的 `python setup.py develop` 会编译 CUDA 扩展，所以最好服务器上有 `nvcc`。

```bash
which nvcc
nvcc -V
```

如果找不到 `nvcc`，再查：

```bash
ls /usr/local | grep cuda
echo $CUDA_HOME
echo $PATH
echo $LD_LIBRARY_PATH
```

如果公司服务器使用 module 管理 CUDA，查：

```bash
module avail cuda
module list
```

加载示例：

```bash
module load cuda/11.8
```

加载后重新确认：

```bash
which nvcc
nvcc -V
```

### 2.3 查看 gcc / g++

```bash
gcc --version
g++ --version
```

Ubuntu 20.04 默认 gcc 9 通常可以用。若编译 CUDA 扩展时报 gcc 版本不支持，需要按公司服务器 CUDA 版本切换 gcc，或者请管理员提供对应编译环境。

---

## 3. 检查公司服务器已有 conda 环境

不要只看环境名判断能不能跑。环境名叫得像不代表能跑，必须实际检查 Python、PyTorch、spconv、pcdet、CUDA 扩展。

### 3.1 找 conda 在哪里

```bash
which conda
conda --version
conda info
conda env list
```

如果 `which conda` 找不到：

```bash
ls ~/anaconda3/bin/conda
ls ~/miniconda3/bin/conda
ls /opt/conda/bin/conda
```

常见初始化方式：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
```

或：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

然后再执行：

```bash
conda env list
```

### 3.2 检查某个已有环境能不能复用

假设你看到一个环境叫 `<ENV_NAME>`：

```bash
conda activate <ENV_NAME>

which python
python -V
which pip
pip -V
pip list | egrep "torch|torchvision|spconv|cumm|numpy|numba|pcdet|easydict|yaml|SharedArray"
```

检查 PyTorch：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

检查 spconv：

```bash
python - <<'PY'
try:
    import spconv
    print("spconv:", spconv.__version__)
except Exception as e:
    print("spconv import failed:", repr(e))
PY
```

检查当前环境是否能导入本项目：

```bash
cd /workspace/OpenPCDet_ljl
python - <<'PY'
import pcdet
print("pcdet:", pcdet.__file__)
PY
```

检查 OpenPCDet CUDA 扩展：

```bash
cd /workspace/OpenPCDet_ljl
python - <<'PY'
from pcdet.ops.iou3d_nms import iou3d_nms_cuda
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_cuda
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_stack_cuda
print("pcdet cuda ops ok")
PY
```

判断标准：

1. `torch.cuda.is_available()` 必须是 `True`。
2. `spconv` 必须能 import。
3. `pcdet` 必须指向 `/workspace/OpenPCDet_ljl/pcdet`。
4. `pcdet cuda ops ok` 必须通过。

如果 4 条都通过，可以考虑复用该环境。只要有一条失败，更建议新建环境。

---

## 4. 拷贝项目到服务器

### 4.1 推荐拷贝内容

推荐拷贝：

```text
OpenPCDet_ljl/
├── pcdet/
├── tools/
├── docs/
├── docker/
├── requirements.txt
├── setup.py
├── COMPANY_NUSCENES_26CLS_GUIDE.md
└── SERVER_RUN_GUIDE.md
```

如果 mini 数据已经在本机准备好，也可以一起拷：

```text
OpenPCDet_ljl/data/company_nuscenes/v1.0-mini/
```

不建议拷贝或不建议依赖：

```text
build/
pcdet.egg-info/
output/
*.so
*.pyd
```

这些是本机或旧环境的编译/训练产物，到 Linux 服务器上要重新生成。

### 4.2 用 rsync 拷贝

在源机器能访问服务器时，可用：

```bash
rsync -avh --progress \
  --exclude build \
  --exclude pcdet.egg-info \
  --exclude output \
  --exclude "__pycache__" \
  OpenPCDet_ljl/ user@server:/workspace/OpenPCDet_ljl/
```

如果数据很大，建议代码和数据分开传。

---

## 5. 新建干净 conda 环境

如果服务器没有可复用环境，建议新建：

```bash
conda create -n openpcdet_ljl python=3.8 -y
conda activate openpcdet_ljl
```

确认：

```bash
which python
python -V
```

升级基础工具：

```bash
pip install -U pip setuptools wheel
```

---

## 6. 安装 PyTorch

PyTorch 要根据公司服务器的驱动、CUDA 和公司内网镜像情况选择。不要盲目照搬其他机器的环境。

先看服务器：

```bash
nvidia-smi
nvcc -V
```

### 6.1 CUDA 11.8 示例

如果服务器 CUDA toolkit 是 11.8，常见安装方式：

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
```

然后验证：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

### 6.2 CUDA 12.x 示例

如果服务器驱动支持 CUDA 12.x，可安装 PyTorch CUDA 12 系 wheel，例如：

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

然后验证：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

说明：

1. `torch.version.cuda` 显示的是 PyTorch 自带 CUDA runtime 版本。
2. `nvidia-smi` 显示的是驱动最高支持版本。
3. 二者不一定完全相同，但驱动必须能支持 PyTorch 使用的 CUDA runtime。

---

## 7. 安装项目 Python 依赖

进入项目根目录：

```bash
cd /workspace/OpenPCDet_ljl
```

安装 requirements：

```bash
pip install -r requirements.txt
```

如果公司服务器没有外网，需要让管理员提供 pip 镜像，或提前下载 wheel 后离线安装。

确认基础依赖：

```bash
python - <<'PY'
import numpy
import numba
import yaml
import easydict
import SharedArray
import skimage
import tqdm
import tensorboardX
print("basic deps ok")
PY
```

---

## 8. 安装 spconv

本项目的 3D sparse convolution backbone 依赖 `spconv`。

先确认当前 PyTorch CUDA：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
PY
```

根据 CUDA 版本选择 spconv 包：

```bash
# PyTorch CUDA 11.8 常用
pip install spconv-cu118

# PyTorch CUDA 12.0 / 12.1 常用
pip install spconv-cu120

# 如果公司镜像提供 CUDA 12.4 包，也可以按实际情况测试
pip install spconv-cu124
```

只保留一个 spconv/cumm 组合。安装前如果环境里已经乱装过，先检查：

```bash
pip list | egrep "spconv|cumm"
```

如需清理：

```bash
pip uninstall -y spconv spconv-cu102 spconv-cu113 spconv-cu114 spconv-cu116 spconv-cu117 spconv-cu118 spconv-cu120 spconv-cu124 cumm cumm-cu102 cumm-cu113 cumm-cu114 cumm-cu116 cumm-cu117 cumm-cu118 cumm-cu120 cumm-cu124
```

再安装一个匹配版本。

验证：

```bash
python - <<'PY'
import spconv
print("spconv:", spconv.__version__)
try:
    import spconv.pytorch as spconv_torch
    print("spconv.pytorch ok")
except Exception as e:
    print("spconv.pytorch import failed:", repr(e))
PY
```

---

## 9. 编译 OpenPCDet CUDA 扩展

服务器上必须重新编译，不能依赖从本机拷过去的 `.so`。

进入项目根目录：

```bash
cd /workspace/OpenPCDet_ljl
```

清理旧产物：

```bash
rm -rf build pcdet.egg-info
find pcdet/ops -name "*.so" -delete
find pcdet/ops -name "*.pyd" -delete
```

如果系统有多个 CUDA，可以手动指定：

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

确认：

```bash
which nvcc
nvcc -V
```

开始编译：

```bash
python setup.py develop
```

如果 CPU 核很多，可以限制或设置并行编译：

```bash
MAX_JOBS=8 python setup.py develop
```

验证 CUDA ops：

```bash
python - <<'PY'
from pcdet.ops.iou3d_nms import iou3d_nms_cuda
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_cuda
from pcdet.ops.roipoint_pool3d import roipoint_pool3d_cuda
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_stack_cuda
from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_batch_cuda
print("pcdet cuda ops ok")
PY
```

如果这里失败，不要继续训练，先看本文最后的常见问题。

---

## 10. 准备公司 nuScenes 数据

本项目默认数据配置文件：

```text
tools/cfgs/dataset_configs/company_nuscenes_dataset.yaml
```

默认关键配置：

```yaml
DATASET: 'CompanyNuScenesDataset'
DATA_PATH: '../data/company_nuscenes'
VERSION: 'v1.0-mini'
MAX_SWEEPS: 1
LIDAR_POINT_DIM: 4
```

注意：训练命令通常从 `tools/` 目录启动，所以 `DATA_PATH: ../data/company_nuscenes` 指的是：

```text
/workspace/OpenPCDet_ljl/data/company_nuscenes
```

### 10.1 使用已经准备好的 mini 数据

如果你已经把本机的 mini 数据拷过去，确认结构：

```bash
cd /workspace/OpenPCDet_ljl
ls data/company_nuscenes/v1.0-mini
ls data/company_nuscenes/v1.0-mini/samples/LIDAR_TOP | head
```

应该能看到：

```text
company_nuscenes_infos_train.pkl
company_nuscenes_infos_val.pkl
sample.json
sample_data.json
sample_annotation.json
samples/LIDAR_TOP/*.bin
ImageSets/train.txt
ImageSets/val.txt
```

检查 infos：

```bash
python tools/company_nuscenes/check_company_infos.py \
  --root data/company_nuscenes/v1.0-mini \
  --strict
```

### 10.2 从原始 mini 数据重新生成

假设原始 mini 数据在：

```text
/data/raw/mini/v1.0-mini
```

运行：

```bash
cd /workspace/OpenPCDet_ljl

python tools/company_nuscenes/prepare_company_mini.py \
  --source /data/raw/mini/v1.0-mini \
  --target data/company_nuscenes/v1.0-mini

python tools/company_nuscenes/create_company_infos.py \
  --version v1.0-mini

python tools/company_nuscenes/check_company_infos.py \
  --root data/company_nuscenes/v1.0-mini \
  --strict
```

### 10.3 使用完整 trainval 数据

推荐结构：

```text
/workspace/OpenPCDet_ljl/data/company_nuscenes/v1.0-trainval/
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
        └── ...
```

如果数据放在大盘，可以软链接：

```bash
cd /workspace/OpenPCDet_ljl
mkdir -p data/company_nuscenes
ln -s /data/company_nuscenes/v1.0-trainval data/company_nuscenes/v1.0-trainval
```

生成 infos：

```bash
python tools/company_nuscenes/create_company_infos.py \
  --version v1.0-trainval
```

检查：

```bash
python tools/company_nuscenes/check_company_infos.py \
  --root data/company_nuscenes/v1.0-trainval \
  --strict
```

训练完整数据时，需要让配置使用 `v1.0-trainval`。有两种方式。

方式 A：临时在命令行覆盖：

```bash
cd /workspace/OpenPCDet_ljl/tools

python train.py \
  --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml \
  --batch_size 1 \
  --epochs 20 \
  --set DATA_CONFIG.VERSION v1.0-trainval
```

方式 B：修改配置文件：

```text
tools/cfgs/dataset_configs/company_nuscenes_dataset.yaml
```

把：

```yaml
VERSION: 'v1.0-mini'
```

改成：

```yaml
VERSION: 'v1.0-trainval'
```

如果只是临时测试，更推荐方式 A，避免忘记把配置改回来。

---

## 11. 跑 dataloader smoke test

mini 数据默认可以直接跑：

```bash
cd /workspace/OpenPCDet_ljl
python tools/company_nuscenes/smoke_test_company_dataloader.py
```

期望输出里应该看到：

```text
dataset_len: 非 0
points_shape: 第二维包含 batch index + 4 个点云特征
gt_boxes_shape: 非空
first_gt_box: 有数值
```

如果你使用的是 `v1.0-trainval`，而配置文件还是 `v1.0-mini`，这个 smoke test 会读错版本。此时要么临时修改 `company_nuscenes_dataset.yaml`，要么复制一个专用配置文件。

---

## 12. 开始训练

训练入口：

```text
/workspace/OpenPCDet_ljl/tools/train.py
```

建议从 `tools/` 目录启动。

### 12.1 单 GPU 跑 26 类 CenterPoint

```bash
cd /workspace/OpenPCDet_ljl/tools

CUDA_VISIBLE_DEVICES=0 python train.py \
  --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml \
  --batch_size 1 \
  --epochs 20 \
  --workers 4 \
  --extra_tag server_centerpoint_26cls
```

如果用完整数据但不想改 yaml：

```bash
cd /workspace/OpenPCDet_ljl/tools

CUDA_VISIBLE_DEVICES=0 python train.py \
  --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml \
  --batch_size 1 \
  --epochs 20 \
  --workers 4 \
  --extra_tag server_centerpoint_26cls_trainval \
  --set DATA_CONFIG.VERSION v1.0-trainval
```

### 12.2 快速验证 12 类 debug 配置

如果只是确认数据能读、loss 能跑，可以先用 12 类配置：

```bash
cd /workspace/OpenPCDet_ljl/tools

CUDA_VISIBLE_DEVICES=0 python train.py \
  --cfg_file cfgs/nuscenes_models/company_centerpoint_12cls.yaml \
  --batch_size 1 \
  --epochs 5 \
  --workers 4 \
  --extra_tag server_debug_12cls
```

### 12.3 跑 26 类 VoxelNeXt

如果目标是后面接自动标注流程，可以跑：

```bash
cd /workspace/OpenPCDet_ljl/tools

CUDA_VISIBLE_DEVICES=0 python train.py \
  --cfg_file cfgs/nuscenes_models/company_voxelnext_26cls.yaml \
  --batch_size 1 \
  --epochs 20 \
  --workers 4 \
  --extra_tag server_voxelnext_26cls
```

### 12.4 多 GPU 训练

查看可用 GPU：

```bash
nvidia-smi
```

例如使用 4 张卡：

```bash
cd /workspace/OpenPCDet_ljl/tools

CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/torch_train.sh 4 \
  --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml \
  --batch_size 4 \
  --epochs 20 \
  --workers 8 \
  --extra_tag server_centerpoint_26cls_4gpu
```

注意：

1. 多 GPU 时，`--batch_size` 是总 batch size。
2. 代码里会自动除以 GPU 数，得到每张卡的 batch size。
3. `--batch_size` 必须能被 GPU 数整除。

---

## 13. 后台运行训练

推荐用 `tmux`：

```bash
tmux new -s pcdet_train
conda activate openpcdet_ljl
cd /workspace/OpenPCDet_ljl/tools
CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml --batch_size 1 --epochs 20 --workers 4 --extra_tag server_test
```

退出但不中断：

```text
Ctrl+b 然后按 d
```

重新进入：

```bash
tmux attach -t pcdet_train
```

也可以用 `nohup`：

```bash
cd /workspace/OpenPCDet_ljl/tools

nohup python train.py \
  --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml \
  --batch_size 1 \
  --epochs 20 \
  --workers 4 \
  --extra_tag server_nohup \
  > train_server_nohup.log 2>&1 &
```

查看：

```bash
tail -f train_server_nohup.log
```

---

## 14. 查看训练输出

训练输出目录一般是：

```text
/workspace/OpenPCDet_ljl/output/nuscenes_models/company_centerpoint_26cls/<extra_tag>/
```

例如：

```text
/workspace/OpenPCDet_ljl/output/nuscenes_models/company_centerpoint_26cls/server_centerpoint_26cls/
```

查看日志：

```bash
cd /workspace/OpenPCDet_ljl
find output -name "log_train_*.txt"
tail -f output/nuscenes_models/company_centerpoint_26cls/server_centerpoint_26cls/log_train_*.txt
```

查看 checkpoint：

```bash
find output -name "checkpoint_epoch_*.pth"
```

常见 checkpoint 路径：

```text
output/nuscenes_models/company_centerpoint_26cls/server_centerpoint_26cls/ckpt/checkpoint_epoch_20.pth
```

---

## 15. 断点续训

假设 checkpoint 在：

```text
/workspace/OpenPCDet_ljl/output/nuscenes_models/company_centerpoint_26cls/server_centerpoint_26cls/ckpt/checkpoint_epoch_10.pth
```

续训：

```bash
cd /workspace/OpenPCDet_ljl/tools

CUDA_VISIBLE_DEVICES=0 python train.py \
  --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml \
  --batch_size 1 \
  --epochs 20 \
  --workers 4 \
  --extra_tag server_centerpoint_26cls \
  --ckpt ../output/nuscenes_models/company_centerpoint_26cls/server_centerpoint_26cls/ckpt/checkpoint_epoch_10.pth
```

如果只是加载预训练模型，不加载 optimizer 状态，用：

```bash
--pretrained_model /path/to/model.pth
```

---

## 16. 测试 checkpoint

训练完成后，可以用 `test.py` 测试：

```bash
cd /workspace/OpenPCDet_ljl/tools

CUDA_VISIBLE_DEVICES=0 python test.py \
  --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml \
  --batch_size 1 \
  --ckpt ../output/nuscenes_models/company_centerpoint_26cls/server_centerpoint_26cls/ckpt/checkpoint_epoch_20.pth
```

本项目公司 26 类评估目前主要是 smoke/count 逻辑，不要把它当官方 nuScenes AP/NDS 指标。

---

## 17. 接自动标注推理

推理入口：

```text
tools/inference/inference_nms.py
```

该文件支持环境变量覆盖配置和 checkpoint：

```bash
export PCDET_CFG_FILE=/workspace/OpenPCDet_ljl/tools/cfgs/nuscenes_models/company_voxelnext_26cls.yaml
export PCDET_CKPT_PATH=/workspace/OpenPCDet_ljl/output/nuscenes_models/company_voxelnext_26cls/server_voxelnext_26cls/ckpt/checkpoint_epoch_20.pth
```

如果你训练的是 CenterPoint，就把 `PCDET_CFG_FILE` 改为：

```bash
export PCDET_CFG_FILE=/workspace/OpenPCDet_ljl/tools/cfgs/nuscenes_models/company_centerpoint_26cls.yaml
```

注意：

1. checkpoint 必须和 cfg 对应。
2. 26 类标签名来自配置里的 `CLASS_NAMES`。
3. 平台如果原本固定使用 VoxelNeXt，优先训练和接入 `company_voxelnext_26cls.yaml`。

---

## 18. 最小验收流程

第一次上服务器，建议按下面顺序验收。

### 18.1 环境验收

```bash
conda activate openpcdet_ljl
cd /workspace/OpenPCDet_ljl

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
PY

python - <<'PY'
import spconv
print("spconv:", spconv.__version__)
PY

python - <<'PY'
from pcdet.ops.iou3d_nms import iou3d_nms_cuda
print("pcdet ops ok")
PY
```

### 18.2 数据验收

```bash
cd /workspace/OpenPCDet_ljl

python tools/company_nuscenes/check_company_infos.py \
  --root data/company_nuscenes/v1.0-mini \
  --strict

python tools/company_nuscenes/smoke_test_company_dataloader.py
```

### 18.3 训练验收

先跑 1 个 epoch：

```bash
cd /workspace/OpenPCDet_ljl/tools

CUDA_VISIBLE_DEVICES=0 python train.py \
  --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml \
  --batch_size 1 \
  --epochs 1 \
  --workers 2 \
  --extra_tag smoke_epoch1
```

通过标准：

1. 能进入 training loop。
2. loss 能正常打印，不是一直 `nan`。
3. 能保存 checkpoint。
4. `output/.../ckpt/checkpoint_epoch_1.pth` 存在。

---

## 19. 常见问题

### 19.1 `ModuleNotFoundError: No module named 'pcdet'`

原因：

1. 没在项目根目录执行过 `python setup.py develop`。
2. 当前 Python 环境不是编译/安装时的环境。
3. 路径不对。

处理：

```bash
conda activate openpcdet_ljl
cd /workspace/OpenPCDet_ljl
python setup.py develop
python -c "import pcdet; print(pcdet.__file__)"
```

### 19.2 `ModuleNotFoundError: No module named 'spconv'`

原因：没有安装 spconv。

处理：

```bash
pip list | egrep "spconv|cumm"
pip install spconv-cu118
```

或根据实际 CUDA 改成：

```bash
pip install spconv-cu120
```

### 19.3 `ImportError` / `undefined symbol` / `.so` 导入失败

原因通常是旧 `.so` 是别的 Python、PyTorch、CUDA 编出来的。

处理：

```bash
cd /workspace/OpenPCDet_ljl
rm -rf build pcdet.egg-info
find pcdet/ops -name "*.so" -delete
python setup.py develop
```

### 19.4 `No CUDA runtime is found`

原因：

1. 找不到 CUDA toolkit。
2. `CUDA_HOME` 没设。
3. 当前环境只有 PyTorch CUDA runtime，但没有 nvcc。

处理：

```bash
which nvcc
nvcc -V
ls /usr/local | grep cuda
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
python setup.py develop
```

如果服务器没有 CUDA toolkit，需要找管理员安装，或使用公司已有 CUDA module。

### 19.5 `torch.cuda.is_available()` 是 False

检查：

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

常见原因：

1. 装了 CPU 版 PyTorch。
2. 当前机器没有 GPU。
3. 驱动不可用。
4. 容器没有挂 GPU。

如果是 Docker，需要确认启动时有：

```bash
--gpus all
```

### 19.6 `cannot reshape array of size ... into shape (-1,4)`

原因：点云不是 4 维 float32，但配置里写了：

```yaml
LIDAR_POINT_DIM: 4
used_feature_list: ['x', 'y', 'z', 'intensity']
src_feature_list: ['x', 'y', 'z', 'intensity']
```

处理：

1. 确认 `.bin` 是否真的是 `x, y, z, intensity`。
2. 如果是 3 维或 5/6 维，修改 `LIDAR_POINT_DIM` 和 `POINT_FEATURE_ENCODING`。
3. 重新生成 infos 并重新 smoke test。

### 19.7 找不到 `company_nuscenes_infos_train.pkl`

原因：

1. 没有生成 infos。
2. `VERSION` 写错。
3. 数据路径不对。

处理：

```bash
cd /workspace/OpenPCDet_ljl
python tools/company_nuscenes/create_company_infos.py --version v1.0-mini
python tools/company_nuscenes/check_company_infos.py --root data/company_nuscenes/v1.0-mini
```

如果是完整数据：

```bash
python tools/company_nuscenes/create_company_infos.py --version v1.0-trainval
python tools/company_nuscenes/check_company_infos.py --root data/company_nuscenes/v1.0-trainval
```

### 19.8 显存不足 `CUDA out of memory`

先降低 batch size：

```bash
--batch_size 1
```

降低 dataloader workers：

```bash
--workers 2
```

如果还是爆显存：

1. 减小 `MAX_NUMBER_OF_VOXELS`。
2. 减小点云范围 `POINT_CLOUD_RANGE`。
3. 先用 CenterPoint 配置跑通，再跑 VoxelNeXt。

### 19.9 多 GPU NCCL 报错

先确认单卡能跑，再尝试多卡。

检查 GPU：

```bash
nvidia-smi
nvidia-smi topo -m
```

临时规避部分 NCCL 网络问题：

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
```

再跑：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/torch_train.sh 2 \
  --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml \
  --batch_size 2
```

---

## 20. 服务器环境记录模板

第一次跑通后，建议把下面信息记录下来，后续别人复现会省很多时间。

```text
服务器:
  hostname:
  OS:
  GPU:
  Driver Version:
  nvidia-smi CUDA Version:
  nvcc -V:
  gcc version:

conda:
  conda path:
  env name:
  python version:

python packages:
  torch:
  torch.version.cuda:
  torchvision:
  spconv:
  cumm:
  numpy:
  numba:

project:
  project path:
  git commit:
  config:
  data version:
  extra_tag:

training:
  command:
  batch_size:
  workers:
  gpu ids:
  output path:
  checkpoint:
```

可以用这些命令快速生成一部分信息：

```bash
cd /workspace/OpenPCDet_ljl

hostname
lsb_release -a
nvidia-smi
nvcc -V
gcc --version
which conda
conda info --envs
which python
python -V

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except Exception as e:
    print("torchvision failed:", repr(e))
try:
    import spconv
    print("spconv:", spconv.__version__)
except Exception as e:
    print("spconv failed:", repr(e))
try:
    import numpy
    print("numpy:", numpy.__version__)
except Exception as e:
    print("numpy failed:", repr(e))
try:
    import numba
    print("numba:", numba.__version__)
except Exception as e:
    print("numba failed:", repr(e))
PY

git rev-parse --short HEAD 2>/dev/null || true
```

---

## 21. 推荐第一次上服务器执行的完整命令顺序

下面是一条从零到 1 epoch smoke train 的参考流程。实际 CUDA/PyTorch/spconv 版本要按服务器情况调整。

```bash
# 0. 登录服务器后确认 GPU
nvidia-smi
nvcc -V

# 1. 准备 conda 环境
conda create -n openpcdet_ljl python=3.8 -y
conda activate openpcdet_ljl
pip install -U pip setuptools wheel

# 2. 进入项目
cd /workspace/OpenPCDet_ljl

# 3. 安装 PyTorch，示例为 CUDA 11.8
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 安装 spconv，示例为 CUDA 11.8
pip install spconv-cu118

# 6. 编译 OpenPCDet
rm -rf build pcdet.egg-info
find pcdet/ops -name "*.so" -delete
python setup.py develop

# 7. 验证环境
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
import spconv
print("spconv", spconv.__version__)
from pcdet.ops.iou3d_nms import iou3d_nms_cuda
print("pcdet ops ok")
PY

# 8. 检查数据
python tools/company_nuscenes/check_company_infos.py --root data/company_nuscenes/v1.0-mini --strict
python tools/company_nuscenes/smoke_test_company_dataloader.py

# 9. 跑 1 epoch
cd tools
CUDA_VISIBLE_DEVICES=0 python train.py \
  --cfg_file cfgs/nuscenes_models/company_centerpoint_26cls.yaml \
  --batch_size 1 \
  --epochs 1 \
  --workers 2 \
  --extra_tag smoke_epoch1
```

如果这套流程跑通，再把 `--epochs` 改成正式训练轮数，把 `--extra_tag` 改成正式实验名。
