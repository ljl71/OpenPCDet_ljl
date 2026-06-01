# OpenPCDet_ljl_plus 接入自动标注平台前置操作记录

> 目标：在不破坏生产自动标注服务的前提下，准备将 `OpenPCDet_ljl_plus` 中训练好的 26 类 VoxelNeXt 模型接入公司本地部署的自动标注平台。  
> 当前阶段：**尚未修改 `ai-nginx` 转发配置**，只完成了 `detection3d_v5_plus` 测试容器的前置准备。  
> 生产容器：`detection3d`  
> 测试容器：`detection3d_v5_plus`  
> 网关容器：`ai-nginx`

---

## 1. 背景说明

公司本地部署的自动标注平台原本使用供应商提供的 3D 自动标注服务。供应商给出的更新方式是：

```text
替换模型权重路径；
替换模型配置文件路径。
```

原生产服务容器为：

```text
detection3d
```

该容器和自动标注平台连接，内部运行 FastAPI 推理服务，接收平台请求并返回自动标注结果。

新版模型来自：

```text
/home/ubuntu/WXY/OpenPCDet_ljl_plus
```

新版模型权重为：

```text
/home/ubuntu/WXY/OpenPCDet_ljl_plus/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/ckpt/checkpoint_epoch_20.pth
```

新版模型配置为：

```text
/home/ubuntu/WXY/OpenPCDet_ljl_plus/tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
```

但实际测试发现，新版模型并不是只替换 `pth + yaml` 就能直接在供应商 `detection3d` 容器中运行。原因是新版配置还依赖：

```text
cfgs/dataset_configs/company_nuscenes_trainval_dataset.yaml
```

并且新版代码可能还依赖：

```text
CompanyNuScenesDataset
26 类类别定义
VoxelNeXt multi-head 适配逻辑
新旧类别名映射
平台输出格式兼容
```

因此，为了避免破坏生产容器，后续转为使用 `detection3d_v5_plus` 作为测试容器进行验证。

---

## 2. 生产容器曾出现的问题

在生产容器 `detection3d` 中尝试将 `inference_nms.py` 的默认模型路径改为新版模型后，容器启动失败。

日志关键报错：

```text
FileNotFoundError: [Errno 2] No such file or directory:
'cfgs/dataset_configs/company_nuscenes_trainval_dataset.yaml'
```

报错链路为：

```text
fastAPI.py startup_event
    -> model = LoadModel()
    -> inference_nms.py LoadModel.__init__()
    -> parse_config(cfg_file)
    -> cfg_from_yaml_file(cfg_file, cfg)
    -> merge_new_config()
    -> open(new_config['_BASE_CONFIG_'])
    -> FileNotFoundError
```

这说明：

```text
新模型配置路径已经生效；
但是配置文件内部引用的 _BASE_CONFIG_ 在原生产容器中不存在；
因此 FastAPI 服务启动失败。
```

---

## 3. 生产服务恢复操作

由于继续在生产 `detection3d` 容器中补配置和代码存在风险，后续先恢复生产容器到供应商原始模型路径。

原供应商路径为：

```text
checkpoint:
/workspace/OpenPCDet/tools/inference/ckpt/checkpoint_epoch_50.pth

config:
/workspace/OpenPCDet/tools/cfgs/argo2_models/cbgs_voxel01_voxelnext.yaml
```

恢复后启动生产容器：

```bash
sudo docker start detection3d
```

检查状态：

```bash
sudo docker ps -a --filter "name=^/detection3d$"
```

恢复成功后状态为：

```text
detection3d   Up
```

随后检查平台相关容器：

```bash
sudo docker ps | grep -E "detection3d|ai-nginx"
```

结果显示：

```text
detection3d   Up
ai-nginx      Up
```

日志中也重新出现供应商模型加载信息：

```text
Loading parameters from checkpoint /workspace/OpenPCDet/tools/inference/ckpt/checkpoint_epoch_50.pth to GPU
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

结论：

```text
生产自动标注服务已恢复到供应商原始状态。
```

---

## 4. ai-nginx 转发关系排查

为了确认自动标注平台请求如何转发，查看 `ai-nginx` 的 Nginx 配置：

```bash
sudo docker exec ai-nginx sh -lc 'grep -R "detection3d\|detection2d\|smart-tool\|proxy_pass" -n /etc/nginx 2>/dev/null'
```

关键输出：

```text
/etc/nginx/nginx.conf:91:        location /smart-tool/detection3d {
/etc/nginx/nginx.conf:92:            proxy_pass http://172.28.5.55:8000;
```

说明当前 3D 自动标注链路为：

```text
自动标注平台
    -> ai-nginx
    -> /smart-tool/detection3d
    -> http://172.28.5.55:8000
    -> detection3d 容器
```

因此，`ai-nginx` 当前并不是通过容器名 `detection3d` 转发，而是通过固定内网 IP：

```text
172.28.5.55:8000
```

---

## 5. 容器网络检查

查看生产 `detection3d` 网络：

```bash
sudo docker inspect -f '{{range $name,$conf := .NetworkSettings.Networks}}{{println $name $conf.IPAddress}}{{end}}' detection3d
```

输出：

```text
mooredata_my-network 172.28.5.55
```

查看 `ai-nginx` 网络：

```bash
sudo docker inspect -f '{{range $name,$conf := .NetworkSettings.Networks}}{{println $name $conf.IPAddress}}{{end}}' ai-nginx
```

输出：

```text
mooredata_my-network 172.28.5.50
```

说明：

```text
ai-nginx 和 detection3d 都在 mooredata_my-network 网络中；
ai-nginx 通过 172.28.5.55:8000 访问 detection3d。
```

查看测试容器 `detection3d_v5_plus` 初始网络：

```bash
sudo docker inspect -f '{{range $name,$conf := .NetworkSettings.Networks}}{{println $name $conf.IPAddress}}{{end}}' detection3d_v5_plus
```

初始输出：

```text
bridge 172.17.0.6
```

说明：

```text
detection3d_v5_plus 最初只在默认 bridge 网络中；
和 ai-nginx / detection3d 不在同一个 mooredata_my-network 网络中。
```

---

## 6. 将 detection3d_v5_plus 接入平台网络

为了后续让 `ai-nginx` 能访问测试容器，需要将 `detection3d_v5_plus` 接入同一个 Docker 网络：

```bash
sudo docker network connect mooredata_my-network detection3d_v5_plus
```

再次检查网络：

```bash
sudo docker inspect -f '{{range $name,$conf := .NetworkSettings.Networks}}{{println $name $conf.IPAddress}}{{end}}' detection3d_v5_plus
```

输出：

```text
bridge 172.17.0.6
mooredata_my-network 172.28.0.1
```

说明：

```text
detection3d_v5_plus 已接入 mooredata_my-network；
后续理论上可以通过 172.28.0.1 访问；
但此时尚未修改 ai-nginx，因此生产平台仍然访问原 detection3d。
```

安全性说明：

```text
该操作只改变 detection3d_v5_plus 的网络连接；
没有修改 ai-nginx；
没有修改 detection3d；
不会影响原 /smart-tool/detection3d 生产链路。
```

---

## 7. 检查 detection3d_v5_plus 是否有平台接口封装

查看测试容器中的推理接口目录：

```bash
sudo docker exec detection3d_v5_plus ls -lh /workspace/OpenPCDet/tools/inference
```

结果显示存在以下文件：

```text
DataSet.py
fastAPI.py
inference_nms.py
nms.py
ckpt/
cfgs/
dataset_demo/
test_pcFile/
```

说明：

```text
detection3d_v5_plus 不是单纯训练容器；
它已经包含类似供应商 detection3d 的平台接口封装文件；
具备进一步测试 FastAPI 自动标注服务的基础。
```

---

## 8. 检查 detection3d_v5_plus 是否已经启动服务

执行：

```bash
sudo docker exec detection3d_v5_plus ps -ef | grep -E "uvicorn|fastAPI|python" | grep -v grep
```

无输出。

执行：

```bash
sudo docker port detection3d_v5_plus
```

无输出。

说明：

```text
detection3d_v5_plus 当前没有运行 FastAPI / Uvicorn 服务；
也没有暴露端口；
目前只是一个具备接口文件的容器，还没有实际提供平台可调用的自动标注服务。
```

---

## 9. 检查 plus 容器中的模型文件和配置文件

确认新版模型权重存在：

```bash
sudo docker exec detection3d_v5_plus ls -lh /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/ckpt/checkpoint_epoch_20.pth
```

输出：

```text
-rw-r--r-- 1 root root 90M ... checkpoint_epoch_20.pth
```

确认新版主配置存在：

```bash
sudo docker exec detection3d_v5_plus ls -lh /workspace/OpenPCDet/tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml
```

输出：

```text
-rwxr-x--- 1 1000 1000 4.3K ... company_voxelnext_26cls_trainval.yaml
```

确认基础数据配置存在：

```bash
sudo docker exec detection3d_v5_plus ls -lh /workspace/OpenPCDet/tools/cfgs/dataset_configs/company_nuscenes_trainval_dataset.yaml
```

输出：

```text
-rwxr-x--- 1 1000 1000 1.9K ... company_nuscenes_trainval_dataset.yaml
```

说明：

```text
detection3d_v5_plus 中训练好的模型权重、主配置文件、base 数据配置文件均存在。
```

---

## 10. 创建独立部署目录 company_test/test1

为了避免污染原始配置目录，在 plus 容器中创建独立部署目录：

```bash
sudo docker exec detection3d_v5_plus mkdir -p /workspace/OpenPCDet/company_test/test1
```

复制模型权重：

```bash
sudo docker exec detection3d_v5_plus cp /workspace/OpenPCDet/output/nuscenes_models/company_voxelnext_26cls_trainval/formal_company_26cls_plus_bs8_e20/ckpt/checkpoint_epoch_20.pth /workspace/OpenPCDet/company_test/test1/
```

复制主配置文件：

```bash
sudo docker exec detection3d_v5_plus cp /workspace/OpenPCDet/tools/cfgs/nuscenes_models/company_voxelnext_26cls_trainval.yaml /workspace/OpenPCDet/company_test/test1/
```

复制基础数据配置文件：

```bash
sudo docker exec detection3d_v5_plus cp /workspace/OpenPCDet/tools/cfgs/dataset_configs/company_nuscenes_trainval_dataset.yaml /workspace/OpenPCDet/company_test/test1/
```

检查目录：

```bash
sudo docker exec detection3d_v5_plus ls -lh /workspace/OpenPCDet/company_test/test1
```

输出：

```text
checkpoint_epoch_20.pth
company_nuscenes_trainval_dataset.yaml
company_voxelnext_26cls_trainval.yaml
```

说明：

```text
测试部署所需的三个核心文件已集中放到 company_test/test1 目录下。
```

---

## 11. 遇到的问题：docker exec python 不可用

尝试使用：

```bash
sudo docker exec detection3d_v5_plus python - <<'PY'
...
PY
```

时报错：

```text
OCI runtime exec failed:
exec: "python": executable file not found in $PATH
```

原因：

```text
detection3d_v5_plus 非交互式 docker exec 环境下没有把 conda Python 加入 PATH；
因此直接执行 python 找不到解释器。
```

解决方式：

```text
不用 python 命令修改文件；
改用 sed 进行字符串替换。
```

---

## 12. 修改主配置中的 _BASE_CONFIG_

原始配置中：

```yaml
_BASE_CONFIG_: cfgs/dataset_configs/company_nuscenes_trainval_dataset.yaml
```

这个相对路径在不同工作目录下容易出错，生产容器之前就是因为这个路径找不到导致启动失败。

因此在独立部署目录中的主配置文件里，将其改为绝对路径：

```bash
sudo docker exec detection3d_v5_plus sed -i 's#cfgs/dataset_configs/company_nuscenes_trainval_dataset.yaml#/workspace/OpenPCDet/company_test/test1/company_nuscenes_trainval_dataset.yaml#g' /workspace/OpenPCDet/company_test/test1/company_voxelnext_26cls_trainval.yaml
```

检查：

```bash
sudo docker exec detection3d_v5_plus grep -n "_BASE_CONFIG_" /workspace/OpenPCDet/company_test/test1/company_voxelnext_26cls_trainval.yaml
```

结果：

```text
31:    _BASE_CONFIG_: /workspace/OpenPCDet/company_test/test1/company_nuscenes_trainval_dataset.yaml
```

说明：

```text
_BASE_CONFIG_ 已改为绝对路径；
后续 FastAPI 启动时不应再因为 base config 相对路径找不到而失败。
```

---

## 13. 修改 detection3d_v5_plus 中的 inference_nms.py

修改前检查：

```bash
sudo docker exec detection3d_v5_plus grep -n "DEFAULT_CKPT_PATH\|DEFAULT_CFG_FILE\|checkpoint_epoch\|company_voxelnext\|cbgs_voxel" /workspace/OpenPCDet/tools/inference/inference_nms.py
```

修改前默认路径：

```text
DEFAULT_CKPT_PATH:
"/workspace/OpenPCDet/tools/inference/ckpt/checkpoint_epoch_50.pth"

DEFAULT_CFG_FILE:
"/workspace/OpenPCDet/tools/cfgs/argo2_models/cbgs_voxel01_voxelnext.yaml"
```

这说明：

```text
如果直接启动 plus API，它仍然会加载供应商默认模型。
```

先备份：

```bash
sudo docker exec detection3d_v5_plus cp /workspace/OpenPCDet/tools/inference/inference_nms.py /workspace/OpenPCDet/tools/inference/inference_nms.py.bak_company_test1
```

使用 `sed` 替换默认 checkpoint 路径：

```bash
sudo docker exec detection3d_v5_plus sed -i 's#/workspace/OpenPCDet/tools/inference/ckpt/checkpoint_epoch_50.pth#/workspace/OpenPCDet/company_test/test1/checkpoint_epoch_20.pth#g' /workspace/OpenPCDet/tools/inference/inference_nms.py
```

使用 `sed` 替换默认 config 路径：

```bash
sudo docker exec detection3d_v5_plus sed -i 's#/workspace/OpenPCDet/tools/cfgs/argo2_models/cbgs_voxel01_voxelnext.yaml#/workspace/OpenPCDet/company_test/test1/company_voxelnext_26cls_trainval.yaml#g' /workspace/OpenPCDet/tools/inference/inference_nms.py
```

修改后检查：

```bash
sudo docker exec detection3d_v5_plus grep -n "DEFAULT_CKPT_PATH\|DEFAULT_CFG_FILE\|checkpoint_epoch\|company_voxelnext\|cbgs_voxel" /workspace/OpenPCDet/tools/inference/inference_nms.py
```

结果：

```text
DEFAULT_CKPT_PATH:
"/workspace/OpenPCDet/company_test/test1/checkpoint_epoch_20.pth"

DEFAULT_CFG_FILE:
"/workspace/OpenPCDet/company_test/test1/company_voxelnext_26cls_trainval.yaml"
```

说明：

```text
detection3d_v5_plus 中的 inference_nms.py 已经改为默认加载我们自己的 26 类模型权重和配置文件。
```

---

## 14. 当前状态总结

截至目前，已经完成：

```text
1. 生产 detection3d 容器已恢复供应商原始模型，平台原功能未被破坏；
2. ai-nginx 仍然指向 172.28.5.55:8000，即原 detection3d；
3. detection3d_v5_plus 已加入 mooredata_my-network；
4. detection3d_v5_plus 具备平台接口封装文件 fastAPI.py / inference_nms.py / DataSet.py / nms.py；
5. 新模型 checkpoint、主配置、base 配置已集中放入 company_test/test1；
6. company_voxelnext_26cls_trainval.yaml 的 _BASE_CONFIG_ 已改成绝对路径；
7. detection3d_v5_plus 中 inference_nms.py 默认加载路径已改为我们自己的模型和配置。
```

尚未执行：

```text
1. 尚未启动 detection3d_v5_plus 的 FastAPI 服务；
2. 尚未测试 plus API 是否能正常加载新模型；
3. 尚未用 curl 或平台请求测试 plus 服务输出；
4. 尚未修改 ai-nginx；
5. 尚未新增 /smart-tool/detection3d_plus 测试路径；
6. 尚未替换原 /smart-tool/detection3d 生产路径。
```

---

## 15. 为什么目前不会影响原有功能

当前生产链路仍然是：

```text
ai-nginx /smart-tool/detection3d
    -> http://172.28.5.55:8000
    -> detection3d
    -> 供应商原模型
```

我们目前修改的对象是：

```text
detection3d_v5_plus
```

并没有修改：

```text
ai-nginx
detection3d
/smart-tool/detection3d
172.28.5.55:8000
```

因此，当前操作不会改变原平台自动标注功能。

---

## 16. 下一步：手动启动 plus API 测试

下一步计划在 `detection3d_v5_plus` 中手动启动 FastAPI 服务：

```bash
sudo docker exec -it detection3d_v5_plus /bin/bash
```

进入容器后执行：

```bash
cd /workspace/OpenPCDet
PYTHONPATH=/workspace/OpenPCDet CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/pcdet/bin/python -m uvicorn tools.inference.fastAPI:app --host 0.0.0.0 --port 8000
```

如果 `/opt/conda/envs/pcdet/bin/python` 不存在，可以先检查：

```bash
ls /opt/conda/envs
which python
which python3
```

启动成功标志：

```text
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

如果启动失败，需要根据报错判断是否还存在：

```text
CompanyNuScenesDataset 缺失
类别名不兼容
模型结构不匹配
平台接口输入输出不兼容
依赖库缺失
```

---

## 17. 后续建议

建议后续继续保持安全策略：

```text
先在 detection3d_v5_plus 中启动测试服务；
确认模型可以加载；
确认接口可以返回结果；
再考虑在 ai-nginx 中新增测试路径 /smart-tool/detection3d_plus；
不要直接替换 /smart-tool/detection3d；
不要直接修改生产 detection3d。
```

推荐未来测试路径设计：

```text
原生产路径：
/smart-tool/detection3d
    -> 172.28.5.55:8000

新增测试路径：
/smart-tool/detection3d_plus
    -> 172.28.0.1:8000
```

这样即使 plus 服务报错，也不会影响原平台生产自动标注功能。

---

## 18. 手动启动 detection3d_v5_plus 的 FastAPI 服务

在前面的准备工作完成后，开始进入实际 API 服务测试阶段。此阶段仍然遵循安全原则：

```text
不停止 detection3d；
不替换 /smart-tool/detection3d；
不修改生产模型；
只在 detection3d_v5_plus 上启动测试服务。
```

首先确认生产容器仍然正常：

```bash
sudo docker ps | grep -E "detection3d|ai-nginx"
```

结果显示：

```text
detection3d_v5_plus   Up
detection3d           Up
ai-nginx              Up
```

然后进入 `detection3d_v5_plus`：

```bash
sudo docker exec -it detection3d_v5_plus /bin/bash
cd /workspace/OpenPCDet
```

检查 Python 环境：

```bash
which python
which python3
ls -lh /opt/conda/bin/python
ls -lh /opt/conda/envs/pcdet/bin/python
```

结果显示：

```text
/opt/conda/bin/python
/opt/conda/bin/python3
/opt/conda/bin/python -> python3.7
/opt/conda/envs/pcdet/bin/python -> python3.8
```

第一次尝试启动服务时执行：

```bash
PYTHONPATH=/workspace/OpenPCDet CUDA_VISIBLE_DEVICES=1 /opt/conda/bin/python -m uvicorn tools.inference.fastAPI:app --host 0.0.0.0 --port 8000
```

出现问题：

```text
ModuleNotFoundError: No module named 'DataSet'
```

原因是 `inference_nms.py` 中使用：

```python
from DataSet import getDataFromFile, getDataFromURL, Dataset
```

而 `DataSet.py` 实际位于：

```text
/workspace/OpenPCDet/tools/inference/DataSet.py
```

因此需要将 `tools/inference` 也加入 `PYTHONPATH`。修正后的启动命令为：

```bash
PYTHONPATH=/workspace/OpenPCDet:/workspace/OpenPCDet/tools/inference CUDA_VISIBLE_DEVICES=1 /opt/conda/bin/python -m uvicorn tools.inference.fastAPI:app --host 0.0.0.0 --port 8000
```

启动成功后日志显示：

```text
Loading parameters from checkpoint /workspace/OpenPCDet/company_test/test1/checkpoint_epoch_20.pth to GPU
==> Done (loaded 443/443)
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

说明：

```text
detection3d_v5_plus 已成功启动 FastAPI 服务；
服务加载的是我们自己的 checkpoint_epoch_20.pth；
该服务在前台运行，不能关闭该终端。
```

---

## 19. plus API 连通性测试

在宿主机测试 plus API：

```bash
curl -I http://172.28.0.1:8000/docs
```

返回：

```text
HTTP/1.1 200 OK
server: uvicorn
```

说明宿主机可以访问 `detection3d_v5_plus` 服务。

随后从同一 Docker 网络中的生产容器 `detection3d` 中访问 plus API：

```bash
sudo docker exec -i detection3d /opt/conda/envs/pcdet/bin/python - <<'PY'
import urllib.request

url = "http://172.28.0.1:8000/docs"
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        print("status:", r.status)
        print("server:", r.headers.get("server"))
        print(r.read(120).decode("utf-8", errors="ignore"))
except Exception as e:
    print("ERROR:", repr(e))
PY
```

返回：

```text
status: 200
server: uvicorn
<!DOCTYPE html>
<html>
...
```

说明：

```text
mooredata_my-network 内部容器可以访问 detection3d_v5_plus；
plus API 网络可达。
```

过程中曾出现一次没有输出的问题，原因是 `docker exec` 未加 `-i`，导致标准输入没有传入容器。修正方式是：

```bash
sudo docker exec -i ...
```

---

## 20. 在 ai-nginx 中新增 detection3d_plus 测试路径

在确认 plus API 可访问后，为了不影响生产接口，采用新增测试路径的方式：

```text
保留原路径：
/smart-tool/detection3d -> 172.28.5.55:8000

新增测试路径：
/smart-tool/detection3d_plus -> 172.28.0.1:8000
```

备份 `ai-nginx` 原配置：

```bash
sudo docker cp ai-nginx:/etc/nginx/nginx.conf /tmp/nginx.conf.bak_detection3d_plus
cp /tmp/nginx.conf.bak_detection3d_plus /tmp/nginx.conf.plus
```

向 `/tmp/nginx.conf.plus` 中新增测试路径：

```bash
python3 - <<'PY'
from pathlib import Path

p = Path("/tmp/nginx.conf.plus")
s = p.read_text()

marker = "        location /smart-tool/detection3d {"

block = """        location = /smart-tool/detection3d_plus {

            proxy_pass http://172.28.0.1:8000/smart-tool/detection3d;

        }



"""

assert marker in s, "没有找到原 /smart-tool/detection3d 配置位置"
assert "/smart-tool/detection3d_plus" not in s, "测试路径已经存在"

s = s.replace(marker, block + marker)
p.write_text(s)
print("added /smart-tool/detection3d_plus")
PY
```

将配置复制到 `ai-nginx` 临时路径并测试：

```bash
sudo docker cp /tmp/nginx.conf.plus ai-nginx:/tmp/nginx.conf.plus
sudo docker exec ai-nginx nginx -t -c /tmp/nginx.conf.plus
```

返回：

```text
nginx: the configuration file /tmp/nginx.conf.plus syntax is ok
nginx: configuration file /tmp/nginx.conf.plus test is successful
```

说明新增配置语法正确。

---

## 21. 遇到的问题：docker cp 覆盖 nginx.conf 失败

直接覆盖正式配置时：

```bash
sudo docker cp /tmp/nginx.conf.plus ai-nginx:/etc/nginx/nginx.conf
```

出现报错：

```text
Error response from daemon: unlinkat /etc/nginx/nginx.conf: device or resource busy
```

说明 `/etc/nginx/nginx.conf` 可能是挂载文件，`docker cp` 无法直接替换。检查后发现正式配置并没有写入新增路径：

```bash
sudo docker exec ai-nginx sh -lc 'grep -n "detection3d_plus\|detection3d" /etc/nginx/nginx.conf'
```

只看到：

```text
91:        location /smart-tool/detection3d {
```

因此当时访问：

```bash
curl -i http://127.0.0.1:8000/smart-tool/detection3d_plus
```

返回：

```text
404 Not Found
```

解决方式是改用容器内部 `cat` 写入：

```bash
sudo docker exec ai-nginx sh -lc 'cp /etc/nginx/nginx.conf /tmp/nginx.conf.before_detection3d_plus'
sudo docker exec ai-nginx sh -lc 'cat /tmp/nginx.conf.plus > /etc/nginx/nginx.conf'
sudo docker exec ai-nginx nginx -t
sudo docker exec ai-nginx nginx -s reload
```

配置测试通过并 reload 成功：

```text
nginx: configuration file /etc/nginx/nginx.conf test is successful
signal process started
```

---

## 22. 新增路径验证

检查正式配置：

```bash
sudo docker exec ai-nginx sh -lc 'grep -n "detection3d_plus\|detection3d" /etc/nginx/nginx.conf'
```

输出：

```text
91:        location = /smart-tool/detection3d_plus {
93:            proxy_pass http://172.28.0.1:8000/smart-tool/detection3d;
99:        location /smart-tool/detection3d {
```

测试新增路径：

```bash
curl -i http://127.0.0.1:8000/smart-tool/detection3d_plus
```

返回：

```text
HTTP/1.1 405 Method Not Allowed
allow: POST
{"detail":"Method Not Allowed"}
```

测试原路径：

```bash
curl -i http://127.0.0.1:8000/smart-tool/detection3d
```

同样返回：

```text
HTTP/1.1 405 Method Not Allowed
allow: POST
{"detail":"Method Not Allowed"}
```

这说明：

```text
新增路径 detection3d_plus 已正确转发到 plus API；
原路径 detection3d 仍然正常；
405 是正常现象，因为接口只接受 POST。
```

容器状态：

```bash
sudo docker ps | grep -E "detection3d|ai-nginx"
```

结果显示：

```text
detection3d_v5_plus   Up
detection3d           Up
ai-nginx              Up
```

---

## 23. POST 接口格式确认

查看 plus 容器中的接口代码：

```bash
sudo docker exec detection3d_v5_plus sed -n '1,180p' /workspace/OpenPCDet/tools/inference/fastAPI.py
```

关键代码：

```python
class InferenceInput(BaseModel):
    pcURL: str = Field(..., example="http://molardata.com/1.pcd", title="预测图片URL")
    shape: Optional[int] = None
    model: Optional[str] = None
    thresh: Optional[float] = None

@app.post("/smart-tool/detection3d")
async def cat_kpt_inference(request: Request, body: InferenceInput):
    ...
    result = await app.package[body.model].inference(path, shape)
    return {"code": 200, "data": result}
```

接口格式为：

```text
路径：/smart-tool/detection3d
方法：POST
请求体：application/json
必填字段：pcURL
可选字段：shape、model、thresh
```

虽然 `fastAPI.py` 定义了 `thresh` 字段，但目前未传入 `inference`，因此请求中带 `thresh` 不一定会实际影响阈值。

通过 OpenAPI 也确认：

```bash
curl -s http://172.28.0.1:8000/openapi.json | python3 -m json.tool | head -200
```

显示 `pcURL` 为必填字段。

---

## 24. POST 空请求测试

测试空 JSON：

```bash
curl -i -X POST http://127.0.0.1:8000/smart-tool/detection3d_plus -H "Content-Type: application/json" -d '{}'
```

返回：

```text
HTTP/1.1 422 Unprocessable Entity
{"detail":[{"loc":["body","pcURL"],"msg":"field required","type":"value_error.missing"}]}
```

说明：

```text
POST 请求已经成功进入 detection3d_v5_plus 的 FastAPI；
只是缺少必填字段 pcURL；
新增测试路径可用于真实推理请求。
```

---

## 25. 获取真实 pcURL

真实 `pcURL` 不是本地数据集路径，而是平台前端加载点云文件时使用的 HTTP 文件地址。原因是 `fastAPI.py` 使用：

```python
resp = requests.get(url, timeout=5)
```

因此接口需要类似：

```text
http://xxx/xxx.pcd
http://xxx/xxx.bin
```

不能直接传本地路径：

```text
/workspace/OpenPCDet/data/nuscenes/xxx.bin
```

获取方式：

```text
1. 打开标注平台页面；
2. 按 F12；
3. 打开 Network；
4. 搜索或过滤 .pcd；
5. 点击点云文件请求；
6. 在 Headers -> General -> Request URL 中复制完整 URL。
```

本次使用的真实点云 URL 为：

```text
http://172.23.131.39:8080/files/import_all/20260312_test1_ds6_1/2026-03-12-18-52-16_uid1/merged/1773312746543418.pcd
```

---

## 26. 使用真实 pcURL 测试 plus 模型接口

使用新增测试路径推理真实点云：

```bash
curl -sS -X POST http://127.0.0.1:8000/smart-tool/detection3d_plus -H "Content-Type: application/json" -d '{"pcURL":"http://172.23.131.39:8080/files/import_all/20260312_test1_ds6_1/2026-03-12-18-52-16_uid1/merged/1773312746543418.pcd","shape":4,"model":"model"}' -o /tmp/detection3d_plus_result.json && echo "saved to /tmp/detection3d_plus_result.json" && head -c 500 /tmp/detection3d_plus_result.json && echo
```

返回：

```text
saved to /tmp/detection3d_plus_result.json
{"code":200,"data":[[{"type":"shape","drawType":"box3d","label":"human_pedestrian_adult",...
```

说明：

```text
plus 测试路径已经成功完成真实点云推理；
detection3d_v5_plus 能下载平台中的真实 pcd；
接口返回 code=200；
返回格式为平台使用的 shape/box3d 格式。
```

注意：

```text
curl 只是直接请求模型接口；
不会写入平台数据库；
不会让前端页面自动显示标注框。
```

平台页面显示框需要：

```text
前端点击自动标注
    -> 后端调用自动标注接口
    -> 模型返回结果
    -> 后端保存标注结果
    -> 前端刷新显示框
```

---

## 27. plus 模型结果统计

统计 plus 返回结果：

```bash
python3 - <<'PY'
import json
p="/tmp/detection3d_plus_result.json"
data=json.load(open(p))
items=data.get("data", [])
if len(items)==1 and isinstance(items[0], list):
    items=items[0]
print("code:", data.get("code"))
print("num_boxes:", len(items))
from collections import Counter
print("labels:", Counter(x.get("label") for x in items))
scores=[x.get("score",0) for x in items]
if scores:
    print("score_min:", min(scores), "score_max:", max(scores), "score_avg:", sum(scores)/len(scores))
    print("score>=0.25:", sum(s>=0.25 for s in scores))
    print("score>=0.30:", sum(s>=0.30 for s in scores))
    print("score>=0.35:", sum(s>=0.35 for s in scores))
PY
```

输出：

```text
code: 200
num_boxes: 48
labels: Counter({'vehicle_car': 24, 'human_pedestrian_adult': 13, 'vehicle_motorcycle': 7, 'vehicle_truck': 4})
score_min: 0.10141897201538086
score_max: 0.5765841603279114
score_avg: 0.16119588244085511
score>=0.25: 3
score>=0.30: 3
score>=0.35: 3
```

结论：

```text
plus 模型接口已经能返回正常 3D 框；
当前样本共返回 48 个框；
类别名为 OpenPCDet_ljl_plus 的新版类别体系；
大部分框 score 低于 0.25。
```

---

## 28. 与供应商原接口对比

使用原接口测试同一点云：

```bash
curl -sS -X POST http://127.0.0.1:8000/smart-tool/detection3d -H "Content-Type: application/json" -d '{"pcURL":"http://172.23.131.39:8080/files/import_all/20260312_test1_ds6_1/2026-03-12-18-52-16_uid1/merged/1773312746543418.pcd","shape":4,"model":"model"}' -o /tmp/detection3d_vendor_result.json && echo "saved to /tmp/detection3d_vendor_result.json" && head -c 500 /tmp/detection3d_vendor_result.json && echo
```

供应商接口返回：

```text
{"code":200,"data":[[{"type":"shape","drawType":"box3d","label":"Regular_vehicle",...
```

统计对比：

```bash
python3 - <<'PY'
import json
from collections import Counter

for name,p in [("plus","/tmp/detection3d_plus_result.json"),("vendor","/tmp/detection3d_vendor_result.json")]:
    data=json.load(open(p))
    items=data.get("data", [])
    if len(items)==1 and isinstance(items[0], list):
        items=items[0]
    scores=[x.get("score",0) for x in items]
    print("
==", name, "==")
    print("code:", data.get("code"))
    print("num_boxes:", len(items))
    print("labels:", Counter(x.get("label") for x in items))
    if scores:
        print("score_min:", min(scores))
        print("score_max:", max(scores))
        print("score_avg:", sum(scores)/len(scores))
        print("score>=0.25:", sum(s>=0.25 for s in scores))
        print("score>=0.30:", sum(s>=0.30 for s in scores))
        print("score>=0.35:", sum(s>=0.35 for s in scores))
PY
```

输出：

```text
== plus ==
code: 200
num_boxes: 48
labels: Counter({'vehicle_car': 24, 'human_pedestrian_adult': 13, 'vehicle_motorcycle': 7, 'vehicle_truck': 4})
score_min: 0.10141897201538086
score_max: 0.5765841603279114
score_avg: 0.16119588244085511
score>=0.25: 3
score>=0.30: 3
score>=0.35: 3

== vendor ==
code: 200
num_boxes: 84
labels: Counter({'Regular_vehicle': 66, 'Pedestrian': 3, 'Stop_sign': 3, 'Bollard': 3, 'Bus': 3, 'Sign': 2, 'Construction_barrel': 2, 'Large_vehicle': 1, 'Wheeled_device': 1})
score_min: 0.10044293850660324
score_max: 0.4419826567173004
score_avg: 0.18372636962504613
score>=0.25: 13
score>=0.30: 8
score>=0.35: 4
```

结论：

```text
plus 和 vendor 接口都能返回 code=200；
两者返回结构都为平台使用的 shape/box3d；
但类别名体系不同。
```

---

## 29. 当前最大风险：类别名体系不一致

plus 返回类别包括：

```text
vehicle_car
human_pedestrian_adult
vehicle_motorcycle
vehicle_truck
```

供应商返回类别包括：

```text
Regular_vehicle
Pedestrian
Stop_sign
Bollard
Bus
Sign
Construction_barrel
Large_vehicle
Wheeled_device
```

这说明：

```text
模型服务已经跑通；
接口格式基本兼容；
但平台是否能识别 plus 的类别名仍需验证。
```

如果平台当前类别模板只包含供应商类别名，则直接使用 plus 输出可能导致：

```text
接口返回 code=200；
但前端不显示框；
或显示未知类别；
或保存时报类别非法；
或导出/统计时类别丢失。
```

---

## 30. 类别名处理路线

当前有两种路线：

### 路线 A：模型输出映射成供应商类别名

示例：

```text
vehicle_car -> Regular_vehicle
human_pedestrian_adult -> Pedestrian
vehicle_truck -> Large_vehicle
vehicle_motorcycle -> Wheeled_device 或其他平台支持类别
```

优点：

```text
最容易兼容当前平台；
不需要改平台类别模板。
```

缺点：

```text
会损失 plus 模型的 26 类细粒度类别信息；
不符合后续使用新版类别体系的目标。
```

### 路线 B：平台类别模板改成 plus 版本类别名

即让平台支持：

```text
vehicle_car
human_pedestrian_adult
vehicle_motorcycle
vehicle_truck
...
```

优点：

```text
保留 plus 模型完整类别体系；
平台显示名称与训练代码一致。
```

风险：

```text
需要确认平台后端、前端、标签模板、保存逻辑、导出逻辑、统计逻辑都支持新版类别名；
不能只改前端显示名称。
```

当前倾向：

```text
如果项目目标是长期使用 OpenPCDet_ljl_plus 的 26 类体系，
建议优先探索路线 B；
但在平台类别模板未确认前，不建议替换原 /smart-tool/detection3d。
```

---

## 31. 当前状态总结

截至目前，已经完成：

```text
1. 原生产 detection3d 服务已恢复并保持正常；
2. detection3d_v5_plus 已成功加载我们自己的 checkpoint_epoch_20.pth；
3. detection3d_v5_plus FastAPI 服务已启动；
4. ai-nginx 已新增测试路径 /smart-tool/detection3d_plus；
5. 原 /smart-tool/detection3d 路径未被替换；
6. /smart-tool/detection3d_plus 已通过 GET 和 POST 连通性测试；
7. plus 接口已用真实平台 pcURL 成功返回 code=200；
8. plus 接口返回了 48 个 box3d；
9. vendor 原接口同样可正常返回；
10. 当前主要待解决问题是平台类别名是否支持 plus 的类别体系。
```

当前安全状态：

```text
生产路径：
/smart-tool/detection3d
    -> 172.28.5.55:8000
    -> vendor detection3d
    -> 正常

测试路径：
/smart-tool/detection3d_plus
    -> 172.28.0.1:8000
    -> detection3d_v5_plus
    -> 正常
```

---

## 32. 下一步计划

下一步建议：

```text
1. 在平台中查找当前项目支持的完整类别列表；
2. 判断平台是否允许新增或替换 3D 标签模板；
3. 如果平台能支持 plus 类别，则在测试项目中导入 plus 26 类标签；
4. 用 /smart-tool/detection3d_plus 测试平台是否能显示、保存、导出 plus 类别；
5. 如果平台短期不能改类别模板，则考虑在 detection3d_v5_plus 输出阶段加 label 映射；
6. 在类别问题解决前，不替换原 /smart-tool/detection3d。
```

当前不建议直接替换生产路径：

```text
/smart-tool/detection3d
```

原因：

```text
虽然 plus 模型接口已经跑通，
但类别名体系尚未与平台配置完全确认；
直接替换存在前端不显示、保存失败或导出异常的风险。
```

---

## 33. 回滚命令

如果需要撤销 `ai-nginx` 新增测试路径，可执行：

```bash
sudo docker exec ai-nginx sh -lc 'cat /tmp/nginx.conf.before_detection3d_plus > /etc/nginx/nginx.conf'
sudo docker exec ai-nginx nginx -t
sudo docker exec ai-nginx nginx -s reload
```

回滚后：

```text
/smart-tool/detection3d_plus 将不再存在；
/smart-tool/detection3d 保持原生产路径。
```

如果需要停止 plus 测试服务：

```text
在运行 detection3d_v5_plus uvicorn 的终端中按 Ctrl + C。
```

该操作只停止测试服务，不影响生产 `detection3d`。

---

## 34. 平台任务类别配置与 aiPower 字段排查

在 plus 接口可以通过 `curl` 正常返回结果后，下一步开始排查平台任务是否能够真正调用 `detection3d_plus`，以及平台类别体系是否兼容 plus 模型输出。

### 34.1 find-labels 返回为空

在浏览器 DevTools 的 Network 中查看 `find-labels` 请求，发现其返回为：

```json
{"code":200,"data":[]}
```

说明：

```text
find-labels 更像是查询当前帧已有标注结果；
当前帧没有保存标注时 data 为空；
它不是完整的项目类别模板接口。
```

因此后续转向查看 `task-info`、`workflow-info`、`get-item-info` 等接口。

### 34.2 task-info 中发现 labelConfig 与 labelAlias

在 `task-info` 的 Response 中发现：

```text
setting.labelConfig
setting.labelAlias
setting.aiPower
```

其中 `labelConfig` 里是平台当前任务的 3D 标注模板，前端显示名多为中文，例如：

```text
成人
儿童
坐轮椅的人
婴儿车
骑滑板车的人
乘用车小于10人
铰链式巴士
普通巴士
卡车
施工车辆
救护车
警用车
挂车
路障
交通锥
推车
杂物
油罐车/拖拉机
摩托车
人力电动两轮车
人群
自行车群
其他
动物
三轮车
自行车
```

同时 `labelAlias` 中包含类似 nuScenes 风格的点号类别，例如：

```text
human.pedestrian.adult
vehicle.car
vehicle.bus.rigid
vehicle.truck
movable_object.barrier
vehicle.motorcycle
vehicle.tricycle
bicycle
```

这说明平台当前任务并非只有供应商英文类别，也不是完全不支持 26 类体系；它内部存在：

```text
中文显示标签 labelConfig
点号标准类别 labelAlias
AI 工具输出 label
```

三套可能相关的命名体系。

### 34.3 供应商自动标注返回的是英文类别

点击平台原自动标注后，在 Network 中看到 `detection3d` 请求返回：

```json
{
  "type": "shape",
  "drawType": "box3d",
  "label": "Regular_vehicle",
  ...
}
```

说明：

```text
供应商自动标注服务返回的 label 是英文；
并不是直接返回 task-info 中的中文 labelConfig；
平台可以接收英文 AI 输出，但其内部可能还有映射或保存转换逻辑。
```

### 34.4 aiPower 字段

在 `task-info` 中搜索 `aiPower`，发现当前任务最初为：

```json
"aiPower": "detection3d"
```

后续为了测试，将当前任务的 `setting.aiPower` 临时改为：

```json
"aiPower": "detection3d_plus"
```

注意：后面验证发现，`aiPower` 虽然在任务配置中存在，但并不是 `main-server` 实际选择 3D 自动标注转发路径的唯一控制因素。

---

## 35. MongoDB 中定位任务配置

### 35.1 MongoDB 认证问题

初次进入 MongoDB 执行 `mongosh --quiet` 并直接执行 `listDatabases` 时出现：

```text
MongoServerError: Command listDatabases requires authentication
```

说明 MongoDB 开启了认证。

通过查看容器环境变量确认存在 MongoDB root 账号：

```bash
sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -Ei 'mongo|user|pass|auth'
```

输出包含：

```text
MONGO_INITDB_ROOT_USERNAME=root
MONGO_INITDB_ROOT_PASSWORD=...
```

安全说明：

```text
实际密码不要写入文档；
后续命令中用 <MONGO_ROOT_PASSWORD> 代替。
```

### 35.2 定位任务文档

由于任务 `_id` 在 MongoDB 中是 `ObjectId` 类型，直接按字符串 `_id` 查询无结果。后续改为同时查询字符串和 `ObjectId`：

```bash
sudo docker exec mongodb mongosh --quiet -u root -p '<MONGO_ROOT_PASSWORD>' --authenticationDatabase admin --eval 'const id="69fe8f840fa06ee86bde27a6"; const oid=ObjectId(id); const name="test3d-260509"; db.adminCommand({listDatabases:1}).databases.forEach(dbInfo=>{const d=db.getSiblingDB(dbInfo.name); let cols=[]; try{cols=d.getCollectionNames()}catch(e){return}; cols.forEach(c=>{let doc=null; try{doc=d.getCollection(c).findOne({$or:[{_id:id},{_id:oid},{id:id},{taskId:id},{name:name}]},{_id:1,id:1,taskId:1,name:1,type:1,status:1,"setting.aiPower":1,createdAt:1,updatedAt:1})}catch(e){}; if(doc){print("DB="+dbInfo.name+" COLL="+c); printjson(doc)}})})'
```

查询结果：

```text
DB=molar
COLL=tasks

{
  _id: ObjectId("69fe8f840fa06ee86bde27a6"),
  status: 'PROCESSING',
  type: 'PCAT',
  name: 'test3d-260509',
  setting: {
    aiPower: 'detection3d'
  },
  createdAt: ISODate("2026-05-09T01:36:04.637Z"),
  updatedAt: ISODate("2026-05-11T07:34:31.571Z")
}
```

结论：

```text
当前测试任务存储位置：
DB: molar
Collection: tasks
_id: ObjectId("69fe8f840fa06ee86bde27a6")
name: test3d-260509
setting.aiPower: detection3d
```

### 35.3 备份任务文档

在修改前备份当前任务文档：

```bash
sudo docker exec mongodb mongosh --quiet -u root -p '<MONGO_ROOT_PASSWORD>' --authenticationDatabase admin --eval 'const d=db.getSiblingDB("molar"); const doc=d.tasks.findOne({_id:ObjectId("69fe8f840fa06ee86bde27a6")}); printjson(doc);' > /tmp/task_69fe8f840fa06ee86bde27a6_backup_before_aiPower_change.json
```

确认备份：

```bash
ls -lh /tmp/task_69fe8f840fa06ee86bde27a6_backup_before_aiPower_change.json
head -c 300 /tmp/task_69fe8f840fa06ee86bde27a6_backup_before_aiPower_change.json && echo
```

备份文件大小约：

```text
51K
```

说明任务文档已完整备份。

### 35.4 修改测试任务 aiPower

仅修改当前测试任务：

```bash
sudo docker exec mongodb mongosh --quiet -u root -p '<MONGO_ROOT_PASSWORD>' --authenticationDatabase admin --eval 'const d=db.getSiblingDB("molar"); const r=d.tasks.updateOne({_id:ObjectId("69fe8f840fa06ee86bde27a6")}, {$set: {"setting.aiPower": "detection3d_plus"}}); printjson(r); printjson(d.tasks.findOne({_id:ObjectId("69fe8f840fa06ee86bde27a6")}, {_id:1,name:1,"setting.aiPower":1,updatedAt:1}));'
```

返回：

```text
matchedCount: 1
modifiedCount: 1

setting.aiPower: detection3d_plus
```

说明：

```text
当前测试任务的 aiPower 已经改为 detection3d_plus；
该操作只影响 test3d-260509 这个任务。
```

但是后续发现，即使 `task-info` 中已经显示：

```json
"aiPower": "detection3d_plus"
```

平台自动标注实际仍然请求：

```text
/smart-tool/detection3d
```

因此继续排查 `main-server`。

---

## 36. main-server 调用链排查

### 36.1 找到实际发起自动标注请求的容器

`ai-nginx` 日志中平台自动标注请求来源 IP 为：

```text
172.28.5.4
```

执行：

```bash
for c in $(sudo docker ps --format '{{.Names}}'); do ip=$(sudo docker inspect -f '{{range $name,$conf := .NetworkSettings.Networks}}{{printf "%s " $conf.IPAddress}}{{end}}' "$c" 2>/dev/null); echo "$c $ip"; done | grep 172.28.5.4
```

输出：

```text
main-server 172.28.5.4
```

结论：

```text
平台前端不是直接请求 ai-nginx；
真正发起 3D 自动标注请求的是 main-server。
```

### 36.2 查看 main-server 代码目录

```bash
sudo docker exec main-server sh -lc 'pwd; ls -lah /; ls -lah /app 2>/dev/null; ls -lah /usr/src/app 2>/dev/null; ls -lah /workspace 2>/dev/null'
```

输出显示当前工作目录为：

```text
/code
```

因此后续在 `/code` 中搜索。

### 36.3 搜索 detection3d 与 aiPower

```bash
sudo docker exec main-server sh -lc 'grep -R "smart-tool/detection3d\|detection3d_plus\|detection3d\|aiPower\|smart-tool" -n /code 2>/dev/null | head -120'
```

关键结果：

```text
/code/dist/main/main.js:2921:    PC_DETECTION: '/smart-tool/detection3d',
/code/dist/main/main.js:39248:    async detection3d(dto, user) {
/code/dist/main/main.js:39251:        const url = this.configService.get('ai.domain') + '/smart-tool/detection3d';
/code/dist/main/main.js:39525:    (0, common_1.Post)('detection3d'),
```

同时日志显示：

```text
module=ToolService.request||url=http://172.28.5.50:8000/smart-tool/detection3d
```

说明：

```text
main-server 内部 detection3d 接口写死调用 /smart-tool/detection3d；
任务配置中的 setting.aiPower 并未被这段代码用于决定转发路径。
```

### 36.4 查看关键代码片段

执行：

```bash
sudo docker exec main-server sh -lc "sed -n '39240,39265p' /code/dist/main/main.js; echo '--- controller ---'; sed -n '39515,39540p' /code/dist/main/main.js; echo '--- constants ---'; sed -n '2910,2925p' /code/dist/main/main.js"
```

关键代码：

```js
async detection3d(dto, user) {
    const taskId = dto.taskId;
    const params = dto.params;
    const url = this.configService.get('ai.domain') + '/smart-tool/detection3d';
    return await this.toolService.start(taskId, url, params, user);
}
```

结论：

```text
main-server 的 /api/smart-tool/detection3d 后端逻辑硬编码了 /smart-tool/detection3d；
这就是修改 aiPower 后仍然请求旧接口的根因。
```

---

## 37. main-server 针对测试任务增加分流逻辑

### 37.1 设计原则

为了不影响原有功能，不能直接把 `ai-nginx` 的原路径 `/smart-tool/detection3d` 改成 plus。

采用更安全的方式：

```text
只在 main-server 中对当前测试 taskId 分流。

如果 taskId == 69fe8f840fa06ee86bde27a6：
    调用 /smart-tool/detection3d_plus

否则：
    仍然调用 /smart-tool/detection3d
```

这样：

```text
当前测试任务走 plus；
其他任务仍然走供应商原模型；
不会全局替换生产 3D 自动标注接口。
```

### 37.2 备份 main-server 文件

```bash
sudo docker exec main-server sh -lc 'cp /code/dist/main/main.js /code/dist/main/main.js.bak_detection3d_plus_task_69fe8f'
```

确认备份：

```bash
sudo docker exec main-server sh -lc 'ls -lh /code/dist/main/main.js /code/dist/main/main.js.bak_detection3d_plus_task_69fe8f'
```

### 37.3 第一次补丁未生效的原因

第一次执行 Node 补丁脚本时使用：

```bash
sudo docker exec main-server node - <<'NODE'
...
NODE
```

由于没有加 `-i`，heredoc 内容没有传入容器，导致 Node 脚本实际没有执行。检查代码仍然是：

```js
const url = this.configService.get('ai.domain') + '/smart-tool/detection3d';
```

因此修正为：

```bash
sudo docker exec -i main-server node - <<'NODE'
...
NODE
```

### 37.4 正式写入分流补丁

执行：

```bash
sudo docker exec -i main-server node - <<'NODE'
const fs = require('fs');

const p = '/code/dist/main/main.js';
let s = fs.readFileSync(p, 'utf8');

const oldLine = "        const url = this.configService.get('ai.domain') + '/smart-tool/detection3d';";
const newBlock = "        const route = taskId === '69fe8f840fa06ee86bde27a6' ? '/smart-tool/detection3d_plus' : '/smart-tool/detection3d';\n        const url = this.configService.get('ai.domain') + route;";

if (!s.includes(oldLine)) {
  console.error('target line not found or already modified');
  process.exit(1);
}

s = s.replace(oldLine, newBlock);
fs.writeFileSync(p, s);
console.log('patched detection3d route for task 69fe8f840fa06ee86bde27a6');
NODE
```

返回：

```text
patched detection3d route for task 69fe8f840fa06ee86bde27a6
```

检查修改：

```bash
sudo docker exec main-server sh -lc "sed -n '39246,39258p' /code/dist/main/main.js"
```

结果：

```js
async detection3d(dto, user) {
    const taskId = dto.taskId;
    const params = dto.params;
    const route = taskId === '69fe8f840fa06ee86bde27a6' ? '/smart-tool/detection3d_plus' : '/smart-tool/detection3d';
    const url = this.configService.get('ai.domain') + route;
    return await this.toolService.start(taskId, url, params, user);
}
```

### 37.5 重启 main-server

```bash
sudo docker restart main-server
```

确认状态：

```bash
sudo docker ps | grep main-server
sudo docker logs --tail 80 main-server
```

关键启动成功信息：

```text
Nest application successfully started
```

说明：

```text
main-server 重启成功；
分流补丁已经加载。
```

---

## 38. 验证 main-server 分流是否生效

### 38.1 页面测试未触发新请求的问题

在浏览器页面点击自动标注时，多次只看到：

```text
*-result.json
```

且有时显示：

```text
disk cache
```

说明：

```text
当前帧已有 result.json；
页面可能只是读取已有自动标注结果；
并未重新触发 /api/smart-tool/detection3d 请求。
```

因此仅靠页面点击无法稳定验证后端分流逻辑。

### 38.2 直接调用 main-server 接口验证

从浏览器请求头中复制 `access-token`，然后在宿主机直接调用：

```bash
curl -sS -X POST http://127.0.0.1:3004/api/smart-tool/detection3d \
  -H "Content-Type: application/json" \
  -H "access-token: <ACCESS_TOKEN>" \
  -d '{"taskId":"69fe8f840fa06ee86bde27a6","params":{"pcURL":"http://172.23.131.39:8080/files/import_all/20260312_test1_ds6_1/2026-03-12-18-52-16_uid1/merged/1773312746543418.pcd","shape":4,"model":"model"}}'
```

注意：

```text
access-token 属于敏感信息；
文档中不记录真实 token。
```

### 38.3 main-server 日志验证

实时查看：

```bash
sudo docker exec main-server sh -lc 'tail -f /code/logs/2026-05-28.log' | grep --line-buffered -E "api/smart-tool|ToolService.request|detection3d|detection3d_plus"
```

调用后日志显示：

```text
requestInfo={"method":"POST","path":"/api/smart-tool/detection3d",...}

module=ToolService.request||url=http://172.28.5.50:8000/smart-tool/detection3d_plus

module=ToolService.request||data={
  "pcURL":"http://172.23.131.39:8080/files/import_all/20260312_test1_ds6_1/2026-03-12-18-52-16_uid1/merged/1773312746543418.pcd",
  "shape":4,
  "model":"model"
}

module=ToolService.request||code=200||result=200

responseInfo={"method":"POST","path":"/api/smart-tool/detection3d","responseTime":"1732ms",...}
```

说明：

```text
main-server 的 /api/smart-tool/detection3d 已经命中当前 taskId 分流；
实际转发到了 /smart-tool/detection3d_plus。
```

### 38.4 ai-nginx 日志验证

查看：

```bash
sudo docker logs -f --tail 0 ai-nginx 2>&1 | grep --line-buffered -E "detection3d|detection3d_plus|smart-tool"
```

出现：

```text
172.28.5.4 - - [28/May/2026:09:25:09 +0000] "POST /smart-tool/detection3d_plus HTTP/1.1" 200 11911 "-" "axios/0.27.2" "-"
```

说明：

```text
main-server 已经通过 ai-nginx 请求 /smart-tool/detection3d_plus；
请求来源 172.28.5.4 为 main-server；
HTTP 200 表示 plus 接口返回成功。
```

### 38.5 main-server 返回结果统计

将 main-server 直调结果保存后统计：

```text
code: 200
num_boxes: 48
labels: Counter({
  'vehicle_car': 24,
  'human_pedestrian_adult': 13,
  'vehicle_motorcycle': 7,
  'vehicle_truck': 4
})
score_min: 0.10141897201538086
score_max: 0.5765841603279114
score_avg: 0.16119658450285593
```

说明：

```text
main-server 分流后拿到的正是 plus 模型结果；
类别仍为 plus 模型的下划线类别名。
```

### 38.6 页面触发后的 ai-nginx 日志

后续页面操作后，ai-nginx 日志中出现：

```text
172.28.5.4 - - [28/May/2026:09:27:33 +0000] "POST /smart-tool/detection3d_plus HTTP/1.1" 200 11909 "-" "axios/0.27.2" "-"
```

说明：

```text
平台页面/后端已经可以触发 detection3d_plus；
测试任务已成功走 plus 自动标注接口。
```

---

## 39. 当前遗留问题：result.json 中 label 仍为 Vehicle

虽然 `main-server -> ai-nginx -> detection3d_plus -> detection3d_v5_plus` 链路已验证成功，但页面加载的 `result.json` 中仍然看到：

```json
"label": "Vehicle"
```

而 plus 原始返回为：

```text
vehicle_car
human_pedestrian_adult
vehicle_motorcycle
vehicle_truck
```

这说明当前问题已经不在接口转发层，而在：

```text
平台结果保存 / 标签转换 / 前端显示兼容层
```

可能原因包括：

```text
1. 平台保存 AI 结果时没有识别 plus 的下划线类别名；
2. 平台内部根据 labelAlias 做映射，但期望的是点号格式，如 vehicle.car；
3. 未匹配到类别时，平台将其兜底为 Vehicle；
4. result.json 仍可能存在旧结果或缓存，但从 plus 调用成功看，更可能是保存/转换阶段的问题。
```

结合 `task-info` 中的 `labelAlias`，平台更可能识别如下点号格式：

```text
vehicle.car
human.pedestrian.adult
vehicle.motorcycle
vehicle.truck
```

而 plus 当前输出为下划线格式：

```text
vehicle_car
human_pedestrian_adult
vehicle_motorcycle
vehicle_truck
```

因此后续重点应放在：

```text
查找平台哪里将未知 label 转换为 Vehicle；
确认是否应在 plus 输出阶段将下划线类别映射为点号类别；
或调整平台 labelAlias / labelConfig。
```

---

## 40. 当前状态总结

截至目前，已经完成：

```text
1. detection3d_v5_plus 已启动并加载 OpenPCDet_ljl_plus 模型；
2. ai-nginx 已新增 /smart-tool/detection3d_plus；
3. 原 /smart-tool/detection3d 生产路径仍保留；
4. plus 接口直接 curl 测试成功；
5. main-server 原来硬编码 /smart-tool/detection3d 的问题已定位；
6. 已对 main-server 做当前测试任务级别分流；
7. main-server 重启成功；
8. main-server 直调 /api/smart-tool/detection3d 时，当前 taskId 已转发到 /smart-tool/detection3d_plus；
9. ai-nginx 日志确认 main-server 请求 /smart-tool/detection3d_plus；
10. plus 模型通过 main-server 返回 48 个框；
11. 页面/后端也已出现 /smart-tool/detection3d_plus 请求；
12. 当前剩余问题是 result.json 中 label 被保存/显示为 Vehicle。
```

当前安全状态：

```text
当前测试任务：
taskId = 69fe8f840fa06ee86bde27a6
-> main-server 分流到 /smart-tool/detection3d_plus

其他任务：
-> 仍然走 /smart-tool/detection3d
```

因此：

```text
当前改动没有全局替换生产 detection3d；
其他任务理论上仍不受影响。
```

---

## 41. 后续工作建议

明天继续时，建议从标签转换逻辑开始排查。

### 41.1 搜索 main-server 中的标签转换逻辑

执行：

```bash
sudo docker exec main-server sh -lc 'grep -R "Vehicle\|Regular_vehicle\|vehicle_car\|human_pedestrian_adult\|labelAlias\|aiPowerMapping\|saveBoxPoint" -n /code/dist/main/main.js /code/logs 2>/dev/null | head -160'
```

目的：

```text
定位哪里将未知 label 转成 Vehicle；
定位平台是否使用 labelAlias 或 aiPowerMapping；
定位保存 box3d 时是否存在默认类别兜底逻辑。
```

### 41.2 查找 ToolService.start

执行：

```bash
sudo docker exec main-server sh -lc 'grep -n "async start" /code/dist/main/main.js | head -20'
```

然后根据返回行号查看附近代码，例如：

```bash
sudo docker exec main-server sh -lc "sed -n '起始行,结束行p' /code/dist/main/main.js"
```

目的：

```text
确认 ToolService.start 在接收 AI 返回后是否做 label 转换；
确认转换发生在 main-server 还是前端保存阶段。
```

### 41.3 检查 save-labels 请求

在浏览器 Network 中点击：

```text
save-labels
```

查看：

```text
Payload
Response
```

重点确认保存时的 label 是：

```text
Vehicle
```

还是：

```text
vehicle_car / human_pedestrian_adult
```

判断依据：

```text
如果 save-labels Payload 已经是 Vehicle：
    前端或 result.json 生成前已经完成转换；

如果 save-labels Payload 是 vehicle_car，但 result.json 是 Vehicle：
    后端保存阶段进行了转换；
```

### 41.4 可能的解决方向

后续可能有三种解决方式：

#### 方案 A：plus 输出改为平台 labelAlias 点号类别

将 plus 输出从：

```text
vehicle_car
human_pedestrian_adult
vehicle_motorcycle
vehicle_truck
```

转换为：

```text
vehicle.car
human.pedestrian.adult
vehicle.motorcycle
vehicle.truck
```

优点：

```text
更接近平台 task-info 中已有的 labelAlias；
改动范围在 detection3d_v5_plus 输出阶段；
不需要改平台中文显示模板。
```

#### 方案 B：main-server 中对 plus 返回 label 做映射

在 main-server 当前测试任务分流逻辑附近或 AI 结果处理阶段，将 plus label 映射成平台可识别 label。

优点：

```text
可只针对当前测试任务做映射；
不改模型容器。
```

缺点：

```text
main-server 是平台核心服务，不宜过多改动；
需要先确认保存逻辑。
```

#### 方案 C：修改平台 labelConfig / labelAlias

让平台直接识别 plus 下划线类别，例如：

```text
vehicle_car
human_pedestrian_adult
...
```

优点：

```text
最终显示名可以与 OpenPCDet_ljl_plus 完全一致。
```

缺点：

```text
涉及平台任务模板、保存逻辑、导出逻辑、统计逻辑；
风险最大；
应仅在测试任务中进行。
```

当前建议优先顺序：

```text
先查转换逻辑；
再尝试方案 A：plus 输出下划线 -> 点号 labelAlias；
暂不直接改正式平台标签模板。
```

---

## 42. 当前回滚方案

如果需要恢复 main-server：

```bash
sudo docker exec main-server sh -lc 'cp /code/dist/main/main.js.bak_detection3d_plus_task_69fe8f /code/dist/main/main.js'
sudo docker restart main-server
```

如果需要恢复当前测试任务的 aiPower：

```bash
sudo docker exec mongodb mongosh --quiet -u root -p '<MONGO_ROOT_PASSWORD>' --authenticationDatabase admin --eval 'const d=db.getSiblingDB("molar"); const r=d.tasks.updateOne({_id:ObjectId("69fe8f840fa06ee86bde27a6")}, {$set: {"setting.aiPower": "detection3d"}}); printjson(r); printjson(d.tasks.findOne({_id:ObjectId("69fe8f840fa06ee86bde27a6")}, {_id:1,name:1,"setting.aiPower":1}));'
```

如果需要撤销 ai-nginx 的测试路径：

```bash
sudo docker exec ai-nginx sh -lc 'cat /tmp/nginx.conf.before_detection3d_plus > /etc/nginx/nginx.conf'
sudo docker exec ai-nginx nginx -t
sudo docker exec ai-nginx nginx -s reload
```

如果需要停止 plus API：

```text
在运行 detection3d_v5_plus uvicorn 的终端中按 Ctrl + C。
```

恢复后状态应为：

```text
平台原 /smart-tool/detection3d -> vendor detection3d；
测试路径 detection3d_plus 不再参与当前任务；
其他生产任务不受影响。
```
---

## 43. 2026-05-29：当前 package 内空白 item 排查

在继续验证平台是否能够显示 plus 类别之前，先排查当前任务中哪些 item 已经存在旧标签，哪些 item 仍为空白，避免继续使用已经保存过供应商旧结果的 item 造成判断混淆。

当前测试任务：

```text
taskId = 69fe8f840fa06ee86bde27a6
packageId = 69fed33d0fa06ee86bdea679
```

执行 MongoDB 查询当前 package 内 item 的标签数量：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p')

sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin <<'MONGO'
const d = db.getSiblingDB("molar");

const taskId = ObjectId("69fe8f840fa06ee86bde27a6");
const packageId = ObjectId("69fed33d0fa06ee86bdea679");

const items = d.items.find(
  {
    taskId,
    $or: [
      { packageId },
      { packageId: packageId.toString() },
      { itemPackageId: packageId },
      { itemPackageId: packageId.toString() }
    ]
  },
  { _id: 1, packageId: 1, itemPackageId: 1, name: 1, index: 1, sort: 1 }
).limit(100).toArray();

print("items in package:", items.length);

for (const item of items) {
  const labelCount = d.labels.countDocuments({ itemId: item._id });
  printjson({
    itemId: item._id,
    labelCount,
    packageId: item.packageId,
    itemPackageId: item.itemPackageId,
    name: item.name,
    index: item.index,
    sort: item.sort
  });
}
MONGO
```

查询结果显示当前 package 内共 10 个 item，其中：

```text
69fed33d0fa06ee86bdea67a   labelCount=84
69fed33d0fa06ee86bdea67b   labelCount=0
69fed33d0fa06ee86bdea67c   labelCount=0
69fed33d0fa06ee86bdea67d   labelCount=0
69fed33d0fa06ee86bdea67e   labelCount=0
69fed33d0fa06ee86bdea67f   labelCount=0
69fed33d0fa06ee86bdea680   labelCount=86
69fed33d0fa06ee86bdea681   labelCount=108
69fed33d0fa06ee86bdea682   labelCount=0
69fed33d0fa06ee86bdea683   labelCount=0
```

结论：

```text
67a、680、681 已经保存过旧标签，不适合继续作为纯净验证样本；
67b、67c、67d、67e、67f、682、683 是当前 package 内空白 item；
后续选择 67b 作为主要测试 item。
```

需要注意：

```text
之前曾尝试使用 69fea6570fa06ee86bde3a4f 作为空白 item；
虽然它 labelCount=0，但它不属于当前 packageId；
直接替换 URL 中 itemId 后，页面会长时间卡在“当前条目加载中”。
```

因此，后续手动切换 item 时必须保证：

```text
itemId 与 packageId 属于同一个 package；
itemId 和 selectIds 要同时替换。
```

67b 页面地址：

```text
http://172.23.131.39:8080/items?version=latest&taskId=69fe8f840fa06ee86bde27a6&vm=batch&batchId=69fed33d0fa06ee86bdea679&dm=all&role=69fe8f840fa06ee86bde27b8&packageId=69fed33d0fa06ee86bdea679&itemId=69fed33d0fa06ee86bdea67b&nodeId=69fe8f840fa06ee86bde27b8&selectIds=69fed33d0fa06ee86bdea67b
```

---

## 44. 67b 自动标注按钮未触发模型请求

在浏览器中打开 67b 后，页面可以正常加载点云和图像，但点击页面上的“自动标注/AI 结果”相关动作后，实时监控日志没有出现新的模型请求。

实时监控命令：

```bash
sudo docker logs -f --tail 0 ai-nginx 2>&1 | grep --line-buffered -E "detection3d|detection3d_plus|smart-tool"
```

```bash
sudo docker exec main-server sh -lc 'tail -f /code/logs/2026-05-29.log' | grep --line-buffered -E "api/smart-tool|ToolService.request|detection3d|detection3d_plus"
```

点击页面后现象：

```text
ai-nginx 没有新增 /smart-tool/detection3d_plus 请求；
main-server 没有新增 /api/smart-tool/detection3d 请求；
MongoDB 中 67b labels 仍然为 0。
```

因此判断：

```text
当前页面点击动作没有真正触发实时 detection3d 模型请求；
问题不在 detection3d_plus 服务，也不在 ai-nginx 转发，而在前端当前动作并非重新调用模型。
```

---

## 45. 67b 点云数据确认

为排除 “67b 没有点云 URL 导致自动标注无法触发” 的可能，查询 67b 的 item 与 item_infos：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p')

sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin <<'MONGO'
const d = db.getSiblingDB("molar");
const itemId = ObjectId("69fed33d0fa06ee86bdea67b");

print("===== items =====");
printjson(d.items.findOne({_id:itemId}));

print("===== item_infos =====");
d.item_infos.find({itemId}).forEach(doc => printjson(doc));
MONGO
```

查询结果显示 67b 的 `item_infos.info.pcdUrl` 中存在真实点云文件路径，例如：

```text
/files/import_all/20260312_test1_ds6_1/2026-03-12-18-53-16_uid1/merged/1773312806509910.pcd
```

因此确认：

```text
67b 有有效点云数据；
点击页面后不触发模型请求，不是因为 pcURL 缺失。
```

---

## 46. 页面当前动作实际读取 OSS result.json

在浏览器 DevTools 的 Network 中观察到，每次点击当前页面动作后，都会出现类似请求：

```text
GET https://molar-public.oss-cn-hangzhou.aliyuncs.com/ai-demo/642568f685975ed9a12372f6-result.json
```

响应头显示：

```text
Server: AliyunOSS
Content-Type: application/json
Request Method: GET
```

这说明：

```text
当前页面动作只是读取阿里云 OSS 上的静态 result.json；
它没有经过 main-server；
也没有经过 ai-nginx；
更没有调用 detection3d_plus 模型。
```

因此解释了为什么两个后端实时日志窗口没有任何反应。

进一步搜索前端、main-server 和任务配置：

```bash
sudo docker exec molar-label-system-fe-v2 sh -lc 'grep -R "642568f685975ed9a12372f6\|ai-demo\|molar-public\|result.json" -n /code/molar-label-system-fe-v2 /usr/share/nginx/html 2>/dev/null | head -200'
```

```bash
sudo docker exec main-server sh -lc 'grep -R "642568f685975ed9a12372f6\|ai-demo\|molar-public\|result.json" -n /code/dist/main/main.js /code/logs 2>/dev/null | head -200'
```

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p')

sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin <<'MONGO'
const d = db.getSiblingDB("molar");
const taskId = ObjectId("69fe8f840fa06ee86bde27a6");

const task = d.tasks.findOne({_id: taskId});
const s = JSON.stringify(task);

print("contains 642568:", s.includes("642568f685975ed9a12372f6"));
print("contains ai-demo:", s.includes("ai-demo"));
print("contains result.json:", s.includes("result.json"));
MONGO
```

搜索结果：

```text
任务配置中不包含 642568 / ai-demo / result.json；
main-server 中没有生成这个 result.json 的相关日志；
前端初步 grep 也没有直接搜到硬编码来源。
```

后续在 `project-server-v6` 中发现：

```text
get-ai-check-result
push-item-ai-check-result
push-item-ai-label-result
```

说明当前页面读取 `result.json` 更接近平台的 AI 检查/预置结果链路，而不是实时 smart-tool/detection3d 链路。

---

## 47. project-server-v6 中 AI check / AI label 链路排查

在 `project-server-v6` 中搜索：

```bash
for c in main-server project-server-v6 open-server; do
  echo "===== $c ====="
  sudo docker exec $c sh -lc 'grep -R "get-ai-check-result\|ai-check-result\|result.json\|642568f685975ed9a12372f6\|ai-demo" -n /code /app /usr/src/app 2>/dev/null | head -120'
done
```

关键发现：

```text
project-server-v6:
/code/dist/main/main.js:21758: Post('get-ai-check-result')
/code/dist/main/main.js:22353: url: `${config.micro['open-platform'].url}/ai-check/get-ai-check-result`
/code/dist/main/main.js:28583: Post('push-item-ai-check-result')
/code/dist/main/main.js:28591: Post('push-item-ai-label-result')
```

查看 `getAICheckResult` 逻辑：

```bash
sudo docker exec project-server-v6 sh -lc "sed -n '22290,22410p' /code/dist/main/main.js"
```

核心逻辑：

```js
async getAICheckResult(itemId) {
    const aiCheckBatchId = await this.itemRecordService.findLatestAICheckSubmitBatchIdAll(itemId);
    if (!aiCheckBatchId) {
        return [];
    }
    try {
        const res = await axios({
            method: 'post',
            url: `${config.micro['open-platform'].url}/ai-check/get-ai-check-result`,
            data: {
                aiCheckBatchId,
                itemIds: [itemId]
            }
        });
        return res?.data || [];
    }
    catch (e) {
        this.logger.error('getAICheckResult||', e);
        return [];
    }
}
```

结论：

```text
页面当前看到的 OSS result.json 属于 AI check result 读取逻辑；
它不是实时调用 detection3d_plus；
因此不能用这个页面动作判断 plus 模型是否被调用。
```

---

## 48. 手动直调 main-server 验证 67b plus 推理

为了绕过前端按钮，直接用 67b 的真实 pcURL 调用 main-server 的 smart-tool 接口。由于 main-server 已经针对当前测试 taskId 做了分流，因此理论上会进入 `/smart-tool/detection3d_plus`。

调用命令示例：

```bash
curl -sS -X POST http://127.0.0.1:3004/api/smart-tool/detection3d \
  -H "Content-Type: application/json" \
  -H "access-token: <ACCESS_TOKEN>" \
  -d '{"taskId":"69fe8f840fa06ee86bde27a6","params":{"pcURL":"http://172.23.131.39:8080/files/import_all/20260312_test1_ds6_1/2026-03-12-18-53-16_uid1/merged/1773312806509910.pcd","shape":4,"model":"model"}}' \
  -o /tmp/67b_plus_result.json && echo "saved /tmp/67b_plus_result.json" && head -c 500 /tmp/67b_plus_result.json && echo
```

安全说明：

```text
access-token 属于敏感信息；
文档中只保留 <ACCESS_TOKEN> 占位符，不记录真实 token。
```

返回结果：

```text
saved /tmp/67b_plus_result.json
{"code":200,"data":[[{"type":"shape","drawType":"box3d","label":"human_pedestrian_adult",...
```

说明：

```text
main-server -> ai-nginx -> detection3d_plus -> detection3d_v5_plus 这条链路对 67b 的 pcURL 仍然可用；
plus 模型可以对 67b 点云返回检测框；
返回类别为 plus 原始下划线类别名。
```

---

## 49. push-item-ai-label-result 路径不适合当前 workflow

继续排查 `project-server-v6` 的 `push-item-ai-label-result` 是否可以用于写入 plus 标签。

Controller 中：

```js
pushItemAILabelResult(dto) {
    return this.itemManager.handleItemAILabelResult(dto);
}
```

真正实现：

```bash
sudo docker exec project-server-v6 sh -lc "sed -n '30080,30480p' /code/dist/main/main.js"
```

核心逻辑：

```js
async handleItemAILabelResult(body) {
    const { taskId, itemIds, nodeId, data, status } = body;
    const workflow = await this.workflowService.findByTaskId(taskId);
    if (!isAILabelNode(workflow, nodeId)) {
        this.logger.error('节点异常');
        return;
    }
    if (status === ITEM_AI_LABEL_STATUS.SUCCESS) {
        await this.labelService.aiLabelInsert(taskId, data);
    }
    ...
}
```

`PushItemAILabelResultDto` 至少要求：

```text
taskId
nodeId
itemIds
status
```

并且 `status` 枚举为：

```text
SUCCESS
FAIL
```

进一步查询当前 workflow：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p')

sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin <<'MONGO'
const d = db.getSiblingDB("molar");
const taskId = ObjectId("69fe8f840fa06ee86bde27a6");

const w = d.workflows.findOne({ taskId });
if (!w) {
  print("workflow not found");
} else {
  print("workflowId:", w._id);
  print("nodes:");
  for (const n of (w.nodes || [])) {
    printjson({
      _id: n._id,
      id: n.id,
      name: n.name,
      type: n.type,
      setting: n.setting
    });
  }
}
MONGO
```

当前 workflow 节点包括：

```text
INITIAL      原始数据
LABEL        标注
CHECK        标注内审
CHECK        审核
ACCEPTANCE   验收
ACCEPTANCE   复核
QUAL         合格数据
```

没有：

```text
AI_LABEL
```

因此判断：

```text
当前任务没有 AI_LABEL 节点；
直接调用 push-item-ai-label-result 会触发“节点异常”并返回；
这条路径不适合当前 workflow。
```

---

## 50. aiLabelInsert 的写入格式

查看 `labelService.aiLabelInsert`：

```bash
sudo docker exec project-server-v6 sh -lc "sed -n '9035,9075p' /code/dist/main/main.js"
```

核心代码：

```js
async aiLabelInsert(taskId, data) {
    const labels = [];
    const preLabels = [];
    for (const item of data) {
        for (const i of item.labels) {
            const label = {
                taskId,
                itemId: new Types.ObjectId(item._id),
                source: LABEL_SOURCE.AI_ASSIST,
                status: LABEL_STATUS.LABELED,
                isUpdate: false,
                data: i
            };
            const preLabel = {
                ...label,
                source: PRE_LABEL_SOURCE.AI_ASSIST
            };
            labels.push(label);
            preLabels.push(preLabel);
        }
    }
    await this.labelModel.insertMany(labels);
    await this.preLabelModel.insertMany(preLabels);
}
```

这说明平台内部 AI label 写入本质上也是：

```text
同时写 labels 与 pre_labels；
每个 label 文档包含 taskId、itemId、source、status、isUpdate、data。
```

由于当前 workflow 不含 AI_LABEL 节点，后续不使用 `push-item-ai-label-result`，而是转为对 67b 进行最小范围的直接写入验证。

---

## 51. 现有 labels 文档结构与 pre_labels 集合确认

查看 67a 中已经存在的旧标签结构：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p')

sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin <<'MONGO'
const d = db.getSiblingDB("molar");

print("===== one existing label from 67a =====");
const doc = d.labels.findOne({
  itemId: ObjectId("69fed33d0fa06ee86bdea67a")
});
printjson(doc);

print("===== possible pre label collections =====");
printjson(d.getCollectionNames().filter(c => /pre.*label|prelabel|pre_labels/i.test(c)));
MONGO
```

代表性标签结构：

```js
{
  _id: ObjectId("..."),
  sys_status: 1,
  taskId: ObjectId("69fe8f840fa06ee86bde27a6"),
  itemId: ObjectId("69fed33d0fa06ee86bdea67a"),
  source: "MANUAL",
  status: "LABELED",
  isUpdate: false,
  data: {
    hash: "1_0",
    id: 1,
    label: "Regular_vehicle",
    drawType: "box3d",
    frameIndex: 0,
    keyframe: true,
    outside: true,
    points: [...],
    pointsInFrame: null,
    empty: false,
    source: "AI_SMART_TOOL",
    count: 1
  },
  createdAt: ISODate("2026-05-28T08:43:41.992Z"),
  updatedAt: ISODate("2026-05-28T08:43:41.992Z")
}
```

同时确认存在预标注集合：

```text
pre_labels
```

因此后续写入 plus 结果时，应同时写入：

```text
labels
pre_labels
```

---

## 52. 67b 直接写入 plus 结果并显示成功

先确认 67b 是空白：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p')

sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin <<'MONGO'
const d = db.getSiblingDB("molar");
const itemId = ObjectId("69fed33d0fa06ee86bdea67b");

print("labels:", d.labels.countDocuments({itemId}));
print("pre_labels:", d.pre_labels.countDocuments({itemId}));
MONGO
```

返回：

```text
labels: 0
pre_labels: 0
```

随后将 `/tmp/67b_plus_result.json` 中的 plus 结果写入 67b，并加上测试标记：

```text
__testTag = plus_67b_test_20260529
```

实际写入后页面刷新，右侧标签列表显示：

```text
标签数: 99 / 99
```

页面上可以看到 3D 框，右侧列表中显示 plus 类别名前缀，例如：

```text
human_...
```

Network 中也出现：

```text
find-labels 200
save-labels 200
```

说明：

```text
平台可以读取并显示直接写入 labels / pre_labels 的 plus box3d 结果。
```

---

## 53. 67b 写入结果统计与结论修正

查询 67b 当前 labels：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p')

sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin <<'MONGO'
const d = db.getSiblingDB("molar");
const itemId = ObjectId("69fed33d0fa06ee86bdea67b");
const tag = "plus_67b_test_20260529";

print("===== counts =====");
print("labels total:", d.labels.countDocuments({itemId}));
print("pre_labels total:", d.pre_labels.countDocuments({itemId}));
print("labels tagged:", d.labels.countDocuments({itemId, __testTag: tag}));
print("pre_labels tagged:", d.pre_labels.countDocuments({itemId, __testTag: tag}));

print("\n===== labels by class =====");
printjson(d.labels.aggregate([
  {$match:{itemId}},
  {$group:{_id:"$data.label", count:{$sum:1}}},
  {$sort:{count:-1}}
]).toArray());
MONGO
```

输出：

```text
labels total: 99
pre_labels total: 99
labels tagged: 99
pre_labels tagged: 99
```

类别统计：

```text
vehicle_car: 51
vehicle_motorcycle: 20
human_pedestrian_adult: 16
vehicle_tricycle: 7
vehicle_truck: 3
vehicle_bicycle: 1
movable_object_trafficcone: 1
```

再次刷新/保存页面后查询类别仍然保持 plus 原始类别名：

```text
vehicle_car
vehicle_motorcycle
human_pedestrian_adult
vehicle_tricycle
vehicle_truck
vehicle_bicycle
movable_object_trafficcone
```

这修正了前一天的判断：

```text
之前怀疑平台可能会把 plus 类别映射或兜底成 Vehicle；
今天通过 67b 直接写入验证，确认平台至少在 labels / pre_labels 读取、显示、保存层面可以保留 plus 下划线类别名；
当前不需要为了“能显示”而强制把 plus 类别映射成 Vehicle 或 Regular_vehicle。
```

更准确的结论是：

```text
plus 类别名本身可以被平台保存和显示；
当前主要未打通的是：
如何让 detection3d_plus 的实时推理结果自动进入 labels / pre_labels。
```

---

## 54. 当前总体状态更新

截至 2026-05-29，目前已经完成：

```text
1. detection3d_v5_plus 服务已启动并加载 OpenPCDet_ljl_plus 模型；
2. ai-nginx 已新增 /smart-tool/detection3d_plus；
3. 原 /smart-tool/detection3d 生产路径未被替换；
4. main-server 已对当前测试 taskId 做任务级分流；
5. main-server 直调 /api/smart-tool/detection3d 已验证会转发到 /smart-tool/detection3d_plus；
6. plus 接口可对真实平台 pcURL 返回 box3d 检测结果；
7. 当前 package 内已找到干净测试 item 67b；
8. 已确认页面当前点击动作读取的是 OSS ai-demo result.json，不是实时模型请求；
9. 已确认 project-server-v6 的 push-item-ai-label-result 需要 AI_LABEL 节点，当前 workflow 不适用；
10. 已确认 labels / pre_labels 的实际写入格式；
11. 已将 67b 的 plus 结果写入 labels / pre_labels；
12. 平台页面可以显示 67b 的 99 个 plus 3D 框；
13. 页面保存/刷新后，plus 类别名仍保持 vehicle_car、human_pedestrian_adult 等原始下划线类别。
```

当前关键结论：

```text
plus 模型结果、类别名、平台显示都已经分别验证可行；
剩余问题不是“类别名无法显示”，而是“如何自动写入平台 labels / pre_labels”。
```

当前 67b 状态：

```text
itemId = 69fed33d0fa06ee86bdea67b
labels = 99
pre_labels = 99
__testTag = plus_67b_test_20260529
```

建议暂时保留该 item 作为成功样例，用于后续测试：

```text
刷新显示
手动保存
导出
统计
类别兼容
```

---

## 55. 67b 测试数据回滚命令

如果需要将 67b 恢复为空白状态，可以使用测试标记一键删除：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p')

sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin <<'MONGO'
const d = db.getSiblingDB("molar");
const tag = "plus_67b_test_20260529";

printjson(d.labels.deleteMany({__testTag: tag}));
printjson(d.pre_labels.deleteMany({__testTag: tag}));
MONGO
```

也可以只查看当前测试数据：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p')

sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin <<'MONGO'
const d = db.getSiblingDB("molar");
const tag = "plus_67b_test_20260529";

print("labels:", d.labels.countDocuments({__testTag: tag}));
print("pre_labels:", d.pre_labels.countDocuments({__testTag: tag}));
MONGO
```

---

## 56. 后续工作建议：做一键写入脚本/接口

由于当前页面按钮并不稳定触发实时模型，而是读取 OSS result.json，因此建议后续不要继续优先从前端按钮入手，而是先做一个可控的命令行脚本或后端接口。

目标：

```text
输入：
taskId
itemId
pcURL
model
shape

流程：
1. 调用 main-server /api/smart-tool/detection3d，利用当前 taskId 分流进入 detection3d_plus；
2. 获取 plus 返回的 box3d 结果；
3. 转换成平台 labels / pre_labels 文档格式；
4. 写入指定 item；
5. 可选：写入前清理该 item 的旧 AI 测试结果；
6. 输出类别统计和写入数量。
```

建议先做命令行版本：

```text
write_plus_labels_to_item.py
```

后续再考虑服务化：

```text
/api/smart-tool/detection3d_plus_save
```

或在 `main-server` 中为当前测试任务增加可控的保存逻辑。

当前推荐策略：

```text
1. 保留原 /smart-tool/detection3d 生产路径；
2. 保留 /smart-tool/detection3d_plus 测试路径；
3. 不再直接修改平台类别模板；
4. 不使用 push-item-ai-label-result，因为当前 workflow 没有 AI_LABEL 节点；
5. 优先实现“调用 plus + 写 labels/pre_labels”的独立脚本；
6. 在更多空白 item 上测试显示、保存、导出；
7. 稳定后再讨论是否接入前端按钮。
```

---

## 57. 当前风险与注意事项

### 57.1 页面点击动作容易误判

当前页面点击后出现的：

```text
642568f685975ed9a12372f6-result.json
```

是 OSS 静态结果文件，不是实时模型请求。以后判断是否触发模型，应以以下日志为准：

```text
main-server:
POST /api/smart-tool/detection3d
ToolService.request||url=http://172.28.5.50:8000/smart-tool/detection3d_plus

ai-nginx:
POST /smart-tool/detection3d_plus HTTP/1.1" 200
```

### 57.2 access-token 不应写入文档

调用 main-server API 需要浏览器 `access-token`，但该 token 是敏感信息。

文档中统一使用：

```text
<ACCESS_TOKEN>
```

不要记录真实值。

### 57.3 直接写 MongoDB 只用于验证

今天对 67b 的直接写入是为了验证平台显示兼容性，不能作为最终生产方案。

最终方案应使用：

```text
明确脚本
后端接口
或平台已有保存接口
```

来保证字段完整、流程可追踪、可回滚。

### 57.4 67b 当前不是空白 item

67b 当前已经有 99 个测试标签：

```text
labels = 99
pre_labels = 99
```

如果需要继续拿空白 item 测试，应选择同 package 下其他 labelCount=0 的 item，例如：

```text
69fed33d0fa06ee86bdea67c
69fed33d0fa06ee86bdea67d
69fed33d0fa06ee86bdea67e
69fed33d0fa06ee86bdea67f
69fed33d0fa06ee86bdea682
69fed33d0fa06ee86bdea683
```

也可以先回滚 67b 的测试数据。

---

## 58. 2026-05-29：确认平台原生 save-labels 保存链路

在前面确认 `detection3d_plus` 推理结果可以写入平台并正常显示后，继续排查平台原生保存接口，目标是避免长期直接写 MongoDB，而是尽量复用平台已有的保存流程，为后续接回前端按钮做准备。

### 58.1 找到原生保存接口

在 `project-server-v6` 中搜索保存标签接口：

```bash
sudo docker exec project-server-v6 sh -lc 'grep -n "save-labels\|saveLabels\|saveLabel\|labelSave\|save-label" /code/dist/main/main.js | head -120'
```

关键输出：

```text
4528:    '/api/label/save-labels',
21677:    (0, common_1.Post)('save-labels'),
31202:        '/api/label/save-labels': { logBodyExcludeFields: ['data'] },
```

说明平台真正保存标注框的接口为：

```text
/api/label/save-labels
```

该接口位于：

```text
project-server-v6
```

而不是 `main-server`。

### 58.2 save-labels Controller 调用链

查看 Controller 附近代码：

```bash
sudo docker exec project-server-v6 sh -lc "sed -n '21595,21625p' /code/dist/main/main.js"
```

关键代码：

```js
async save(dto, user, req) {
    return this.labelManager.save(user, dto, req.workflow, req.task);
}
```

结合装饰器：

```js
Post('save-labels')
UseGuards(ItemMember)
```

说明调用链为：

```text
/api/label/save-labels
    -> LabelController.save(dto, user, req)
    -> LabelManager.save(user, dto, req.workflow, req.task)
```

### 58.3 LabelManager.save 调用 LabelService.save

继续查看：

```bash
sudo docker exec project-server-v6 sh -lc "sed -n '22230,22330p' /code/dist/main/main.js"
```

关键代码：

```js
async save(user, labelData, workflow, task) {
    const { itemId, workTime, data, nodeId } = labelData;
    const item = await this.itemService.findById(itemId);
    const res = await this.labelService.save(user, item, data);
    await this.invalidFrameService.filterDelLabel(data.delete, itemId);
    ...
    this.itemRecordService.insert({
        user,
        taskId: item.taskId,
        itemIds: [itemId],
        type: ITEM_OPERATE_TYPE.SAVE,
        nodeId,
        nodeType: findNodeById(workflow, nodeId).type,
        workTime
    })
    ...
    return res;
}
```

说明 `save-labels` 会：

```text
1. 根据 itemId 找 item；
2. 调用 LabelService.save(user, item, data)；
3. 处理删除无效帧关联；
4. 记录一次 SAVE 操作；
5. 刷新统计计数；
6. 返回保存结果。
```

因此后续如果通过 `/api/label/save-labels` 保存 plus 自动标注结果，比直接写 MongoDB 更接近平台正式流程。

### 58.4 LabelService.save 的 create / update / delete 结构

查看：

```bash
sudo docker exec project-server-v6 sh -lc "sed -n '8670,8745p' /code/dist/main/main.js"
```

关键代码：

```js
async save(user, item, data) {
    const result = {
        insertData: [],
        updateCount: 0,
        updateData: [],
        deleteCount: 0
    };
    if (!isEmpty(data.create)) {
        const res = await this.insertMany(user, item.taskId, item._id, data.create);
        result.insertData = res.map((el) => {
            return {
                _id: el._id,
                hash: el.data?.hash,
                dataId: el.data.id
            };
        });
    }
    if (!isEmpty(data.update)) {
        const updateRes = await this.updateMany(user, item, data.update);
        result.updateCount = updateRes.count;
        result.updateData = updateRes.data;
    }
    if (!isEmpty(data.delete)) {
        const count1 = await this.deleteMany(data.delete);
        const count2 = await this.deleteInvalidLabel(data.delete);
        result.deleteCount = count1 + count2;
    }
    ...
    return result;
}
```

由此确认，`save-labels` 的请求体中 `data` 应为：

```json
{
  "create": [],
  "update": [],
  "delete": []
}
```

新增自动标注框时应放入：

```text
data.create
```

更新已有框时放入：

```text
data.update
```

删除已有框时放入：

```text
data.delete
```

### 58.5 SaveLabelDataDto 参数结构

查看 DTO：

```bash
sudo docker exec project-server-v6 sh -lc 'L=$(grep -n "class SaveLabelDataDto" /code/dist/main/main.js | head -1 | cut -d: -f1); echo "LINE=$L"; if [ -n "$L" ]; then sed -n "$((L-20)),$((L+120))p" /code/dist/main/main.js; fi'
```

确认 `save-labels` 外层参数至少包含：

```text
nodeId
itemId
data
workTime
```

浏览器 Network 中实际 `save-labels` Payload 也显示为：

```json
{
  "nodeId": "69fe8f840fa06ee86bde27b8",
  "itemId": "69fed33d0fa06ee86bdea67b",
  "taskId": "69fe8f840fa06ee86bde27a6",
  "workTime": 309,
  "data": {
    "create": [],
    "update": [],
    "delete": []
  }
}
```

其中 `taskId` 在 DTO 中不是强校验字段，但前端实际会携带。

### 58.6 insertMany 确认 create 只需传标签 data 本体

继续查看 `insertMany`：

```bash
sudo docker exec project-server-v6 sh -lc "sed -n '9108,9135p' /code/dist/main/main.js"
```

关键代码：

```js
async insertMany(user, taskId, itemId, dataList, source = LABEL_SOURCE.MANUAL) {
    const labels = [];
    for (const data of dataList) {
        const label = {
            taskId,
            itemId,
            source,
            status: LABEL_STATUS.LABELED,
            isUpdate: false,
            data
        };
        labels.push(label);
    }
    return this.labelModel.insertMany(labels);
}
```

由此确认：

```text
data.create 中只需要传标签 data 本体；
后端会自动补 taskId、itemId、source、status、isUpdate；
不需要在 create 中传完整 MongoDB label 文档。
```

也就是说，plus 检测框可以转换成如下结构后放入 `data.create`：

```json
{
  "hash": "plus_save_1_0",
  "id": 1,
  "label": "vehicle_car",
  "drawType": "box3d",
  "frameIndex": 0,
  "keyframe": true,
  "outside": true,
  "points": [ ... ],
  "pointsInFrame": null,
  "empty": false,
  "source": "AI_SMART_TOOL",
  "count": 1,
  "score": 0.8
}
```

### 58.7 重要差异：save-labels 只写 labels，不写 pre_labels

`insertMany` 只调用：

```js
this.labelModel.insertMany(labels)
```

没有写入：

```text
pre_labels
```

这与前面 `aiLabelInsert()` 不同。`aiLabelInsert()` 会同时写 `labels` 和 `pre_labels`，但它要求当前 workflow 存在 `AI_LABEL` 节点，当前任务不满足。

当前判断：

```text
save-labels 是正式人工保存接口；
写 labels 即可让平台前端通过 find-labels 加载；
pre_labels 更偏向导入/AI预标注缓存，不是当前原生保存流程必需项。
```

---

## 59. 67c 作为原生 save-labels 测试 item

由于 67b 已经被直接写入 99 个 plus 测试框，后续使用当前 package 内另一个空白 item `67c` 测试平台原生 `save-labels`。

### 59.1 67c 当前为空白

执行：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p')

sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin <<'MONGO'
const d = db.getSiblingDB("molar");
const itemId = ObjectId("69fed33d0fa06ee86bdea67c");

print("labels before:", d.labels.countDocuments({itemId}));
print("pre_labels before:", d.pre_labels.countDocuments({itemId}));

printjson(d.labels.deleteMany({itemId}));
printjson(d.pre_labels.deleteMany({itemId}));

print("labels after:", d.labels.countDocuments({itemId}));
print("pre_labels after:", d.pre_labels.countDocuments({itemId}));
MONGO
```

输出：

```text
labels before: 0
pre_labels before: 0
deletedCount: 0
deletedCount: 0
labels after: 0
pre_labels after: 0
```

说明：

```text
67c 是当前 package 内干净的空白 item；
可以用于测试 save-labels 的 create 流程。
```

### 59.2 67c 点云地址

查询 67c 的点云地址：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p')

sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin <<'MONGO'
const d = db.getSiblingDB("molar");
const itemId = ObjectId("69fed33d0fa06ee86bdea67c");
const info = d.item_infos.findOne({itemId});

printjson({
  itemId,
  firstPcdUrl: info?.info?.pcdUrl?.[0],
  pcdUrlCount: info?.info?.pcdUrl?.length
});
MONGO
```

输出：

```text
itemId: ObjectId("69fed33d0fa06ee86bdea67c")
firstPcdUrl: /files/import_all/20260312_test1_ds6_1/2026-03-12-18-54-16_uid1/merged/1773312866512658.pcd
pcdUrlCount: 59
```

因此 67c 的完整 `pcURL` 为：

```text
http://172.23.131.39:8080/files/import_all/20260312_test1_ds6_1/2026-03-12-18-54-16_uid1/merged/1773312866512658.pcd
```

67c 页面地址为：

```text
http://172.23.131.39:8080/items?version=latest&taskId=69fe8f840fa06ee86bde27a6&vm=batch&batchId=69fed33d0fa06ee86bdea679&dm=all&role=69fe8f840fa06ee86bde27b8&packageId=69fed33d0fa06ee86bdea679&itemId=69fed33d0fa06ee86bdea67c&nodeId=69fe8f840fa06ee86bde27b8&selectIds=69fed33d0fa06ee86bdea67c
```

该页面可用于从浏览器 Network 请求头中获取当前登录用户的 `access-token`。

---

## 60. 下一步计划：用 save-labels 保存 67c 的 plus 前 5 个框

下一步要做的验证是：

```text
不直接写 MongoDB；
通过平台原生 /api/label/save-labels；
将 detection3d_plus 对 67c 点云返回的前 5 个 box3d 写入 labels；
刷新 67c 页面确认显示。
```

### 60.1 调用 plus 推理生成 67c 结果

命令模板如下，`<ACCESS_TOKEN>` 不写入文档：

```bash
ACCESS_TOKEN='<ACCESS_TOKEN>'
PCURL='http://172.23.131.39:8080/files/import_all/20260312_test1_ds6_1/2026-03-12-18-54-16_uid1/merged/1773312866512658.pcd'

curl -sS -X POST http://127.0.0.1:3004/api/smart-tool/detection3d   -H "Content-Type: application/json"   -H "access-token: $ACCESS_TOKEN"   -d "{\"taskId\":\"69fe8f840fa06ee86bde27a6\",\"params\":{\"pcURL\":\"$PCURL\",\"shape\":4,\"model\":\"model\"}}"   -o /tmp/67c_plus_result.json   && echo "saved /tmp/67c_plus_result.json"   && head -c 500 /tmp/67c_plus_result.json && echo
```

### 60.2 生成 save-labels 请求体

只取前 5 个 box3d，用于最小验证：

```bash
python3 - <<'PY'
import json

src = "/tmp/67c_plus_result.json"
dst = "/tmp/67c_save_labels_payload.json"

raw = json.load(open(src))
boxes = raw.get("data", [])
if len(boxes) == 1 and isinstance(boxes[0], list):
    boxes = boxes[0]

boxes = [
    x for x in boxes
    if x.get("drawType") == "box3d" and isinstance(x.get("points"), list)
]
boxes = sorted(boxes, key=lambda x: x.get("score", 0), reverse=True)[:5]

create = []
for idx, x in enumerate(boxes, 1):
    create.append({
        "hash": f"plus_save_{idx}_0",
        "id": idx,
        "label": x.get("label"),
        "drawType": "box3d",
        "frameIndex": 0,
        "keyframe": True,
        "outside": True,
        "points": x.get("points"),
        "pointsInFrame": None,
        "empty": False,
        "source": "AI_SMART_TOOL",
        "count": 1,
        "score": x.get("score")
    })

payload = {
    "nodeId": "69fe8f840fa06ee86bde27b8",
    "itemId": "69fed33d0fa06ee86bdea67c",
    "taskId": "69fe8f840fa06ee86bde27a6",
    "workTime": 1,
    "data": {
        "create": create,
        "update": [],
        "delete": []
    }
}

json.dump(payload, open(dst, "w"), ensure_ascii=False)
print("create count:", len(create))
print("saved:", dst)
print(json.dumps(payload, ensure_ascii=False)[:1000])
PY
```

### 60.3 调用原生 save-labels 保存

命令模板：

```bash
ACCESS_TOKEN='<ACCESS_TOKEN>'

curl -sS -X POST http://172.23.131.39:8080/api/label/save-labels   -H "Content-Type: application/json"   -H "access-token: $ACCESS_TOKEN"   --data-binary @/tmp/67c_save_labels_payload.json   -o /tmp/67c_save_labels_response.json   && echo "saved response"   && cat /tmp/67c_save_labels_response.json && echo
```

预期结果：

```text
接口返回 code=200；
或 data 中包含 insertData；
MongoDB 中 67c 的 labels 变为 5；
pre_labels 仍为 0 是正常现象。
```

### 60.4 验证写入结果

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p')

sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin <<'MONGO'
const d = db.getSiblingDB("molar");
const itemId = ObjectId("69fed33d0fa06ee86bdea67c");

print("labels:", d.labels.countDocuments({itemId}));
print("pre_labels:", d.pre_labels.countDocuments({itemId}));

printjson(d.labels.aggregate([
  {$match:{itemId}},
  {$group:{_id:"$data.label", count:{$sum:1}}},
  {$sort:{count:-1}}
]).toArray());

d.labels.find(
  {itemId},
  {_id:1, source:1, status:1, "data.label":1, "data.score":1, "data.source":1}
).limit(10).forEach(doc => printjson(doc));
MONGO
```

### 60.5 页面验证

打开 67c 页面：

```text
http://172.23.131.39:8080/items?version=latest&taskId=69fe8f840fa06ee86bde27a6&vm=batch&batchId=69fed33d0fa06ee86bdea679&dm=all&role=69fe8f840fa06ee86bde27b8&packageId=69fed33d0fa06ee86bdea679&itemId=69fed33d0fa06ee86bdea67c&nodeId=69fe8f840fa06ee86bde27b8&selectIds=69fed33d0fa06ee86bdea67c
```

强制刷新：

```text
Ctrl + Shift + R
```

预期：

```text
页面显示 5 个 plus 类别的 3D 框。
```

如果成功，则证明完整闭环为：

```text
detection3d_plus 推理
    -> 组装 save-labels data.create
    -> 调用 /api/label/save-labels
    -> labels 正式保存
    -> find-labels 加载并显示
```

---

## 61. 当前阶段性结论

截至目前，前端按钮最终仍然应该保留给标注员使用，但不应一开始就直接改前端按钮。

更稳妥的工程路线是：

```text
第一步：证明 plus 模型结果能显示；
    已完成，67b 已验证。

第二步：证明 plus 模型结果能通过平台原生 save-labels 保存；
    正在进行，当前目标为 67c。

第三步：将“推理 + 组装 create + save-labels”封装成一个后端接口；
    未开始。

第四步：将前端按钮接入该后端接口；
    未开始。
```

当前不建议放弃前端按钮。最终用户流程必须是：

```text
标注人员打开平台
    -> 点击自动标注 / 智能标注按钮
    -> 平台自动调用 plus 模型
    -> 自动保存/展示框
    -> 标注人员人工修正
    -> 保存/提交
```

当前暂时通过命令行验证后端闭环，只是为了降低风险、排清楚每一层接口，不代表最终方案绕开前端。


---

## 62. 2026-05-29 后续：save-labels 真实入口修正

在 67c 空白 item 上继续验证“plus 推理结果能否通过平台原生保存接口写入正式 labels”。

### 62.1 问题现象

最初按如下路径调用保存接口：

```bash
curl -sS -X POST http://172.23.131.39:8080/api/label/save-labels \
  -H "Content-Type: application/json" \
  -H "access-token: $ACCESS_TOKEN" \
  --data-binary @/tmp/67c_save_labels_payload.json
```

返回：

```json
{
  "status": 404,
  "path": "/api/label/save-labels",
  "exception": "Cannot POST /api/label/save-labels",
  "message": "Cannot POST /api/label/save-labels"
}
```

随后改用 `main-server:3004` 调用：

```bash
curl -sS -X POST http://127.0.0.1:3004/api/label/save-labels ...
```

同样返回 404。

结论：

```text
/api/label/save-labels 不属于 main-server；
也不是 8080 当前直接暴露的旧路径；
需要定位真实保存服务。
```

### 62.2 错误方向：main-server 的 /api/item/work/save

在 `main-server` 中搜索保存相关路由：

```bash
sudo docker exec main-server sh -lc 'grep -n "(0, common_1.Post).*save\|common_1.Post.*label\|Post).*label\|Post).*save" /code/dist/main/main.js | head -120'
```

发现：

```text
Post('/work/save')
Post('/work/label')
```

尝试调用：

```bash
curl -sS -i -X POST http://127.0.0.1:3004/api/item/work/save \
  -H "Content-Type: application/json" \
  -H "access-token: $ACCESS_TOKEN" \
  --data-binary @/tmp/67c_save_labels_payload.json
```

返回：

```json
{"message":"条目不存在","code":1000}
```

进一步查看代码：

```js
async workSave(dto, req, user) {
    const { itemId, data, workTime } = dto;
    const batchFlow = req.req_batch_flow;
    const task = req.req_task;
    return this.workService.save(itemId, user, batchFlow, workTime, data, task);
}
```

说明：

```text
/api/item/work/save 是 main-server 的旧 ItemController 工作流接口；
虽然接口存在，但它无法正确处理当前 v2 页面 item 上下文；
不是本次平台 v2 前端使用的 save-labels 入口。
```

---

## 63. 定位 project-server-v6 为 save-labels 所属服务

根据上午记录，重新检查 `project-server-v6`：

```bash
sudo docker exec project-server-v6 sh -lc 'grep -n "save-labels\|saveLabels\|saveLabel\|labelSave\|save-label" /code/dist/main/main.js | head -120'
```

关键输出：

```text
4528:    '/api/label/save-labels',
21677:    (0, common_1.Post)('save-labels'),
31202:        '/api/label/save-labels': { logBodyExcludeFields: ['data'] },
```

查看服务启动配置：

```js
const GLOBAL_PREFIX = '/api';
...
app.setGlobalPrefix(GLOBAL_PREFIX);
```

确认：

```text
project-server-v6 中存在 LabelController.save；
真实路径为 /api/label/save-labels；
该服务监听容器内 3000 端口。
```

查看容器网络与端口：

```bash
sudo docker inspect -f '{{range $name,$conf := .NetworkSettings.Networks}}{{println $name $conf.IPAddress}}{{end}}' project-server-v6
sudo docker port project-server-v6
sudo docker exec project-server-v6 sh -lc 'echo PORT=$PORT; ps -ef | grep node | grep -v grep'
```

输出：

```text
mooredata_my-network 172.28.5.30

3000/tcp -> 0.0.0.0:3000
3000/tcp -> [::]:3000

node /code/dist/main/main.js
```

结论：

```text
平台正式标注保存接口属于 project-server-v6；
宿主机可通过 http://127.0.0.1:3000/api/label/save-labels 调用；
8080 前端也可通过 /api/v2/label/save-labels 代理到保存接口。
```

---

## 64. 67c：通过 project-server-v6 原生接口保存 5 个 plus 框

### 64.1 生成可见框 payload

为了验证显示，重新生成保存 payload，并将 `outside` 设置为 `false`：

```bash
python3 -c 'import json; src="/tmp/67c_plus_result.json"; dst="/tmp/67c_save_labels_payload_visible.json"; raw=json.load(open(src)); boxes=raw.get("data", []); boxes=boxes[0] if len(boxes)==1 and isinstance(boxes[0], list) else boxes; boxes=sorted([x for x in boxes if x.get("drawType")=="box3d" and isinstance(x.get("points"), list)], key=lambda x:x.get("score",0), reverse=True)[:5]; create=[{"hash":f"plus_save_visible_{i}_0","id":i,"label":x.get("label"),"drawType":"box3d","frameIndex":0,"keyframe":True,"outside":False,"points":x.get("points"),"pointsInFrame":None,"empty":False,"source":"AI_SMART_TOOL","count":1,"score":x.get("score")} for i,x in enumerate(boxes,1)]; payload={"nodeId":"69fe8f840fa06ee86bde27b8","itemId":"69fed33d0fa06ee86bdea67c","taskId":"69fe8f840fa06ee86bde27a6","workTime":1,"data":{"create":create,"update":[],"delete":[]}}; json.dump(payload, open(dst,"w"), ensure_ascii=False); print("create count:",len(create)); print("saved:",dst); print(json.dumps(payload,ensure_ascii=False)[:1000])'
```

输出：

```text
create count: 5
saved: /tmp/67c_save_labels_payload_visible.json
```

### 64.2 调用 project-server-v6 保存

调用真实保存接口：

```bash
curl -sS -i -X POST http://127.0.0.1:3000/api/label/save-labels \
  -H "Content-Type: application/json" \
  -H "access-token: $ACCESS_TOKEN" \
  --data-binary @/tmp/67c_save_labels_payload_visible.json \
  -o /tmp/67c_project_save_labels_response.txt \
  && echo "saved response" \
  && cat /tmp/67c_project_save_labels_response.txt && echo
```

返回：

```text
HTTP/1.1 200 OK
```

响应包含：

```json
{
  "code": 200,
  "data": {
    "insertData": [
      {"_id":"6a192ddc26ccd0bb7792caf3","hash":"plus_save_visible_1_0","dataId":1},
      {"_id":"6a192ddc26ccd0bb7792caf4","hash":"plus_save_visible_2_0","dataId":2},
      {"_id":"6a192ddc26ccd0bb7792caf5","hash":"plus_save_visible_3_0","dataId":3},
      {"_id":"6a192ddc26ccd0bb7792caf6","hash":"plus_save_visible_4_0","dataId":4},
      {"_id":"6a192ddc26ccd0bb7792caf7","hash":"plus_save_visible_5_0","dataId":5}
    ],
    "updateCount": 0,
    "updateData": [],
    "deleteCount": 0
  }
}
```

### 64.3 MongoDB 验证

查询：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p'); sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin --eval 'const d=db.getSiblingDB("molar"); const itemId=ObjectId("69fed33d0fa06ee86bdea67c"); print("labels:", d.labels.countDocuments({itemId})); print("pre_labels:", d.pre_labels.countDocuments({itemId})); printjson(d.labels.aggregate([{$match:{itemId}},{$group:{_id:"$data.label", count:{$sum:1}}},{$sort:{count:-1}}]).toArray()); d.labels.find({itemId},{_id:1, source:1, status:1, "data.label":1, "data.score":1, "data.outside":1, "data.source":1}).limit(10).forEach(doc => printjson(doc));'
```

输出：

```text
labels: 5
pre_labels: 0
[
  {
    _id: 'vehicle_car',
    count: 5
  }
]
```

单条 label 示例：

```json
{
  "_id": ObjectId("6a192ddc26ccd0bb7792caf3"),
  "source": "MANUAL",
  "status": "LABELED",
  "data": {
    "label": "vehicle_car",
    "outside": false,
    "source": "AI_SMART_TOOL",
    "score": 0.9141050577163696
  }
}
```

说明：

```text
通过 project-server-v6 的原生 save-labels 接口，plus 框已经成功写入正式 labels；
外层 source=MANUAL 是平台正式保存后的来源状态；
data.source=AI_SMART_TOOL 保留了自动标注来源。
```

### 64.4 前端验证

刷新 67c 页面后，能看到 5 个 plus 类别框。

结论：

```text
plus 推理结果 -> project-server-v6 原生 save-labels -> 正式 labels -> 前端显示
该闭环已成功打通。
```

---

## 65. 验证 8080 v2 保存入口

后续又验证了前端同源保存入口：

```bash
curl -sS -i -X POST http://172.23.131.39:8080/api/v2/label/save-labels \
  -H "Content-Type: application/json" \
  -H "access-token: $ACCESS_TOKEN" \
  --data-binary @/tmp/67c_save_labels_payload_visible.json \
  -o /tmp/8080_v2_save_labels_test.txt \
  && echo "saved response" \
  && cat /tmp/8080_v2_save_labels_test.txt && echo
```

返回：

```text
HTTP/1.1 200 OK
Server: nginx/1.16.1
```

响应中同样包含：

```text
code: 200
insertData: 5 条
```

查询 MongoDB：

```text
labels: 5
pre_labels: 0
vehicle_car: 5
```

结论：

```text
浏览器前端可以通过同源路径 /api/v2/label/save-labels 完成正式保存；
不需要浏览器跨端口访问 3000；
这为后续前端按钮接入提供了稳定入口。
```

---

## 66. 供应商原 detection3d 线仍然正常

为了确认测试改动没有破坏供应商原接口，直接测试原路径：

```bash
curl -sS -X POST http://127.0.0.1:8000/smart-tool/detection3d \
  -H "Content-Type: application/json" \
  -d "{\"pcURL\":\"$PCURL\",\"shape\":4,\"model\":\"model\"}" \
  -o /tmp/vendor_direct_check.json \
  && python3 -c 'import json; from collections import Counter; data=json.load(open("/tmp/vendor_direct_check.json")); items=data.get("data", []); items=items[0] if len(items)==1 and isinstance(items[0], list) else items; print("code:",data.get("code")); print("num_boxes:",len(items)); print("labels:",Counter(x.get("label") for x in items))'
```

输出：

```text
code: 200
num_boxes: 126
labels: Counter({
  'Regular_vehicle': 77,
  'Pedestrian': 23,
  'Bus': 7,
  'Bollard': 6,
  'Sign': 3,
  'Construction_barrel': 3,
  'Stop_sign': 2,
  'Motorcycle': 2,
  'Truck_cab': 1,
  'Large_vehicle': 1,
  'Wheeled_device': 1
})
```

结论：

```text
供应商原 /smart-tool/detection3d 接口仍然正常；
当前 plus 接入没有全局替换或破坏生产 detection3d。
```

---

## 67. AI 标注按钮实际逻辑排查：发现前端硬编码 OSS demo result

### 67.1 点击按钮后无后端日志

初始点击页面“AI标注”按钮时，`ai-nginx` 和 `main-server` 中没有新的 detection3d 请求日志。

在浏览器 Network 中发现新增请求为：

```text
https://molar-publish.oss-cn-hangzhou.aliyuncs.com/ai-demo/642568f685975ed9a12372f6-result.json
```

这说明按钮当时并没有实时调用：

```text
/api/smart-tool/detection3d
```

而是读取 OSS 上的历史 demo result.json。

### 67.2 task-info 中 aiPower 曾被改为 detection3d_plus，但前端不识别

MongoDB 中当前任务配置一度显示：

```json
"setting": {
  "aiPower": "detection3d_plus"
}
```

但是前端按钮依旧读取 OSS demo result，说明：

```text
前端 getAIPower 不一定识别自定义的 detection3d_plus；
前端原本识别的是 detection3d 等内置 aiPower；
main-server 的 taskId 分流才是安全接 plus 的位置。
```

### 67.3 使用 DevTools XHR/fetch 断点定位

在 DevTools 中添加 XHR/fetch Breakpoint：

```text
URL contains: result.json
```

刷新或点击 AI 标注按钮后，断点停在：

```text
index.vue_vue_type_style_index_0_lang-5fbe0bdf.js
```

Call Stack 中关键函数为：

```text
Ue
```

在源码附近发现硬编码逻辑：

```js
const Se = "https://molar-publish.oss-cn-hangzhou.aliyuncs.com/ai-demo/642568f685975ed9a12372f6-result.json",
g = (await qs.get(Se,{responseType:"json"})).data;
```

同时附近存在：

```js
const me = P(()=>{var Se; return ((Se=s.item)==null?void 0:Se._id)==="642568f685975ed9a12372f6"})
```

说明：

```text
前端本来有 demo item 判断 me；
但 OSS demo result 的读取被无条件放在了 Ue 函数开头；
导致非 demo item 点击 AI 标注时也读取固定 demo result。
```

---

## 68. 前端按钮补丁：取消非 demo item 的 OSS result 读取

### 68.1 找到前端真实文件

真实前端入口为：

```text
http://172.23.131.39:8080/assets/index-5093d980.js
```

AI 标注按钮所在 chunk 文件为：

```text
/code/molar-label-system-fe-v2/assets/index.vue_vue_type_style_index_0_lang-5fbe0bdf.js
```

查找命令：

```bash
sudo docker exec molar-label-system-fe-v2 sh -lc 'find / -name "index.vue_vue_type_style_index_0_lang-5fbe0bdf.js" 2>/dev/null'
```

输出：

```text
/code/molar-label-system-fe-v2/assets/index.vue_vue_type_style_index_0_lang-5fbe0bdf.js
```

### 68.2 备份前端 JS

```bash
sudo docker exec molar-label-system-fe-v2 sh -lc 'cp /code/molar-label-system-fe-v2/assets/index.vue_vue_type_style_index_0_lang-5fbe0bdf.js /code/molar-label-system-fe-v2/assets/index.vue_vue_type_style_index_0_lang-5fbe0bdf.js.bak_ai_demo_oss_20260529'
```

确认：

```bash
sudo docker exec molar-label-system-fe-v2 sh -lc 'ls -lh /code/molar-label-system-fe-v2/assets/index.vue_vue_type_style_index_0_lang-5fbe0bdf.js*'
```

输出：

```text
index.vue_vue_type_style_index_0_lang-5fbe0bdf.js
index.vue_vue_type_style_index_0_lang-5fbe0bdf.js.bak_ai_demo_oss_20260529
```

### 68.3 拷贝到宿主机分析

由于前端容器无 `node`，因此将 JS 拷贝到宿主机：

```bash
sudo docker cp molar-label-system-fe-v2:/code/molar-label-system-fe-v2/assets/index.vue_vue_type_style_index_0_lang-5fbe0bdf.js /tmp/fe_ai_button_chunk.js
```

打印硬编码附近代码：

```bash
python3 -c 'p="/tmp/fe_ai_button_chunk.js"; s=open(p,encoding="utf-8",errors="ignore").read(); key="molar-publish"; i=s.find(key); print("index=",i); print(s[max(0,i-2500):i+5000])'
```

确认硬编码位置：

```text
index=43982
```

### 68.4 修改逻辑

将原来的无条件读取：

```js
const Se="https://molar-publish.oss-cn-hangzhou.aliyuncs.com/ai-demo/642568f685975ed9a12372f6-result.json",
g=(await qs.get(Se,{responseType:"json"})).data;
if(s.getCheckLock()
```

改为只在 demo item 时读取：

```js
let g=null;
if(me.value){
    const Se="https://molar-publish.oss-cn-hangzhou.aliyuncs.com/ai-demo/642568f685975ed9a12372f6-result.json";
    g=(await qs.get(Se,{responseType:"json"})).data
}
if(s.getCheckLock()
```

执行补丁：

```bash
python3 -c 'p="/tmp/fe_ai_button_chunk.js"; s=open(p,encoding="utf-8",errors="ignore").read(); old="const Se=\"https://molar-publish.oss-cn-hangzhou.aliyuncs.com/ai-demo/642568f685975ed9a12372f6-result.json\",g=(await qs.get(Se,{responseType:\"json\"})).data;if(s.getCheckLock()"; new="let g=null;if(me.value){const Se=\"https://molar-publish.oss-cn-hangzhou.aliyuncs.com/ai-demo/642568f685975ed9a12372f6-result.json\";g=(await qs.get(Se,{responseType:\"json\"})).data}if(s.getCheckLock()"; assert old in s, "target string not found"; s=s.replace(old,new,1); open("/tmp/fe_ai_button_chunk.patched.js","w",encoding="utf-8").write(s); print("patched")'
```

拷回容器：

```bash
sudo docker cp /tmp/fe_ai_button_chunk.patched.js molar-label-system-fe-v2:/code/molar-label-system-fe-v2/assets/index.vue_vue_type_style_index_0_lang-5fbe0bdf.js
```

### 68.5 任务 aiPower 改回 detection3d

因为前端识别的是内置 AI 工具名 `detection3d`，真正的 plus 分流在 main-server 中按 taskId 完成，所以将当前测试任务 `aiPower` 改回：

```text
detection3d
```

命令：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p'); sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin --eval 'const d=db.getSiblingDB("molar"); const taskId=ObjectId("69fe8f840fa06ee86bde27a6"); print("before:"); printjson(d.tasks.findOne({_id:taskId},{_id:1,name:1,"setting.aiPower":1})); printjson(d.tasks.updateOne({_id:taskId},{$set:{"setting.aiPower":"detection3d"}})); print("after:"); printjson(d.tasks.findOne({_id:taskId},{_id:1,name:1,"setting.aiPower":1}));'
```

设计逻辑：

```text
前端按钮看到 aiPower=detection3d；
前端调用 /api/smart-tool/detection3d；
main-server 检查 taskId；
当前测试 taskId 分流到 /smart-tool/detection3d_plus；
其他任务仍然走 /smart-tool/detection3d。
```

---

## 69. 验证前端同源 smart-tool 入口

测试 8080 同源入口：

```bash
PCURL='http://172.23.131.39:8080/files/import_all/20260312_test1_ds6_1/2026-03-12-18-54-16_uid1/merged/1773312866512658.pcd'; curl -sS -X POST http://172.23.131.39:8080/api/smart-tool/detection3d -H "Content-Type: application/json" -H "access-token: $ACCESS_TOKEN" -d "{\"taskId\":\"69fe8f840fa06ee86bde27a6\",\"params\":{\"pcURL\":\"$PCURL\",\"shape\":4,\"model\":\"model\"}}" -o /tmp/8080_smart_tool_check.json && python3 -c 'import json; from collections import Counter; data=json.load(open("/tmp/8080_smart_tool_check.json")); items=data.get("data", []); items=items[0] if len(items)==1 and isinstance(items[0], list) else items; print("code:",data.get("code")); print("num_boxes:",len(items)); print("labels:",Counter(x.get("label") for x in items)); print(open("/tmp/8080_smart_tool_check.json").read()[:300])'
```

输出：

```text
code: 200
num_boxes: 143
labels: Counter({
  'vehicle_car': 58,
  'vehicle_motorcycle': 36,
  'human_pedestrian_adult': 20,
  'vehicle_tricycle': 13,
  'vehicle_truck': 4,
  'group_human_pedestrian': 3,
  'group_vehicle_bicycle': 3,
  'vehicle_bicycle': 2,
  'vehicle_construction': 1,
  'vehicle_bus_rigid': 1,
  'bicycle': 1,
  'movable_object_trafficcone': 1
})
```

结论：

```text
前端同源 /api/smart-tool/detection3d 已经能触发 main-server；
main-server 对当前测试 taskId 已成功分流到 detection3d_plus；
返回的是 plus 类别体系。
```

---

## 70. AI 标注按钮成功走 plus 线

### 70.1 测试前准备

清空 67c 旧 labels：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p'); sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin --eval 'const d=db.getSiblingDB("molar"); const itemId=ObjectId("69fed33d0fa06ee86bdea67c"); print("labels before:", d.labels.countDocuments({itemId})); print("pre_labels before:", d.pre_labels.countDocuments({itemId})); printjson(d.labels.deleteMany({itemId})); printjson(d.pre_labels.deleteMany({itemId})); print("labels after:", d.labels.countDocuments({itemId})); print("pre_labels after:", d.pre_labels.countDocuments({itemId}));'
```

页面强制刷新：

```text
Ctrl + Shift + R
```

### 70.2 点击 AI 标注按钮后的日志

`main-server` 日志：

```text
requestInfo={"method":"POST","path":"/api/smart-tool/detection3d",...,"user-agent":"Mozilla/5.0 ..."}
module=ToolService.request||url=http://172.28.5.50:8000/smart-tool/detection3d_plus
module=ToolService.request||data={
  "pcURL":"http://172.28.5.5:8080/files/import_all/20260312_test1_ds6_1/2026-03-12-18-54-16_uid1/merged/1773312869512658.pcd",
  "shape":5
}
module=ToolService.request||code=200||result=200
responseInfo={"method":"POST","path":"/api/smart-tool/detection3d","responseTime":"7411ms",...}
```

`ai-nginx` 日志：

```text
172.28.5.4 - - [29/May/2026:07:27:48 +0000] "POST /smart-tool/detection3d_plus HTTP/1.1" 200 37593 "-" "axios/0.27.2" "-"
```

说明：

```text
浏览器点击 AI 标注按钮后，已经不再读取 OSS demo result；
前端调用 /api/smart-tool/detection3d；
main-server 根据当前 taskId 分流；
实际进入 /smart-tool/detection3d_plus；
plus 模型返回成功。
```

### 70.3 前端临时显示

点击 AI 标注后，页面右侧出现大量自动标注对象，右下角显示：

```text
标签数 151/151
```

此时查询 MongoDB 一度仍为：

```text
labels: 0
pre_labels: 0
```

解释：

```text
AI 标注按钮先将结果加载到前端临时状态；
正式落库需要点击页面右上角“暂存”。
```

---

## 71. 点击“暂存”后正式保存 151 个 plus 标签

点击页面右上角“暂存”后，再次查询 MongoDB：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p'); sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin --eval 'const d=db.getSiblingDB("molar"); const itemId=ObjectId("69fed33d0fa06ee86bdea67c"); print("labels:", d.labels.countDocuments({itemId})); print("pre_labels:", d.pre_labels.countDocuments({itemId})); printjson(d.labels.aggregate([{$match:{itemId}},{$group:{_id:"$data.label", count:{$sum:1}}},{$sort:{count:-1}}]).toArray());'
```

输出：

```text
labels: 151
pre_labels: 0
[
  {
    _id: 'vehicle_car',
    count: 56
  },
  {
    _id: 'human_pedestrian_adult',
    count: 37
  },
  {
    _id: 'vehicle_motorcycle',
    count: 32
  },
  {
    _id: 'vehicle_tricycle',
    count: 10
  },
  {
    _id: 'group_vehicle_bicycle',
    count: 8
  },
  {
    _id: 'vehicle_truck',
    count: 3
  },
  {
    _id: 'vehicle_bus_rigid',
    count: 2
  },
  {
    _id: 'group_human_pedestrian',
    count: 2
  },
  {
    _id: 'vehicle_construction',
    count: 1
  }
]
```

结论：

```text
AI 标注按钮 -> detection3d_plus -> 前端临时显示 -> 点击暂存 -> 正式 labels
完整闭环已成功。
```

---

## 72. 当前最终状态

截至 2026-05-29，本轮接入已验证成功：

```text
1. detection3d_v5_plus 已加载 OpenPCDet_ljl_plus 26 类 VoxelNeXt 模型；
2. ai-nginx 已新增 /smart-tool/detection3d_plus 测试路径；
3. 原供应商 /smart-tool/detection3d 仍然正常；
4. main-server 已做当前测试 taskId 级别分流；
5. 前端 AI 标注按钮已取消非 demo item 的 OSS result.json 读取；
6. 当前测试任务 aiPower 使用 detection3d；
7. 浏览器 AI 标注按钮会调用 /api/smart-tool/detection3d；
8. main-server 将当前测试任务分流到 /smart-tool/detection3d_plus；
9. 前端可显示 plus 自动标注结果；
10. 点击“暂存”后，plus 结果正式写入 labels；
11. 67c 验证样本最终 labels=151，pre_labels=0；
12. plus 类别名可以被平台保存和前端显示。
```

当前实际成功链路：

```text
浏览器 AI标注按钮
    -> 8080 /api/smart-tool/detection3d
    -> main-server
    -> taskId == 69fe8f840fa06ee86bde27a6
    -> ai-nginx /smart-tool/detection3d_plus
    -> detection3d_v5_plus
    -> OpenPCDet_ljl_plus 推理
    -> 前端加载 151 个候选框
    -> 点击“暂存”
    -> 8080 /api/v2/label/save-labels
    -> project-server-v6
    -> MongoDB molar.labels
    -> 前端正式显示
```

原供应商线验证：

```text
127.0.0.1:8000/smart-tool/detection3d
    -> code=200
    -> 返回 Regular_vehicle / Pedestrian / Bus 等供应商类别
```

说明：

```text
当前操作未破坏供应商原模型服务；
只有当前测试任务通过 main-server 分流走 plus。
```

---

## 73. 当前仍需优化的问题：结果数量与过滤策略

本次 67c 自动标注并暂存后共保存：

```text
labels = 151
```

这说明按钮链路已通，但直接保存全部 plus 返回结果可能过多。

后续建议增加过滤策略，例如：

```text
score >= 0.25；
或 score >= 0.20；
或最多保存 Top-N；
或按类别设置不同阈值；
或低分框仅前端显示，不自动暂存。
```

可先统计当前 151 个框的分数分布：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p'); sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin --eval 'const d=db.getSiblingDB("molar"); const itemId=ObjectId("69fed33d0fa06ee86bdea67c"); printjson(d.labels.aggregate([{$match:{itemId}},{$group:{_id:"$data.label", count:{$sum:1}, minScore:{$min:"$data.score"}, maxScore:{$max:"$data.score"}, avgScore:{$avg:"$data.score"}}},{$sort:{count:-1}}]).toArray());'
```

不同阈值下数量：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p'); sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin --eval 'const d=db.getSiblingDB("molar"); const itemId=ObjectId("69fed33d0fa06ee86bdea67c"); [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5].forEach(t=>print("score>="+t+":", d.labels.countDocuments({itemId,"data.score":{$gte:t}})));'
```

当前建议：

```text
先不要直接全局加过滤；
先基于多个样本统计 score 分布；
再决定过滤加在 detection3d_v5_plus 输出阶段，还是 main-server 当前测试分流阶段，还是前端暂存前。
```

---

## 74. 当前回滚方案

### 74.1 回滚前端 AI 标注按钮补丁

```bash
sudo docker exec molar-label-system-fe-v2 sh -lc 'cp /code/molar-label-system-fe-v2/assets/index.vue_vue_type_style_index_0_lang-5fbe0bdf.js.bak_ai_demo_oss_20260529 /code/molar-label-system-fe-v2/assets/index.vue_vue_type_style_index_0_lang-5fbe0bdf.js'
```

作用：

```text
恢复 AI 标注按钮读取原 OSS demo result.json 的行为。
```

### 74.2 回滚 main-server 分流

```bash
sudo docker exec main-server sh -lc 'cp /code/dist/main/main.js.bak_detection3d_plus_task_69fe8f /code/dist/main/main.js'
sudo docker restart main-server
```

作用：

```text
恢复 main-server /api/smart-tool/detection3d 全部走 /smart-tool/detection3d；
当前测试任务不再分流到 plus。
```

### 74.3 回滚 ai-nginx 测试路径

```bash
sudo docker exec ai-nginx sh -lc 'cat /tmp/nginx.conf.before_detection3d_plus > /etc/nginx/nginx.conf'
sudo docker exec ai-nginx nginx -t
sudo docker exec ai-nginx nginx -s reload
```

作用：

```text
删除 /smart-tool/detection3d_plus 测试路径；
原 /smart-tool/detection3d 不受影响。
```

### 74.4 清空 67c 自动标注结果

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p'); sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin --eval 'const d=db.getSiblingDB("molar"); const itemId=ObjectId("69fed33d0fa06ee86bdea67c"); print("before labels:", d.labels.countDocuments({itemId})); printjson(d.labels.deleteMany({itemId})); print("after labels:", d.labels.countDocuments({itemId}));'
```

---

## 75. 后续工程化建议

后续如果准备从“测试可用”推进到“稳定可用”，建议按以下顺序整理：

```text
1. 不再手工 patch dist/main/main.js 和压缩前端 JS；
   应回到源码仓库中修改，再重新构建镜像或前端产物。

2. 将 taskId 分流逻辑改成配置化；
   例如根据任务配置、环境变量或白名单控制是否走 plus。

3. 将前端 demo result.json 逻辑整理为正式条件；
   demo item 才走 OSS demo；
   普通任务一律走 getAIPower 对应工具。

4. 增加 plus 结果过滤策略；
   根据 score、类别、Top-N 控制自动保存框数量。

5. 明确类别体系；
   当前 plus 下划线类别已能保存显示，但仍需确认导出、统计、审核、标签模板是否全部兼容。

6. 保留供应商 detection3d 原路径；
   在 plus 充分验证前，不直接替换生产 /smart-tool/detection3d。
```

当前阶段性结论：

```text
OpenPCDet_ljl_plus 已成功接入当前测试任务的 AI 标注按钮链路；
供应商原 detection3d 仍可正常使用；
下一阶段重点从“接通链路”转为“过滤策略、配置化和工程化上线”。
```

---

## 76. 关于其他标注人员电脑是否可以使用当前 AI 标注按钮

在当前测试完成后，进一步讨论了一个实际使用问题：

```text
当前这个 AI 标注按钮是在服务器浏览器上验证成功的；
如果其他标注人员在其他电脑上访问平台，是否也能点击并走 plus 线？
```

结论如下：

```text
可以生效，但需要满足几个前提条件。
```

### 76.1 为什么其他电脑也可以生效

本次修改并不是只修改了服务器浏览器的本地状态，而是修改了服务器端平台相关组件：

```text
1. molar-label-system-fe-v2 前端打包 JS；
2. main-server 中当前测试 taskId 的 detection3d 分流逻辑；
3. ai-nginx 中新增的 /smart-tool/detection3d_plus 测试路径；
4. detection3d_v5_plus 中运行的 plus FastAPI 服务。
```

因此，其他标注人员如果访问同一个平台地址：

```text
http://172.23.131.39:8080
```

他们下载到的也是同一份服务器前端资源，请求的也是同一套后端服务。只要进入当前测试任务，AI 标注按钮理论上也会走 plus 链路。

当前实际链路为：

```text
其他电脑浏览器
    -> http://172.23.131.39:8080
    -> 下载服务器上的前端 JS
    -> 点击 AI 标注按钮
    -> /api/smart-tool/detection3d
    -> main-server
    -> 当前测试 taskId 分流到 /smart-tool/detection3d_plus
    -> ai-nginx
    -> detection3d_v5_plus
    -> 返回 plus 检测结果
    -> 前端显示
    -> 点击暂存
    -> /api/v2/label/save-labels
    -> 正式写入 labels
```

### 76.2 生效前提

其他电脑可以使用当前按钮的前提包括：

```text
1. 其他标注人员能访问同一个平台地址：
   http://172.23.131.39:8080

2. 其他标注人员进入的是当前测试任务：
   taskId = 69fe8f840fa06ee86bde27a6

3. 其他标注人员浏览器加载的是最新前端资源；
   如果之前打开过平台，需要 Ctrl + Shift + R 强刷或清理缓存。

4. detection3d_v5_plus 的 FastAPI 服务必须保持运行；
   如果 plus 服务是在终端中手动启动的，不能关闭该终端。

5. main-server 分流补丁和 ai-nginx 的 detection3d_plus 测试路径仍然存在。
```

其中最容易忽略的是浏览器缓存问题。由于本次是直接替换了服务器上的已打包 JS 文件，而文件名仍然是：

```text
index.vue_vue_type_style_index_0_lang-5fbe0bdf.js
```

如果其他电脑曾经访问过该平台，浏览器可能仍然缓存旧版本 JS，导致其点击 AI 标注按钮时仍走旧逻辑。因此第一次测试时建议强刷：

```text
Ctrl + Shift + R
```

### 76.3 当前任务与其他任务的影响范围

当前 main-server 的分流逻辑是按照任务 ID 判断：

```js
taskId === '69fe8f840fa06ee86bde27a6'
    ? '/smart-tool/detection3d_plus'
    : '/smart-tool/detection3d'
```

因此影响范围为：

```text
当前测试任务：
taskId = 69fe8f840fa06ee86bde27a6
-> AI 标注按钮走 detection3d_plus

其他任务：
-> 仍然走供应商原 /smart-tool/detection3d
```

这说明当前改动没有全局替换供应商模型路径，也没有破坏其他任务的原自动标注逻辑。

已经额外验证过供应商原接口仍正常：

```text
/smart-tool/detection3d
-> code = 200
-> 返回 Regular_vehicle、Pedestrian、Bus 等供应商类别
```

### 76.4 建议的异机验证流程

如果要让其他同事或标注人员在其他电脑上验证，可按以下流程：

```text
1. 其他电脑打开：
   http://172.23.131.39:8080

2. 登录平台。

3. 进入测试任务 test3d-260509，
   或包含 taskId = 69fe8f840fa06ee86bde27a6 的页面。

4. 打开 67c 或同任务下其他测试 item。

5. 按 Ctrl + Shift + R 强制刷新页面。

6. 点击 AI 标注按钮。

7. 观察页面是否出现 plus 类别体系下的 3D 框，
   如 vehicle_car、human_pedestrian_adult、vehicle_motorcycle 等。

8. 点击“暂存”。

9. 在 MongoDB 中检查 labels 是否增加，
   或刷新页面确认标注结果是否仍然存在。
```

可用于验证的 MongoDB 查询命令：

```bash
MONGO_PWD=$(sudo docker inspect mongodb --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^MONGO_INITDB_ROOT_PASSWORD=//p'); sudo docker exec -i mongodb mongosh --quiet -u root -p "$MONGO_PWD" --authenticationDatabase admin --eval 'const d=db.getSiblingDB("molar"); const itemId=ObjectId("<ITEM_ID>"); print("labels:", d.labels.countDocuments({itemId})); print("pre_labels:", d.pre_labels.countDocuments({itemId})); printjson(d.labels.aggregate([{$match:{itemId}},{$group:{_id:"$data.label", count:{$sum:1}}},{$sort:{count:-1}}]).toArray());'
```

其中 `<ITEM_ID>` 替换为实际测试 item 的 ID。

### 76.5 当前仍不建议大范围正式使用

虽然其他电脑理论上可以使用当前按钮，但目前仍建议只做小范围验证，不建议直接让大量标注人员正式使用。原因如下：

```text
1. 当前前端是直接 patch 打包后的 JS 文件；
   尚未回到前端源码中正式修改并重新构建。

2. main-server 分流逻辑目前是写死测试 taskId；
   尚未配置化。

3. detection3d_v5_plus 服务可能仍是手动启动；
   如果终端关闭，plus 服务会停止。

4. plus 返回框数量偏多；
   当前 67c 暂存后 labels = 151，
   后续还需要增加 score 阈值、类别阈值或 Top-N 限制。

5. 虽然 plus 下划线类别已经能够保存显示，
   但导出、统计、审核等后续流程仍需继续验证。
```

因此当前建议：

```text
可以让一两台其他电脑验证“是否也能点击 AI 标注并走 plus”；
但在过滤策略、服务自启动、前端正式构建、任务级配置化完成前，
不建议作为正式生产标注流程开放给大量标注人员。
```

### 76.6 该问题的阶段性结论

最终结论：

```text
其他标注人员在其他电脑上可以使用当前 AI 标注按钮，
前提是访问同一个平台、进入当前测试任务、强刷前端缓存，并且 plus 服务保持运行。

当前改动对其他任务仍保持供应商原 detection3d 线，
所以测试任务以外的任务理论上不会受到 plus 分流影响。
```

这也进一步说明：

```text
当前改动已经从“单机浏览器临时验证”
推进到了“服务器端平台链路级别生效”；
后续重点是工程化、配置化和上线前稳定性验证。
```

