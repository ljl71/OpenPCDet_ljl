# 自定义数据集教程
对于自定义数据集模板，我们只考虑最基本的场景：原始点云及其对应标注。点云应以 `.npy` 格式存储。

## 标签格式
在标签模板中，我们只考虑最基本的信息——类别和边界框。
标注存储在 `.txt` 文件中。每一行表示给定场景中的一个框，如下所示：
```
# format: [x y z dx dy dz heading_angle category_name]
1.50 1.46 0.10 5.12 1.85 4.13 1.56 Vehicle
5.54 0.57 0.41 1.08 0.74 1.95 1.57 Pedestrian
```
该框应采用统一的 3D 框定义（见 [README](../README.md)）。

## 文件结构
文件应按照以下文件夹结构放置：
```
OpenPCDet
├── data
│   ├── custom
│   │   │── ImageSets
│   │   │   │── train.txt
│   │   │   │── val.txt
│   │   │── points
│   │   │   │── 000000.npy
│   │   │   │── 999999.npy
│   │   │── labels
│   │   │   │── 000000.txt
│   │   │   │── 999999.txt
├── pcdet
├── tools
```
数据集划分需要预先定义，并放置在 `ImageSets` 中。

## 超参数配置

### 点云特征
修改 `custom_dataset.yaml` 中以下配置，使其适配你自己的点云。
```yaml
POINT_FEATURE_ENCODING: {
    encoding_type: absolute_coordinates_encoding,
    used_feature_list: ['x', 'y', 'z', 'intensity'],
    src_feature_list: ['x', 'y', 'z', 'intensity'],
}
...
# In gt_sampling data augmentation
NUM_POINT_FEATURES: 4

```

#### 点云范围和体素尺寸
对于基于体素的检测器，例如 SECOND、PV-RCNN 和 CenterPoint，点云范围和体素尺寸应满足：
1. z 轴方向的点云范围 / voxel_size 为 40。
2. x 和 y 轴方向的点云范围 / voxel_size 为 16 的倍数。

注意，第二条规则也适用于基于 pillar 的检测器，例如 PointPillar 和 CenterPoint-Pillar。

### 类别名称和 anchor 尺寸
类别名称和 anchor 尺寸需要根据自定义数据集进行调整。
 ```yaml
CLASS_NAMES: ['Vehicle', 'Pedestrian', 'Cyclist']  
...
MAP_CLASS_TO_KITTI: {
    'Vehicle': 'Car',
    'Pedestrian': 'Pedestrian',
    'Cyclist': 'Cyclist',
}
...
'anchor_sizes': [[3.9, 1.6, 1.56]],
...
# In gt sampling data augmentation
PREPARE: {
 filter_by_min_points: ['Vehicle:5', 'Pedestrian:5', 'Cyclist:5'],
 filter_by_difficulty: [-1],
}
SAMPLE_GROUPS: ['Vehicle:20','Pedestrian:15', 'Cyclist:15']
...
 ```
此外，请在 `custom_dataset.py` 中修改用于创建 infos 的默认类别名称：
```
create_custom_infos(
    dataset_cfg=dataset_cfg,
    class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
    data_path=ROOT_DIR / 'data' / 'custom',
    save_path=ROOT_DIR / 'data' / 'custom',
)
```


## 创建数据信息
运行以下命令生成数据 infos：
```shell
python -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/custom_dataset.yaml
```


## 评估
这里我们只提供 KITTI 风格评估的实现。
自定义数据集与 KITTI 之间的类别映射需要在 `custom_dataset.yaml` 中定义：
```yaml
MAP_CLASS_TO_KITTI: {
    'Vehicle': 'Car',
    'Pedestrian': 'Pedestrian',
    'Cyclist': 'Cyclist',
}
```
