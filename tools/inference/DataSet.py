import os
import glob
import numpy as np
from pcdet.datasets import DatasetTemplate
from pcdet.datasets.company_nuscenes import point_io
import os
import copy
from pathlib import Path
from torch.utils.data import DistributedSampler as _DistributedSampler
# from datasets import load_dataset


def format_points_for_config(points, dataset_cfg, point_fields=None):
    points = np.asarray(points, dtype=np.float32)
    points = points[~np.isnan(points).any(axis=1), :]

    src_feature_list = point_io.get_src_feature_list(dataset_cfg)
    if point_fields is not None:
        return point_io.align_points_to_features(points, point_fields, src_feature_list)

    target_dim = len(src_feature_list)
    if points.shape[1] == target_dim:
        return points
    if points.shape[1] > target_dim:
        return points[:, :target_dim]

    formatted = np.zeros((points.shape[0], target_dim), dtype=np.float32)
    formatted[:, :points.shape[1]] = points
    return formatted


class Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names,data_file_list, training=True, shape = 4):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training
        )
        self.infos = []
        self.shape = dataset_cfg.get('LIDAR_POINT_DIM', shape)
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        # return data_dict
        sample_path = Path(self.sample_file_list[index])
        suffix = sample_path.suffix.lower()
        if suffix == '.bin':
            points = point_io.read_lidar_bin(
                sample_path, self.dataset_cfg, fallback_dim=self.shape
            )
        elif suffix == '.npy':
            points = np.load(sample_path)
        elif suffix == '.pcd':
            from pyntcloud import PyntCloud
            path = str(sample_path)

            try:
                pcd_load = PyntCloud.from_file(path)
                point_fields = list(pcd_load.points.columns)
                points = format_points_for_config(
                    np.asarray(pcd_load.points), self.dataset_cfg, point_fields=point_fields
                )
            except:
                from pypcd import pypcd
                import pandas as pd
                pc = pypcd.PointCloud.from_path(path)
                points = np.asarray(pd.DataFrame(pc.pc_data, columns=['x', 'y', 'z']))

        else:
            raise NotImplementedError

        # points = np.pad(points, (0, 2), 'constant', constant_values=(0, 0))

        points = format_points_for_config(points, self.dataset_cfg)

        input_dict = {
            'points': points,
            'frame_id': index,
            'ori_points': points[:,:3]
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def getDataFromFile(datapath):

    assert os.path.exists(datapath)

    data_file_list = list(Path(datapath).glob('*.*'))
    data_file_list.sort()
    # print(data_file_list)
    return data_file_list


def getDataFromURL(URL):
    # 待操作
    data_url_list = list(Path(URL).glob('*.*'))
    data_url_list.sort()
    return data_url_list


