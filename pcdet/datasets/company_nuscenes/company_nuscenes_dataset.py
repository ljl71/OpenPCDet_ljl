import copy
import pickle
from collections import Counter
from pathlib import Path

from ..dataset import DatasetTemplate
from . import company_nuscenes_utils
from . import point_io


class CompanyNuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        self.data_path = Path(root_path if root_path is not None else dataset_cfg.DATA_PATH)
        self.metadata_path = self.data_path / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=self.data_path, logger=logger
        )
        self.infos = []
        self.point_dim = len(point_io.get_src_feature_list(self.dataset_cfg))
        self.lidar_point_dim = point_io.get_lidar_point_dim(self.dataset_cfg, default=self.point_dim)
        self.include_company_nuscenes_data(self.mode)

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def include_company_nuscenes_data(self, mode):
        self._log('Loading CompanyNuScenes dataset')
        company_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.metadata_path / info_path
            if not info_path.exists():
                self._log(f'Missing info file: {info_path}')
                continue
            with open(info_path, 'rb') as f:
                company_infos.extend(pickle.load(f))
        self.infos.extend(company_infos)
        self._log('Total samples for CompanyNuScenes dataset: %d' % len(company_infos))

    def read_lidar(self, lidar_path):
        lidar_path = Path(lidar_path)
        if lidar_path.is_absolute():
            candidates = [lidar_path]
        else:
            candidates = [self.data_path / lidar_path, self.metadata_path / lidar_path]

        resolved_path = None
        for candidate in candidates:
            bin_path = candidate.with_suffix('.bin') if candidate.suffix == '.pcd' else candidate
            if bin_path.exists():
                resolved_path = bin_path
                break
            if candidate.exists():
                resolved_path = candidate
                break

        if resolved_path is None:
            raise FileNotFoundError(f'Lidar file not found; checked: {candidates}')
        lidar_path = resolved_path

        if lidar_path.suffix == '.pcd':
            return point_io.read_binary_pcd(lidar_path, self.dataset_cfg)

        return point_io.read_lidar_bin(
            lidar_path, self.dataset_cfg, fallback_dim=self.lidar_point_dim
        )

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        if max_sweeps != 1:
            raise NotImplementedError('CompanyNuScenesDataset currently supports MAX_SWEEPS=1 only.')
        info = self.infos[index]
        return self.read_lidar(info['lidar_path'])

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']},
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                min_points = self.dataset_cfg.FILTER_MIN_POINTS_IN_GT
                mask = [
                    count is None or count >= min_points
                    for count in info['num_lidar_pts']
                ]
            else:
                mask = None
            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask],
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        if not self.dataset_cfg.get('PRED_VELOCITY', False) and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        gt_counter = Counter()
        pred_counter = Counter()
        for info in self.infos:
            gt_counter.update([x for x in info.get('gt_names', []) if x in class_names])
        for anno in det_annos:
            pred_counter.update([x for x in anno.get('name', []) if x in class_names])

        result = ['---------------CompanyNuScenes smoke evaluation---------------']
        result.append('This first-stage evaluation only reports GT/pred counts; AP is not implemented yet.')
        details = {}
        for name in class_names:
            gt_count = int(gt_counter.get(name, 0))
            pred_count = int(pred_counter.get(name, 0))
            result.append(f'{name}: gt={gt_count}, pred={pred_count}')
            details[f'{name}_gt'] = gt_count
            details[f'{name}_pred'] = pred_count
        return '\n'.join(result) + '\n', details


def create_company_nuscenes_infos(dataset_cfg, data_path, save_path, version):
    return company_nuscenes_utils.create_company_nuscenes_infos(
        version=version,
        data_path=data_path,
        save_path=save_path,
        max_sweeps=dataset_cfg.MAX_SWEEPS,
        train_ratio=dataset_cfg.get('TRAIN_SPLIT_RATIO', 0.8),
        seed=dataset_cfg.get('SPLIT_SEED', 0),
        min_lidar_points=dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', 1),
    )


if __name__ == '__main__':
    import argparse
    import yaml
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='Company nuScenes data prep')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--func', type=str, default='create_company_nuscenes_infos')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    args = parser.parse_args()

    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    root_dir = (Path(__file__).resolve().parent / '../../../').resolve()
    dataset_cfg.VERSION = args.version

    if args.func == 'create_company_nuscenes_infos':
        create_company_nuscenes_infos(
            dataset_cfg=dataset_cfg,
            data_path=root_dir / 'data' / 'nuscenes',
            save_path=root_dir / 'data' / 'nuscenes',
            version=args.version,
        )
    else:
        raise NotImplementedError(args.func)
