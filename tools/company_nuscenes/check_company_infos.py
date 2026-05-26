import argparse
import importlib.util
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def load_default_class_names(repo_root):
    utils_path = repo_root / 'pcdet' / 'datasets' / 'company_nuscenes' / 'company_nuscenes_utils.py'
    spec = importlib.util.spec_from_file_location('company_nuscenes_utils', utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COMPANY_26_CLASS_NAMES


def find_lidar_path(root, data_root, lidar_path):
    lidar_path = Path(lidar_path)
    candidates = [lidar_path] if lidar_path.is_absolute() else [data_root / lidar_path, root / lidar_path]
    for candidate in candidates:
        bin_path = candidate.with_suffix('.bin') if candidate.suffix == '.pcd' else candidate
        if bin_path.exists():
            return bin_path
        if candidate.exists():
            return candidate
    return None


def check_info_file(root, data_root, info_name, min_lidar_points):
    info_path = root / info_name
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    counter = Counter()
    retained_counter = Counter()
    missing = []
    empty_gt = 0
    empty_gt_after_filter = 0
    annotation_count_missing = 0
    boxes_with_unknown_lidar_pts = 0
    for info in infos:
        names = np.asarray(info.get('gt_names', []))
        counter.update(names)
        if len(names) == 0:
            empty_gt += 1
        point_counts = info.get('num_lidar_pts')
        if point_counts is None or len(point_counts) != len(names):
            annotation_count_missing += 1
            retained_names = np.asarray([])
        else:
            retained_mask = np.asarray([
                count is None or count >= min_lidar_points for count in point_counts
            ])
            boxes_with_unknown_lidar_pts += sum(count is None for count in point_counts)
            retained_names = names[retained_mask]
        retained_counter.update(retained_names)
        if len(retained_names) == 0:
            empty_gt_after_filter += 1
        if find_lidar_path(root, data_root, info['lidar_path']) is None:
            missing.append(info['lidar_path'])

    print(info_name)
    print('  samples:', len(infos))
    print('  empty_gt_samples:', empty_gt)
    print(f'  empty_gt_after_min_points_{min_lidar_points}:', empty_gt_after_filter)
    print('  invalid_num_lidar_pts_samples:', annotation_count_missing)
    print('  boxes_with_unknown_lidar_pts_kept:', boxes_with_unknown_lidar_pts)
    print('  missing_lidar_paths:', len(missing))
    if missing:
        for path in missing[:20]:
            print('   ', path)
    print('  classes:', len(counter))
    for name, count in counter.most_common():
        print(f'    {name}: {count}')
    print(f'  retained_boxes_min_points_{min_lidar_points}:', sum(retained_counter.values()))
    return infos, counter, retained_counter, missing, annotation_count_missing


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Check CompanyNuScenes info files')
    parser.add_argument('--root', type=Path, default=repo_root / 'data' / 'nuscenes' / 'v1.0-trainval')
    parser.add_argument(
        '--data_root', type=Path, default=None,
        help='data root containing samples/; defaults to the parent directory of --root'
    )
    parser.add_argument('--class_names', nargs='*', default=None)
    parser.add_argument('--strict', action='store_true', help='fail on paths or classes outside configured names')
    parser.add_argument(
        '--require_all_classes', action='store_true',
        help='also fail if a configured class has no annotations in these infos'
    )
    parser.add_argument(
        '--min_lidar_points', type=int, default=1,
        help='mirror FILTER_MIN_POINTS_IN_GT when checking retained training boxes'
    )
    args = parser.parse_args()
    data_root = args.data_root or args.root.parent
    class_names = args.class_names or load_default_class_names(repo_root)

    _, train_counter, train_retained_counter, train_missing, train_invalid_counts = check_info_file(
        args.root, data_root, 'company_nuscenes_infos_train.pkl', args.min_lidar_points
    )
    _, val_counter, _, val_missing, val_invalid_counts = check_info_file(
        args.root, data_root, 'company_nuscenes_infos_val.pkl', args.min_lidar_points
    )

    if class_names:
        all_counter = train_counter + val_counter
        outside = sorted(set(all_counter) - set(class_names))
        absent = sorted(set(class_names) - set(all_counter))
        absent_from_train = sorted(set(all_counter) - set(train_counter))
        absent_from_val = sorted(set(all_counter) - set(val_counter))
        absent_after_train_filter = sorted(set(all_counter) - set(train_retained_counter))
        print('Class-name check')
        print('  outside_config:', outside)
        print('  absent_from_infos:', absent)
        print('  annotated_but_absent_from_train:', absent_from_train)
        print('  annotated_but_absent_from_val:', absent_from_val)
        print('  annotated_but_unlearnable_after_train_filter:', absent_after_train_filter)
        if args.strict and outside:
            raise SystemExit(3)
        if args.require_all_classes and absent:
            raise SystemExit(3)
        if absent_after_train_filter:
            raise SystemExit(4)

    if train_missing or val_missing:
        raise SystemExit(2)
    if train_invalid_counts or val_invalid_counts:
        raise SystemExit('num_lidar_pts is absent or inconsistent in one or more infos.')
    if not train_retained_counter:
        raise SystemExit('No training GT boxes remain after num_lidar_pts filtering.')


if __name__ == '__main__':
    main()
