import argparse
import importlib.util
import pickle
from collections import Counter
from pathlib import Path


def load_default_class_names(repo_root):
    utils_path = repo_root / 'pcdet' / 'datasets' / 'company_nuscenes' / 'company_nuscenes_utils.py'
    spec = importlib.util.spec_from_file_location('company_nuscenes_utils', utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COMPANY_26_CLASS_NAMES


def check_info_file(root, info_name):
    info_path = root / info_name
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    counter = Counter()
    missing = []
    empty_gt = 0
    for info in infos:
        counter.update(info.get('gt_names', []))
        if len(info.get('gt_names', [])) == 0:
            empty_gt += 1
        lidar_path = root / info['lidar_path']
        if not lidar_path.exists():
            missing.append(info['lidar_path'])

    print(info_name)
    print('  samples:', len(infos))
    print('  empty_gt_samples:', empty_gt)
    print('  missing_lidar_paths:', len(missing))
    if missing:
        for path in missing[:20]:
            print('   ', path)
    print('  classes:', len(counter))
    for name, count in counter.most_common():
        print(f'    {name}: {count}')
    return infos, counter, missing


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Check CompanyNuScenes info files')
    parser.add_argument('--root', type=Path, default=repo_root / 'data' / 'company_nuscenes' / 'v1.0-mini')
    parser.add_argument('--class_names', nargs='*', default=None)
    parser.add_argument('--strict', action='store_true', help='fail when info classes and class_names do not match')
    args = parser.parse_args()
    class_names = args.class_names or load_default_class_names(repo_root)

    _, train_counter, train_missing = check_info_file(args.root, 'company_nuscenes_infos_train.pkl')
    _, val_counter, val_missing = check_info_file(args.root, 'company_nuscenes_infos_val.pkl')

    if class_names:
        all_counter = train_counter + val_counter
        outside = sorted(set(all_counter) - set(class_names))
        absent = sorted(set(class_names) - set(all_counter))
        print('Class-name check')
        print('  outside_config:', outside)
        print('  absent_from_infos:', absent)
        if args.strict and (outside or absent):
            raise SystemExit(3)

    if train_missing or val_missing:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
