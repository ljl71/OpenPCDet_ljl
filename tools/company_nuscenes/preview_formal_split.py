import argparse
import importlib.util
from collections import Counter
from pathlib import Path


def load_utils(repo_root):
    utils_path = repo_root / 'pcdet' / 'datasets' / 'company_nuscenes' / 'company_nuscenes_utils.py'
    spec = importlib.util.spec_from_file_location('company_nuscenes_utils', utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Preview the formal scene-level train/val split without writing files.')
    parser.add_argument('--data_path', type=Path, default=repo_root / 'data' / 'nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--min_lidar_points', type=int, default=1)
    args = parser.parse_args()

    utils = load_utils(repo_root)
    metadata_path = args.data_path / args.version
    tables = utils.load_company_tables(metadata_path)
    train, val = utils.make_scene_split(
        tables,
        train_ratio=args.train_ratio,
        seed=args.seed,
        min_lidar_points=args.min_lidar_points,
    )
    train, val = set(train), set(val)

    sample_to_scene = {
        sample['token']: tables['scenes'][sample['scene_token']]['name']
        for sample in tables['samples']
    }
    sample_counts = Counter(sample_to_scene.values())
    train_counts = Counter()
    val_counts = Counter()

    for sample_token, annotations in tables['annos_by_sample'].items():
        scene_name = sample_to_scene[sample_token]
        for annotation in annotations:
            if not utils.annotation_passes_point_filter(annotation, args.min_lidar_points):
                continue
            class_name = utils.raw_category_for_annotation(annotation, tables)
            if scene_name in train:
                train_counts[class_name] += 1
            elif scene_name in val:
                val_counts[class_name] += 1

    annotated_classes = sorted(set(train_counts) | set(val_counts))
    train_absent = sorted(set(annotated_classes) - set(train_counts))
    val_absent = sorted(set(annotated_classes) - set(val_counts))

    print('===== Formal Split Preview (read-only) =====')
    print('train_scenes:', len(train))
    print('val_scenes:', len(val))
    print('train_samples:', sum(sample_counts[x] for x in train))
    print('val_samples:', sum(sample_counts[x] for x in val))
    print(f'min_lidar_points: {args.min_lidar_points}')
    print(f'{"class":42s} {"train_boxes":>13s} {"val_boxes":>11s}')
    for class_name in annotated_classes:
        print(f'{class_name:42s} {train_counts[class_name]:13d} {val_counts[class_name]:11d}')
    print('annotated_but_absent_from_train:', train_absent)
    print('annotated_but_absent_from_val:', val_absent)
    if train_absent:
        raise SystemExit('FAIL: one or more learnable classes are absent from training.')
    print('PASS: all learnable annotated classes are present in training.')


if __name__ == '__main__':
    main()
