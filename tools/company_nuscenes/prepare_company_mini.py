import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


JSON_FILES = [
    'attribute.json',
    'calibrated_sensor.json',
    'category.json',
    'ego_pose.json',
    'instance.json',
    'log.json',
    'map.json',
    'sample.json',
    'sample_annotation.json',
    'sample_data.json',
    'scene.json',
    'sensor.json',
    'visibility.json',
]


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def copy_if_needed(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    shutil.copy2(src, dst)


def normalize_sample_data(target_root):
    sample_data_path = target_root / 'sample_data.json'
    sample_data = load_json(sample_data_path)
    changed = 0
    for item in sample_data:
        filename = item.get('filename', '')
        if filename.startswith('samples/LIDAR_TOP/') and filename.endswith('.pcd'):
            item['filename'] = filename[:-4] + '.bin'
            item['fileformat'] = 'bin'
            changed += 1
    with open(sample_data_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    return changed


def create_image_sets(target_root, train_ratio=0.8, seed=0):
    import random

    scenes = load_json(target_root / 'scene.json')
    names = [x['name'] for x in scenes]
    target_train_count = max(1, int(len(names) * train_ratio))
    if len(names) > 1:
        target_train_count = min(target_train_count, len(names) - 1)

    scene_classes = collect_scene_classes(target_root)
    class_scene_counts = Counter()
    for classes in scene_classes.values():
        class_scene_counts.update(classes)

    required_train = set()
    for scene_name, classes in scene_classes.items():
        if any(class_scene_counts[name] == 1 for name in classes):
            required_train.add(scene_name)

    remaining = [x for x in names if x not in required_train]
    random.Random(seed).shuffle(remaining)

    train = sorted(required_train)
    for scene_name in remaining:
        if len(train) >= target_train_count:
            break
        train.append(scene_name)

    if len(train) == len(names) and len(names) > 1:
        movable = [x for x in train if x not in required_train]
        if movable:
            train.remove(movable[-1])

    val = [x for x in names if x not in set(train)]

    image_sets = target_root / 'ImageSets'
    image_sets.mkdir(parents=True, exist_ok=True)
    (image_sets / 'train.txt').write_text('\n'.join(train) + '\n', encoding='utf-8')
    (image_sets / 'val.txt').write_text('\n'.join(val) + '\n', encoding='utf-8')
    return train, val


def collect_scene_classes(target_root):
    scenes = {x['token']: x['name'] for x in load_json(target_root / 'scene.json')}
    samples = {x['token']: x for x in load_json(target_root / 'sample.json')}
    categories = {x['token']: x['name'] for x in load_json(target_root / 'category.json')}
    instances = {x['token']: categories[x['category_token']] for x in load_json(target_root / 'instance.json')}
    scene_classes = defaultdict(set)
    for annotation in load_json(target_root / 'sample_annotation.json'):
        sample = samples[annotation['sample_token']]
        scene_name = scenes[sample['scene_token']]
        scene_classes[scene_name].add(instances[annotation['instance_token']])
    return scene_classes


def print_class_stats(target_root):
    categories = {x['token']: x['name'] for x in load_json(target_root / 'category.json')}
    instances = {x['token']: categories[x['category_token']] for x in load_json(target_root / 'instance.json')}
    annotations = load_json(target_root / 'sample_annotation.json')
    counter = Counter(instances[x['instance_token']] for x in annotations)
    print('Raw annotation classes:', len(counter))
    for name, count in counter.most_common():
        print(f'  {name}: {count}')


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Prepare company mini nuScenes data')
    parser.add_argument('--source', type=Path, default=repo_root.parent / 'mini' / 'v1.0-mini')
    parser.add_argument('--target', type=Path, default=repo_root / 'data' / 'company_nuscenes' / 'v1.0-mini')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    source_root = args.source
    source_meta = source_root / 'v1.0-mini'
    if not source_meta.exists():
        source_meta = source_root

    args.target.mkdir(parents=True, exist_ok=True)
    for name in JSON_FILES:
        src = source_meta / name
        if src.exists():
            copy_if_needed(src, args.target / name)

    lidar_src = source_root / 'samples' / 'LIDAR_TOP'
    lidar_dst = args.target / 'samples' / 'LIDAR_TOP'
    lidar_dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(lidar_src.glob('*.bin')):
        dst = lidar_dst / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
            copied += 1

    changed = normalize_sample_data(args.target)
    train_scenes, val_scenes = create_image_sets(args.target, train_ratio=args.train_ratio, seed=args.seed)

    print('Prepared target:', args.target)
    print('Copied lidar bins:', copied)
    print('Normalized lidar sample_data rows:', changed)
    print('Train scenes:', train_scenes)
    print('Val scenes:', val_scenes)
    print_class_stats(args.target)


if __name__ == '__main__':
    main()
