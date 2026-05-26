import json
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


COMPANY_NAME_MAPPING = {
    'human.pedestrian.adult': 'human_pedestrian_adult',
    'human.pedestrian.child': 'human_pedestrian_child',
    'human.pedestrian.wheelchair': 'human_pedestrian_wheelchair',
    'human.pedestrian.stroller': 'human_pedestrian_stroller',
    'human.pedestrian.personal_mobility': 'human_pedestrian_personal_mobility',
    'vehicle.car': 'vehicle_car',
    'vehicle.bus.bendy': 'vehicle_bus_bendy',
    'vehicle.bus.rigid': 'vehicle_bus_rigid',
    'vehicle.truck': 'vehicle_truck',
    'vehicle.construction': 'vehicle_construction',
    'vehicle.emergency.ambulance': 'vehicle_emergency_ambulance',
    'vehicle.emergency.police': 'vehicle_emergency_police',
    'vehicle.trailer': 'vehicle_trailer',
    'movable_object.barrier': 'movable_object_barrier',
    'movable_object.trafficcone': 'movable_object_trafficcone',
    'movable_object.pushable_pullable': 'movable_object_pushable_pullable',
    'movable_object.debris': 'movable_object_debris',
    'vehicle.emergency.other': 'vehicle_emergency_other',
    'vehicle.motorcycle': 'vehicle_motorcycle',
    'vehicle.bicycle': 'vehicle_bicycle',
    'group.human.pedestrian': 'group_human_pedestrian',
    'group.vehicle.bicycle': 'group_vehicle_bicycle',
    'other': 'other',
    'animal': 'animal',
    'vehicle.tricycle': 'vehicle_tricycle',
    'bicycle': 'bicycle',
}

COMPANY_RAW_CLASS_NAMES = [
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.wheelchair',
    'human.pedestrian.stroller',
    'human.pedestrian.personal_mobility',
    'vehicle.car',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.truck',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'vehicle.trailer',
    'movable_object.barrier',
    'movable_object.trafficcone',
    'movable_object.pushable_pullable',
    'movable_object.debris',
    'vehicle.emergency.other',
    'vehicle.motorcycle',
    'vehicle.bicycle',
    'group.human.pedestrian',
    'group.vehicle.bicycle',
    'other',
    'animal',
    'vehicle.tricycle',
    'bicycle',
]

COMPANY_26_CLASS_NAMES = [COMPANY_NAME_MAPPING[name] for name in COMPANY_RAW_CLASS_NAMES]


def annotation_point_count(annotation, key='num_lidar_pts'):
    value = annotation.get(key)
    return None if value is None else int(value)


def annotation_passes_point_filter(annotation, min_lidar_points=1):
    point_count = annotation_point_count(annotation)
    return point_count is None or point_count >= min_lidar_points


def normalize_category_name(raw_name):
    if raw_name in COMPANY_NAME_MAPPING:
        return COMPANY_NAME_MAPPING[raw_name]
    return re.sub(r'[^0-9A-Za-z]+', '_', raw_name).strip('_')


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def dump_pickle(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def quat_inverse(q):
    q = np.asarray(q, dtype=np.float64)
    inv = q.copy()
    inv[1:] *= -1
    return inv / np.dot(q, q)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quat_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rotate(q, points):
    return quat_to_rotmat(q).dot(np.asarray(points, dtype=np.float64))


def quaternion_yaw(q):
    direction = quat_to_rotmat(q).dot(np.array([1.0, 0.0, 0.0]))
    return math.atan2(direction[1], direction[0])


def annotation_to_lidar_box(annotation, ego_pose, calibrated_sensor):
    center = np.asarray(annotation['translation'], dtype=np.float64)
    center = rotate(quat_inverse(ego_pose['rotation']), center - np.asarray(ego_pose['translation']))
    center = rotate(
        quat_inverse(calibrated_sensor['rotation']),
        center - np.asarray(calibrated_sensor['translation'])
    )

    orientation = np.asarray(annotation['rotation'], dtype=np.float64)
    orientation = quat_multiply(quat_inverse(ego_pose['rotation']), orientation)
    orientation = quat_multiply(quat_inverse(calibrated_sensor['rotation']), orientation)

    width, length, height = annotation['size']
    yaw = quaternion_yaw(orientation)
    return np.array([center[0], center[1], center[2], length, width, height, yaw], dtype=np.float32)


def read_image_sets(root_path, train_ratio=0.8, seed=0, min_lidar_points=1):
    image_sets = root_path / 'ImageSets'
    train_file = image_sets / 'train.txt'
    val_file = image_sets / 'val.txt'

    if train_file.exists() and val_file.exists():
        train = {x.strip() for x in train_file.read_text(encoding='utf-8').splitlines() if x.strip()}
        val = {x.strip() for x in val_file.read_text(encoding='utf-8').splitlines() if x.strip()}
        return train, val

    tables = load_company_tables(root_path)
    train, val = make_scene_split(
        tables, train_ratio=train_ratio, seed=seed, min_lidar_points=min_lidar_points
    )
    image_sets.mkdir(parents=True, exist_ok=True)
    train_file.write_text('\n'.join(train) + '\n', encoding='utf-8')
    val_file.write_text('\n'.join(val) + '\n', encoding='utf-8')
    return set(train), set(val)


def make_scene_split(tables, train_ratio=0.8, seed=0, min_lidar_points=1):
    scene_names = [x['name'] for x in tables['scenes'].values()]
    scene_classes = collect_scene_classes(tables, min_lidar_points=min_lidar_points)

    target_train_count = max(1, int(len(scene_names) * train_ratio))
    if len(scene_names) > 1:
        target_train_count = min(target_train_count, len(scene_names) - 1)
    target_val_count = len(scene_names) - target_train_count

    class_scene_counts = Counter()
    for classes in scene_classes.values():
        class_scene_counts.update(classes)

    required_train = set([
        scene_name for scene_name, classes in scene_classes.items()
        if any(class_scene_counts[name] == 1 for name in classes)
    ])
    train_covered = set().union(*(scene_classes[x] for x in required_train)) if required_train else set()
    additional_train, train_uncovered = select_cover_scenes(
        scene_classes=scene_classes,
        class_names=set(class_scene_counts) - train_covered,
        candidate_scenes=[x for x in scene_names if x not in required_train],
    )
    if train_uncovered:
        raise ValueError(f'Cannot allocate training scenes for annotated classes: {sorted(train_uncovered)}')
    required_train.update(additional_train)
    if len(required_train) > target_train_count:
        raise ValueError(
            f'Train split target {target_train_count} cannot cover required scenes: {sorted(required_train)}'
        )

    val_candidates = [x for x in scene_names if x not in required_train]
    val_classes = {
        class_name for class_name in class_scene_counts
        if any(class_name in scene_classes.get(scene_name, set()) for scene_name in val_candidates)
    }
    required_val, val_uncovered = select_cover_scenes(
        scene_classes=scene_classes,
        class_names=val_classes,
        candidate_scenes=val_candidates,
    )
    if val_uncovered:
        raise ValueError(f'Cannot allocate validation scenes for annotated classes: {sorted(val_uncovered)}')
    if len(required_val) > target_val_count:
        raise ValueError(
            f'Validation split target {target_val_count} cannot cover required scenes: {sorted(required_val)}'
        )

    remaining = [
        x for x in scene_names
        if x not in required_train and x not in set(required_val)
    ]
    random.Random(seed).shuffle(remaining)

    val = required_val + remaining[:target_val_count - len(required_val)]
    val_set = set(val)
    train = [x for x in scene_names if x not in val_set]
    return train, val


def select_cover_scenes(scene_classes, class_names, candidate_scenes):
    uncovered = set(class_names)
    candidates = set(candidate_scenes)
    selected = []

    while uncovered:
        ranked = sorted(
            candidates,
            key=lambda scene_name: (
                -len(scene_classes.get(scene_name, set()) & uncovered),
                scene_name,
            )
        )
        if not ranked:
            break
        best_scene = ranked[0]
        covered = scene_classes.get(best_scene, set()) & uncovered
        if not covered:
            break
        selected.append(best_scene)
        candidates.remove(best_scene)
        uncovered -= covered

    return selected, uncovered


def collect_scene_classes(tables, min_lidar_points=1):
    sample_to_scene = {
        sample['token']: tables['scenes'][sample['scene_token']]['name']
        for sample in tables['samples']
    }
    scene_classes = defaultdict(set)
    for sample_token, annotations in tables['annos_by_sample'].items():
        scene_name = sample_to_scene[sample_token]
        for annotation in annotations:
            if not annotation_passes_point_filter(annotation, min_lidar_points):
                continue
            scene_classes[scene_name].add(raw_category_for_annotation(annotation, tables))
    return scene_classes


def load_company_tables(root_path):
    categories = {x['token']: x for x in load_json(root_path / 'category.json')}
    instances = {x['token']: x for x in load_json(root_path / 'instance.json')}
    calibrated_sensors = {x['token']: x for x in load_json(root_path / 'calibrated_sensor.json')}
    ego_poses = {x['token']: x for x in load_json(root_path / 'ego_pose.json')}
    sample_data = {x['token']: x for x in load_json(root_path / 'sample_data.json')}
    scenes = {x['token']: x for x in load_json(root_path / 'scene.json')}
    samples = load_json(root_path / 'sample.json')
    annotations = load_json(root_path / 'sample_annotation.json')

    annos_by_sample = defaultdict(list)
    for anno in annotations:
        annos_by_sample[anno['sample_token']].append(anno)

    return {
        'categories': categories,
        'instances': instances,
        'calibrated_sensors': calibrated_sensors,
        'ego_poses': ego_poses,
        'sample_data': sample_data,
        'scenes': scenes,
        'samples': samples,
        'annos_by_sample': annos_by_sample,
    }


def raw_category_for_annotation(annotation, tables):
    instance = tables['instances'][annotation['instance_token']]
    category = tables['categories'][instance['category_token']]
    return category['name']


def build_sample_info(sample, tables, root_path, data_path=None, max_sweeps=1):
    if max_sweeps != 1:
        raise NotImplementedError('CompanyNuScenes first-stage adapter supports MAX_SWEEPS=1 only.')

    root_path = Path(root_path)
    data_path = Path(data_path) if data_path is not None else root_path
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_sd = tables['sample_data'][lidar_token]
    lidar_path = resolve_lidar_path(root_path, lidar_sd['filename'], data_path=data_path)

    calibrated_sensor = tables['calibrated_sensors'][lidar_sd['calibrated_sensor_token']]
    ego_pose = tables['ego_poses'][lidar_sd['ego_pose_token']]

    boxes = []
    names = []
    raw_names = []
    num_lidar_pts = []
    num_radar_pts = []
    tokens = []

    for annotation in tables['annos_by_sample'].get(sample['token'], []):
        raw_name = raw_category_for_annotation(annotation, tables)
        boxes.append(annotation_to_lidar_box(annotation, ego_pose, calibrated_sensor))
        names.append(normalize_category_name(raw_name))
        raw_names.append(raw_name)
        num_lidar_pts.append(annotation_point_count(annotation))
        num_radar_pts.append(annotation_point_count(annotation, key='num_radar_pts'))
        tokens.append(annotation['token'])

    info = {
        'lidar_path': lidar_path.as_posix(),
        'token': sample['token'],
        'scene_token': sample['scene_token'],
        'timestamp': sample['timestamp'] * 1e-6,
        'sweeps': [],
        'gt_boxes': np.asarray(boxes, dtype=np.float32).reshape(-1, 7),
        'gt_names': np.asarray(names),
        'gt_raw_names': np.asarray(raw_names),
        'gt_boxes_token': np.asarray(tokens),
        'num_lidar_pts': np.asarray(num_lidar_pts, dtype=object),
        'num_radar_pts': np.asarray(num_radar_pts, dtype=object),
    }
    info['lidar_path_exists'] = (data_path / info['lidar_path']).exists()
    return info


def resolve_lidar_path(root_path, filename, data_path=None):
    metadata_path = Path(root_path)
    data_path = Path(data_path) if data_path is not None else metadata_path
    lidar_path = Path(filename)
    path_candidates = [lidar_path]
    if lidar_path.suffix == '.pcd':
        path_candidates.insert(0, lidar_path.with_suffix('.bin'))

    for candidate in path_candidates:
        for base_path in [data_path, metadata_path]:
            absolute_path = base_path / candidate
            if absolute_path.exists():
                try:
                    return absolute_path.relative_to(data_path)
                except ValueError:
                    return absolute_path
    return lidar_path


def build_company_infos(
        root_path, data_path=None, max_sweeps=1, train_ratio=0.8, seed=0, min_lidar_points=1
):
    root_path = Path(root_path)
    data_path = Path(data_path) if data_path is not None else root_path
    tables = load_company_tables(root_path)
    train_scene_names, val_scene_names = read_image_sets(
        root_path, train_ratio=train_ratio, seed=seed, min_lidar_points=min_lidar_points
    )

    train_infos = []
    val_infos = []
    missing_lidar_paths = []
    raw_counter = Counter()
    mapped_counter = Counter()

    samples = sorted(tables['samples'], key=lambda x: x.get('timestamp', 0))
    for sample in samples:
        scene_name = tables['scenes'][sample['scene_token']]['name']
        info = build_sample_info(
            sample, tables, root_path=root_path, data_path=data_path, max_sweeps=max_sweeps
        )
        raw_counter.update(info['gt_raw_names'].tolist())
        mapped_counter.update(info['gt_names'].tolist())

        if not info['lidar_path_exists']:
            missing_lidar_paths.append(info['lidar_path'])

        if scene_name in train_scene_names:
            train_infos.append(info)
        elif scene_name in val_scene_names:
            val_infos.append(info)

    stats = {
        'train_samples': len(train_infos),
        'val_samples': len(val_infos),
        'missing_lidar_paths': missing_lidar_paths,
        'raw_class_counts': dict(raw_counter),
        'mapped_class_counts': dict(mapped_counter),
    }
    return train_infos, val_infos, stats


def create_company_nuscenes_infos(
        version, data_path, save_path, max_sweeps=1, train_ratio=0.8, seed=0, min_lidar_points=1
):
    data_path = Path(data_path)
    root_path = data_path / version
    save_path = Path(save_path) / version

    train_infos, val_infos, stats = build_company_infos(
        root_path=root_path, data_path=data_path, max_sweeps=max_sweeps, train_ratio=train_ratio, seed=seed,
        min_lidar_points=min_lidar_points
    )

    train_path = save_path / 'company_nuscenes_infos_train.pkl'
    val_path = save_path / 'company_nuscenes_infos_val.pkl'
    dump_pickle(train_infos, train_path)
    dump_pickle(val_infos, val_path)

    print('Company nuScenes train infos:', len(train_infos))
    print('Company nuScenes val infos:', len(val_infos))
    print('Saved:', train_path)
    print('Saved:', val_path)
    if stats['missing_lidar_paths']:
        print('Missing lidar paths:', len(stats['missing_lidar_paths']))
        for item in stats['missing_lidar_paths'][:20]:
            print('  ', item)
    print('Mapped class counts:')
    for name, count in Counter(stats['mapped_class_counts']).most_common():
        print(f'  {name}: {count}')
    return stats
