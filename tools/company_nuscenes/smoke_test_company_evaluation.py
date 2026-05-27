import importlib.util
from pathlib import Path

import numpy as np


def load_eval_module(repo_root):
    path = repo_root / 'pcdet' / 'datasets' / 'company_nuscenes' / 'company_nuscenes_eval.py'
    spec = importlib.util.spec_from_file_location('company_nuscenes_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    repo_root = Path(__file__).resolve().parents[2]
    evaluator = load_eval_module(repo_root)
    class_names = ['vehicle_car', 'human_pedestrian_adult', 'absent_class']
    infos = [{
        'token': 'sample-1',
        'gt_names': np.asarray(['vehicle_car', 'human_pedestrian_adult']),
        'gt_boxes': np.asarray([
            [5.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
            [10.0, 0.0, 0.0, 0.8, 0.8, 1.7, 0.0],
        ], dtype=np.float32),
        'num_lidar_pts': np.asarray([10, 0]),
    }]
    gt_annos, stats = evaluator.build_ground_truth_annos(infos, class_names, min_lidar_points=1)
    assert stats['evaluated_gt_boxes'] == 1
    assert stats['filtered_min_points_gt_boxes'] == 1

    perfect_predictions = [{
        'metadata': {'token': 'sample-1'},
        'name': np.asarray(['vehicle_car']),
        'score': np.asarray([0.9]),
        'boxes_lidar': np.asarray([[5.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]], dtype=np.float32),
    }]
    evaluator.validate_prediction_alignment(perfect_predictions, infos)
    perfect = evaluator.evaluate_company_predictions(
        gt_annos, perfect_predictions, class_names, gt_stats=stats
    )
    assert perfect['mAP'] == 1.0
    assert perfect['operating']['precision'] == 1.0
    assert perfect['operating']['recall'] == 1.0
    assert perfect['tp_quality'] == {'mATE': 0.0, 'mASE': 0.0, 'mAOE': 0.0}

    duplicate_predictions = [{
        'metadata': {'token': 'sample-1'},
        'name': np.asarray(['vehicle_car', 'vehicle_car', 'absent_class']),
        'score': np.asarray([0.9, 0.8, 0.7]),
        'boxes_lidar': np.asarray([
            [5.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
            [5.1, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
            [20.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
        ], dtype=np.float32),
    }]
    duplicate = evaluator.evaluate_company_predictions(
        gt_annos, duplicate_predictions, class_names, gt_stats=stats
    )
    assert duplicate['operating']['tp'] == 1
    assert duplicate['operating']['fp'] == 2
    assert np.isclose(duplicate['operating']['precision'], 1.0 / 3.0)

    symmetric_gt = [{
        'name': np.asarray(['movable_object_barrier', 'movable_object_trafficcone']),
        'boxes_lidar': np.asarray([
            [2.0, 0.0, 0.0, 2.0, 0.5, 1.0, 0.0],
            [4.0, 0.0, 0.0, 0.4, 0.4, 0.8, 0.0],
        ], dtype=np.float32),
    }]
    symmetric_predictions = [{
        'name': np.asarray(['movable_object_barrier', 'movable_object_trafficcone']),
        'score': np.asarray([0.9, 0.8]),
        'boxes_lidar': np.asarray([
            [2.0, 0.0, 0.0, 2.0, 0.5, 1.0, np.pi],
            [4.0, 0.0, 0.0, 0.4, 0.4, 0.8, np.pi / 2],
        ], dtype=np.float32),
    }]
    symmetric = evaluator.evaluate_company_predictions(
        symmetric_gt, symmetric_predictions,
        ['movable_object_barrier', 'movable_object_trafficcone']
    )
    assert np.isclose(symmetric['tp_quality']['mAOE'], 0.0, atol=1e-6)
    print('company_evaluation_smoke: PASS')


if __name__ == '__main__':
    main()
