import argparse
import importlib.util
import json
import pickle
from pathlib import Path


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description='Evaluate saved CompanyNuScenes predictions with custom 26-class distance AP metrics.'
    )
    parser.add_argument('--result', type=Path, required=True, help='OpenPCDet result.pkl file.')
    parser.add_argument('--infos', type=Path, required=True, help='Matching company_nuscenes_infos_val.pkl file.')
    parser.add_argument('--class_names', nargs='*', default=None)
    parser.add_argument('--min_lidar_points', type=int, default=1)
    parser.add_argument('--distance_thresholds', nargs='+', type=float, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument('--tp_distance_threshold', type=float, default=2.0)
    parser.add_argument(
        '--min_score', type=float, default=None,
        help='Optionally drop predictions below this score in addition to filtering already applied during inference.'
    )
    parser.add_argument(
        '--output_json', type=Path, default=None,
        help='Output JSON path; defaults to company_metrics_summary.json next to --result.'
    )
    args = parser.parse_args()

    eval_module = load_module(
        'company_nuscenes_eval',
        repo_root / 'pcdet' / 'datasets' / 'company_nuscenes' / 'company_nuscenes_eval.py'
    )
    utils_module = load_module(
        'company_nuscenes_utils',
        repo_root / 'pcdet' / 'datasets' / 'company_nuscenes' / 'company_nuscenes_utils.py'
    )
    class_names = args.class_names or utils_module.COMPANY_26_CLASS_NAMES

    with open(args.result, 'rb') as f:
        det_annos = pickle.load(f)
    with open(args.infos, 'rb') as f:
        infos = pickle.load(f)

    eval_module.validate_prediction_alignment(det_annos, infos)
    gt_annos, gt_stats = eval_module.build_ground_truth_annos(
        infos, class_names, min_lidar_points=args.min_lidar_points
    )
    metrics = eval_module.evaluate_company_predictions(
        gt_annos=gt_annos,
        det_annos=det_annos,
        class_names=class_names,
        distance_thresholds=args.distance_thresholds,
        tp_distance_threshold=args.tp_distance_threshold,
        min_score=args.min_score,
        gt_stats=gt_stats,
    )
    print(eval_module.format_company_results(metrics), end='')

    output_json = args.output_json or args.result.parent / 'company_metrics_summary.json'
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=True, allow_nan=False)
    print(f'Metrics JSON saved to: {output_json}')


if __name__ == '__main__':
    main()
