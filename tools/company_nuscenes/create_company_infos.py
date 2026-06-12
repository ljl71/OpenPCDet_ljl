import argparse
import importlib.util
from pathlib import Path


def load_utils(repo_root):
    utils_path = repo_root / 'pcdet' / 'datasets' / 'company_nuscenes' / 'company_nuscenes_utils.py'
    spec = importlib.util.spec_from_file_location('company_nuscenes_utils', utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Create CompanyNuScenes info files without importing torch/pcdet.')
    parser.add_argument('--data_path', type=Path, default=repo_root / 'data' / 'NuScenes-develop_t23_2026')
    parser.add_argument('--save_path', type=Path, default=repo_root / 'data' / 'NuScenes-develop_t23_2026')
    parser.add_argument('--version', type=str, default='v1.0-develop')
    parser.add_argument('--max_sweeps', type=int, default=1)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--min_lidar_points', type=int, default=1)
    args = parser.parse_args()

    utils = load_utils(repo_root)
    utils.create_company_nuscenes_infos(
        version=args.version,
        data_path=args.data_path,
        save_path=args.save_path,
        max_sweeps=args.max_sweeps,
        train_ratio=args.train_ratio,
        seed=args.seed,
        min_lidar_points=args.min_lidar_points,
    )


if __name__ == '__main__':
    main()
