import argparse
from pathlib import Path

import numpy as np


def read_pcd_header(path):
    header = {}
    with open(path, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f'PCD DATA header not found: {path}')
            decoded = line.decode('ascii', errors='replace').strip()
            if not decoded or decoded.startswith('#'):
                continue
            key, *values = decoded.split()
            header[key.upper()] = values
            if key.upper() == 'DATA':
                break
    return header


def inspect_pair(pcd_path, bin_path):
    header = read_pcd_header(pcd_path)
    point_count = int(header['POINTS'][0])
    fields = header.get('FIELDS', [])
    raw = np.fromfile(str(bin_path), dtype=np.float32)
    if point_count == 0 or raw.size % point_count != 0:
        print(f'{pcd_path.name}: cannot align {raw.size} float32 values with {point_count} PCD points')
        return False

    point_dim = raw.size // point_count
    print(f'{pcd_path.name}: PCD fields={fields}, points={point_count}, bin_dim={point_dim}')
    if point_dim == 4:
        fourth = raw.reshape(-1, 4)[:, 3]
        print(
            '  bin fourth column: '
            f'min={fourth.min():.6g}, max={fourth.max():.6g}, mean={fourth.mean():.6g}'
        )
    return fields == ['x', 'y', 'z'] and point_dim == 4


def main():
    parser = argparse.ArgumentParser(description='Compare source company PCD files with converted LiDAR bins.')
    parser.add_argument('--pcd_dir', type=Path, required=True)
    parser.add_argument('--bin_dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=3)
    args = parser.parse_args()

    checked = 0
    expected_format = True
    for pcd_path in sorted(args.pcd_dir.glob('*.pcd')):
        bin_path = args.bin_dir / f'{pcd_path.stem}.bin'
        if not bin_path.exists():
            continue
        expected_format = inspect_pair(pcd_path, bin_path) and expected_format
        checked += 1
        if checked >= args.limit:
            break

    if not checked:
        raise SystemExit('No matching .pcd/.bin pairs were found.')
    if not expected_format:
        raise SystemExit('One or more pairs do not match the expected XYZ PCD -> four-float bin conversion.')


if __name__ == '__main__':
    main()
