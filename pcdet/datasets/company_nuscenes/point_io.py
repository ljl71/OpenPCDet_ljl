from pathlib import Path

import numpy as np


DEFAULT_COMPANY_POINT_FIELDS = ['x', 'y', 'z', 'intensity', 'ring', 'timestamp']
FIELD_ALIASES = {
    'timestamp': ['timestamp', 'time', 't'],
    'time': ['time', 'timestamp', 't'],
}


def get_src_feature_list(dataset_cfg):
    return list(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)


def get_lidar_point_dim(dataset_cfg, default=None):
    if default is None:
        default = len(get_src_feature_list(dataset_cfg))
    return int(dataset_cfg.get('LIDAR_POINT_DIM', default))


def get_lidar_point_fields(dataset_cfg):
    fields = dataset_cfg.get('LIDAR_POINT_FIELDS', None)
    if fields is not None:
        return list(fields)

    point_dim = get_lidar_point_dim(dataset_cfg)
    if point_dim == len(DEFAULT_COMPANY_POINT_FIELDS):
        return DEFAULT_COMPANY_POINT_FIELDS[:]
    return get_src_feature_list(dataset_cfg)[:point_dim]


def _candidate_names(name):
    return [name] + FIELD_ALIASES.get(name, [])


def align_points_to_features(points, point_fields, src_feature_list):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f'Expected point array with shape (N, C), got {points.shape}')

    field_to_idx = {name: idx for idx, name in enumerate(point_fields)}
    output = np.zeros((points.shape[0], len(src_feature_list)), dtype=np.float32)
    for out_idx, feature_name in enumerate(src_feature_list):
        for candidate in _candidate_names(feature_name):
            if candidate in field_to_idx:
                output[:, out_idx] = points[:, field_to_idx[candidate]]
                break
    return output


def _struct_array_to_points(arr, src_feature_list):
    output = np.zeros((arr.shape[0], len(src_feature_list)), dtype=np.float32)
    names = set(arr.dtype.names or [])
    for out_idx, feature_name in enumerate(src_feature_list):
        for candidate in _candidate_names(feature_name):
            if candidate in names:
                output[:, out_idx] = arr[candidate].astype(np.float32)
                break
    return output


def _pcl_xyzirt_dtypes():
    packed_dtype = np.dtype([
        ('x', '<f4'),
        ('y', '<f4'),
        ('z', '<f4'),
        ('intensity', '<f4'),
        ('ring', '<u2'),
        ('timestamp', '<f4'),
    ], align=False)
    aligned_dtype = np.dtype({
        'names': ['x', 'y', 'z', 'intensity', 'ring', 'timestamp'],
        'formats': ['<f4', '<f4', '<f4', '<f4', '<u2', '<f4'],
        'offsets': [0, 4, 8, 12, 16, 20],
        'itemsize': 24,
    })
    return packed_dtype, aligned_dtype


def _ring_column_looks_float32(ring_values):
    if ring_values.size == 0:
        return False
    sample = ring_values[:min(ring_values.shape[0], 4096)]
    if not np.isfinite(sample).all():
        return False
    sample_min = sample.min()
    sample_max = sample.max()
    if sample_max <= 0.5:
        return False
    if sample_min < 0.0 or sample_max > 65535.0:
        return False
    return np.mean(np.abs(sample - np.round(sample)) < 1e-3) > 0.9


def _score_xyzirt_points(points):
    if points.shape[0] == 0:
        return -1.0
    sample = points[:min(points.shape[0], 4096)]
    finite_mask = np.isfinite(sample)
    if not finite_mask.all():
        return float(finite_mask.mean())

    xyz_abs = np.abs(sample[:, :3])
    intensity = sample[:, 3]
    ring = sample[:, 4]
    timestamp = sample[:, 5]

    score = 0.0
    score += float(np.mean(xyz_abs < 1e6)) * 3.0
    score += float(np.mean(np.abs(intensity) < 1e6))
    score += float(np.mean((ring >= 0.0) & (ring <= 65535.0))) * 2.0
    score += float(np.mean(np.abs(ring - np.round(ring)) < 1e-3)) * 2.0
    score += float(np.mean(np.abs(timestamp) < 1e6))
    if np.unique(ring[:min(ring.shape[0], 512)]).size > 1:
        score += 0.25
    return score


def _read_xyzirt_binary(path, src_feature_list):
    byte_count = Path(path).stat().st_size
    packed_dtype, aligned_dtype = _pcl_xyzirt_dtypes()
    candidates = []

    if byte_count % packed_dtype.itemsize == 0:
        arr = np.fromfile(str(path), dtype=packed_dtype)
        candidates.append(_struct_array_to_points(arr, DEFAULT_COMPANY_POINT_FIELDS))

    if byte_count % aligned_dtype.itemsize == 0:
        arr = np.fromfile(str(path), dtype=aligned_dtype)
        candidates.append(_struct_array_to_points(arr, DEFAULT_COMPANY_POINT_FIELDS))

        float_points = np.fromfile(str(path), dtype=np.float32).reshape(-1, len(DEFAULT_COMPANY_POINT_FIELDS))
        if _ring_column_looks_float32(float_points[:, DEFAULT_COMPANY_POINT_FIELDS.index('ring')]):
            candidates.append(float_points.astype(np.float32, copy=False))

    if candidates:
        points = max(candidates, key=_score_xyzirt_points)
        return align_points_to_features(points, DEFAULT_COMPANY_POINT_FIELDS, src_feature_list)

    legacy_float32_dim = 4
    if byte_count % (legacy_float32_dim * np.dtype(np.float32).itemsize) == 0:
        points = np.fromfile(str(path), dtype=np.float32).reshape(-1, legacy_float32_dim)
        return align_points_to_features(
            points, DEFAULT_COMPANY_POINT_FIELDS[:legacy_float32_dim], src_feature_list
        )

    raise ValueError(
        f'Cannot read {path} as x/y/z/intensity/ring/timestamp binary. '
        f'File has {byte_count} bytes, expected a multiple of 22 packed bytes, '
        f'24 aligned bytes, or 16 legacy float32 bytes per point.'
    )


def read_lidar_bin(path, dataset_cfg, fallback_dim=None):
    src_feature_list = get_src_feature_list(dataset_cfg)
    point_format = str(dataset_cfg.get('LIDAR_POINT_FORMAT', 'float32')).lower()

    if point_format in ['xyzirt', 'pcl_xyzirt', 'pointxyzirt']:
        return _read_xyzirt_binary(path, src_feature_list)

    point_dim = get_lidar_point_dim(dataset_cfg, default=fallback_dim)
    point_fields = get_lidar_point_fields(dataset_cfg)
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % point_dim != 0:
        raise ValueError(
            f'Cannot reshape {path} with {raw.size} float32 values to LIDAR_POINT_DIM={point_dim}'
        )

    points = raw.reshape(-1, point_dim)
    return align_points_to_features(points, point_fields, src_feature_list)


def _pcd_field_dtype(size, field_type):
    field_type = field_type.upper()
    if field_type == 'F':
        if size == 4:
            return '<f4'
        if size == 8:
            return '<f8'
    if field_type == 'U':
        return f'<u{size}'
    if field_type == 'I':
        return f'<i{size}'
    raise ValueError(f'Unsupported PCD field type: size={size}, type={field_type}')


def _pcd_fields_to_dtype(fields, sizes, types):
    return np.dtype([
        (field, _pcd_field_dtype(size, field_type))
        for field, size, field_type in zip(fields, sizes, types)
    ])


def _lzf_decompress(data, expected_size):
    """Decompress PCL PCD binary_compressed payloads without extra deps."""
    data = memoryview(data)
    out = bytearray(expected_size)
    in_pos = 0
    out_pos = 0

    while in_pos < len(data):
        ctrl = data[in_pos]
        in_pos += 1

        if ctrl < 32:
            length = ctrl + 1
            if in_pos + length > len(data) or out_pos + length > expected_size:
                raise ValueError('Invalid LZF literal block in PCD payload')
            out[out_pos:out_pos + length] = data[in_pos:in_pos + length]
            in_pos += length
            out_pos += length
            continue

        length = ctrl >> 5
        ref = out_pos - ((ctrl & 0x1f) << 8) - 1
        if length == 7:
            if in_pos >= len(data):
                raise ValueError('Invalid LZF length block in PCD payload')
            length += data[in_pos]
            in_pos += 1
        if in_pos >= len(data):
            raise ValueError('Invalid LZF reference block in PCD payload')
        ref -= data[in_pos]
        in_pos += 1
        length += 2

        if ref < 0 or out_pos + length > expected_size:
            raise ValueError('Invalid LZF back-reference in PCD payload')
        for _ in range(length):
            out[out_pos] = out[ref]
            out_pos += 1
            ref += 1

    if out_pos != expected_size:
        raise ValueError(f'LZF decompressed size mismatch: got {out_pos}, expected {expected_size}')
    return bytes(out)


def _decode_binary_compressed_pcd(payload, fields, sizes, types, counts, points):
    if len(payload) < 8:
        raise ValueError('Invalid binary_compressed PCD payload: missing size header')
    compressed_size = int(np.frombuffer(payload[:4], dtype='<u4', count=1)[0])
    uncompressed_size = int(np.frombuffer(payload[4:8], dtype='<u4', count=1)[0])
    compressed = payload[8:8 + compressed_size]
    if len(compressed) != compressed_size:
        raise ValueError(
            f'Invalid binary_compressed PCD payload: expected {compressed_size} bytes, got {len(compressed)}'
        )

    decompressed = _lzf_decompress(compressed, uncompressed_size)
    expected_size = points * sum(size * count for size, count in zip(sizes, counts))
    if uncompressed_size != expected_size:
        raise ValueError(
            f'Unexpected binary_compressed PCD size: header={uncompressed_size}, expected={expected_size}'
        )

    dtype = _pcd_fields_to_dtype(fields, sizes, types)
    arr = np.empty(points, dtype=dtype)
    offset = 0
    for field, size, field_type, count in zip(fields, sizes, types, counts):
        if count != 1:
            raise ValueError('PCD COUNT > 1 is not supported for binary_compressed payloads')
        field_size = size * points
        field_dtype = np.dtype(_pcd_field_dtype(size, field_type))
        arr[field] = np.frombuffer(decompressed[offset:offset + field_size], dtype=field_dtype, count=points)
        offset += field_size
    return arr


def read_binary_pcd(path, dataset_cfg):
    src_feature_list = get_src_feature_list(dataset_cfg)
    with open(path, 'rb') as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f'Invalid PCD file without DATA line: {path}')
            decoded = line.decode('ascii', errors='ignore').strip()
            header.append(decoded)
            if decoded.startswith('DATA'):
                break
        payload = f.read()

    fields = None
    sizes = None
    types = None
    counts = None
    points = None
    width = None
    height = None
    data_mode = None
    for line in header:
        tokens = line.split()
        if not tokens:
            continue
        key = tokens[0].upper()
        if key == 'FIELDS':
            fields = tokens[1:]
        elif key == 'SIZE':
            sizes = [int(x) for x in tokens[1:]]
        elif key == 'TYPE':
            types = tokens[1:]
        elif key == 'COUNT':
            counts = [int(x) for x in tokens[1:]]
        elif key == 'WIDTH':
            width = int(tokens[1])
        elif key == 'HEIGHT':
            height = int(tokens[1])
        elif key == 'POINTS':
            points = int(tokens[1])
        elif key == 'DATA':
            data_mode = tokens[1].lower()

    if points is None and width is not None:
        points = width * (height if height is not None else 1)

    if fields is None or points is None or data_mode is None:
        raise ValueError(f'Unsupported PCD header: {path}')
    if sizes is None:
        sizes = [4] * len(fields)
    if types is None:
        types = ['F'] * len(fields)
    if counts is None:
        counts = [1] * len(fields)
    if any(count != 1 for count in counts):
        raise ValueError(f'PCD COUNT > 1 is not supported for {path}')

    if data_mode == 'binary':
        dtype = _pcd_fields_to_dtype(fields, sizes, types)
        arr = np.frombuffer(payload, dtype=dtype, count=points)
    elif data_mode == 'binary_compressed':
        arr = _decode_binary_compressed_pcd(payload, fields, sizes, types, counts, points)
    elif data_mode == 'ascii':
        values = np.fromstring(payload.decode('ascii', errors='ignore'), sep=' ', dtype=np.float32)
        expected_values = points * len(fields)
        if values.size < expected_values:
            raise ValueError(
                f'Invalid ASCII PCD payload for {path}: got {values.size} values, expected {expected_values}'
            )
        return align_points_to_features(values[:expected_values].reshape(points, len(fields)), fields, src_feature_list)
    else:
        raise ValueError(f'Unsupported PCD DATA mode {data_mode!r}: {path}')

    return _struct_array_to_points(arr, src_feature_list)
