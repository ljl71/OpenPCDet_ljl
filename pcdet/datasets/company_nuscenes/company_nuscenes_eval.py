import math

import numpy as np


DEFAULT_DISTANCE_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_TP_DISTANCE_THRESHOLD = 2.0
DEFAULT_DISTANCE_RANGES = ((0.0, 30.0), (30.0, 50.0), (50.0, float('inf')))
ORIENTATION_PERIOD_BY_CLASS = {'movable_object_barrier': np.pi}
ORIENTATION_IGNORED_CLASSES = {'movable_object_trafficcone'}


def build_ground_truth_annos(infos, class_names, min_lidar_points=1):
    """Build filtered evaluation annotations from CompanyNuScenes info records."""
    class_names = set(class_names)
    gt_annos = []
    stats = {
        'raw_gt_boxes': 0,
        'evaluated_gt_boxes': 0,
        'filtered_min_points_gt_boxes': 0,
        'ignored_unknown_class_gt_boxes': 0,
        'retained_unknown_point_count_gt_boxes': 0,
    }

    for info in infos:
        names = np.asarray(info.get('gt_names', []))
        boxes = _boxes_from_anno(info, 'gt_boxes')
        if len(names) != len(boxes):
            raise ValueError('The number of GT names does not match gt_boxes.')

        class_mask = np.asarray([name in class_names for name in names], dtype=bool)
        point_mask = np.ones(len(names), dtype=bool)
        point_counts = info.get('num_lidar_pts')
        if min_lidar_points and point_counts is not None:
            if len(point_counts) != len(names):
                raise ValueError('The number of num_lidar_pts values does not match GT boxes.')
            point_mask = np.asarray([
                _point_count_is_unknown(count) or count >= min_lidar_points
                for count in point_counts
            ], dtype=bool)
            stats['retained_unknown_point_count_gt_boxes'] += int(sum(
                class_mask[idx] and _point_count_is_unknown(count)
                for idx, count in enumerate(point_counts)
            ))

        retained_mask = class_mask & point_mask
        stats['raw_gt_boxes'] += int(len(names))
        stats['evaluated_gt_boxes'] += int(retained_mask.sum())
        stats['filtered_min_points_gt_boxes'] += int((class_mask & ~point_mask).sum())
        stats['ignored_unknown_class_gt_boxes'] += int((~class_mask).sum())
        gt_annos.append({
            'name': names[retained_mask],
            'boxes_lidar': boxes[retained_mask],
        })

    return gt_annos, stats


def validate_prediction_alignment(det_annos, infos):
    if len(det_annos) != len(infos):
        raise ValueError(
            f'Prediction count ({len(det_annos)}) does not match info count ({len(infos)}).'
        )
    for idx, (det_anno, info) in enumerate(zip(det_annos, infos)):
        token = det_anno.get('metadata', {}).get('token')
        if token is not None and token != info.get('token'):
            raise ValueError(f'Prediction/info token mismatch at sample {idx}: {token} != {info.get("token")}.')


def evaluate_company_predictions(
        gt_annos,
        det_annos,
        class_names,
        distance_thresholds=DEFAULT_DISTANCE_THRESHOLDS,
        tp_distance_threshold=DEFAULT_TP_DISTANCE_THRESHOLD,
        distance_ranges=DEFAULT_DISTANCE_RANGES,
        min_score=None,
        gt_stats=None):
    """Evaluate custom classes with nuScenes-style XY center-distance matching."""
    if len(gt_annos) != len(det_annos):
        raise ValueError('The number of ground-truth samples must match predictions.')

    distance_thresholds = tuple(float(x) for x in distance_thresholds)
    if not distance_thresholds or any(x <= 0 for x in distance_thresholds):
        raise ValueError('distance_thresholds must contain positive values.')
    tp_distance_threshold = float(tp_distance_threshold)
    if tp_distance_threshold <= 0:
        raise ValueError('tp_distance_threshold must be positive.')

    threshold_labels = [_threshold_label(x) for x in distance_thresholds]
    class_metrics = {}
    ap_values = {label: [] for label in threshold_labels}
    operating_totals = {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'pred': 0}
    operating_pairs = []
    present_classes = []

    for class_name in class_names:
        prepared = _prepare_class_data(
            gt_annos, det_annos, class_name, min_score=min_score
        )
        per_threshold = {}
        for distance_threshold, label in zip(distance_thresholds, threshold_labels):
            matched = _match_prepared(*prepared, distance_threshold)
            per_threshold[label] = matched['ap']
            if matched['num_gt']:
                ap_values[label].append(matched['ap'])

        operating = _match_prepared(
            *prepared, tp_distance_threshold, collect_pairs=True, pair_class_name=class_name
        )
        if operating['num_gt']:
            present_classes.append(class_name)
        for key in operating_totals:
            operating_totals[key] += operating[key if key not in ('gt', 'pred') else f'num_{key}']
        operating_pairs.extend(operating.pop('pairs'))

        class_metrics[class_name] = {
            'gt': operating['num_gt'],
            'pred': operating['num_pred'],
            'ap': per_threshold,
            'mAP': _mean_or_none([value for value in per_threshold.values() if value is not None]),
            'precision': _ratio_or_none(operating['tp'], operating['tp'] + operating['fp']),
            'recall': _ratio_or_none(operating['tp'], operating['tp'] + operating['fn']),
            'tp': operating['tp'],
            'fp': operating['fp'],
            'fn': operating['fn'],
        }

    ap_by_threshold = {
        label: _mean_or_none(values) for label, values in ap_values.items()
    }
    result = {
        'metric_definition': 'XY-center-distance AP with class-matched greedy assignment',
        'distance_thresholds': list(distance_thresholds),
        'tp_distance_threshold': tp_distance_threshold,
        'min_score': min_score,
        'present_classes': present_classes,
        'num_present_classes': len(present_classes),
        'class_metrics': class_metrics,
        'AP_by_distance': ap_by_threshold,
        'mAP': _mean_or_none([value for value in ap_by_threshold.values() if value is not None]),
        'operating': _operating_metrics(operating_totals),
        'tp_quality': _quality_metrics(operating_pairs),
        'distance_ranges': [],
        'gt_stats': gt_stats or {},
        'official_NDS': None,
        'official_NDS_reason': (
            'Not available: custom 26-class taxonomy and no velocity/attribute predictions.'
        ),
    }

    for lower, upper in distance_ranges:
        result['distance_ranges'].append(_evaluate_distance_range(
            gt_annos=gt_annos,
            det_annos=det_annos,
            class_names=class_names,
            distance_thresholds=distance_thresholds,
            tp_distance_threshold=tp_distance_threshold,
            lower=float(lower),
            upper=float(upper),
            min_score=min_score,
        ))
    return result


def metrics_to_log_dict(metrics):
    details = {
        'company/mAP': _number_or_zero(metrics['mAP']),
        'company/evaluated_classes': metrics['num_present_classes'],
    }
    for key, value in metrics.get('gt_stats', {}).items():
        details[f'company/{key}'] = value
    for label, value in metrics['AP_by_distance'].items():
        details[f'company/mAP@{label}'] = _number_or_zero(value)

    operating = metrics['operating']
    tp_distance = _threshold_label(metrics['tp_distance_threshold'])
    for key in ('precision', 'recall', 'f1'):
        details[f'company/{key}@{tp_distance}'] = _number_or_zero(operating[key])
    for key in ('tp', 'fp', 'fn', 'gt', 'pred'):
        details[f'company/{key}@{tp_distance}'] = operating[key]
    for key, value in metrics['tp_quality'].items():
        details[f'company/{key}@{tp_distance}'] = _number_or_zero(value)

    for class_name, values in metrics['class_metrics'].items():
        details[f'company/{class_name}_gt'] = values['gt']
        details[f'company/{class_name}_pred'] = values['pred']
        if values['mAP'] is not None:
            details[f'company/{class_name}_mAP'] = values['mAP']
            for label, value in values['ap'].items():
                details[f'company/{class_name}_AP@{label}'] = value
    for distance_range in metrics['distance_ranges']:
        details[f'company/distance_{distance_range["label"]}_mAP'] = _number_or_zero(distance_range['mAP'])
    return details


def format_company_results(metrics):
    thresholds = metrics['distance_thresholds']
    labels = [_threshold_label(value) for value in thresholds]
    tp_label = _threshold_label(metrics['tp_distance_threshold'])
    operating = metrics['operating']
    quality = metrics['tp_quality']
    stats = metrics.get('gt_stats', {})

    lines = ['---------------CompanyNuScenes distance evaluation---------------']
    lines.append('Metric: class-matched XY-center distance AP (nuScenes-style matching; custom taxonomy).')
    lines.append('This is not official nuScenes AP/NDS: velocity and attribute outputs are unavailable.')
    if stats:
        lines.append(
            'GT retained for evaluation: %d (removed by min-points rule: %d, retained with unknown point count: %d)'
            % (
                stats.get('evaluated_gt_boxes', 0),
                stats.get('filtered_min_points_gt_boxes', 0),
                stats.get('retained_unknown_point_count_gt_boxes', 0),
            )
        )
    lines.append(
        'mAP across %d classes with validation GT and distances [%s]: %s'
        % (
            metrics['num_present_classes'],
            ', '.join(labels),
            _display(metrics['mAP']),
        )
    )
    lines.append(
        'Emitted-prediction P/R/F1 @%s: %s / %s / %s (TP=%d, FP=%d, FN=%d)'
        % (
            tp_label,
            _display(operating['precision']),
            _display(operating['recall']),
            _display(operating['f1']),
            operating['tp'],
            operating['fp'],
            operating['fn'],
        )
    )
    lines.append(
        'Matched TP quality @%s: mATE=%s mASE=%s mAOE=%s rad (cone yaw ignored; barrier yaw pi-periodic)'
        % (
            tp_label,
            _display(quality['mATE']),
            _display(quality['mASE']),
            _display(quality['mAOE']),
        )
    )
    lines.append('')
    lines.append('class                                     GT    Pred  ' + '  '.join(
        f'AP@{label:>5}' for label in labels
    ) + '    mAP      P      R')
    for class_name, result in metrics['class_metrics'].items():
        ap_columns = '  '.join(f'{_display(result["ap"][label]):>8}' for label in labels)
        lines.append(
            f'{class_name:<38} {result["gt"]:>6} {result["pred"]:>7}  '
            f'{ap_columns}  {_display(result["mAP"]):>6} '
            f'{_display(result["precision"]):>6} {_display(result["recall"]):>6}'
        )
    lines.append('')
    lines.append('Distance breakdown (mAP across present GT classes in each range):')
    for distance_range in metrics['distance_ranges']:
        range_op = distance_range['operating']
        lines.append(
            f'  {distance_range["label"]:<12} GT={range_op["gt"]:>7} Pred={range_op["pred"]:>7} '
            f'mAP={_display(distance_range["mAP"])} '
            f'P@{tp_label}={_display(range_op["precision"])} '
            f'R@{tp_label}={_display(range_op["recall"])}'
        )
    return '\n'.join(lines) + '\n'


def _evaluate_distance_range(
        gt_annos, det_annos, class_names, distance_thresholds, tp_distance_threshold,
        lower, upper, min_score):
    distance_range = (lower, upper)
    ap_values = []
    totals = {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'pred': 0}
    for class_name in class_names:
        prepared = _prepare_class_data(
            gt_annos, det_annos, class_name,
            distance_range=distance_range, min_score=min_score
        )
        class_aps = []
        for threshold in distance_thresholds:
            matched = _match_prepared(*prepared, threshold)
            if matched['num_gt']:
                class_aps.append(matched['ap'])
        if class_aps:
            ap_values.append(float(np.mean(class_aps)))
        operating = _match_prepared(*prepared, tp_distance_threshold)
        totals['tp'] += operating['tp']
        totals['fp'] += operating['fp']
        totals['fn'] += operating['fn']
        totals['gt'] += operating['num_gt']
        totals['pred'] += operating['num_pred']
    return {
        'label': _range_label(lower, upper),
        'lower': lower,
        'upper': None if math.isinf(upper) else upper,
        'mAP': _mean_or_none(ap_values),
        'operating': _operating_metrics(totals),
    }


def _prepare_class_data(gt_annos, det_annos, class_name, distance_range=None, min_score=None):
    gt_by_sample = []
    predictions = []
    for sample_idx, (gt_anno, det_anno) in enumerate(zip(gt_annos, det_annos)):
        gt_boxes = _boxes_from_anno(gt_anno, 'boxes_lidar')
        gt_names = np.asarray(gt_anno.get('name', []))
        gt_mask = (gt_names == class_name) & _distance_mask(gt_boxes, distance_range)
        gt_by_sample.append(gt_boxes[gt_mask])

        det_boxes = _boxes_from_anno(det_anno, 'boxes_lidar')
        det_names = np.asarray(det_anno.get('name', []))
        scores = np.asarray(det_anno.get('score', []), dtype=np.float64)
        if len(det_names) != len(det_boxes) or len(scores) != len(det_boxes):
            raise ValueError('The number of prediction names, boxes and scores must match.')
        det_mask = (det_names == class_name) & _distance_mask(det_boxes, distance_range)
        if min_score is not None:
            det_mask &= scores >= min_score
        for box, score in zip(det_boxes[det_mask], scores[det_mask]):
            predictions.append((float(score), sample_idx, box))

    predictions.sort(key=lambda prediction: prediction[0], reverse=True)
    return gt_by_sample, predictions


def _match_prepared(
        gt_by_sample, predictions, distance_threshold, collect_pairs=False, pair_class_name=None):
    matched_gt = [np.zeros(len(boxes), dtype=bool) for boxes in gt_by_sample]
    tp = np.zeros(len(predictions), dtype=np.float64)
    fp = np.zeros(len(predictions), dtype=np.float64)
    pairs = []
    for idx, (_, sample_idx, pred_box) in enumerate(predictions):
        gt_boxes = gt_by_sample[sample_idx]
        if len(gt_boxes) == 0:
            fp[idx] = 1.0
            continue
        distances = np.linalg.norm(gt_boxes[:, :2] - pred_box[None, :2], axis=1)
        distances[matched_gt[sample_idx]] = np.inf
        match_idx = int(np.argmin(distances))
        if distances[match_idx] <= distance_threshold:
            tp[idx] = 1.0
            matched_gt[sample_idx][match_idx] = True
            if collect_pairs:
                pairs.append((pred_box[:7], gt_boxes[match_idx, :7], pair_class_name))
        else:
            fp[idx] = 1.0

    num_gt = int(sum(len(boxes) for boxes in gt_by_sample))
    num_tp = int(tp.sum())
    num_fp = int(fp.sum())
    return {
        'ap': _average_precision(tp, fp, num_gt),
        'num_gt': num_gt,
        'num_pred': len(predictions),
        'tp': num_tp,
        'fp': num_fp,
        'fn': num_gt - num_tp,
        'pairs': pairs,
    }


def _quality_metrics(pairs):
    if not pairs:
        return {'mATE': None, 'mASE': None, 'mAOE': None}
    pred_boxes = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    gt_boxes = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    translation_error = np.linalg.norm(pred_boxes[:, :2] - gt_boxes[:, :2], axis=1)

    pred_dims = np.maximum(pred_boxes[:, 3:6], 1e-8)
    gt_dims = np.maximum(gt_boxes[:, 3:6], 1e-8)
    intersection = np.prod(np.minimum(pred_dims, gt_dims), axis=1)
    union = np.prod(pred_dims, axis=1) + np.prod(gt_dims, axis=1) - intersection
    scale_error = 1.0 - intersection / np.maximum(union, 1e-8)

    orientation_error = []
    for pred_box, gt_box, class_name in pairs:
        if class_name in ORIENTATION_IGNORED_CLASSES:
            continue
        period = ORIENTATION_PERIOD_BY_CLASS.get(class_name, 2 * np.pi)
        yaw_delta = (pred_box[6] - gt_box[6] + period / 2) % period - period / 2
        orientation_error.append(abs(yaw_delta))
    return {
        'mATE': float(np.mean(translation_error)),
        'mASE': float(np.mean(scale_error)),
        'mAOE': _mean_or_none(orientation_error),
    }


def _operating_metrics(totals):
    precision = _ratio_or_none(totals['tp'], totals['tp'] + totals['fp'])
    recall = _ratio_or_none(totals['tp'], totals['tp'] + totals['fn'])
    f1 = None
    if precision is not None and recall is not None and precision + recall:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        **totals,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def _average_precision(tp, fp, num_gt):
    if num_gt == 0:
        return None
    if len(tp) == 0:
        return 0.0
    recall = np.cumsum(tp) / float(num_gt)
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-12)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for idx in range(len(precision) - 1, 0, -1):
        precision[idx - 1] = max(precision[idx - 1], precision[idx])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def _boxes_from_anno(anno, key):
    boxes = np.asarray(anno.get(key, []), dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((0, 7), dtype=np.float64)
    boxes = boxes.reshape((-1, boxes.shape[-1]))
    if boxes.shape[1] < 7:
        raise ValueError(f'{key} must contain at least seven box values.')
    return boxes[:, :7]


def _distance_mask(boxes, distance_range):
    if distance_range is None:
        return np.ones(len(boxes), dtype=bool)
    distance = np.linalg.norm(boxes[:, :2], axis=1)
    lower, upper = distance_range
    return (distance >= lower) & (distance < upper)


def _point_count_is_unknown(value):
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))


def _ratio_or_none(numerator, denominator):
    return None if denominator == 0 else float(numerator) / float(denominator)


def _mean_or_none(values):
    return None if not values else float(np.mean(values))


def _number_or_zero(value):
    return 0.0 if value is None else value


def _threshold_label(value):
    return f'{float(value):.2f}m'


def _range_label(lower, upper):
    return f'{lower:g}m-inf' if math.isinf(upper) else f'{lower:g}m-{upper:g}m'


def _display(value):
    return 'n/a' if value is None else f'{value:.4f}'
