from pathlib import Path
import argparse
import pickle
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def get_pred_boxes(pred):
    for k in ["boxes_lidar", "pred_boxes", "box3d_lidar"]:
        if k in pred:
            return to_numpy(pred[k])
    return np.zeros((0, 7), dtype=np.float32)


def get_pred_scores(pred):
    for k in ["score", "scores", "pred_scores"]:
        if k in pred:
            return to_numpy(pred[k]).reshape(-1)
    boxes = get_pred_boxes(pred)
    return np.ones((len(boxes),), dtype=np.float32)


def get_pred_names(pred):
    if "name" in pred:
        return np.asarray(pred["name"]).astype(str)
    boxes = get_pred_boxes(pred)
    return np.array(["pred"] * len(boxes))


def get_gt_boxes(info):
    if "gt_boxes" in info:
        return to_numpy(info["gt_boxes"])
    if "annos" in info and isinstance(info["annos"], dict):
        annos = info["annos"]
        for k in ["gt_boxes_lidar", "boxes_lidar", "gt_boxes"]:
            if k in annos:
                return to_numpy(annos[k])
    return np.zeros((0, 7), dtype=np.float32)


def get_gt_names(info):
    if "gt_names" in info:
        return np.asarray(info["gt_names"]).astype(str)
    if "annos" in info and isinstance(info["annos"], dict):
        annos = info["annos"]
        for k in ["name", "gt_names"]:
            if k in annos:
                return np.asarray(annos[k]).astype(str)
    boxes = get_gt_boxes(info)
    return np.array(["gt"] * len(boxes))


def get_lidar_path(info, data_root):
    candidates = []
    for k in ["lidar_path", "lidar_file", "point_cloud_path"]:
        if k in info:
            candidates.append(info[k])

    if "point_cloud" in info and isinstance(info["point_cloud"], dict):
        pc = info["point_cloud"]
        for k in ["lidar_path", "lidar_file", "velodyne_path"]:
            if k in pc:
                candidates.append(pc[k])

    for c in candidates:
        if c is None:
            continue
        p = Path(str(c))
        if p.is_absolute() and p.exists():
            return p
        p1 = Path(data_root) / p
        if p1.exists():
            return p1
        p2 = Path("/workspace/OpenPCDet") / p
        if p2.exists():
            return p2

    return None


def read_points(path):
    if path is None or not Path(path).exists():
        return None
    arr = np.fromfile(str(path), dtype=np.float32)
    if arr.size % 4 == 0:
        return arr.reshape(-1, 4)
    if arr.size % 5 == 0:
        return arr.reshape(-1, 5)
    return None


def box_corners_bev(box):
    x, y, z, dx, dy, dz, yaw = box[:7]
    local = np.array([
        [ dx / 2,  dy / 2],
        [ dx / 2, -dy / 2],
        [-dx / 2, -dy / 2],
        [-dx / 2,  dy / 2],
        [ dx / 2,  dy / 2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def draw_boxes(ax, boxes, kind="pred"):
    if boxes is None or len(boxes) == 0:
        return

    for box in boxes:
        if len(box) < 7:
            continue
        corners = box_corners_bev(box)
        if kind == "gt":
            ax.plot(corners[:, 0], corners[:, 1], linewidth=1.2, color="lime")
        else:
            ax.plot(corners[:, 0], corners[:, 1], linewidth=1.0, color="red")

        x, y, _, dx, _, _, yaw = box[:7]
        ax.plot(
            [x, x + np.cos(yaw) * dx / 2],
            [y, y + np.sin(yaw) * dx / 2],
            linewidth=0.8,
            color="yellow" if kind == "pred" else "cyan",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--infos", required=True)
    parser.add_argument("--data_root", default="data/nuscenes")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--score_thresh", type=float, default=0.2)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--range", type=float, default=55.0)
    args = parser.parse_args()

    preds = pickle.load(open(args.result, "rb"))
    infos = pickle.load(open(args.infos, "rb"))

    if isinstance(infos, dict) and "infos" in infos:
        infos = infos["infos"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert len(preds) == len(infos), f"pred len {len(preds)} != info len {len(infos)}"

    if args.indices:
        indices = args.indices
    else:
        indices = list(range(min(args.num, len(preds))))

    lines = []
    for idx in indices:
        pred = preds[idx]
        info = infos[idx]

        pred_boxes = get_pred_boxes(pred)
        pred_scores = get_pred_scores(pred)
        keep = pred_scores >= args.score_thresh
        pred_boxes = pred_boxes[keep]

        gt_boxes = get_gt_boxes(info)

        lidar_path = get_lidar_path(info, args.data_root)
        points = read_points(lidar_path)

        fig, ax = plt.subplots(figsize=(10, 10))

        if points is not None and len(points) > 0:
            pts = points[:, :2]
            r = args.range
            mask = (
                (pts[:, 0] >= -r) & (pts[:, 0] <= r) &
                (pts[:, 1] >= -r) & (pts[:, 1] <= r)
            )
            pts = pts[mask]
            if len(pts) > 100000:
                sel = np.random.choice(len(pts), 100000, replace=False)
                pts = pts[sel]
            ax.scatter(pts[:, 0], pts[:, 1], s=0.1, c="gray", alpha=0.35)

        draw_boxes(ax, gt_boxes, kind="gt")
        draw_boxes(ax, pred_boxes, kind="pred")

        r = args.range
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3)
        ax.set_title(
            f"idx={idx} | GT={len(gt_boxes)} | Pred(score>={args.score_thresh})={len(pred_boxes)}\n"
            f"green=GT, red=Pred"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        out_path = out_dir / f"bev_idx_{idx:06d}_gt{len(gt_boxes)}_pred{len(pred_boxes)}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        frame_id = info.get("frame_id", info.get("token", idx))
        lines.append(f"{idx}\t{frame_id}\tgt={len(gt_boxes)}\tpred={len(pred_boxes)}\t{out_path}")

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {len(indices)} BEV visualizations to: {out_dir}")
    print(f"Summary: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
