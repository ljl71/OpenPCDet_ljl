import numpy as np
import networkx as nx
from scipy.spatial import ConvexHull


def polygon_clip(subjectPolygon, clipPolygon):
    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList


def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convex_hull_intersection(p1, p2):
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def box3d_vol(corners):
    """corners: (8,3) no assumption on axis direction"""
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def is_clockwise(p):
    x = p[:, 0]
    y = p[:, 1]
    return np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)) > 0


def box3d_iou(corners1, corners2):
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]

    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])

    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])

    inter_vol = inter_area * max(0.0, ymax - ymin)

    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def get_3d_box(box_size, heading_angle, center):
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # pos x axis
        # return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        # return np.array([[c,0,-s], [0,1,0], [s,0,c]])

    R = roty(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def filter_point_cloud_to_bbox_3D_vectorized(
    bbox: np.ndarray, pc_raw: np.ndarray
) -> np.ndarray:

    # get 3 principal directions (edges) of the cuboid
    u = bbox[2] - bbox[6]
    v = bbox[2] - bbox[3]
    w = bbox[2] - bbox[1]

    # point x lies within the box when the following
    # constraints are respected

    # IN BETWEEN

    # do i need to check the other direction as well?
    valid_u1 = np.logical_and(
        u.dot(bbox[2]) <= pc_raw.dot(u), pc_raw.dot(u) <= u.dot(bbox[6])
    )
    valid_v1 = np.logical_and(
        v.dot(bbox[2]) <= pc_raw.dot(v), pc_raw.dot(v) <= v.dot(bbox[3])
    )
    valid_w1 = np.logical_and(
        w.dot(bbox[2]) <= pc_raw.dot(w), pc_raw.dot(w) <= w.dot(bbox[1])
    )

    valid_u2 = np.logical_and(
        u.dot(bbox[2]) >= pc_raw.dot(u), pc_raw.dot(u) >= u.dot(bbox[6])
    )
    valid_v2 = np.logical_and(
        v.dot(bbox[2]) >= pc_raw.dot(v), pc_raw.dot(v) >= v.dot(bbox[3])
    )
    valid_w2 = np.logical_and(
        w.dot(bbox[2]) >= pc_raw.dot(w), pc_raw.dot(w) >= w.dot(bbox[1])
    )

    valid_u = np.logical_or(valid_u1, valid_u2)
    valid_v = np.logical_or(valid_v1, valid_v2)
    valid_w = np.logical_or(valid_w1, valid_w2)

    is_valid = np.logical_and(np.logical_and(valid_u, valid_v), valid_w)
    # segment_pc = pc_raw[is_valid]
    # return segment_pc, is_valid
    # return is_valid
    return np.sum(is_valid != 0)


def nms(corners, count, volume, labels):
    indexCount = 0
    delList = []
    for item in corners:
        compare = []
        for bbox1 in range(0, len(item)):
            # if indexCount + bbox1 in list(_flatten(compare)):
            #     continue
            tempCompare = []
            for bbox2 in range(bbox1 + 1, len(item)):
                # if box3d_iou(np.array(item[bbox1]), np.array(item[bbox2])) > 0.3:
                # print(item[bbox1],type(item[bbox1]))
                # tempNum = box3d_iou(item[bbox1], item[bbox2])
                if box3d_iou(item[bbox1], item[bbox2])[0] > 0.1:
                    tempCompare.extend([indexCount + bbox1, indexCount + bbox2])
            compare.append(list(set(tempCompare)))
        indexCount += len(item)

        compare = [x for x in compare if x != []]

        G = nx.Graph()
        G.add_nodes_from(sum(compare, []))
        q = [[(s[i], s[i + 1]) for i in range(len(s) - 1)] for s in compare]
        for i in q:
            G.add_edges_from(i)
        compare = [list(i) for i in nx.connected_components(G)]

        for ele in compare:

            eleCount = [count[i] for i in ele]
            # 找到点数最多的框的索引
            maxCount = ele[eleCount.index(max(eleCount))]

            eleVolumeCount = [count[i] / volume[i] for i in ele]
            # 找到单位体积内点最多的索引
            maxVolumeCount = ele[eleVolumeCount.index(max(eleVolumeCount))]

            del ele[eleCount.index(max(eleCount))]
            delList.extend(ele)

    #     allCompare.append(compare)
    delList.sort(reverse=True)
    for i in delList:
        del labels[0][i]

    return labels
