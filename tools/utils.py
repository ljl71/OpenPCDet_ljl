import glob
import json
import os
import re
import time
from multiprocessing.pool import ThreadPool

import requests
import urllib
import numpy as np
from scipy.spatial import ConvexHull
import networkx as nx
from tkinter import _flatten


def loadJson(path):
    with open(path, 'r', encoding='utf-8') as f:
        fileData = json.load(f)
    return fileData


def saveJson(path, infos):
    with open(path, 'w', encoding='utf8') as fp:
        fp.write(json.dumps(infos))
    return True


def findFileType(path, fileType):
    return glob.glob(os.path.join(path, '*.{}'.format(fileType)))


def dicSortNumber(dic):
    return dic.sort(key=lambda x: int(x.split('.')[0]))


def mkdir(filePath):
    try:
        if not os.path.exists(filePath):
            os.makedirs(filePath)
        return True
    except:
        return False


def downloadItem(item, filePath, retryTime=0):
    try:
        resp = requests.get(item['url'], timeout=5)
        if resp.status_code == 200:
            fileIndex = re.search('.aliyuncs.com/', item['url']).span()[-1]
            imgPath = item['url'][fileIndex:]
            # path = filePath + urllib.parse.quote(url.split('/')[-1])
            path = filePath + '/'.join(imgPath.split('/')[:-1]) + '/'
            mkdir(path)
            fp = open(path + urllib.parse.unquote(imgPath.split('/')[-1]), 'wb')
            fp.write(resp.content)
            fp.close()

            item['result'] = True
            item['link'] = '/'.join(imgPath.split('/')[:-1]) + '/' + urllib.parse.unquote(imgPath.split('/')[-1])

            return item
    except Exception as e:
        print(e)
        if retryTime < 3:
            retryTime += 1
            return downloadItem(item, filePath, retryTime)
        else:

            item['error'] = e
            item['result'] = False

            return item


def downloadItemList(itemList):
    tStamp = time.gmtime()
    theTime = time.strftime("%Y%m%d_%H%M%S", tStamp)
    filePath = '/home/molardata/dataset/data/{}/'.format(theTime)
    if not os.path.exists(filePath):
        os.makedirs(filePath)

    failList = []
    for item in itemList:
        if not downloadItem(item, filePath):
            failList.append(item)

    return failList


def downloadItemListThread(itemList, num_processes, Async=True):
    '''
    多线程下载图片
    :param itemList: image url list
    :param num_processes: 开启线程个数
    :param Async:是否异步
    :return: 返回图片的存储地址列表
    '''

    tStamp = time.gmtime()
    theTime = time.strftime("%Y%m%d_%H%M%S", tStamp)
    filePath = '/home/molardata/dataset/data/{}/'.format(theTime)

    mkdir(filePath)

    pool = ThreadPool(processes=num_processes)
    threadList = []
    for item in itemList:
        if Async:
            out = pool.apply_async(func=downloadItem, args=(item, filePath))  # 异步
        else:
            out = pool.apply(func=downloadItem, args=(item, filePath))  # 同步
        threadList.append(out)

    pool.close()
    pool.join()

    # 获取输出结果
    resultList = []
    if Async:
        for p in threadList:
            image = p.get()  # get会阻塞
            resultList.append(image)
    else:
        resultList = threadList

    successList = [i for i in resultList if i['result']]
    failList = [i for i in resultList if not i['result']]

    return successList, failList, theTime


def matchImageSize(infoList):
    sortedList = sorted(infoList, key=lambda e: (e.__getitem__('info').__getitem__('ImageHeight').__getitem__('value'),
                                                 e.__getitem__('info').__getitem__('ImageWidth').__getitem__('value')))
    sizeList = []
    eachList = [sortedList[0]]
    for i in range(1, len(sortedList)):
        if sortedList[i]['info']['ImageHeight']['value'] == sortedList[i - 1]['info']['ImageHeight']['value'] and \
                sortedList[i]['info']['ImageWidth']['value'] == sortedList[i - 1]['info']['ImageWidth']['value']:
            eachList.append(sortedList[i])
        else:
            sizeList.append(eachList)
            eachList = [sortedList[i]]
    sizeList.append(eachList)
    return sizeList


def matchImageDivisor(infoList, scope=64):
    for i in infoList:
        i['info']['ImageHeight']['value'] = int(i['info']['ImageHeight']['value'])
        i['info']['ImageWidth']['value'] = int(i['info']['ImageWidth']['value'])
        if i['info']['ImageHeight']['value'] % scope >= scope / 2:
            i['heightDivisor'] = int(int(i['info']['ImageHeight']['value'] / scope + 1) * scope)
        else:
            i['heightDivisor'] = int(int(i['info']['ImageHeight']['value'] / scope) * scope)

        if i['info']['ImageWidth']['value'] % scope >= scope / 2:
            i['widthDivisor'] = int(int(i['info']['ImageWidth']['value'] / scope + 1) * scope)
        else:
            i['widthDivisor'] = int(int(i['info']['ImageWidth']['value'] / scope) * scope)

    sortedList = sorted(infoList, key=lambda e: (e.__getitem__('heightDivisor'),
                                                e.__getitem__('widthDivisor')))

    sizeList = []
    eachList = [sortedList[0]]
    for i in range(1, len(sortedList)):
        if sortedList[i]['heightDivisor'] == sortedList[i - 1]['heightDivisor'] and \
                sortedList[i]['widthDivisor'] == sortedList[i - 1]['widthDivisor']:
            eachList.append(sortedList[i])
        else:
            sizeList.append(eachList)
            eachList = [sortedList[i]]
    sizeList.append(eachList)
    return sizeList

def polygon_clip(subjectPolygon, clipPolygon):
    """Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**
    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

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
    """Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convex_hull_intersection(p1, p2):
    """Compute area of two convex hull's intersection area.
    p1,p2 are a list of (x,y) tuples of hull vertices.
    return a list of (x,y) for the intersection and its volume
    """
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
    """Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (kent): add more description on corner points' orders.
    """
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
    """Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    """

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
    """
    Args:
       bbox: Numpy array pf shape (8,3) representing 3d cuboid vertices, ordered
                as shown below.
       pc_raw: Numpy array of shape (N,3), representing a point cloud
    Returns:
       segment: Numpy array of shape (K,3) representing 3d points that fell
                within 3d cuboid volume.
       is_valid: Numpy array of shape (N,) of type bool
    https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d
    ::
            5------4
            |\\    |\\
            | \\   | \\
            6--\\--7  \\
            \\  \\  \\ \\
        l    \\  1-------0    h
         e    \\ ||   \\ ||   e
          n    \\||    \\||   i
           g    \\2------3    g
            t      width.     h
             h.               t.
    """
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

def nms(corners,count,volume,labels):
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
        del labels[i]

    return labels
