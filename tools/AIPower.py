import json
import os
from tkinter import N

import numpy as np
import pymongo
from operator import itemgetter
import sys
sys.path.append('/home/molardata/pcModel/OpenPCDet/')
from bson.objectid import ObjectId
import requests
from yaml import load
from tools.batch_deal import predict
from pyntcloud import PyntCloud

from tools.utils import downloadItemListThread, filter_point_cloud_to_bbox_3D_vectorized, get_3d_box, loadJson, matchImageDivisor, mkdir, saveJson, nms


class mongodbHandle:
    def getConn_uri(self, uri, dbName, tableName):
        """
        建立数据库连接
        """
        self.collection = pymongo.MongoClient(uri)
        self.db = self.collection[dbName]
        self.collection = self.db[tableName]

    def getConn_keyword(self, dbHost, dbPort, dbUser, dbPswd, tableName, dbAuthen):
        """
        建立数据库连接
        """
        self.conn = pymongo.MongoClient(host=dbHost, port=dbPort, username=dbUser, password=dbPswd,
                                        authSource=dbAuthen)
        self.db = eval('self.conn.{}'.format(tableName))
        self.collection = self.db['pre_label_process']

    def getData(self, processId):
        infos = self.collection.find_one({"_id": ObjectId(processId)})
        return infos


def getItemList(infos):
    batchInfo = []
    for item in infos['items']:
        for i in range(len(item['info']['pcdUrl'])):
            batchInfo.append({'taskID': str(item['_id']),
                              'url': item['info']['pcdUrl'][i],
                              'index': i})
    return batchInfo


def getNPFItemList(infos):
    batchInfo = []
    for item in infos['items']:
        for i in range(len(item['info']['pcdUrl'])):
            batchInfo.append({'taskID': str(item['itemId']),
                              'url': item['info']['pcdUrl'][i],
                              'index': i})
    return batchInfo


def getPredictClass(itemInfos, segClas):
    predictList = []
    mappingInfo = itemInfos['mapping']
    for i in mappingInfo:
        for i2 in i.values():
            for i3 in i2:
                predictList.append(list(segClas.keys())[list(segClas.values()).index(i3.lower())])
    return predictList, itemInfos['mapping']


def getLabelTag(originTag, mappingInfo):
    for i in mappingInfo:
        for i2 in i.values():
            for i3 in i2:
                if originTag == i3.lower():
                    labelTag = ''.join(list(i.keys()))
    return labelTag

def predictPCDet():
    processId = sys.argv[1]
    env = sys.argv[2]


    if env == 'work':
        DATABASE_HOST = "47.98.222.26"
        DATABASE_PORT = 27017
        DATABASE_USER = "auto_server"
        DATABASE_PASSWORD = "1asFr2fFwFds8Li0Bc1Dgfm"
        DATABASE_NAME = "MolarLabelSystemProd"
        DATABASE_AUTH = "admin"
        x = mongodbHandle()
        x.getConn_keyword(DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD, DATABASE_NAME, DATABASE_AUTH)

        itemInfos = x.getData(processId)
        batchInfo = getItemList(itemInfos)

    elif env == 'daily':
        DATABASE_HOST = '47.98.222.26'
        DATABASE_PORT = 27017
        DATABASE_USER = 'MolarLabelSystemDaily'
        DATABASE_PASSWORD = 'hellomolar'
        DATABASE_NAME = 'MolarLabelSystemDaily'
        DATABASE_AUTH = 'MolarLabelSystemDaily'
        x = mongodbHandle()
        x.getConn_keyword(DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD, DATABASE_NAME, DATABASE_AUTH)

        itemInfos = x.getData(processId)
        batchInfo = getItemList(itemInfos)

    elif env == 'new_daily':
        uri = 'mongodb://molar_data_developer:hfHhdSCl32salcDJHsfxm@59.111.93.126:5001,59.111.93.126:5002,59.111.93.126:5003/molar_data?authSource=admin&replicaSet=molar_label_rs0'
        DATABASE_NAME = 'molar_data'
        TABLE_NAME = 'pre_label_process'
        x = mongodbHandle()
        x.getConn_uri(uri, DATABASE_NAME, TABLE_NAME)

        itemInfos = x.getData(processId)
        batchInfo = getNPFItemList(itemInfos)
        
    elif env == 'new_prod':
        uri = 'mongodb://root:vfa7N8eYnwL9Paiy@47.96.65.228:5001,47.96.65.228:5002,47.96.65.228:5003/molar_data?authSource=admin&replicaSet=molar_label_prod_rs0'
        DATABASE_NAME = 'molar_data'
        TABLE_NAME = 'pre_label_process'
        x = mongodbHandle()
        x.getConn_uri(uri, DATABASE_NAME, TABLE_NAME)
        
        itemInfos = x.getData(processId)
        batchInfo = getNPFItemList(itemInfos)

    model, scene = itemInfos['model'].split('.')[0], itemInfos['model'].split('.')[1]
    if scene == 'common':
        labelClas = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle',
                   'pedestrian', 'traffic_cone']
        modelPath = '/home/molardata/segModel/SegFormer/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_160k_ade20k.py'
        checkpointPath = '/home/molardata/segModel/SegFormer/pth/deeplabv3plus/deeplabv3plus_r101-d8_512x512_160k_ade20k_20200615_123232-38ed86bb.pth'
    elif scene == 'large':
        labelClas = {1:'vehicle', 2:'pedestrian', 3:'cyclist'}
        modelPath = '/home/molardata/pcModel/OpenPCDet/tools/cfgs/waymo_models/pv_rcnn_plusplus_resnet.yaml'
        checkpointPath = '/home/molardata/pcModel/OpenPCDet/output/home/molardata/pcModel/OpenPCDet/tools/cfgs/waymo_models/pv_rcnn_plusplus_resnet/default/ckpt/checkpoint_epoch_39.pth'

    predictInfo, mappingInfo = getPredictClass(itemInfos, labelClas)

    successList, failList, theTime = downloadItemListThread(batchInfo, 8)
    batchPCD = [theTime, successList]

    # mkdir('./data/results/{}'.format(theTime))

    total_res = predict(batchPCD,modelPath,checkpointPath)

    labelData = []

    bboxCornersA = []
    bboxPointCountA = []
    bboxVolumeA = []


    for pcdIndex in range(len(total_res)):
        boxes = list(total_res[pcdIndex]['pred_boxes'].cpu().numpy())
        score = list(total_res[pcdIndex]['pred_scores'].cpu().numpy())
        labels = list(total_res[pcdIndex]['pred_labels'].cpu().numpy())

        bboxCorners = []

        for boxIndex in range(len(labels)):
            if labels[boxIndex] in predictInfo:
                BBOX = list(boxes[boxIndex])

                # 转换为平台所需的顺序，即x, y, z, w, h, l, Θ转为x, y, z, p, y, r, w, h, l
                BBOX.insert(6,0)
                BBOX.insert(6,0)

                BBOX[3], BBOX[4], BBOX[5], BBOX[6], BBOX[7], BBOX[8] = BBOX[6], BBOX[7], BBOX[8], BBOX[3], BBOX[4], BBOX[5]

                BBOX = [float(i) for i in BBOX]

                boxCorners = get_3d_box(
                        (BBOX[7], BBOX[8], BBOX[6]),
                        BBOX[5],
                        (BBOX[0], BBOX[1], BBOX[2]),
                    )

                bboxCorners.append(boxCorners)

                if batchPCD[1][pcdIndex]['link'].endswith('.bin'):
                    points = np.fromfile(os.path.join(os.path.join(os.environ["HOME"], "dataset/data/{}/".format(theTime)) + batchPCD[1][pcdIndex]['link']), dtype=np.float32).reshape(-1, 4)
                    points = np.delete(points, -1, axis=1)
                elif batchPCD[1][pcdIndex]['link'].endswith('.pcd'):
                    try:
                        points = np.asarray(PyntCloud.from_file(os.path.join(os.path.join(os.environ["HOME"], "dataset/data/{}/".format(theTime)) + batchPCD[1][pcdIndex]['link'])).xyz)
                    except:
                        from pypcd import pypcd
                        import pandas as pd
                        pc = pypcd.PointCloud.from_path(os.path.join(os.path.join(os.environ["HOME"], "dataset/data/{}/".format(theTime)) + batchPCD[1][pcdIndex]['link']))
                        points = np.asarray(pd.DataFrame(pc.pc_data,columns = ['x','y','z']))
                        # points = np.asarray(PyntCloud.from_file(os.path.join(os.path.join(os.environ["HOME"], "dataset/data/{}/".format(theTime)) + batchPCD[1][pcdIndex]['link'])).xyz)
                points = points[~np.isnan(points).any(axis=1), :]
                bboxPointCountA.append(int(filter_point_cloud_to_bbox_3D_vectorized(boxCorners,points)))
                bboxVolumeA.append(float(BBOX[7]) * float(BBOX[8]) * float(BBOX[6]))

                if env == 'daily' or env == 'word':
                    dicObj = {
                                "itemId": batchPCD[1][pcdIndex]['taskID'],
                                "data": [{
                                    "label": getLabelTag(labelClas[labels[boxIndex]], mappingInfo),
                                    "type": "track",
                                    "frames": [
                                        {
                                            "frame": batchPCD[1][pcdIndex]['index'],
                                            "points": BBOX,
                                            "outside": True
                                        }
                                    ],
                                    "count": 1,
                                    "drawType": "CUBOID",
                                    "group": "0",
                                    "score":float(score[boxIndex])
                                }]
                            }
                else:
                    dicObj = {
                                "itemId": batchPCD[1][pcdIndex]['taskID'],
                                "preData": [{
                                    "label": getLabelTag(labelClas[labels[boxIndex]], mappingInfo),
                                    "frames": [
                                        {
                                            "frame": batchPCD[1][pcdIndex]['index'],
                                            "points": BBOX,
                                            "outside": True
                                        }
                                    ],
                                    "count": 1,
                                    "drawType": "box3d",
                                    "group": "0",
                                    "score":float(score[boxIndex])
                                }]
                            }

                labelData.append(dicObj)

        bboxCornersA.append(bboxCorners)

    mkdir('./data/results/{}'.format(theTime))

    print(len(labelData))
    labelData = nms(bboxCornersA,bboxPointCountA,bboxVolumeA,labelData)
    print(len(labelData))

    # format数据成平台能接受的格式
    labelData.sort(key=itemgetter('itemId'))

    postData = {"processId": processId, 'data': []}
    itemMap = []

    if env == 'daily' or env == 'word':
        for i in range(len(itemInfos['items'])):
            postData['data'].append({'info':itemInfos['items'][i]['info'],'data':[],'itemId':str(itemInfos['items'][i]['_id'])})
            itemMap.append(str(itemInfos['items'][i]['_id']))
    else:
        for i in range(len(itemInfos['items'])):
            postData['data'].append({'info':itemInfos['items'][i]['info'],'data':[],'itemId':str(itemInfos['items'][i]['itemId'])})
            itemMap.append(str(itemInfos['items'][i]['itemId']))

    for i in labelData:
        postData['data'][itemMap.index(i['itemId'])]['data'].append(i['preData'][0])

    # for i in range(len(labelData)):
    #     if i == 0:
    #         infos = labelData[i]['data'][0]
    #         postData['data'].append(labelData[i])
    #     else:
    #         if postData['data'][-1]['itemId'] == labelData[i]['itemId']:
    #             infos = labelData[i]['data'][0]
    #             postData['data'][-1]['data'].append(infos)
    #         else:
    #             postData['data'].append(labelData[i])
    
    saveJson('./data/results/{}/{}_final.json'.format(theTime, theTime), postData)

    header = {'Content-Type': 'application/json'}
    r = requests.post(itemInfos['callback'], data=json.dumps(postData), headers=header)

    print(r.status_code)


if __name__ == '__main__':
    predictPCDet()
