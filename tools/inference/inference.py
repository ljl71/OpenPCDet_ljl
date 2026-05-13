import torch
import datetime, os

from pathlib import Path
from pcdet.utils import common_utils
import numpy as np
import torch

from pathlib import Path
from pcdet.models import build_network
from pcdet.models import load_data_to_gpu
from DataSet import getDataFromFile, getDataFromURL, Dataset
from pcdet.config import cfg, cfg_from_yaml_file
from torch.utils.data import DataLoader
from functools import partial
from pcdet.utils import common_utils


def parse_config():
    cfg_file = (
        "/workspace/OpenPCDet/tools/cfgs/waymo_models/pv_rcnn_plusplus_resnet.yaml"
    )
    cfg_from_yaml_file(cfg_file, cfg)
    cfg.TAG = Path(cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(
        cfg_file.split("/")[1:-1]
    )  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    return cfg


# formatList = ["x","y","z","w","h","l","yew"]


class LoadModel:
    def __init__(self):

        self.cfg = parse_config()
        self.checkpoint = (
            "/workspace/OpenPCDet/tools/inference/ckpt/checkpoint_epoch_40.pth"
        )
        self.cfg_Model = self.cfg.MODEL
        self.num_class = self.cfg.CLASS_NAMES

        self.data_set_demo, _ = self.encapData(
            "/workspace/OpenPCDet/tools/inference/dataset_demo", 4
        )

        output_dir = cfg.ROOT_DIR / "output" / cfg.EXP_GROUP_PATH / cfg.TAG / "default"
        output_dir.mkdir(parents=True, exist_ok=True)

        eval_output_dir = output_dir / "eval"
        eval_output_dir = (
            eval_output_dir / ("epoch_%s" % 5) / cfg.DATA_CONFIG.DATA_SPLIT["test"]
        )
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        log_file = eval_output_dir / (
            "log_eval_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
        # log to file
        self.logger.info("**********************Start logging**********************")

        model = build_network(
            model_cfg=self.cfg_Model,
            num_class=self.num_class,
            dataset=self.data_set_demo,
        )
        model.load_params_from_file(
            filename=self.checkpoint, logger=self.logger, to_cpu=False
        )
        model.cuda()

        self.model = model

    def encapData(self, FilePath, shape=4):

        batch_size = self.cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU

        # 文件夹模式需要用这个
        # data_file_list = getDataFromFile(FilePath)

        # 单文件模式用这个
        data_file_list = [FilePath]

        data_set = Dataset(
            dataset_cfg=self.cfg.DATA_CONFIG,
            class_names=self.cfg.CLASS_NAMES,
            training=False,
            data_file_list=data_file_list,
            shape=shape,
        )

        data_loader = DataLoader(
            dataset=data_set,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4,
            shuffle=False,
            collate_fn=data_set.collate_batch,
            drop_last=False,
            timeout=0,
            worker_init_fn=partial(common_utils.worker_init_fn, seed=None),
        )
        # result_dir.mkdir(parents=True, exist_ok=True)
        # final_output_dir = result_dir / 'final_result' / 'data'

        return data_set, data_loader

    async def inference(self, FilePath, shape=4):

        data_sets, data_loader = self.encapData(FilePath, shape)

        final_output_dir = "/final_result/data"

        class_names = data_sets.class_names
        det_annos = []
        self.logger.info("*************** EPOCH %s EVALUATION *****************" % 5)
        self.model.eval()

        for i, batch_dict in enumerate(data_loader):
            pointcloud = batch_dict["ori_points"]
            try:
                pointcloud = pointcloud[~np.isnan(pointcloud).any(axis=1), :]
            except:
                pass
            del batch_dict["ori_points"]
            load_data_to_gpu(batch_dict)

            with torch.no_grad():
                pred_dicts, ret_dict = self.model.forward(batch_dict)

            annos = data_sets.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names, output_path=final_output_dir
            )

            annos = {k: v.tolist() for k, v in annos[0].items()}
            det_annos.append(annos)

        result = []
        labelData = []
        for a in range(len(det_annos)):
            for i in range(len(det_annos[a]["name"])):
                dicObj = {
                    "type": "shape",
                    "drawType": "box3d",
                    "label": det_annos[a]["name"][i],
                    "group": 0,
                    "score": det_annos[a]["score"][i],
                    # "points": {k:v for k,v in zip(formatList,det_annos[a]['boxes_lidar'][i])}}
                    "points": [
                        det_annos[a]["boxes_lidar"][i][0],
                        det_annos[a]["boxes_lidar"][i][1],
                        det_annos[a]["boxes_lidar"][i][2],
                        0,
                        0,
                        det_annos[a]["boxes_lidar"][i][6],
                        det_annos[a]["boxes_lidar"][i][3],
                        det_annos[a]["boxes_lidar"][i][4],
                        det_annos[a]["boxes_lidar"][i][5],
                    ],
                }
                labelData.append(dicObj)
            result.append(labelData)

        return result


if __name__ == "__main__":

    cfg = parse_config()

    path = Path("/workspace/OpenPCDet/tools/inference/test_pcFile")
    ####
    model = LoadModel()

    with torch.no_grad():
        result_annos = model.inference(FilePath=path)
        print("*" * 1000, result_annos)
