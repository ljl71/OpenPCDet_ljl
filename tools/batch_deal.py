import datetime
import os
import torch
from pathlib import Path
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_molar_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from tools.eval_utils.eval_utils import statistics_info


def predict(data_file_list,cfg_file,ckpt):

    dist_test = False
    batch_size = 2
    workers = 0


    cfg_from_yaml_file(cfg_file, cfg)
    cfg.TAG = Path(cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    # np.gettext.random.seed(1024)

    # if args.set_cfgs is not None:
    #     cfg_from_list(args.set_cfgs, cfg)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / '.pcd'
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    cfg.DATA_CONFIG.DATASET = 'molarDataset'

    test_set, test_loader, sampler = build_molar_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=dist_test,
        data_file_list=data_file_list,
        workers=workers, logger=logger, training=False
    )

    # metric = {
    #     'gt_num': 0,
    # }

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()
    model.eval()

    dataset = test_loader.dataset
    # class_names = dataset.class_names
    # det_annos = []
    
    total_res = []

    for i, batch_dict in enumerate(test_loader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        # disp_dict = {}
        total_res.extend(pred_dicts)

        # statistics_info(cfg, ret_dict, metric, disp_dict)
        # annos = dataset.generate_prediction_dicts(
        #     batch_dict, pred_dicts, class_names,
        #     output_path=final_output_dir
        # )
        # det_annos += annos
    return total_res
