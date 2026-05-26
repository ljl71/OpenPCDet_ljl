import argparse
import os
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Run one formal CompanyNuScenes VoxelNeXt training step.')
    parser.add_argument(
        '--cfg_file',
        type=Path,
        default=repo_root / 'tools' / 'cfgs' / 'nuscenes_models' / 'company_voxelnext_26cls_trainval.yaml'
    )
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root / 'tools')

    import torch

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    cfg_from_yaml_file(str(args.cfg_file), cfg)
    logger = common_utils.create_logger()
    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=True
    )
    if len(dataset) == 0:
        raise RuntimeError('Training dataset is empty. Generate and check formal infos first.')

    batch = next(iter(dataloader))
    if batch['gt_boxes'].shape[1] == 0:
        raise RuntimeError('The sampled training batch has no GT boxes after filtering.')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset).cuda()
    model.train()
    load_data_to_gpu(batch)
    ret_dict, tb_dict, _ = model(batch)
    loss = ret_dict['loss']
    if not torch.isfinite(loss):
        raise RuntimeError(f'Non-finite loss detected: {loss.item()}')
    loss.backward()

    print('formal_voxelnext_smoke: PASS')
    print('classes:', len(cfg.CLASS_NAMES))
    print('dataset_len:', len(dataset))
    print('points_shape:', tuple(batch['points'].shape))
    print('gt_boxes_shape:', tuple(batch['gt_boxes'].shape))
    print('loss:', float(loss.detach().cpu()))
    print('loss_terms:', {name: float(value) for name, value in tb_dict.items()})


if __name__ == '__main__':
    main()
