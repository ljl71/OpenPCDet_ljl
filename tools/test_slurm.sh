#!/bin/bash

#SBATCH --job-name=train_pvrcnn

#SBATCH --mail-user=minghao.liu@molardata.com

#SBATCH --nodes=1

#SBATCH --ntasks=8

#SBATCH --gres=gpu:1

source ~/.bashrc
cd /home/molardata/pcModel/OpenPCDet/tools
conda activate molar-opc
# python test.py --cfg_file /home/molardata/pcModel/OpenPCDet/tools/cfgs/waymo_models/pv_rcnn_plusplus_resnet.yaml --batch_size 4 --eval_all --ckpt_dir /home/molardata/pcModel/OpenPCDet/output/home/molardata/pcModel/OpenPCDet/tools/cfgs/waymo_models/pv_rcnn_plusplus_resnet/default/ckpt/  --save_to_file
python test.py --cfg_file /home/molardata/pcModel/OpenPCDet/tools/cfgs/waymo_models/pv_rcnn_plusplus_resnet.yaml --batch_size 4 --ckpt /home/molardata/pcModel/OpenPCDet/output/home/molardata/pcModel/OpenPCDet/tools/cfgs/waymo_models/pv_rcnn_plusplus_resnet/default/ckpt/checkpoint_epoch_40.pth