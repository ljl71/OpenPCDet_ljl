#!/bin/bash

#SBATCH --job-name=train_pvrcnn

#SBATCH --mail-user=minghao.liu@molardata.com

#SBATCH --nodes=1

#SBATCH --ntasks=8

#SBATCH --gres=gpu:1

source ~/.bashrc
cd /home/molardata/pcModel/OpenPCDet/tools
conda activate molar-opc
python train.py --cfg_file /home/molardata/pcModel/OpenPCDet/tools/cfgs/waymo_models/pv_rcnn_plusplus_resnet.yaml --batch_size 2 --epochs 40

