#!/usr/bin/env bash
#nvidia-smi
export DETECTRON2_DATASETS=/home/swx/local/codes/lr_dt2/datasets
export NGPUS=3

python train_net.py --num-gpus $NGPUS --config-file /home/swx/local/codes/lr_dt2/configs/BDH/instance-segmentation/swin/BDHNet_swin_small_bs16_50ep.yaml MODEL.WEIGHTS /home/swx/local/codes/BDHNet/pretrained_weights/phase1.pth OUTPUT_DIR ./experiments/BDH/BDHNet_swin_small_adabins_it80k