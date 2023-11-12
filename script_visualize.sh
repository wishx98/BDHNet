#!/usr/bin/env bash
#nvidia-smi
export DETECTRON2_DATASETS=/home/swx/local/codes/lr_dt2/datasets
export NGPUS=3
#cd tools

python visualize_json_results.py --input /home/swx/local/codes/lr_dt2/experiments/BDH/.../coco_instances_results.json \
--output /home/swx/local/codes/lr_dt2/experiments/BDH/.../res --dataset bdh_val