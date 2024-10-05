# !/bin/bash

# Example training shell script.
MODEL='mamba2-130m'
EXP_NAME='Example'
DATA_PATH='datapath'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py \
  --seed 0 \
  --model state-spaces/${MODEL} \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --data_path ${DATA_PATH} \
  --exp_name ${EXP_NAME} \
