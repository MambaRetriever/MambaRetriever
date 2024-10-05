#!/bin/bash

# Example evaluation shell script.
MODEL='mamba2-130m'
EXP_NAME='Example'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 evaluate.py \
  --seed 0 \
  --logit_path rag_pipeline/prediction_logits \
  --model state-spaces/${MODEL} \
  --eval_data_path ... \
  --exp_name ${EXP_NAME} \
  --window_size 120000 \
