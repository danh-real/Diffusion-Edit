#!/bin/bash -l

# usage: ./train.sh [CONFIG_FILE] [PORT]
# example: ./train.sh normal_lora.yaml 41353

CONFIG_FILE=${1:-"moe_lora_task_stage2.yaml"}
PORT=${2:-41353}

export XFL_CONFIG=./config/${CONFIG_FILE}
echo "Using config: $XFL_CONFIG"
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=.

# CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port ${PORT} -m src.train.train_moe
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port ${PORT} -m src.train.train_moe