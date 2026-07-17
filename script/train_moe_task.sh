#!/bin/bash -l

# usage: ./train.sh [CONFIG_FILE] [PORT]
# example: ./train.sh normal_lora.yaml 41353

CONFIG_FILE=${1:-"moe_lora_task_stage1.yaml"}
PORT=${2:-41353}

export XFL_CONFIG=./config/${CONFIG_FILE}
echo "Using config: $XFL_CONFIG"
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1

# # Debugging variable
# export NCCL_DEBUG=INFO
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

accelerate launch --main_process_port ${PORT} -m src.train.train_moe