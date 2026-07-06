#!/bin/bash -l

# RL routing training for MoE LoRA (GRPO-style)
#
# usage:  ./train_route_rl.sh [CONFIG_FILE] [PORT]
# example: ./train_route_rl.sh moe_lora_rl.yaml 41354

CONFIG_FILE=${1:-"moe_lora_rl.yaml"}
PORT=${2:-41354}

export XFL_CONFIG=./train/config/${CONFIG_FILE}
echo "Using config: $XFL_CONFIG"
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port ${PORT} -m src.train.train_route_rl
