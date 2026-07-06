#!/bin/bash

# usage: ./train.sh [CONFIG_FILE] [PORT]
# example: ./train.sh normal_lora.yaml 41353

CONFIG_FILE=${1:-"normal_lora.yaml"}
PORT=${2:-41353}

export XFL_CONFIG=./train/config/${CONFIG_FILE}
echo "Using config: $XFL_CONFIG"
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port ${PORT} -m src.train.train