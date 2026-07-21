#!/bin/bash -l

# usage: ./train.sh [CONFIG_FILE] [PORT]
# example: ./train.sh normal_lora.yaml 41353

CONFIG_FILE=${1:-"moe_lora_task_stage2.yaml"}
PORT=${2:-41353}

export XFL_CONFIG=./config/${CONFIG_FILE}
echo "Using config: $XFL_CONFIG"
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# # Debugging variable
# export NCCL_DEBUG=INFO
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# This box has 2 physical GPUs (see `nvidia-smi`). With no accelerate config file present,
# accelerate defaults --num_processes to the visible device count and turns on multi_gpu, so
# this launches plain 2-way DDP: one full Flux Kontext replica per GPU, no GPU left over.
# That is intended for supervised runs (rl_coeff: 0.0), where train_moe.py never constructs
# the EditScore reward model at all. Enabling RL needs a third GPU for the reward model, or
# an explicit --num_processes 1 here to free one up (config: rl.reward.device).
accelerate launch --main_process_port ${PORT} -m src.train.train_moe