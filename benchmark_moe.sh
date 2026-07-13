CUDA_VISIBLE_DEVICES=1 python benchmark_moe.py \
    --data-root /data/datasets/AnyEdit \
    --output-dir /data/repos/models/0-output/AnyEdit-Test \
    --edit-file /data/datasets/AnyEdit/edit.json \
    --lora-path runs/train_moe_lora_stage1overfit/20260711-071003/ckpt/50000
