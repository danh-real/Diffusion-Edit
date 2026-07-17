CUDA_VISIBLE_DEVICES=0 python benchmark_moe.py \
    --data-root /data/datasets/AnyEdit \
    --output-dir /data/repos/models/0-output/AnyEdit-Test \
    --edit-file /data/datasets/AnyEdit/edit.json \
    --lora-path runs/task-specific-moe-lora-stage2combine-softmax/20260716-025748/ckpt/30000 \
    --saved-suffix-model "stage2-softmax"