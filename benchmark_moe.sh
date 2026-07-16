CUDA_VISIBLE_DEVICES=0 python benchmark_moe.py \
    --data-root /data/datasets/AnyEdit \
    --output-dir /data/repos/models/0-output/AnyEdit-Test \
    --edit-file /data/datasets/AnyEdit/edit.json \
    --lora-path runs/train_moe_lora_stage1overfit_softmax/20260714-010507/ckpt/30000 \
    --saved-suffix-model "stage1-softmax"