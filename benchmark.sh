# CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_moe.py \
#     --top-k 1 \
#     --data-root /data/datasets/emu_edit_test_set \
#     --output-dir /data/repos/models/0-output/emu_edit_test_set \
#     --edit-file /data/datasets/emu_edit_test_set/edit.json \
#     --saved-suffix-model "" \
#     --save-every 1 \
#     --token-stride 32 2>&1 > benchmark_emu.log &

CUDA_VISIBLE_DEVICES=0 python benchmark_moe.py \
    --top-k 1 \
    --data-root /data/datasets/AnyEdit \
    --output-dir /data/repos/models/0-output/AnyEdit-Test \
    --edit-file /data/datasets/AnyEdit/edit.json \
    --saved-suffix-model "" \
    --save-every 1 \
    --save-tensors \
    --save-token-inputs \
    --token-stride 64 # 2>&1 > benchmark_anyedit.log &

# CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_moe.py \
#     --top-k 1 \
#     --data-root /data/datasets/GEdit-Bench \
#     --output-dir /data/repos/models/0-output/GEdit-Bench \
#     --edit-file /data/datasets/GEdit-Bench/edit.json \
#     --saved-suffix-model "" \
#     --save-every 1 \
#     --save-tensors \
#     --save-token-inputs \
#     --token-stride 64 2>&1 > benchmark_gedit.log &