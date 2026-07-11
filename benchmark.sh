CUDA_VISIBLE_DEVICES=1 python benchmark_moe.py \
    --data-root /data/datasets/emu_edit_test_set \
    --output-dir /data/repos/models/0-output/emu_edit_test_set \
    --edit-file /data/datasets/emu_edit_test_set/edit.json \

CUDA_VISIBLE_DEVICES=1 python benchmark_moe.py \
    --data-root /data/datasets/GEdit-Bench \
    --output-dir /data/repos/models/0-output/GEdit-Bench \
    --edit-file /data/datasets/GEdit-Bench/edit.json \

CUDA_VISIBLE_DEVICES=1 python benchmark_moe.py \
    --data-root /data/datasets/MagicBrush-test \
    --output-dir /data/repos/models/0-output/MagicBrush-test \
    --edit-file /data/datasets/MagicBrush-test/edit.json \

