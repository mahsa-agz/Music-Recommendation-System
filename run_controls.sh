#!/bin/bash
cd "$(dirname "$0")"
C="--train_path data/train_cold.csv --valid_path data/test_cold.txt --dim 64 --epoch 8 --save_batch_time 250 --num_workers 6 --eval_items 3000"
echo "=== control: shuffled item->content ==="
python -u model.py $C --shuffle_content 42 --tag ctrlA_shuffled > run_ctrlA.log 2>&1
echo "=== control: warm vs cold ==="
python -u model.py $C --warm_path data/warm_probe.txt --tag ctrlB_warm > run_ctrlB.log 2>&1
echo "=== DONE ==="
