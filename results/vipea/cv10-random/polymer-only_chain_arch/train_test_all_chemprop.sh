#!/bin/bash

agg='mean'
gpu=0

for k in 0 1 2 3 4 5 6 7 8 9; do
  chemprop_train --data_path input_train_${k}.csv --separate_val_path input_val_${k}.csv --separate_test_path input_test_${k}.csv --dataset_type regression --save_dir chemprop_checkpoints --aggregation $agg --gpu $gpu --polymer --pytorch_seed 0
  chemprop_predict --test_path input_test_${k}.csv --checkpoint_dir chemprop_checkpoints --preds_path predictions_${k}.csv --gpu $gpu
  rm -r chemprop_checkpoints
done
