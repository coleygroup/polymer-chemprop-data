#!/bin/bash

agg='mean'
gpu=0

# 80% 20% 5% 2% 1% 0.5% 0.2% 0.1%
for train_size in 0.8 0.2 0.05 0.02 0.01 0.005 0.002 0.001; do
  for i in {0..9}; do
    echo "************************************************************"
    echo "TRAIN SIZE $train_size, REPEAT $i "
    echo "************************************************************"
    sleep 2
    test_size=`echo "0.9-$train_size" | bc`

    chemprop_train --data_path dataset-master_chemprop.csv --split_sizes $train_size 0.1 $test_size --dataset_type regression --save_dir chemprop_checkpoints --aggregation $agg --gpu $gpu --pytorch_seed 0 --seed $i --metric rmse --extra_metrics r2

    mv chemprop_checkpoints/test_scores.csv test_scores_train-${train_size}_${i}.csv
    rm -r chemprop_checkpoints
  done
done

