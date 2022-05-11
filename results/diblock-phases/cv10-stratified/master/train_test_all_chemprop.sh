#!/bin/bash

polymer=false
num_folds=10
aggregation='mean'
pytorch_seed=0
gpu=0

END=$(expr $num_folds - 1)

if [ "$polymer" = true ] ; then
  extra_args='--polymer'
else
  extra_args=''
fi

for k in $(seq 0 $END); do
  chemprop_train --data_path input_train_${k}.csv --separate_val_path input_val_${k}.csv --separate_test_path input_test_${k}.csv --dataset_type classification --save_dir chemprop_checkpoints --aggregation ${aggregation} --gpu ${gpu} --pytorch_seed ${pytorch_seed} ${extra_args}
  chemprop_predict --test_path input_test_${k}.csv --checkpoint_dir chemprop_checkpoints --preds_path predictions_${k}.csv --gpu ${gpu}
  rm -r chemprop_checkpoints
done
