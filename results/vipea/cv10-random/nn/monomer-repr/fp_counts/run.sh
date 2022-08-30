#!/bin/bash

FPS="dataset-mono_fps_counts.pkl"
SPLIT="cv10-random_split.pkl"

python ../../../../train_test_nn.py -f ~/polymer-chemprop-data/datasets/vipea/rf_input/${FPS} \
     -k ~/polymer-chemprop-data/datasets/vipea/${SPLIT} \
     --hidden_size 128 \
     --batch_size 256 \
     --num_epochs 200 \
     --patience 50 \
     --gpu_id 1 \
     --hopt 20
