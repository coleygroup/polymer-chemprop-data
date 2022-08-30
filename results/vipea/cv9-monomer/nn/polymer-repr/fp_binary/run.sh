#!/bin/bash

FPS="dataset-poly_fps_binary.pkl"
SPLIT="cv9-monomer_split.pkl"

python ../../../../train_test_nn.py -f ~/polymer-chemprop-data/datasets/vipea/rf_input/${FPS} \
     -k ~/polymer-chemprop-data/datasets/vipea/${SPLIT} \
     --hidden_size 128 \
     --batch_size 256 \
     --num_epochs 200 \
     --patience 50 \
     --gpu_id 0 \
     --hopt 20
