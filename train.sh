#!/bin/bash
python ./src/main.py \
    --mode=train \
    --dataset=WaterDropSample \
    --gnn_type=interaction_net \
    --loss_type=one_step \
    --num_steps=500000 \
    --batch_size=2 \
    --use_bn
