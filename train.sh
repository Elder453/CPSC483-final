#!/bin/bash
python main.py \
    --mode=train \
    --dataset=WaterDropSample \
    --gnn_type=interaction_net \
    --loss_type=one_step \
    --num_steps=250000 \
    --batch_size=2 \
    "$@"
