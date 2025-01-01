#!/bin/bash
python ./src/main.py \
    --mode=eval \
    --dataset=WaterDropSample \
    --gnn_type=interaction_net \
    --loss_type=one_step \
    --eval_split=test \
    --compute_all_metrics
