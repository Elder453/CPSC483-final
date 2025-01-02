#!/bin/bash
python ./src/main.py \
    --mode=eval \
    --dataset=WaterDropSample \
    --gnn_type=interaction_net \
    --loss_type=one_step \
    --eval_split=test \
    --checkpoint=1 \
    --compute_all_metrics
