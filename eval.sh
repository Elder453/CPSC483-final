#!/bin/bash
python main.py \
    --mode=eval \
    --dataset=WaterDropSample \
    --gnn_type=interaction_net \
    --loss_type=one_step \
    --eval_split=test \
    "$@"  # allow passing additional args (if needed)
