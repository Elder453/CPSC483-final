#!/bin/bash

DATASET="WaterDropSample"
GNN_TYPE="interaction_net"
LOSS_TYPE="multi_step"
EVAL_SPLIT="test"


python ./src/main.py \
    --mode=eval_rollout \
    --dataset=$DATASET \
    --gnn_type=$GNN_TYPE \
    --loss_type=$LOSS_TYPE \
    --eval_split=$EVAL_SPLIT


python ./src/render_rollout.py \
    --output_path=rollouts \
    --dataset=$DATASET \
    --gnn_type=$GNN_TYPE \
    --loss_type=$LOSS_TYPE \
    --eval_split=$EVAL_SPLIT \
    --step_stride=3