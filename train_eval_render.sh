#!/bin/bash

DATASET="Goop"
GNN_TYPE="gat"
LOSS_TYPE="multi_step"


python ./src/main.py \
    --mode=train \
    --dataset=$DATASET \
    --gnn_type=$GNN_TYPE \
    --loss_type=$LOSS_TYPE \
    --num_steps=500000 \
    --batch_size=2 \
    --use_bn

python ./src/main.py \
    --mode=eval \
    --dataset=$DATASET \
    --gnn_type=$GNN_TYPE \
    --loss_type=$LOSS_TYPE \
    --eval_split=test \
    --compute_all_metrics


python ./src/main.py \
    --mode=eval_rollout \
    --dataset=$DATASET \
    --gnn_type=$GNN_TYPE \
    --loss_type=$LOSS_TYPE \
    --eval_split=test


python ./src/render_rollout.py \
    --output_path=rollouts \
    --dataset=$DATASET \
    --gnn_type=$GNN_TYPE \
    --loss_type=$LOSS_TYPE \
    --eval_split=test \
    --step_stride=3
