#!/bin/bash

DATASET="WaterDropSample"
GNN_TYPE="interaction_net"
LOSS_TYPE="one_step"
CHECKPOINT=1


python ./src/main.py \
    --mode=eval_rollout \
    --dataset=$DATASET \
    --gnn_type=$GNN_TYPE \
    --loss_type=$LOSS_TYPE \
    --eval_split=test \
    --checkpoint=$CHECKPOINT


python ./src/render_rollout.py \
    --output_path=rollouts \
    --dataset=$DATASET \
    --gnn_type=$GNN_TYPE \
    --loss_type=$LOSS_TYPE \
    --eval_split=test \
    --step_stride=3