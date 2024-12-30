#!/bin/bash
python render_rollout.py \
    --output_path=rollouts \
    --dataset=WaterDropSample \
    --gnn_type=interaction_net \
    --loss_type=one_step \
    --eval_split=test \
    --time_step=0 \
    --step_stride=3 \
    --block_on_show=True \
    "$@"  # allow passing additional args (if needed)
