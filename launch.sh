#!/bin/bash
python main.py \
    --mode=train \
    --dataset=WaterDropSample \
    --num_steps=500000 \
    --batch_size=2 \
    "$@"  # allow passing additional args (if needed)