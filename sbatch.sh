#!/bin/bash
#SBATCH --job-name=learn_sim_train_gpu
#SBATCH --partition=scavenge_gpu
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --gpus=a100:1
#SBATCH --mail-type=ALL
#SBATCH --requeue
#SBATCH --output=./slogs/2full_%j.out
#SBATCH --error=./slogs/2full_%j.err

module load miniconda

ENV_NAME="learn-to-sim"
if ! conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Environment ${ENV_NAME} not found. Creating from environment.yml..."
    conda env create -f environment.yml
fi

source activate $ENV_NAME

echo "======================================="
echo "Job started on $(hostname) at $(date)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs allocated: $SLURM_JOB_GPUS"
echo "Using Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Current directory: $(pwd)"
echo "======================================="

DATASET="WaterDropSample"
GNN_TYPE="interaction_net"
LOSS_TYPE="one_step"

echo "Dataset: ${DATASET}"
echo "GNN: ${GNN_TYPE}"
echo "Loss: ${LOSS_TYPE}"

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

echo "======================================="
echo "Job completed on $(hostname) at $(date)"
echo "======================================="