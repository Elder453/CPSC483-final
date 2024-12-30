#!/bin/bash
#SBATCH --job-name=learn_sim_train_gpu
#SBATCH --partition=scavenge_gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH -C "a100|l40s"
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --requeue
#SBATCH --output=/vast/palmer/home.grace/cpsc483_egv6/CPSC483-final/slogs/train_%j.out
#SBATCH --error=/vast/palmer/home.grace/cpsc483_egv6/CPSC483-final/slogs/train_%j.err

cd ${SLURM_SUBMIT_DIR}
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

# training script
python ./src/main.py \
    --mode=train \
    --dataset=WaterDropSample \
    --gnn_type=interaction_net \
    --loss_type=multi_step \
    --num_steps=500000 \
    --batch_size=2

echo "======================================="
echo "Job completed on $(hostname) at $(date)"
echo "======================================="
