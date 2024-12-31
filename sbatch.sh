#!/bin/bash
#SBATCH --job-name=learn_sim_train_gpu
#SBATCH --partition=scavenge_gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH -C "a100"
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --requeue
#SBATCH --output=./slogs/full_%j.out
#SBATCH --error=./slogs/full_%j.err

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

bash ./train_eval_render.sh

echo "======================================="
echo "Job completed on $(hostname) at $(date)"
echo "======================================="
