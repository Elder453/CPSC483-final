#!/bin/bash
# SBATCH --job-name=learn_sim_train_gpu  # Job name
# SBATCH --partition=scavenge_gpu        # Partition name
# SBATCH --time=24:00:00                 # Time limit
# SBATCH --ntasks=1                      # Number of tasks
# SBATCH --cpus-per-task=6               # Number of CPU cores per task
# SBATCH --mem-per-cpu=10G               # Memory per CPU core
# SBATCH --gpus=1                        # Number of GPUs
# SBATCH --mail-type=ALL                 # Email notifications
# SBATCH --requeue                       # Requeue if preempted
# SBATCH --output=./logs/train_%j.out    # Standard output log
# SBATCH --error=./logs/train_%j.err     # Standard error log

# Load necessary modules
module load miniconda

# Define your conda environment name
ENV_NAME="learn-to-sim"

# Check if the conda environment exists; if not, create it
if ! conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Environment ${ENV_NAME} not found. Creating from environment.yml..."
    conda env create -f environment.yml
fi

# Activate the conda environment
conda activate $ENV_NAME

# Navigate to your project's root directory
cd /home/cpsc483_egv6/CPSC483-final

# Print job start information
echo "======================================="
echo "Job started on $(hostname) at $(date)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs allocated: $SLURM_JOB_GPUS"
echo "Using Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Current directory: $(pwd)"
echo "======================================="

# Execute your training script with desired arguments
bash ./train.sh

# Print job completion information
echo "======================================="
echo "Job completed on $(hostname) at $(date)"
echo "======================================="