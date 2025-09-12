#!/usr/bin/env bash

#SBATCH --job-name=jina
#SBATCH --partition=medimgfmod
#SBATCH --account=medimgfmod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=28
#SBATCH --time=36:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fredliam99@hotmail.com

# Set error handling
set -euo pipefail

# Load environment
source ~/.bashrc
source activate FYP2526_JINA

# Change to the project directory (use current directory)
cd /home/shebd/4_Collaboration/FYP2526/FYP2526_fred

# Print environment info for debugging
echo "========================================="
echo "SLURM Job Information:"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Number of GPUs: $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "========================================="

# Load SLURM configuration from unified config system
eval $(python tools/get_config.py --section runtime)

# SLURM-specific settings (override runtime defaults for cluster environment)
export RUN_MODE="${SLURM_RUN_MODE:-$DEFAULT_RUN_MODE}"     # Use distributed training for multiple GPUs
export GPUS="${SLURM_GPUS:-$DEFAULT_GPUS}"                # Use SLURM GPU allocation
export NUM_PROC="${SLURM_NUM_PROC:-$SLURM_GPUS_ON_NODE}"  # Use SLURM GPU count

# Load data and training configuration
eval $(python tools/get_config.py --section data)
eval $(python tools/get_config.py --section training)

# Configuration is now loaded from unified system as:
# TRAIN_DATA, EVAL_DATA, OUTPUT_DIR, etc.
# These can still be overridden by environment variables for job customization

echo "=== Configuration from unified system ==="
echo "Run mode: $RUN_MODE"
echo "GPUs: $GPUS"
echo "Train data: $TRAIN_DATA"
echo "Output dir: $OUTPUT_DIR"
echo "========================================="

./tools/run_train.sh

echo "Training completed successfully!"