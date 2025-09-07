#!/usr/bin/env bash

#SBATCH --job-name=jina
#SBATCH --partition=medimgfmod
#SBATCH --account=medimgfmod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=28
#SBATCH --time=96:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fredliam99@hotmail.com

# Set error handling
set -euo pipefail

# Load environment
source ~/.bashrc
source activate FYP2526_JINA

# Change to the project directory
cd /home/shebd/4_Collaboration/FYP2526/SmartEmbed_liam

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

# User-configurable settings for training
export RUN_MODE="distributed"   # Use distributed training for 4 GPUs
export GPUS="0,1,2,3"          # Use all 4 GPUs
export NUM_PROC="4"            # 4 processes for 4 GPUs

# Optional override paths (usually set in project_config.yaml)
export TRAIN_DATA="/home/shebd/4_Collaboration/FYP2526/data/train_full_path.jsonl"
export EVAL_DATA=""              # optional
export OUTPUT_DIR="/home/shebd/4_Collaboration/FYP2526/output/models/run_9.6"

./tools/run_train.sh

echo "Training completed successfully!"