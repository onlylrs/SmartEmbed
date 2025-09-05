#!/bin/bash

#SBATCH -J meddr #Slurm job name
#SBATCH -t 72:00:00
#SBATCH -p medimgfmod
#SBATCH --nodes=1 # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1 # This needs to match Trainer(devices=...)

#SBATCH --mail-user=shebd@ust.hk #Update your email address
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --output=srun_output/%j/output.txt
#SBATCH --error=srun_output/%j/error.txt
#SBATCH --account=medimgfmod
source ~/.bashrc
source activate torch22


# Go to the job submission directory and run your application

cd /home/shebd/0_Research/1_MedicalLLM/medical_mllm
# python3 8_DataGeneration/database_build/index.py
python3 8_DataGeneration/database_build/retrieved.py
# python3 src/gsco/eval_qwen.py