#!/bin/bash
#SBATCH --job-name=jina_train
#SBATCH --partition=preempt               # 或 preempt（免费但可被中断）
#SBATCH --nodes=1                        # 单机多卡，只需 1 个节点
#SBATCH --ntasks-per-node=1              # 只启动一个任务（torchrun 会管理多进程）
#SBATCH --gres=gpu:8                     # 请求 8 个 GPU（DGX 节点有 8 个）
#SBATCH --time=50:00:00                  # normal 分区支持无限时长
#SBATCH --account=medimgfmod
#SBATCH --cpus-per-task=14 
#SBATCH --output=%x-%j.out               # 输出日志文件名
#SBATCH --error=%x-%j.err

# 可选：邮件通知
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fredliam99@hotmail.com

# === 加载环境 ===
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate FYP2526_new

# === 设置路径 ===
SCRIPT_DIR="/home/shebd/4_Collaboration/FYP2526"    # 你的脚本所在路径
REPO_ROOT="/home/shebd/4_Collaboration/FYP2526/SmartEmbed_liam"
ENTRYPOINT="${REPO_ROOT}/tools/train.py"

TRAIN_DATA="/home/shebd/4_Collaboration/FYP2526/data/openi_retrieval.jsonl"
EVAL_DATA=""
OUTPUT_DIR="/home/shebd/4_Collaboration/FYP2526/output/models/run_new"

# === 启动训练 ===
cd "$REPO_ROOT"

# 使用 torchrun，nproc 自动从 GPU 数量获取
srun torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --standalone \
    "$ENTRYPOINT" \
    --train_data "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    ${EVAL_DATA:+--eval_data "$EVAL_DATA"}