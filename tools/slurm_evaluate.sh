#!/bin/bash
#SBATCH --job-name=jina_eval
#SBATCH --partition=preempt               # 或 preempt（免费但可被中断）
#SBATCH --nodes=1                        # 单机多卡，只需 1 个节点
#SBATCH --ntasks-per-node=1              # 只启动一个任务（torchrun 会管理多进程）
#SBATCH --gres=gpu:8                     # 请求 8 个 GPU（DGX 节点有 8 个）
#SBATCH --time=02:00:00                  # 评估通常比训练快，设置 2 小时
#SBATCH --account=medimgfmod
#SBATCH --cpus-per-task=28
#SBATCH --output=%x-%j.out               # 输出日志文件名
#SBATCH --error=%x-%j.err

# 可选：邮件通知
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fredliam99@hotmail.com

# === 加载环境 ===
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate FYP2526_qwen

# === 设置路径 ===
SCRIPT_DIR="/home/shebd/4_Collaboration/FYP2526"    # 你的脚本所在路径
REPO_ROOT="/home/shebd/4_Collaboration/FYP2526/FYP2526_fred"
ENTRYPOINT="${REPO_ROOT}/jina/eval/cross_retrieval_eval.py"

# Load evaluation configuration from unified config
cd "$REPO_ROOT"
eval $(python tools/get_config.py --section evaluation)
eval $(python tools/get_config.py --section runtime)

# === 评估参数配置 ===
DATA_JSONL="${EVAL_DATA:-/home/shebd/4_Collaboration/FYP2526/data/eval.jsonl}"
MODEL_PATH="${EVAL_MODEL_PATH:-/home/shebd/4_Collaboration/FYP2526/output/models/run_0/checkpoint-1000}"  # 使用训练好的模型
BASE_MODEL_PATH="${EVAL_BASE_MODEL_PATH:-/home/shebd/4_Collaboration/FYP2526/jina-embeddings-v4}"              # 基础模型路径
IMAGE_BASE_DIR="${EVAL_IMAGE_BASE_DIR:-/scratch/medimgfmod/Generalist/medical}"     # 图片基础目录
BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
DEVICE="${EVAL_DEVICE:-cuda}"

# === 启动评估 ===
cd "$REPO_ROOT"

echo "=== 评估配置 ==="
echo "Repo:          ${REPO_ROOT}"
echo "Entry:         ${ENTRYPOINT}"
echo "Data JSONL:    ${DATA_JSONL}"
echo "Model path:    ${MODEL_PATH}"
echo "Base model:    ${BASE_MODEL_PATH}"
echo "Image base:    ${IMAGE_BASE_DIR}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Device:        ${DEVICE}"
echo "GPUs:          ${SLURM_GPUS_ON_NODE}"
echo "=================="

# 使用 torchrun 进行分布式评估
srun torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --standalone \
    "$ENTRYPOINT" \
    --data_jsonl "$DATA_JSONL" \
    --model_path "$MODEL_PATH" \
    --base_model_path "$BASE_MODEL_PATH" \
    --image_base_dir "$IMAGE_BASE_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"
