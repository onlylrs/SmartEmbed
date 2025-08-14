#!/bin/bash
# 进入项目目录（动态获取脚本所在目录）
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")  # Get the parent directory of tools
cd "$PROJECT_ROOT"

# 检查基础模型目录是否存在
BASE_MODEL_DIR="../jina-embeddings-v4-base"
if [ ! -d "$BASE_MODEL_DIR" ]; then
    echo "错误: 基础模型目录不存在: $BASE_MODEL_DIR"
    echo "请确保 jina-embeddings-v4-base 文件夹存在于父目录中"
    exit 1
fi

echo "项目目录: $PROJECT_ROOT"
echo "基础模型目录: $BASE_MODEL_DIR"

# 设置环境变量以确保输出显示
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=warning
export DATASETS_VERBOSITY=warning

# 创建必要的输出目录
mkdir -p outputs/{models,logs,results}
mkdir -p data/{raw,processed,examples}

# # the following 2 lines are only for dealing with cuda out of mem issues - use GPU from 6
# # should be disabled when no one is using the GPU
# export CUDA_VISIBLE_DEVICES=7  # 使用空闲的GPU 6
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 减少内存碎片

echo "=== 开始训练 ==="
echo "使用新的统一训练脚本..."

# 使用新的主训练脚本
python tools/train.py 2>&1 | tee outputs/logs/training_output.log

echo "=== 训练完成 ==="
