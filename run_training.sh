#!/bin/bash
# 运行 Jina 训练脚本 - 详细输出版本

# 进入项目目录（动态获取脚本所在目录）
SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

# 设置环境变量以确保输出显示
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=warning
export DATASETS_VERBOSITY=warning

echo "=== 开始训练 ==="
echo "当前目录: $(pwd)"
echo "训练数据: data0/train.jsonl"
echo "配置文件: configs/jina_training_config.json"
echo "输出目录: /project/fyp25_hc2/results/jina_test_run"
echo "================"

# 运行训练并强制显示输出
python -u scripts/train/train_jina.py \
    --train_data data0/train.jsonl \
    --data_format jsonl \
    --config_file configs/jina_training_config.json \
    --fp16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-5 \
    --max_seq_length 256 \
    --output_dir /project/fyp25_hc2/results/jina_test_run \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 16 2>&1 | tee training_output.log

echo "=== 训练完成 ==="
echo "日志已保存到: training_output.log"
