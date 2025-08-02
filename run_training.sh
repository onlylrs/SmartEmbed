#!/bin/bash
# 运行 Jina 训练脚本

# 进入项目目录（动态获取脚本所在目录）
SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"
#cd /homes/rgongac/Desktop/FYP/SmartEmbed

# 直接运行训练 (假设已经在正确的环境中)

# 运行训练
python scripts/train/train_jina.py \
    --train_data data0/train.jsonl \
    --data_format jsonl \
    --config_file configs/jina_training_config.json \
    --fp16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-5 \
    --max_seq_length 256 \
    --output_dir ./localdata/rgongac/results/jina_test_run \

echo "Training completed!"
