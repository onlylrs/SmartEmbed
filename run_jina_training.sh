#!/bin/bash

# Jina Embeddings V4 Training Script

set -e

echo "Starting Jina Embeddings V4 training..."

# Configuration
CONFIG_FILE="configs/jina_training_config.json"
TRAIN_DATA="data/train.json"
EVAL_DATA="data/eval.json"
OUTPUT_DIR="results/jina_embeddings_v4"

# Create directories
mkdir -p data
mkdir -p results
mkdir -p logs

# Check if training data exists, create sample if not
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Training data not found. Will create sample data during training."
fi

# Run training
python scripts/train/train_jina.py \
    --config_file "$CONFIG_FILE" \
    --train_data "$TRAIN_DATA" \
    --eval_data "$EVAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --use_lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --bf16 \
    --seed 42

echo "Training completed! Model saved to $OUTPUT_DIR"

# Run inference example
echo "Running inference example..."
python scripts/inference/infer_jina.py \
    --model_path "$OUTPUT_DIR" \
    --texts "What is machine learning?" "Deep learning tutorial" \
    --task "retrieval" \
    --prompt_name "query" \
    --output_file "results/sample_embeddings.npy"

echo "Inference completed! Embeddings saved to results/sample_embeddings.npy"
