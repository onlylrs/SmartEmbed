#!/bin/tcsh

echo "=== 运行wandb测试脚本 ==="

# Wandb 配置
setenv WANDB_ENTITY "smart-search-fyp"
setenv WANDB_PROJECT "jina-embeddings-finetune-test"
# 如果你有API密钥，可以取消下面这行的注释并填入
# setenv WANDB_API_KEY "你的API密钥"

echo "运行测试脚本..."
python test_wandb.py
