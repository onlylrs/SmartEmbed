#!/bin/tcsh

echo "=== 启动训练脚本 ==="
echo "正在设置环境变量..."

# Wandb 配置
setenv WANDB_ENTITY "smart-search-fyp"
setenv WANDB_PROJECT "jina-embeddings-finetune"
# 如果你有API密钥，可以取消下面这行的注释并填入
# setenv WANDB_API_KEY "你的API密钥"

# 添加调试选项
setenv PYTHONVERBOSE 1

echo "正在运行训练脚本，请耐心等待库的导入过程..."
echo "首次运行时，库的导入可能需要较长时间..."

# 运行带有更多调试信息的训练脚本
python -c "
import sys
import time

def print_status(msg):
    sys.stdout.write(f'\r{msg}')
    sys.stdout.flush()

print('正在导入基础库...')
import os
import yaml
import argparse
import logging
from pathlib import Path
from typing import Optional

print('正在导入PyTorch...')
import torch

print('正在导入Transformers...')
import transformers
from transformers import TrainingArguments, set_seed

print('正在导入Wandb...')
import wandb

print('正在导入PEFT...')
import peft

print('正在导入项目自定义模块...')
sys.path.insert(0, '$(pwd)')

print('完成所有导入，现在开始运行脚本！')
print('-'*50)
"

if ($status == 0) then
    echo "库导入成功，现在运行完整训练脚本..."
    python train.py
else
    echo "导入过程出现问题，请检查错误信息"
endif
