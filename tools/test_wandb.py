#!/usr/bin/env python3
"""
简单的wandb测试脚本，用于验证环境是否正常
"""

import os
import time
import random
import wandb

print("正在初始化wandb...")
wandb_entity = os.getenv("WANDB_ENTITY", "smart-search-fyp")
wandb_project = os.getenv("WANDB_PROJECT", "jina-embeddings-finetune-test")

# 初始化wandb
wandb.init(
    project=wandb_project,
    entity=wandb_entity,
    name=f"测试运行-{time.strftime('%m%d-%H%M')}",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5,
    },
)

# 模拟一些训练数据
print("模拟训练过程...")
for epoch in range(5):
    loss = 1.0 - 0.1 * epoch + random.random() * 0.1
    accuracy = 0.7 + 0.05 * epoch + random.random() * 0.05
    
    print(f"Epoch {epoch+1}/5: loss={loss:.4f}, accuracy={accuracy:.4f}")
    
    # 记录到wandb
    wandb.log({
        "epoch": epoch + 1,
        "loss": loss,
        "accuracy": accuracy
    })
    
    # 暂停一下，模拟训练过程
    time.sleep(1)

print("测试完成！")
wandb.finish()
