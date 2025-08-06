#!/usr/bin/env python3
"""
Main training script for Jina Embeddings V4 fine-tuning
Uses the base model from jina-embeddings-v4-base folder
"""

import sys
import os
import yaml
from pathlib import Path

def main():
    # 获取项目根目录
    project_root = Path(__file__).parent
    
    # 加载项目配置
    with open(project_root / "project_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 获取基础模型路径
    base_model_path = Path(config['base_model_path'])
    
    # 检查基础模型是否存在
    if not base_model_path.exists():
        print(f"错误: 基础模型目录不存在: {base_model_path}")
        sys.exit(1)
    
    # 添加基础模型路径到Python路径，这样可以导入Jina模型
    sys.path.insert(0, str(base_model_path))
    
    print(f"项目根目录: {project_root}")
    print(f"基础模型路径: {base_model_path}")
    print(f"训练配置: {config['training']}")
    
    try:
        # 导入Jina模型（从基础模型目录）
        from modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model, JinaEmbeddingsV4Processor
        from configuration_jina_embeddings_v4 import JinaEmbeddingsV4Config
        
        print("✓ 成功导入Jina模型")
        
        # 这里可以添加实际的训练逻辑
        # 或者调用现有的训练脚本
        
        # 调用现有的训练脚本
        from scripts.train.train_jina import main as train_main
        
        # 设置训练参数
        sys.argv = [
            'train.py',
            '--train_data', str(project_root / config['data']['processed_dir'] / 'train.jsonl'),
            '--output_dir', str(project_root / config['training']['output_dir'] / 'finetuned'),
            '--num_train_epochs', str(config['training']['epochs']),
            '--per_device_train_batch_size', str(config['training']['batch_size']),
            '--learning_rate', str(config['training']['learning_rate']),
            '--max_seq_length', str(config['system']['max_seq_length']),
        ]
        
        if config['training']['use_lora']:
            sys.argv.extend([
                '--use_lora',
                '--lora_r', str(config['lora']['r']),
                '--lora_alpha', str(config['lora']['alpha']),
                '--lora_dropout', str(config['lora']['dropout'])
            ])
        
        if config['system']['bf16']:
            sys.argv.append('--bf16')
        
        # 运行训练
        train_main()
        
    except ImportError as e:
        print(f"错误: 无法导入Jina模型: {e}")
        print("请确保基础模型目录包含正确的文件")
        sys.exit(1)
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
