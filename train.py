#!/usr/bin/env python3
"""
Main training script for Jina Embeddings V4 fine-tuning

"""

import sys
import os
import yaml
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from transformers import TrainingArguments, set_seed

# Import our modules
from jina.models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model, JinaEmbeddingsV4Processor
from jina.models.configuration_jina_embeddings_v4 import JinaEmbeddingsV4Config
from jina.training.jina_trainer import JinaEmbeddingTrainer, setup_model_for_training
from jina.training.training_config import JinaTrainingConfig

# Import Liam's data loading solution
from src.datasets.multimodal_dataset import get_training_dataloader

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Jina Embeddings V4 model")
    
    # If no arguments provided, use config file mode
    parser.add_argument("--config_mode", action="store_true", 
                       help="Use project_config.yaml for all settings (default behavior)")
    
    # Data arguments (optional, override config)
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--eval_data", type=str, help="Path to evaluation data")
    
    # Training arguments (optional, override config)
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    
    return parser.parse_args()


def load_config():
    """Load configuration from project_config.yaml"""
    config_path = project_root / "project_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_training_config(project_config, args):
    """
    配置转换桥接函数
    
    功能：将用户友好的 project_config.yaml 转换为代码需要的 JinaTrainingConfig 对象
    
    转换映射：
    ┌─────────────────────────────────────────────────────────────────────┐
    │ project_config.yaml (用户编辑)  →  JinaTrainingConfig (代码使用)    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ base_model_path                 →  model_name_or_path               │
    │ training.epochs                 →  num_train_epochs                 │
    │ training.batch_size             →  per_device_train_batch_size      │
    │ training.learning_rate          →  learning_rate                    │
    │ system.max_seq_length           →  max_seq_length                   │
    │ lora.*                          →  lora_* (直接映射)                │
    │ (默认值)                        →  temperature, margin, 等损失函数参数 │
    └─────────────────────────────────────────────────────────────────────┘
    
    为什么需要这个函数：
    1. JinaEmbeddingTrainer 必须要 JinaTrainingConfig 对象来初始化损失函数
    2. 用户不应该编辑包含100+参数的复杂配置文件
    3. 技术参数(如 temperature)使用经过验证的默认值
    
    Args:
        project_config: 从 project_config.yaml 加载的字典
        args: 命令行参数，可以覆盖配置文件设置
        
    Returns:
        JinaTrainingConfig: 包含所有必需参数的完整配置对象
    """
    
    # Override with command line arguments if provided
    epochs = args.epochs or project_config['training']['epochs']
    batch_size = args.batch_size or project_config['training']['batch_size']
    learning_rate = args.learning_rate or project_config['training']['learning_rate']
    
    # Create JinaTrainingConfig with project_config values
    training_config = JinaTrainingConfig(
        # Model settings (from project_config)
        model_name_or_path=project_config['base_model_path'],
        trust_remote_code=True,
        
        # Training hyperparameters (from project_config + args override)
        output_dir=str(project_root / project_config['training']['output_dir'] / 'finetuned'),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        
        # System settings (from project_config)
        max_seq_length=project_config['system']['max_seq_length'],
        
        # LoRA settings (from project_config)
        use_lora=project_config['training']['use_lora'],
        lora_r=project_config['lora']['r'],
        lora_alpha=project_config['lora']['alpha'],
        lora_dropout=project_config['lora']['dropout'],
        
        # Loss function settings (使用经过验证的默认值)
        # 这些参数影响对比学习的效果，使用 Jina 推荐的值
        temperature=0.02,                    # 对比学习温度参数
        margin=0.0,                         # 损失函数边界
        matryoshka_dims=[128, 256, 512, 1024],  # 多维度嵌入支持
        use_matryoshka=False,               # 暂时禁用，专注基础训练
    )
    
    return training_config


def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    config = load_config()
    
    # Create unified training configuration
    training_config = create_training_config(config, args)
    
    print("=== Jina Embeddings V4 Training ===")
    print(f"Project root: {project_root}")
    print(f"Base model path: {training_config.model_name_or_path}")
    
    # Validate base model path
    base_model_path = Path(training_config.model_name_or_path)
    if not base_model_path.exists():
        print(f"❌ Base model path does not exist: {base_model_path}")
        sys.exit(1)
    
    # Set up training parameters from unified config
    train_data_path = args.train_data or str(project_root / config['data']['processed_dir'] / 'train.jsonl')
    eval_data_path = args.eval_data if args.eval_data else None
    
    output_dir = Path(training_config.output_dir)
    
    print(f"📊 Training data: {train_data_path}")
    print(f"📈 Training config: {training_config.num_train_epochs} epochs, batch_size={training_config.per_device_train_batch_size}, lr={training_config.learning_rate}")
    print(f"💾 Output directory: {output_dir}")
    
    # Check training data exists
    if not Path(train_data_path).exists():
        print(f"❌ Training data not found: {train_data_path}")
        print("💡 Please create training data in the specified path")
        sys.exit(1)
    
    # Set random seed
    set_seed(42)
    
    try:
        # Load data using Liam's solution
        # ---- NEW: build real dataloader via src.datasets.multimodal_dataset ---- #
        from src.datasets.multimodal_dataset import get_training_dataloader

        train_dataloader = get_training_dataloader(data_config)
        train_dataset = train_dataloader.dataset
        
        # Load model and processor
        logger.info("Loading model and processor...")
        processor = JinaEmbeddingsV4Processor.from_pretrained(str(base_model_path))
        model = JinaEmbeddingsV4Model.from_pretrained(str(base_model_path))
        
        # Setup model for training
        if training_config.use_lora:
            model = setup_model_for_training(
                model,
                use_lora=True,
                lora_r=training_config.lora_r,
                lora_alpha=training_config.lora_alpha,
                lora_dropout=training_config.lora_dropout
            )
        
        # Create dataset (will be replaced with Liam's dataloader)
        # from jina.data.jina_dataset import JinaContrastiveDataset
        
        # train_dataset = JinaContrastiveDataset(
        #     examples=train_examples,
        #     processor=processor,
        #     max_length=training_config.max_seq_length,
        # )
        
        eval_dataset = None
        # if eval_examples:
        #     eval_dataset = JinaContrastiveDataset(
        #         examples=eval_examples,
        #         processor=processor,
        #         max_length=training_config.max_seq_length,
        #     )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=int(training_config.num_train_epochs),
            per_device_train_batch_size=int(training_config.per_device_train_batch_size),
            per_device_eval_batch_size=int(training_config.per_device_train_batch_size),
            learning_rate=float(training_config.learning_rate),
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            eval_steps=500 if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",  # Fixed back to correct parameter name
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            report_to="none",
            bf16=bool(config['system']['bf16']),  # Ensure boolean type
            dataloader_pin_memory=False,
        )
        
        # Initialize trainer with both TrainingArguments and JinaTrainingConfig
        trainer = JinaEmbeddingTrainer(
            model=model,
            training_config=training_config,  # 🔥 This was missing!
            training_args=training_args,
            tokenizer=processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving final model to {output_dir}")
        trainer.save_model()
        processor.save_pretrained(str(output_dir))
        
        print("🎉 Training completed successfully!")
        print(f"📁 Model saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
