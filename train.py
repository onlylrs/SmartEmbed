#!/usr/bin/env python3
"""
Main training script for Jina Embeddings V4 fine-tuning
Simplified unified training script
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
from jina.data.jina_dataset import load_training_data, create_dataloaders

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


def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    config = load_config()
    
    print("=== Jina Embeddings V4 Training ===")
    print(f"Project root: {project_root}")
    print(f"Base model path: {config['base_model_path']}")
    
    # Validate base model path
    base_model_path = Path(config['base_model_path'])
    if not base_model_path.exists():
        print(f"❌ Base model path does not exist: {base_model_path}")
        sys.exit(1)
    
    # Set up training parameters (config file + optional overrides)
    train_data_path = args.train_data or str(project_root / config['data']['processed_dir'] / 'train.jsonl')
    eval_data_path = args.eval_data if args.eval_data else None
    
    epochs = args.epochs or config['training']['epochs']
    batch_size = args.batch_size or config['training']['batch_size']
    learning_rate = args.learning_rate or config['training']['learning_rate']
    
    output_dir = project_root / config['training']['output_dir'] / 'finetuned'
    
    print(f"📊 Training data: {train_data_path}")
    print(f"📈 Training config: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
    print(f"💾 Output directory: {output_dir}")
    
    # Check training data exists
    if not Path(train_data_path).exists():
        print(f"❌ Training data not found: {train_data_path}")
        print("💡 Please create training data in the specified path")
        sys.exit(1)
    
    # Set random seed
    set_seed(42)
    
    try:
        # Load data
        logger.info("Loading training data...")
        train_examples = load_training_data(train_data_path, data_format="jsonl")
        eval_examples = None
        
        if eval_data_path and Path(eval_data_path).exists():
            logger.info("Loading evaluation data...")
            eval_examples = load_training_data(eval_data_path, data_format="jsonl")
        elif len(train_examples) > 10:
            # Split for evaluation
            split_idx = int(0.8 * len(train_examples))
            eval_examples = train_examples[split_idx:]
            train_examples = train_examples[:split_idx]
        
        logger.info(f"Training examples: {len(train_examples)}")
        logger.info(f"Evaluation examples: {len(eval_examples) if eval_examples else 0}")
        
        # Load model and processor
        logger.info("Loading model and processor...")
        processor = JinaEmbeddingsV4Processor.from_pretrained(str(base_model_path))
        model = JinaEmbeddingsV4Model.from_pretrained(str(base_model_path))
        
        # Setup model for training
        if config['training']['use_lora']:
            model = setup_model_for_training(
                model,
                use_lora=True,
                lora_r=config['lora']['r'],
                lora_alpha=config['lora']['alpha'],
                lora_dropout=config['lora']['dropout']
            )
        
        # Create data loaders
        train_dataloader, eval_dataloader = create_dataloaders(
            train_examples=train_examples,
            eval_examples=eval_examples,
            processor=processor,
            batch_size=batch_size,
            max_length=config['system']['max_seq_length'],
            dataset_type="contrastive"
        )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps" if eval_dataloader else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataloader else False,
            metric_for_best_model="eval_loss" if eval_dataloader else None,
            report_to="none",
            bf16=config['system']['bf16'],
            dataloader_pin_memory=False,
        )
        
        # Initialize trainer
        trainer = JinaEmbeddingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=eval_dataloader.dataset if eval_dataloader else None,
            processor=processor,
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
