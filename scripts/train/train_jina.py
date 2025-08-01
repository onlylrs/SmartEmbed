#!/usr/bin/env python3
"""
Training script for Jina Embeddings V4 model
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import (
    TrainingArguments,
    set_seed,
    AutoTokenizer,
)

# Import our modules
from src.models.modeling_qwen2_5_vl import JinaEmbeddingsV4Model, JinaEmbeddingsV4Processor
from src.models.configuration_jina_embeddings_v4 import JinaEmbeddingsV4Config
from src.training.training_config import JinaTrainingConfig
from src.trainer.jina_trainer import JinaEmbeddingTrainer, setup_model_for_training
from src.datasets.jina_dataset import (
    load_training_data, 
    create_dataloaders,
    JinaTrainingExample
)

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
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--eval_data", type=str, help="Path to evaluation data")
    parser.add_argument("--data_format", type=str, default="json", choices=["json", "jsonl", "tsv"], 
                       help="Format of data files")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="/homes/rliuar/Desktop/FYP/jina-embedding-v4",
                       help="Model name or path")
    parser.add_argument("--config_file", type=str, help="Path to training config file")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA adapters")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # System arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 precision")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume from checkpoint")
    
    return parser.parse_args()


def load_config(args) -> JinaTrainingConfig:
    """Load training configuration"""
    
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = JinaTrainingConfig(**config_dict)
    else:
        config = JinaTrainingConfig()
    
    # Override with command line arguments
    config.model_name_or_path = args.model_name_or_path
    config.output_dir = args.output_dir
    config.num_train_epochs = args.num_train_epochs
    config.per_device_train_batch_size = args.per_device_train_batch_size
    config.per_device_eval_batch_size = args.per_device_eval_batch_size
    config.learning_rate = args.learning_rate
    config.max_seq_length = args.max_seq_length
    config.use_lora = args.use_lora
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.lora_dropout = args.lora_dropout
    config.seed = args.seed
    config.bf16 = args.bf16
    config.fp16 = args.fp16
    config.resume_from_checkpoint = args.resume_from_checkpoint
    
    return config


def create_sample_data(output_path: str):
    """Create sample training data for testing"""
    
    sample_data = [
        {
            "query": "What is machine learning?",
            "positive": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "task": "retrieval"
        },
        {
            "query": "How does deep learning work?",
            "positive": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            "task": "retrieval"
        },
        {
            "query": "Python programming tutorial",
            "positive": "def hello_world():\n    print('Hello, World!')\n\nhello_world()",
            "task": "code"
        },
        {
            "query": "What is natural language processing?",
            "positive": "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language.",
            "task": "text-matching"
        },
        {
            "query": "Explain computer vision",
            "positive": "Computer vision is a field of AI that trains computers to interpret and understand the visual world using digital images and videos.",
            "task": "retrieval"
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Sample data created at {output_path}")


def main():
    """Main training function"""
    
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args)
    
    logger.info("="*50)
    logger.info("Jina Embeddings V4 Training")
    logger.info("="*50)
    logger.info(f"Model: {config.model_name_or_path}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Use LoRA: {config.use_lora}")
    logger.info(f"Training epochs: {config.num_train_epochs}")
    logger.info(f"Batch size: {config.per_device_train_batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Check if data exists, create sample if not
    if not os.path.exists(args.train_data):
        logger.warning(f"Training data not found at {args.train_data}")
        logger.info("Creating sample training data...")
        create_sample_data(args.train_data)
    
    # Load training data
    logger.info("Loading training data...")
    train_examples = load_training_data(args.train_data, args.data_format)
    eval_examples = None
    
    if args.eval_data and os.path.exists(args.eval_data):
        logger.info("Loading evaluation data...")
        eval_examples = load_training_data(args.eval_data, args.data_format)
    elif len(train_examples) > 10:
        # Split training data for evaluation
        split_idx = int(0.9 * len(train_examples))
        eval_examples = train_examples[split_idx:]
        train_examples = train_examples[:split_idx]
        logger.info(f"Split data: {len(train_examples)} train, {len(eval_examples)} eval")
    
    logger.info(f"Total training examples: {len(train_examples)}")
    if eval_examples:
        logger.info(f"Total evaluation examples: {len(eval_examples)}")
    
    # Load model and processor
    logger.info("Loading model and processor...")
    
    try:
        # Load Jina Embeddings V4 model from local path
        logger.info(f"Loading Jina model from: {config.model_name_or_path}")
        model = JinaEmbeddingsV4Model.from_pretrained(
            config.model_name_or_path,
            torch_dtype="auto",
            trust_remote_code=config.trust_remote_code,
        )
        logger.info("Successfully loaded Jina Embeddings V4 model")
        
        # Enable gradient checkpointing if configured
        if hasattr(config, 'gradient_checkpointing') and config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Set model to use less memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache
            logger.info("Cleared GPU cache")
        
        # Apply LoRA if configured
        if config.use_lora:
            logger.info("Setting up LoRA adapters...")
            model = setup_model_for_training(config, processor.tokenizer)
            logger.info("LoRA adapters successfully applied")
        else:
            # If not using LoRA, manually enable gradients for specific layers
            logger.info("Setting up model for full fine-tuning...")
            # Enable gradients for projection layers
            for name, param in model.named_parameters():
                if any(keyword in name.lower() for keyword in ['projector', 'embed', 'layer_norm']):
                    param.requires_grad = True
                    logger.info(f"Enabled gradients for: {name}")
        
        # Force enable gradients for projection layers regardless of LoRA
        projection_layers = ['multi_vector_projector', 'single_vector_projector']
        for name, param in model.named_parameters():
            if any(proj_name in name for proj_name in projection_layers):
                param.requires_grad = True
                logger.info(f"Force enabled gradients for projection layer: {name}")
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        actual_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {actual_trainable_params:,}")
        logger.info(f"Trainable percentage: {100 * actual_trainable_params / total_params:.2f}%")
        
        # Print details of trainable parameters
        logger.info("Trainable parameter breakdown:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"  {name}: {param.numel():,} parameters")
        
        # Verify that we have trainable parameters
        if actual_trainable_params == 0:
            logger.error("No trainable parameters found! This will cause gradient errors.")
            raise ValueError("No trainable parameters found. Check LoRA configuration.")
        
        # Load processor
        processor = JinaEmbeddingsV4Processor.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        logger.info("Successfully loaded Jina processor")
        
    except Exception as e:
        logger.error(f"Failed to load Jina model: {e}")
        raise e
    
    # Create datasets and dataloaders
    logger.info("Creating datasets...")
    train_dataloader, eval_dataloader = create_dataloaders(
        train_examples=train_examples,
        eval_examples=eval_examples,
        processor=processor,
        batch_size=config.per_device_train_batch_size,
        max_length=config.max_seq_length,
        num_workers=config.dataloader_num_workers,
        dataset_type="contrastive"
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        warmup_steps=config.warmup_steps,
        logging_strategy=config.logging_strategy,
        logging_steps=config.logging_steps,
        eval_strategy=config.evaluation_strategy if eval_examples else "no",
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        dataloader_drop_last=getattr(config, 'dataloader_drop_last', True),
        bf16=config.bf16,
        fp16=config.fp16,
        tf32=config.tf32,
        seed=config.seed,
        data_seed=config.data_seed,
        remove_unused_columns=False,
        report_to=config.report_to,
        run_name=config.run_name,
        logging_dir=config.logging_dir,
        resume_from_checkpoint=config.resume_from_checkpoint,
        ignore_data_skip=config.ignore_data_skip,
        gradient_checkpointing=getattr(config, 'gradient_checkpointing', True),
        ddp_find_unused_parameters=getattr(config, 'ddp_find_unused_parameters', False),
    )
    
    # Create trainer
    logger.info("Setting up trainer...")
    trainer = JinaEmbeddingTrainer(
        model=model,
        training_config=config,
        training_args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=eval_dataloader.dataset if eval_dataloader else None,
        tokenizer=getattr(processor, 'tokenizer', None),
    )
    
    # Start training
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    
    # Save training results
    with open(os.path.join(config.output_dir, "train_results.json"), "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    
    # Run evaluation if eval data is available
    if eval_examples:
        logger.info("Running evaluation...")
        eval_result = trainer.evaluate()
        
        with open(os.path.join(config.output_dir, "eval_results.json"), "w") as f:
            json.dump(eval_result, f, indent=2)
        
        logger.info(f"Evaluation results: {eval_result}")
    
    logger.info("Training completed!")
    logger.info(f"Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
