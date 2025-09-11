#!/usr/bin/env python3
"""
Main training script for Jina Embeddings V4 fine-tuning

"""

import sys
import os
import argparse
import logging
from pathlib import Path

# proj root location for later use
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import set_seed
import os
from datetime import datetime
import wandb
import traceback

from jina.models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model, JinaEmbeddingsV4Processor
from jina.training.jina_trainer import JinaEmbeddingTrainer
from jina.utils.config_manager import load_config, create_training_config_from_unified
from jina.data.multimodal_dataset import create_multimodal_dataloader

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Define command line arguments
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Jina Embeddings V4 model")
    
    # If no arguments provided, use config file mode
    parser.add_argument("--config_mode", action="store_true", 
                       help="Use project_config.yaml for all settings (default behavior)")
    
    # Data arguments (optional, override config)
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--eval_data", type=str, help="Path to evaluation data")
    parser.add_argument("--output_dir", type=str, help="Path to save training outputs")
    
    # Training arguments (optional, override config)
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    
    # Load unified configuration
    config = load_config()
    
    # Create training configuration from unified config + args
    training_config = create_training_config_from_unified(config, args)

    # Pre-compute wandb_enabled so that it's always defined, even if an
    # exception is raised before the later assignment inside the try block.
    wandb_enabled = config.get('wandb', {}).get('enabled', True)
    
    # Only print info from main process to avoid duplicate output in distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("=== Jina Embeddings V4 Training ===")
        print(f"Project root: {project_root}")
        print(f"Base model path: {training_config.model_name_or_path}")
    

    base_model_path = training_config.base_model_path
    train_data_path = training_config.train_data
    output_dir = training_config.output_dir

    if local_rank == 0:
        print(f"📊 Training data: {train_data_path}")
        print(f"📈 Training config: {training_config.num_train_epochs} epochs, batch_size={training_config.per_device_train_batch_size}, lr={training_config.learning_rate}")
        print(f"💾 Output directory: {output_dir}")

    set_seed(42)
    
    try:
        """
        call func create_multimodal_dataloader defined in file multimodal_dataset.py
        opens the data source specified in data_config, 
        then creates a new pytorch Dataset containing a processor inside
        finally Dataset is wrapped in a DataLoader - batching, shuffling
        """
        processor = JinaEmbeddingsV4Processor.from_pretrained(base_model_path, trust_remote_code=True, use_fast=True)
        train_dataloader = create_multimodal_dataloader(
            jsonl_path=train_data_path,
            processor=processor,
            batch_size=training_config.per_device_train_batch_size,
            text_max_length=training_config.max_seq_length,
            image_max_patches=256,
            task_name="retrieval",
            shuffle=True,
            num_workers=training_config.dataloader_num_workers,
            pin_memory=training_config.dataloader_pin_memory,
            image_base_dir=training_config.image_base_dir,
        )
        train_dataset = train_dataloader.dataset
        
        # Load model
        model = JinaEmbeddingsV4Model.from_pretrained(base_model_path)
        
        # Move model to GPU immediately after loading to fix Flash Attention warning
        if torch.cuda.is_available():
            model = model.to('cuda')

        model.train()
        
        lora_params_enabled = 0
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                lora_params_enabled += 1
        
        # Also disable inference mode in PEFT config if applicable
        if hasattr(model, 'peft_config'):
            for peft_config in model.peft_config.values():
                peft_config.inference_mode = False
        
        if local_rank == 0:
            logger.info(f"🔧 Force enabled {lora_params_enabled} LoRA parameters")
        
        # Count trainable parameters
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        total_params = sum(1 for p in model.parameters())
        if local_rank == 0:
            logger.info(f"Total trainable parameters: {trainable_count} / {total_params}")
        
        # Initialize Weights & Biases run
        wandb_entity = os.getenv("WANDB_ENTITY", config.get('wandb', {}).get('entity', "smart-search-fyp"))
        wandb_project = os.getenv("WANDB_PROJECT", config.get('wandb', {}).get('project', "jina-embeddings-finetune"))
        wandb_dir = os.getenv("WANDB_DIR", None)
        
        # Initialize wandb only on main process (rank 0)
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        if wandb_enabled and (world_size == 1 or rank == 0):
            print(f"🔄 Initializing wandb with entity={wandb_entity}, project={wandb_project}")
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                dir=wandb_dir,
                name=training_config.run_name or f"smart-search-fyp-{datetime.now().strftime('%m%d-%H%M')}",
                config={
                    "model_name_or_path": training_config.model_name_or_path,
                    "epochs": int(training_config.num_train_epochs),
                    "batch_size": int(training_config.per_device_train_batch_size),
                    "learning_rate": float(training_config.learning_rate),
                    "max_seq_length": int(training_config.max_seq_length),
                    "use_lora": bool(training_config.use_lora),
                    "lora_r": int(training_config.lora_r),
                    "lora_alpha": int(training_config.lora_alpha),
                    "lora_dropout": float(training_config.lora_dropout),
                    "temperature": float(training_config.temperature),
                    "margin": float(training_config.margin),
                },
            )
        elif local_rank == 0:
            print("ℹ️ Wandb logging disabled in configuration")

        # Initialize trainer with both TrainingArguments and JinaTrainingConfig
        trainer = JinaEmbeddingTrainer(
            model=model,
            training_config=training_config,
            tokenizer=train_dataloader.dataset.processor,
            data_collator=train_dataloader.collate_fn,
            train_dataset=train_dataset,
        )
        
        # Start training
        if local_rank == 0:
            logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving final model to {output_dir}")
        trainer.save_model()
        # Save processor only on rank 0 to avoid file write collisions
        if world_size == 1 or rank == 0:
            train_dataloader.dataset.processor.save_pretrained(str(output_dir))
        
        if local_rank == 0:
            print("🎉 Training completed successfully!")
            print(f"📁 Model saved to: {output_dir}")
        if wandb_enabled and (world_size == 1 or rank == 0):
            try:
                wandb.finish()
                if local_rank == 0:
                    print("📊 Wandb tracking finalized")
            except Exception:
                pass
        
    except Exception as e:
        
        logger.exception("Training failed:")  # logs full traceback
        if local_rank == 0:
            print("❌ Training failed – see traceback above.")
        traceback.print_exc()
        if wandb_enabled and (int(os.environ.get("WORLD_SIZE", "1")) == 1 or int(os.environ.get("RANK", "0")) == 0):
            try:
                wandb.finish()
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
