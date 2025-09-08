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

# proj root location for later use
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import TrainingArguments, set_seed
import os
from datetime import datetime
import wandb

from transformers import AutoModel

# Import our modules
from jina.models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model, JinaEmbeddingsV4Processor
from jina.models.configuration_jina_embeddings_v4 import JinaEmbeddingsV4Config
from jina.training.jina_trainer import JinaEmbeddingTrainer, setup_model_for_training
from jina.training.config_schema import JinaTrainingConfig

# Import unified configuration manager
from jina.utils.config_manager import load_config, create_training_config_from_unified

# Import Liam's data loading solution
from jina.data.multimodal_dataset import get_training_dataloader

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
    
    # path checking
    base_model_path = Path(training_config.model_name_or_path)
    if not base_model_path.exists():
        if local_rank == 0:
            print(f"❌ Base model path does not exist: {base_model_path}")
        sys.exit(1)
    
    # Specify data paths for training and eval
    # highest priority: cmd line args
    train_data_path = args.train_data or str(project_root / 'data' / 'train.jsonl')
    eval_data_path = args.eval_data if args.eval_data else None

    output_dir = Path(training_config.output_dir)

    if local_rank == 0:
        print(f"📊 Training data: {train_data_path}")
        print(f"📈 Training config: {training_config.num_train_epochs} epochs, batch_size={training_config.per_device_train_batch_size}, lr={training_config.learning_rate}")
        print(f"💾 Output directory: {output_dir}")

    # Check training data exists
    if not Path(train_data_path).exists():
        if local_rank == 0:
            print(f"❌ Training data not found: {train_data_path}")
        sys.exit(1)

    # pack all configs related to data loading into data_config
    # later on passed on to get_training_dataloader function
    data_config = {
        "jsonl_path": train_data_path,
        "batch_size": training_config.per_device_train_batch_size,
        "text_max_length": training_config.max_seq_length,
        "image_max_patches": 256,
        "task_name": "retrieval",
        "shuffle": True,
        "num_workers": 0,
        "image_base_dir": config.get('data', {}).get('image_base_dir', None), 
    }
    
    # Set random seed
    set_seed(42)
    
    try:
        """
        call func get_training_dataloader defined in file multimodal_dataset.py
        opens the data source specified in data_config, 
        then creates a new pytorch Dataset containing a processor inside
        finally Dataset is wrapped in a DataLoader - batching, shuffling
        """
        train_dataloader = get_training_dataloader(data_config, model_path=str(base_model_path))
        train_dataset = train_dataloader.dataset
        
        # Load model (processor is already inside Dataset)
        # import pdb; pdb.set_trace()
        base_model_path = "/project/medimgfmod/Generalist/0_Pretrained/jina-embeddings-v4"
        # logger.info(f"Loading model... {str(base_model_path)}")
        
        # model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True, torch_dtype=torch.float16)
        # model = AutoModel.from_pretrained(str(base_model_path), trust_remote_code=True)
        model = JinaEmbeddingsV4Model.from_pretrained(str(base_model_path))
        
        # Move model to GPU immediately after loading to fix Flash Attention warning
        if torch.cuda.is_available():
            model = model.to('cuda')
        
        # import pdb; pdb.set_trace()
        # Setup model for training
        # if training_config.use_lora:
        #     model = setup_model_for_training(
        #         model,
        #         use_lora=True,
        #         lora_r=training_config.lora_r,
        #         lora_alpha=training_config.lora_alpha,
        #         lora_dropout=training_config.lora_dropout,
        #         enable_visual_lora=getattr(training_config, 'enable_visual_lora', False)
        #         # enable 这个待定，目前是config里为false
        #     )
        
        # Force model into training mode
        # train() is defined in pytorch
        # under training mode: dropout and batchnorm layers are active
        model.train()
        
        # CRITICAL FIX: Force enable ALL LoRA parameters for training
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
        
        eval_dataset = None
        
        # Derive a readable run name for wandb / logs
        auto_run_name = training_config.run_name or f"smart-search-fyp-{datetime.now().strftime('%m%d-%H%M')}"

        # Set up training arguments (enable wandb reporting)
        # Check if wandb is enabled in config
        report_to = ["wandb"] if wandb_enabled else []
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=int(training_config.num_train_epochs),
            per_device_train_batch_size=int(training_config.per_device_train_batch_size),
            per_device_eval_batch_size=int(training_config.per_device_train_batch_size),
            learning_rate=float(training_config.learning_rate),
            gradient_accumulation_steps=getattr(training_config, 'gradient_accumulation_steps', 1),
            warmup_steps=100,
            logging_steps=10,
            save_steps=1,
            eval_steps=500 if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",  # Fixed back to correct parameter name
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            report_to=report_to,
            run_name=auto_run_name,
            bf16=bool(config['system']['bf16']),  # Ensure boolean type
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,  # Critical: Don't remove our columns!
            dataloader_drop_last=getattr(training_config, 'dataloader_drop_last', True),
            # Important for models with branches or conditional paths
            ddp_find_unused_parameters=True,
            gradient_checkpointing=False,  # CRITICAL: Must be disabled for LoRA to work with this model arch.
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        )

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
                name=training_args.run_name,
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
            training_args=training_args,
            tokenizer=train_dataloader.dataset.processor,
            data_collator=train_dataloader.collate_fn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
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
        import traceback
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
