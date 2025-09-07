"""
Unified configuration management for SmartEmbed project.

This module provides a centralized way to load and manage all configuration parameters.
Configuration priority (high to low):
1. Command line arguments 
2. user_config.yaml (if exists)
3. system_config.yaml (default system configuration)

Usage:
    from jina.utils.config_manager import load_config
    config = load_config()
    # Access any configuration parameter:
    batch_size = config['training']['per_device_train_batch_size']
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import copy


def deep_merge(base_dict: Dict[Any, Any], override_dict: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Deep merge two dictionaries. override_dict values take precedence.
    
    Args:
        base_dict: Base dictionary (lower priority)
        override_dict: Override dictionary (higher priority)
        
    Returns:
        Merged dictionary
    """
    result = copy.deepcopy(base_dict)
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def load_config(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load unified configuration from system_config.yaml and optional user_config.yaml.
    
    Args:
        project_root: Path to project root. If None, auto-detect from this file location.
        
    Returns:
        Complete configuration dictionary
        
    Raises:
        FileNotFoundError: If system_config.yaml is not found
        yaml.YAMLError: If configuration files are malformed
    """
    if project_root is None:
        # Auto-detect project root (assumes this file is in jina/utils/)
        project_root = Path(__file__).resolve().parents[2]
    
    # Load system default configuration (required)
    system_config_path = project_root / "system_config.yaml"
    if not system_config_path.exists():
        raise FileNotFoundError(f"System configuration not found: {system_config_path}")
    
    try:
        with open(system_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing system_config.yaml: {e}")
    
    # Load user override configuration (optional)
    user_config_path = project_root / "user_config.yaml"
    if user_config_path.exists():
        try:
            with open(user_config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f) or {}
            
            # Deep merge user config into system config
            config = deep_merge(config, user_config)
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing user_config.yaml: {e}")
    
    return config


def get_model_path(config: Dict[str, Any]) -> str:
    """
    Get the model path from unified configuration.
    
    Args:
        config: Configuration dictionary from load_config()
        
    Returns:
        Model path string
        
    Raises:
        ValueError: If no model path found in configuration
    """
    model_path = config.get('model', {}).get('base_model_path')
    
    if not model_path:
        raise ValueError("No model path found in configuration. Please set model.base_model_path in user_config.yaml")
    
    return model_path


def create_training_config_from_unified(config: Dict[str, Any], args: Any = None) -> 'JinaTrainingConfig':
    """
    Create JinaTrainingConfig object from unified configuration.
    
    Args:
        config: Unified configuration dictionary
        args: Command line arguments (optional, for overrides)
        
    Returns:
        JinaTrainingConfig object
    """
    from ..training.training_config import JinaTrainingConfig
    
    # Handle command line overrides
    if args:
        # Override core training parameters if provided via command line
        if hasattr(args, 'epochs') and args.epochs:
            config['training']['num_train_epochs'] = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size:
            config['training']['per_device_train_batch_size'] = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate:
            config['training']['learning_rate'] = args.learning_rate
        if hasattr(args, 'output_dir') and args.output_dir:
            config['training']['output_dir'] = args.output_dir
    
    # Create JinaTrainingConfig with all parameters from unified config
    training_config = JinaTrainingConfig(
        # Model settings
        model_name_or_path=get_model_path(config),
        config_name=config['model']['config_name'],
        tokenizer_name=config['model']['tokenizer_name'],
        cache_dir=config['model']['cache_dir'],
        model_revision=config['model']['model_revision'],
        use_auth_token=config['model']['use_auth_token'],
        torch_dtype=config['model']['torch_dtype'],
        trust_remote_code=config['model']['trust_remote_code'],
        
        # Training hyperparameters
        output_dir=config['training']['output_dir'],
        overwrite_output_dir=config['training']['overwrite_output_dir'],
        do_train=config['training']['do_train'],
        do_eval=config['training']['do_eval'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        eval_steps=config['training']['eval_steps'],
        logging_strategy=config['training']['logging_strategy'],
        logging_steps=config['training']['logging_steps'],
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        
        adam_beta1=config['training']['adam_beta1'],
        adam_beta2=config['training']['adam_beta2'],
        adam_epsilon=config['training']['adam_epsilon'],
        max_grad_norm=config['training']['max_grad_norm'],
        
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        warmup_ratio=config['training']['warmup_ratio'],
        warmup_steps=config['training']['warmup_steps'],
        
        # Data settings
        max_seq_length=config['data']['max_seq_length'],
        preprocessing_num_workers=config['data']['preprocessing_num_workers'],
        dataloader_num_workers=config['data']['dataloader_num_workers'],
        dataloader_pin_memory=config['data']['dataloader_pin_memory'],
        dataloader_drop_last=config['data']['dataloader_drop_last'],
        
        # LoRA settings
        use_lora=config['training']['use_lora'],
        lora_r=config['training']['lora_r'],
        lora_alpha=config['training']['lora_alpha'],
        lora_dropout=config['training']['lora_dropout'],
        lora_bias=config['training']['lora_bias'],
        lora_target_modules=config['training']['lora_target_modules'],
        task_names=config['training']['task_names'],
        enable_visual_lora=config['training']['enable_visual_lora'],
        
        # Jina-specific settings
        single_vector_pool_strategy=config['jina_model']['single_vector_pool_strategy'],
        multi_vector_projector_dim=config['jina_model']['multi_vector_projector_dim'],
        matryoshka_dims=config['jina_model']['matryoshka_dims'],
        use_matryoshka=config['jina_model']['use_matryoshka'],
        
        # Loss function settings
        temperature=config['loss']['temperature'],
        margin=config['loss']['margin'],
        use_miner=config['loss']['use_miner'],
        miner_margin=config['loss']['miner_margin'],
        type_of_triplets=config['loss']['type_of_triplets'],
        use_simplified_contrastive=config['loss']['use_simplified_contrastive'],
        negative_sampling_strategy=config['loss']['negative_sampling_strategy'],
        num_negatives=config['loss']['num_negatives'],
        
        # Evaluation settings
        eval_batch_size=config['evaluation']['eval_batch_size'],
        eval_max_length=config['evaluation']['eval_max_length'],
        
        # System settings
        seed=config['system']['seed'],
        data_seed=config['system']['data_seed'],
        local_rank=config['system']['local_rank'],
        use_cpu=config['system']['use_cpu'],
        fp16=config['system']['fp16'],
        bf16=config['system']['bf16'],
        tf32=config['system']['tf32'],
        gradient_checkpointing=config['system']['gradient_checkpointing'],
        max_memory_MB=config['system']['max_memory_MB'],
        ddp_find_unused_parameters=config['system']['ddp_find_unused_parameters'],
        
        # Logging and monitoring
        report_to=config['logging']['report_to'],
        run_name=config['logging']['run_name'],
        logging_dir=config['logging']['logging_dir'],
        
        # Resume training
        resume_from_checkpoint=config['resume']['resume_from_checkpoint'],
        ignore_data_skip=config['resume']['ignore_data_skip'],
    )
    
    return training_config
