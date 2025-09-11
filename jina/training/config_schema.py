"""
Configuration schema for Jina Embeddings V4 training.

This file contains the data structure definitions for training configurations.
All default values are managed through the unified configuration system:
- system_config.yaml (system defaults)
- user_config.yaml (user overrides)

Use jina.utils.config_manager.create_training_config_from_unified() to create instances.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch
from transformers import TrainingArguments

@dataclass
class JinaTrainingConfig(TrainingArguments):
    """Configuration schema for Jina Embeddings V4 training.
    
    This class defines the structure of training configuration parameters.
    Default values are now managed by the unified configuration system.
    Use config_manager.create_training_config_from_unified() to create instances.
    """
    
    # Model settings
    model_name_or_path: Optional[str] = None  # Model path or name (provided via config system)
    base_model_path: Optional[str] = None     # Base model path (provided via config system)
    model_revision: str = "main"              # Git branch/tag for model version control
    torch_dtype: Optional[str] = "auto"       # Data type for model weights (auto/float16/bfloat16)
    trust_remote_code: bool = True            # Required for Jina custom model architecture
    
    # Training hyperparameters
    output_dir: str = "./results"
    overwrite_output_dir: bool = False
    do_train: bool = True
    do_eval: bool = True
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    logging_strategy: str = "steps"
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 2
    
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    
    # Data settings
    max_seq_length: int = 256
    preprocessing_num_workers: int = 1
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    
    # Data paths
    train_data: Optional[str] = None
    eval_data: Optional[str] = None
    image_base_dir: Optional[str] = None
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16                                         # LoRA rank - controls adaptation capacity
    lora_alpha: int = 16                                     # LoRA scaling factor
    lora_dropout: float = 0.1                                # Dropout rate for LoRA layers
    lora_bias: str = "none"                                  # LoRA bias configuration
    task_names: List[str] = field(default_factory=lambda: ["retrieval", "text-matching", "code"])  # Task names for multi-adapter LoRA
    enable_visual_lora: bool = False                         # Apply LoRA to visual components
    gradient_checkpointing=False,
    
    # Jina-specific settings
    single_vector_pool_strategy: str = "mean"
    multi_vector_projector_dim: int = 128
    matryoshka_dims: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048])
    use_matryoshka: bool = False
    
    # Loss function settings
    temperature: float = 0.02
    margin: float = 0.0
    use_miner: bool = False
    miner_margin: float = 0.2
    type_of_triplets: str = "all"
    use_simplified_contrastive: bool = True
    
    # Loss weights (Phase 1 Pair Training)
    w1_single: float = 1.0
    w2_multi: float = 1.0
    w3_kl: float = 1.0
    
    # Contrastive learning settings
    negative_sampling_strategy: str = "random"
    num_negatives: int = 7
    
    # Evaluation settings
    eval_batch_size: int = 32
    eval_max_length: int = 512
    eval_device: str = "cuda"
    eval_data_jsonl: Optional[str] = None
    eval_model_path: Optional[str] = None
    eval_base_model_path: Optional[str] = None
    # batch_eval_metrics: bool = False
    # eval_strategy: str = 'no'
    # load_best_model_at_end: bool = False
    # full_determinism: bool = False
    
    # System settings
    seed: int = 42
    data_seed: int = None
    local_rank: int = -1
    use_cpu: bool = False
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True
    
    # GPU Memory optimization
    dataloader_drop_last: bool = True
    ddp_find_unused_parameters: bool = True
    gradient_checkpointing: bool = True
    max_memory_MB: int = 20000
    
    # Logging and monitoring
    report_to: List[str] = field(default_factory=lambda: [])
    run_name: Optional[str] = None
    logging_dir: Optional[str] = None
    
    # WandB Configuration
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_enabled: bool = True
    
    # Inference Configuration
    infer_batch_size: int = 4
    infer_device: str = "cuda"
    save_topk: bool = False
    topk: int = 10
    prompt_name: str = "query"
    save_dir: Optional[str] = None
    
    # Runtime Configuration
    default_run_mode: str = "distributed"
    default_gpus: str = "0,1,2,3"
    default_num_proc: int = 4
    slurm_run_mode: str = "distributed"
    slurm_gpus: str = "0,1,2,3"
    slurm_num_proc: int = 4
    slurm_partition: str = "medimgfmod"
    slurm_account: str = "medimgfmod"
    slurm_nodes: int = 1
    slurm_ntasks_per_node: int = 1
    slurm_gres_gpu: int = 2
    slurm_cpus_per_task: int = 28
    slurm_time: str = "96:00:00"
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    ignore_data_skip: bool = False
    
    def __post_init__(self):
        # Convert string numeric parameters to proper types
        # This handles cases where YAML config has quoted numbers like "1e-4"
        if isinstance(self.learning_rate, str):
            self.learning_rate = float(self.learning_rate)
        if isinstance(self.weight_decay, str):
            self.weight_decay = float(self.weight_decay)
        if isinstance(self.adam_beta1, str):
            self.adam_beta1 = float(self.adam_beta1)
        if isinstance(self.adam_beta2, str):
            self.adam_beta2 = float(self.adam_beta2)
        if isinstance(self.adam_epsilon, str):
            self.adam_epsilon = float(self.adam_epsilon)
        if isinstance(self.max_grad_norm, str):
            self.max_grad_norm = float(self.max_grad_norm)
        if isinstance(self.warmup_ratio, str):
            self.warmup_ratio = float(self.warmup_ratio)
        if isinstance(self.temperature, str):
            self.temperature = float(self.temperature)
        if isinstance(self.margin, str):
            self.margin = float(self.margin)
        if isinstance(self.miner_margin, str):
            self.miner_margin = float(self.miner_margin)
        if isinstance(self.lora_dropout, str):
            self.lora_dropout = float(self.lora_dropout)
        if isinstance(self.w1_single, str):
            self.w1_single = float(self.w1_single)
        if isinstance(self.w2_multi, str):
            self.w2_multi = float(self.w2_multi)
        if isinstance(self.w3_kl, str):
            self.w3_kl = float(self.w3_kl)
            
        super().__post_init__()
        if self.data_seed is None:
            self.data_seed = self.seed
