"""
Training configuration for Jina Embeddings V4 training
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch
from jina.utils.local_paths import get_path  # new utility


@dataclass
class JinaTrainingConfig:
    """Configuration for Jina Embeddings V4 training"""
    
    # Model settings
    model_name_or_path: str = get_path("base_model_path") or "./jina-embeddings-v4-base"
    config_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
    model_revision: str = "main"
    use_auth_token: bool = False
    torch_dtype: Optional[str] = "auto"
    trust_remote_code: bool = True
    
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
    per_device_train_batch_size: int = 1  # Further reduced for 3090 GPUs
    per_device_eval_batch_size: int = 1   # Further reduced
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    
    # Data settings
    max_seq_length: int = 128  # Further reduced from 256 to save memory
    preprocessing_num_workers: int = 1
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(single_vector_projector|multi_vector_projector).*$)"
    ])
    lora_bias: str = "none"
    task_names: List[str] = field(default_factory=lambda: ["retrieval", "text-matching", "code"])
    
    # Jina-specific settings
    single_vector_pool_strategy: str = "mean"
    multi_vector_projector_dim: int = 128
    matryoshka_dims: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048])
    use_matryoshka: bool = False  # Add matryoshka loss support
    
    # Loss function settings
    temperature: float = 0.02
    margin: float = 0.0
    use_miner: bool = False
    miner_margin: float = 0.2
    type_of_triplets: str = "all"
    
    # Contrastive learning settings
    negative_sampling_strategy: str = "random"  # "random" or "hard"
    num_negatives: int = 7
    
    # Evaluation settings
    eval_batch_size: int = 32
    eval_max_length: int = 512
    
    # System settings
    seed: int = 42
    data_seed: int = None
    local_rank: int = -1
    use_cpu: bool = False
    fp16: bool = True   # Use fp16 for 3090 GPU compatibility
    bf16: bool = False  # 3090 doesn't support bf16
    tf32: bool = True
    
    # GPU Memory optimization
    dataloader_drop_last: bool = True
    ddp_find_unused_parameters: bool = False
    gradient_checkpointing: bool = True  # Enable gradient checkpointing to save memory
    max_memory_MB: int = 20000  # Limit memory usage per GPU (adjust for 3090s)
    
    # Logging and monitoring
    report_to: List[str] = field(default_factory=lambda: [])
    run_name: Optional[str] = None
    logging_dir: Optional[str] = None
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    ignore_data_skip: bool = False
    
    def __post_init__(self):
        if self.data_seed is None:
            self.data_seed = self.seed
