#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# 导入我们自己的模块
from src.datasets.qwen_utils.data_utils import MultimodalDataset, contrastive_collate_fn
from src.models.jina_model import JinaEmbeddingModel

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    pooling_strategy: str = field(
        default="mean",
        metadata={"help": "Pooling strategy to use for embedding extraction"}
    )
    matryoshka_dims: str = field(
        default="128,256,512,1024,2048",
        metadata={"help": "Dimensions for Matryoshka embedding"}
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Whether to use Flash Attention"}
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_path: str = field(
        default="data/processed/multimodal_retrieval.jsonl",
        metadata={"help": "Path to the training data"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for text"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the evaluation data"}
    )

class ContrastiveTrainer(transformers.Trainer):
    """自定义Trainer，实现对比学习损失"""
    
    def __init__(self, *args, temperature: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # 获取嵌入
        outputs = model(**inputs)
        embeddings = outputs.pooler_output  # [batch_size, hidden_size]
        
        # 归一化嵌入
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(embeddings, embeddings.transpose(0, 1)) / self.temperature
        
        # 创建标签（假设批次内成对样本是正样本）
        batch_size = embeddings.size(0)
        labels = torch.arange(batch_size).to(embeddings.device)
        
        # 计算对比损失
        loss = torch.nn.CrossEntropyLoss()(sim_matrix, labels)
        
        return (loss, outputs) if return_outputs else loss

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the --output_dir or add --resume_from_checkpoint."
            )
    
    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": False,
        "trust_remote_code": True
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are not specifying a tokenizer, which is required for this task. "
            "You have to set either --tokenizer_name or --model_name_or_path."
        )
    
    # 加载图像处理器
    from src.utils.qwen_utils.image_processing import Qwen25VLImageProcessor
    image_processor = Qwen25VLImageProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    
    # Load dataset
    train_dataset = MultimodalDataset(
        json_file=data_args.data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=data_args.max_length
    )
    
    eval_dataset = None
    if data_args.eval_data_path and training_args.do_eval:
        eval_dataset = MultimodalDataset(
            json_file=data_args.eval_data_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_length=data_args.max_length
        )
    
    # 转换 Matryoshka 维度字符串为列表
    matryoshka_dims = [int(x) for x in model_args.matryoshka_dims.split(",")]
    
    # Load model
    model = JinaEmbeddingModel.from_pretrained(
        model_args.model_name_or_path,
        pooling_strategy=model_args.pooling_strategy,
        matryoshka_dims=matryoshka_dims,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention else "sdpa"
    )
    
    # Initialize Trainer
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=contrastive_collate_fn,
        temperature=0.05
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy inference
        
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
