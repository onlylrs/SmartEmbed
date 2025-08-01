"""
Trainer for Jina Embeddings V4 model
"""

import os
import json
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass

from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.trainer_utils import PredictionOutput
from peft import LoraConfig, get_peft_model, TaskType

from ..models.modeling_qwen2_5_vl import JinaEmbeddingsV4Model, JinaEmbeddingsV4ModelOutput
from ..models.configuration_jina_embeddings_v4 import JinaEmbeddingsV4Config
from ..models.losses import JinaContrastiveLoss, JinaMultiTaskLoss, JinaMatryoshkaLoss
from ..models.training_config import JinaTrainingConfig

logger = logging.getLogger(__name__)


class JinaEmbeddingTrainer(Trainer):
    """
    Custom trainer for Jina Embeddings V4 model
    """
    
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        training_config: JinaTrainingConfig = None,
        training_args: TrainingArguments = None,
        **kwargs
    ):
        self.training_config = training_config
        
        # Set up loss functions
        self.contrastive_loss = JinaContrastiveLoss(
            temperature=training_config.temperature,
            margin=training_config.margin,
        )
        
        self.multi_task_loss = JinaMultiTaskLoss(
            temperature=training_config.temperature,
            margin=training_config.margin,
        )
        
        self.matryoshka_loss = JinaMatryoshkaLoss(
            matryoshka_dims=training_config.matryoshka_dims,
            base_loss_fn=self.contrastive_loss,
        )
        
        super().__init__(
            model=model,
            args=training_args,
            **kwargs
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss
        """
        
        # Extract inputs
        task_labels = inputs.pop("task_labels", None)
        labels = inputs.pop("labels", None)
        
        # Forward pass
        outputs = model(**inputs)
        
        if isinstance(outputs, JinaEmbeddingsV4ModelOutput):
            single_vec_emb = outputs.single_vec_emb
            multi_vec_emb = outputs.multi_vec_emb
        else:
            # Fallback for other model types
            single_vec_emb = outputs.get("single_vec_emb", None)
            multi_vec_emb = outputs.get("multi_vec_emb", None)
        
        loss = 0.0
        loss_dict = {}
        
        # Use single vector embeddings for loss computation
        if single_vec_emb is not None:
            batch_size = single_vec_emb.size(0)
            
            # Split batch into queries and documents
            query_embeddings = single_vec_emb[:batch_size//2]
            doc_embeddings = single_vec_emb[batch_size//2:]
            
            if task_labels is not None:
                # Multi-task loss
                embeddings = {"retrieval": single_vec_emb}  # Simplified for now
                total_loss, task_losses = self.multi_task_loss(
                    embeddings, task_labels, labels
                )
                loss += total_loss
                loss_dict.update(task_losses)
            else:
                # Standard contrastive loss
                if len(query_embeddings) > 0 and len(doc_embeddings) > 0:
                    # Matryoshka loss
                    matryoshka_loss, dim_losses = self.matryoshka_loss(
                        query_embeddings, doc_embeddings, labels
                    )
                    loss += matryoshka_loss
                    loss_dict["matryoshka_loss"] = matryoshka_loss
                    
                    # Log individual dimension losses
                    for dim, dim_loss in dim_losses.items():
                        loss_dict[f"loss_dim_{dim}"] = dim_loss
        
        # Log losses
        if len(loss_dict) > 0:
            self.log(loss_dict)
            
        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Custom evaluation loop
        """
        
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        
        batch_size = dataloader.batch_size
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        logger.info(f"  Batch size = {batch_size}")
        
        model.eval()
        
        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = getattr(dataloader, "dataset", None)
        
        if self.args.past_index >= 0:
            self._past = None
            
        # Initialize containers
        all_losses = []
        all_preds = []
        all_labels = []
        
        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            
            if loss is not None:
                all_losses.append(loss)
            if logits is not None:
                all_preds.append(logits)
            if labels is not None:
                all_labels.append(labels)
                
        # Aggregate results
        if len(all_losses) > 0:
            eval_loss = torch.stack(all_losses).mean().item()
        else:
            eval_loss = None
            
        if len(all_preds) > 0:
            predictions = torch.cat(all_preds, dim=0)
        else:
            predictions = None
            
        if len(all_labels) > 0:
            label_ids = torch.cat(all_labels, dim=0)
        else:
            label_ids = None
            
        return PredictionOutput(
            predictions=predictions,
            label_ids=label_ids,
            metrics={"eval_loss": eval_loss} if eval_loss is not None else {}
        )
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform a prediction step
        """
        
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
                
        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                    
        if prediction_loss_only:
            return (loss, None, None)
            
        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
            
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None
            
        return (loss, logits, labels)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the model and training state
        """
        
        if output_dir is None:
            output_dir = self.args.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            
        # Save tokenizer if available
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            
        # Save training config
        if self.training_config is not None:
            config_dict = self.training_config.__dict__.copy()
            with open(os.path.join(output_dir, "training_config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)
                
        logger.info(f"Model saved to {output_dir}")


def setup_model_for_training(
    training_config: JinaTrainingConfig,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> JinaEmbeddingsV4Model:
    """
    Set up the model for training with LoRA adapters
    """
    
    # Load base model
    model = JinaEmbeddingsV4Model.from_pretrained(
        training_config.model_name_or_path,
        torch_dtype=getattr(torch, training_config.torch_dtype) if training_config.torch_dtype != "auto" else "auto",
        trust_remote_code=training_config.trust_remote_code,
        cache_dir=training_config.cache_dir,
    )
    
    if training_config.use_lora:
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=training_config.lora_r,
            lora_alpha=training_config.lora_alpha,
            lora_dropout=training_config.lora_dropout,
            target_modules=training_config.lora_target_modules,
            bias=training_config.lora_bias,
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        logger.info("LoRA adapters added to model")
        logger.info(f"Trainable parameters: {model.num_parameters()}")
        
    return model


def nested_detach(tensors):
    """Detach tensors from computation graph"""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()
