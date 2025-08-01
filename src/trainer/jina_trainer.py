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
from ..training.training_config import JinaTrainingConfig
from ..datasets.data_collator import JinaContrastiveDataCollator

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
        tokenizer: PreTrainedTokenizer = None,
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
        
        # Set up data collator
        if 'data_collator' not in kwargs:
            kwargs['data_collator'] = JinaContrastiveDataCollator(
                tokenizer=tokenizer,
                padding=True,
                max_length=training_config.max_seq_length,
                return_tensors="pt"
            )
        
        super().__init__(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            **kwargs
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training loss
        """
        
        # Extract metadata
        task_labels = inputs.pop("task_labels", None)
        
        # Separate query and positive inputs
        query_inputs = {}
        positive_inputs = {}
        
        for key, value in inputs.items():
            if key.startswith('query_'):
                query_inputs[key.replace('query_', '')] = value
            elif key.startswith('positive_'):
                positive_inputs[key.replace('positive_', '')] = value
        
        # Forward pass for queries
        query_outputs = model(
            task_label="retrieval",  # Add required task_label parameter
            **query_inputs
        )
        if isinstance(query_outputs, JinaEmbeddingsV4ModelOutput):
            query_embeddings = query_outputs.single_vec_emb
        else:
            query_embeddings = query_outputs.get("single_vec_emb", None)
            
        # Forward pass for positives
        positive_outputs = model(
            task_label="retrieval",  # Add required task_label parameter
            **positive_inputs
        )
        if isinstance(positive_outputs, JinaEmbeddingsV4ModelOutput):
            positive_embeddings = positive_outputs.single_vec_emb
        else:
            positive_embeddings = positive_outputs.get("single_vec_emb", None)
        
        loss = 0.0
        loss_dict = {}
        
        # Compute contrastive loss
        if query_embeddings is not None and positive_embeddings is not None:
            # Ensure embeddings require gradients
            if not query_embeddings.requires_grad:
                logger.warning("Query embeddings do not require grad. This may cause training issues.")
            if not positive_embeddings.requires_grad:
                logger.warning("Positive embeddings do not require grad. This may cause training issues.")
            
            contrastive_loss = self.contrastive_loss(
                query_embeddings, positive_embeddings
            )
            
            # Ensure loss requires gradients
            if not contrastive_loss.requires_grad:
                logger.error("Contrastive loss does not require grad! Check model parameters.")
                # Add small regularization to ensure gradients flow
                regularization = 1e-6 * sum(p.pow(2.0).sum() for p in model.parameters() if p.requires_grad)
                contrastive_loss = contrastive_loss + regularization
                logger.info("Added regularization to enable gradient computation")
            
            loss += contrastive_loss
            loss_dict["contrastive_loss"] = contrastive_loss.item()
            
            # Add Matryoshka loss if configured
            if self.training_config.use_matryoshka:
                matryoshka_loss = self.matryoshka_loss(
                    query_embeddings, positive_embeddings
                )
                loss += matryoshka_loss
                loss_dict["matryoshka_loss"] = matryoshka_loss.item()
        else:
            logger.error("Query or positive embeddings are None!")
            # Create a loss with gradients
            dummy_loss = sum(p.pow(2.0).sum() for p in model.parameters() if p.requires_grad) * 1e-8
            loss = dummy_loss if dummy_loss.requires_grad else torch.tensor(0.0, requires_grad=True, device=next(model.parameters()).device)
        
        # Log losses
        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
            for loss_name, loss_value in loss_dict.items():
                self.log({loss_name: loss_value})
        
        if return_outputs:
            return loss, {
                "query_outputs": query_outputs,
                "positive_outputs": positive_outputs,
                "loss_dict": loss_dict
            }
        
        return loss

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
        # Configure LoRA with corrected target modules
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=training_config.lora_r,
            lora_alpha=training_config.lora_alpha,
            lora_dropout=training_config.lora_dropout,
            target_modules=[
                # Core Qwen attention and MLP layers
                "q_proj", "k_proj", "v_proj", "o_proj",  # attention
                "gate_proj", "up_proj", "down_proj",     # MLP
                # Jina-specific projection layers
                "multi_vector_projector",
            ],
            bias=training_config.lora_bias,
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Ensure projection layers are trainable even if not in LoRA targets
        for name, param in model.named_parameters():
            if "projector" in name and not param.requires_grad:
                param.requires_grad = True
                logger.info(f"Force enabled gradients for projection layer: {name}")
        
        logger.info("LoRA adapters added to model")
        
        # Print trainable parameter details
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # Log which parameters are trainable
        logger.info("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"  {name}: {param.shape}")
    
    return model


def nested_detach(tensors):
    """Detach tensors from computation graph"""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()
