"""
Trainer for Jina Embeddings V4 model
"""

import os
import json
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
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

from ..models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model, JinaEmbeddingsV4ModelOutput
from ..models.custom_lora_module import MultiAdapterLinear
from ..models.configuration_jina_embeddings_v4 import JinaEmbeddingsV4Config
from ..models.losses import JinaContrastiveLoss, JinaMultiTaskLoss, JinaMatryoshkaLoss
from ..training.training_config import JinaTrainingConfig
from ..data.data_collator import JinaContrastiveDataCollator

logger = logging.getLogger(__name__)
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import IterableDataset
import os


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
        
        # Fix for newer transformers versions
        super().__init__(
            model=model,
            args=training_args,
            processing_class=tokenizer,  # Use processing_class instead of tokenizer
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
        
        # Extract task labels from batch data
        if task_labels is not None and len(task_labels) > 0:
            # Use the first task label in the batch (assuming all items in batch have same task)
            current_task_label = task_labels[0] if isinstance(task_labels, (list, tuple)) else task_labels
        else:
            # Fallback to default task
            current_task_label = "retrieval"
        
        # Forward pass for queries (image branch) and positives (text branch)
        query_outputs = model(task_label=current_task_label, **query_inputs)
        positive_outputs = model(task_label=current_task_label, **positive_inputs)

        # Extract single- and multi-vector embeddings
        if isinstance(query_outputs, JinaEmbeddingsV4ModelOutput):
            query_single = query_outputs.single_vec_emb
            query_multi = query_outputs.multi_vec_emb
        else:
            query_single = query_outputs.get("single_vec_emb", None)
            query_multi = query_outputs.get("multi_vec_emb", None)

        if isinstance(positive_outputs, JinaEmbeddingsV4ModelOutput):
            pos_single = positive_outputs.single_vec_emb
            pos_multi = positive_outputs.multi_vec_emb
        else:
            pos_single = positive_outputs.get("single_vec_emb", None)
            pos_multi = positive_outputs.get("multi_vec_emb", None)

        loss = 0.0
        loss_dict = {}

        temperature = getattr(self.training_config, "temperature", 0.02)

        def _is_dist_initialized() -> bool:
            return dist.is_available() and dist.is_initialized()

        def _concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
            """Gather tensors from all ranks and concatenate along dim 0.
            Gradients only flow to local chunk (standard behavior for DDP contrastive losses).
            """
            if not _is_dist_initialized():
                return tensor
            world_size = dist.get_world_size()
            if world_size == 1:
                return tensor
            tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(tensors_gather, tensor)
            return torch.cat(tensors_gather, dim=0)

        def _gather_concat_keep_local(tensor: torch.Tensor) -> torch.Tensor:
            """Like concat(all_gather(tensor)) but preserves autograd for the local rank's slice
            by replacing gathered local copy with the original tensor.
            """
            if not _is_dist_initialized():
                return tensor
            world_size = dist.get_world_size()
            if world_size == 1:
                return tensor
            gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered, tensor)
            rank = dist.get_rank()
            gathered[rank] = tensor
            return torch.cat(gathered, dim=0)

        def _pad_to_length_2d(x: torch.Tensor, target_len: int, pad_value: float = 0.0) -> torch.Tensor:
            # Pad along dim=1 to target_len; x: (B, T) -> (B, target_len)
            if x.size(1) >= target_len:
                return x
            pad_amount = target_len - x.size(1)
            pad = x.new_full((x.size(0), pad_amount), pad_value)
            return torch.cat([x, pad], dim=1)

        def _pad_to_length_3d(x: torch.Tensor, target_len: int, pad_value: float = 0.0) -> torch.Tensor:
            # Pad along dim=1 to target_len; x: (B, T, D) -> (B, target_len, D)
            if x.size(1) >= target_len:
                return x
            pad_amount = target_len - x.size(1)
            pad = x.new_full((x.size(0), pad_amount, x.size(2)), pad_value)
            return torch.cat([x, pad], dim=1)

        def _info_nce_from_dense_cosine(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            # q: (B, D), p: (B_or_global, D). Both assumed L2-normalized so dot==cosine
            if _is_dist_initialized() and dist.get_world_size() > 1:
                local_bs = q.size(0)
                # Check equal batch size across ranks; if not equal, fall back to local loss
                bs_t = torch.tensor([local_bs], device=q.device)
                bs_all = [torch.zeros_like(bs_t) for _ in range(dist.get_world_size())]
                dist.all_gather(bs_all, bs_t)
                bs_list = [int(b.item()) for b in bs_all]
                equal_bs = all(b == bs_list[0] for b in bs_list)
                if equal_bs:
                    gathered_p = _gather_concat_keep_local(p)  # (B*world, D)
                    logits = (q @ gathered_p.t()) / temperature  # (B, B*world)
                    rank = dist.get_rank()
                    B = bs_list[0]
                    labels = torch.arange(local_bs, device=q.device) + rank * B
                    return torch.nn.functional.cross_entropy(logits, labels)
                # Fallback to local positives only
            logits = (q @ p.t()) / temperature  # (B, B)
            labels = torch.arange(logits.size(0), device=logits.device)
            return torch.nn.functional.cross_entropy(logits, labels)

        def _late_interaction_similarity(q_tokens: torch.Tensor, q_mask: torch.Tensor,
                                         p_tokens: torch.Tensor, p_mask: torch.Tensor) -> torch.Tensor:
            # q_tokens: (Bq, Tq, D), p_tokens: (Bp, Tp, D); already L2-normalized; masked positions are zeroed
            Bq = q_tokens.size(0)
            Bp = p_tokens.size(0)
            S = q_tokens.new_zeros((Bq, Bp))
            for i in range(Bq):
                q_valid_mask = q_mask[i].bool()
                qi = q_tokens[i][q_valid_mask]  # (t_i, D)
                t_i = max(int(q_valid_mask.sum().item()), 1)
                if qi.numel() == 0:
                    continue
                for j in range(Bp):
                    pj = p_tokens[j][p_mask[j].bool()]  # (t_j, D)
                    if pj.numel() == 0:
                        continue
                    sim_ij = qi @ pj.t()  # (t_i, t_j)
                    # max over passage tokens per query token, then average over query tokens
                    s = sim_ij.max(dim=1).values.sum() / t_i
                    S[i, j] = s
            return S

        # Validate tensors
        if query_single is None or pos_single is None or query_multi is None or pos_multi is None:
            logger.error("Missing embeddings from model outputs. Ensure model returns both single and multi vector embeddings.")
            # Create a small regularized loss to keep graph
            dummy_loss = sum(p.pow(2.0).sum() for p in model.parameters() if p.requires_grad) * 1e-8
            loss = dummy_loss if dummy_loss.requires_grad else torch.tensor(0.0, requires_grad=True, device=next(model.parameters()).device)
        else:
            # Sanity: check grads for single branch
            if not query_single.requires_grad:
                logger.warning("Query embeddings (single) do not require grad. Check that LoRA is enabled on the decoder.")
            if not pos_single.requires_grad:
                logger.warning("Positive embeddings (single) do not require grad. Check that LoRA is enabled on the decoder.")

            # Single-vector InfoNCE (dense cosine matrix)
            single_loss = _info_nce_from_dense_cosine(query_single, pos_single)
            loss_dict["loss_single"] = float(single_loss.detach().item())

            # Multi-vector late-interaction InfoNCE
            q_mask = query_inputs.get("attention_mask")
            if q_mask is None:
                q_mask = query_inputs.get("query_attention_mask")
            p_mask = positive_inputs.get("attention_mask")
            if p_mask is None:
                p_mask = positive_inputs.get("positive_attention_mask")
            if q_mask is None or p_mask is None:
                # Fallback: infer mask from non-zero rows
                q_mask = (query_multi.abs().sum(dim=-1) > 0).to(query_multi.dtype)
                p_mask = (pos_multi.abs().sum(dim=-1) > 0).to(pos_multi.dtype)

            # Late-interaction uses dense token-token sims per pair. For global negatives under DDP,
            # we gather the passage side tokens and masks across ranks when shapes are consistent.
            if _is_dist_initialized() and dist.get_world_size() > 1:
                local_bs = query_multi.size(0)
                bs_t = torch.tensor([local_bs], device=query_multi.device)
                bs_all = [torch.zeros_like(bs_t) for _ in range(dist.get_world_size())]
                dist.all_gather(bs_all, bs_t)
                bs_list = [int(b.item()) for b in bs_all]
                if all(b == bs_list[0] for b in bs_list):
                    # Align sequence lengths across ranks for safe all_gather
                    local_T = pos_multi.size(1)
                    T_t = torch.tensor([local_T], device=pos_multi.device)
                    T_all = [torch.zeros_like(T_t) for _ in range(dist.get_world_size())]
                    dist.all_gather(T_all, T_t)
                    max_T = int(torch.stack(T_all).max().item())

                    pos_multi_pad = _pad_to_length_3d(pos_multi, max_T, 0.0)
                    p_mask_pad = _pad_to_length_2d(p_mask, max_T, 0.0)

                    gathered_p = _gather_concat_keep_local(pos_multi_pad)  # (B*world, T*, D)
                    gathered_pm = _concat_all_gather(p_mask_pad)           # (B*world, T*)
                    S_late = _late_interaction_similarity(query_multi, q_mask, gathered_p, gathered_pm)
                    logits_late = S_late / temperature
                    rank = dist.get_rank()
                    B = bs_list[0]
                    labels = torch.arange(local_bs, device=logits_late.device) + rank * B
                    multi_loss = torch.nn.functional.cross_entropy(logits_late, labels)
                else:
                    S_late = _late_interaction_similarity(query_multi, q_mask, pos_multi, p_mask)
                    logits_late = S_late / temperature
                    labels = torch.arange(logits_late.size(0), device=logits_late.device)
                    multi_loss = torch.nn.functional.cross_entropy(logits_late, labels)
            else:
                S_late = _late_interaction_similarity(query_multi, q_mask, pos_multi, p_mask)
                logits_late = S_late / temperature
                labels = torch.arange(logits_late.size(0), device=logits_late.device)
                multi_loss = torch.nn.functional.cross_entropy(logits_late, labels)
            loss_dict["loss_multi"] = float(multi_loss.detach().item())

            loss = single_loss + multi_loss
        
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

    def get_train_dataloader(self) -> DataLoader:
        """
        Build the training dataloader with DistributedSampler when running under torch.distributed.
        Keeps collate_fn and per-device batch size semantics intact.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset")

        # IterableDataset: do not use sampler/shuffle here
        if isinstance(self.train_dataset, IterableDataset):
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=self.args.dataloader_drop_last,
            )

        # Determine distributed context from env to avoid requiring explicit local_rank flag
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))

        if world_size > 1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=self.args.seed,
                drop_last=self.args.dataloader_drop_last,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )
        return dataloader

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
                    # Extract task_label from inputs for model forward pass
                    task_labels = inputs.get("task_labels", None)
                    if task_labels is not None and len(task_labels) > 0:
                        current_task_label = task_labels[0] if isinstance(task_labels, (list, tuple)) else task_labels
                    else:
                        current_task_label = "retrieval"  # Default fallback
                    
                    # Process inputs similar to compute_loss
                    model_inputs = {}
                    
                    # Check if we have query/positive prefixed inputs (training format)
                    has_query_inputs = any(k.startswith('query_') for k in inputs.keys())
                    
                    if has_query_inputs:
                        # Extract query inputs for evaluation (use query as the main input)
                        for key, value in inputs.items():
                            if key.startswith('query_'):
                                model_inputs[key.replace('query_', '')] = value
                            elif not key.startswith('positive_') and key != "task_labels":
                                model_inputs[key] = value
                    else:
                        # Standard inputs without prefixes
                        model_inputs = {k: v for k, v in inputs.items() if k != "task_labels"}
                    
                    # Add task_label
                    model_inputs["task_label"] = current_task_label
                    
                    outputs = model(**model_inputs)
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
        # Only save on main process to avoid write collisions under DDP
        if hasattr(self.args, "should_save") and not self.args.should_save:
            return
        if hasattr(self, "is_world_process_zero") and not self.is_world_process_zero():
            return

        if output_dir is None:
            output_dir = self.args.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model (unwrap DDP/Accelerator wrappers if present)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model_to_save, 'save_pretrained'):
            model_to_save.save_pretrained(output_dir)
        else:
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            
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
    model_or_path,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    task_names: List[str] = None,
    adapter_name: str = "retrieval",
    enable_visual_lora: bool = False,
) -> JinaEmbeddingsV4Model:
    """
    Simplified setup function for model training with LoRA adapters
    Compatible with train.py calling convention
    """
    
    # If a model is passed, use it; otherwise load from path
    if isinstance(model_or_path, str):
        model = JinaEmbeddingsV4Model.from_pretrained(
            model_or_path,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        model = model_or_path
    
    if use_lora:
        # NOTE: enable_visual_lora is currently a no-op here.  The visual tower in the base
        # Jina Embeddings V4 model is already frozen for memory reasons.  We accept the
        # argument so that higher-level callers (tools/train.py) can pass it without
        # breaking, but we intentionally do not inject LoRA layers into the vision
        # transformer at this stage.  This avoids changing call signatures that would
        # require passing task_label through the vision path.  Support can be added
        # later by matching visual.* linears and applying the same MultiAdapterLinear
        # mechanism.

        # If the model already has PEFT, it was likely loaded by JinaEmbeddingsV4Model.from_pretrained,
        # which does not apply the custom MultiAdapterLinear module. We need to unwrap it and re-apply
        # PEFT with the correct configuration.
        if hasattr(model, 'peft_config') and hasattr(model, 'model'):
            model = model.model  # Unwrap the base model from PeftModel

        # Now, apply PEFT with the custom MultiAdapterLinear module
        if use_lora:
            if task_names is None:
                task_names = ["retrieval", "text-matching", "code"]
            target_regex = r"(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(single_vector_projector|multi_vector_projector).*$)"
            # target_regex = r".*(model\.layers\.\d+\.mlp\.(down_proj|gate_proj|up_proj)|model\.layers\.\d+\.self_attn\.(k_proj|q_proj|v_proj|o_proj)).*$|.*(single_vector_projector|multi_vector_projector).*$"
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_regex,
                modules_to_save=None,
                inference_mode=False,
            )
            from torch.nn.modules.linear import Linear
            from functools import partial
            lora_config._custom_modules = {  # type: ignore[attr-defined]
                Linear: partial(MultiAdapterLinear, task_names=task_names)
            }
            model = get_peft_model(model, lora_config)
            try:
                model.set_adapter(adapter_name)
            except Exception:
                pass
            for cfg in getattr(model, 'peft_config', {}).values():
                if hasattr(cfg, 'inference_mode'):
                    cfg.inference_mode = False
            for name, param in model.named_parameters():
                if ('lora_' in name) or ('multi_vector_projector' in name) or ('single_vector_projector' in name):
                    param.requires_grad = True
            # Make sure all LoRA layers are in unmerged (training) state
            from peft.tuners.lora import LoraLayer as _LL
            for mod in model.modules():
                if isinstance(mod, _LL) and getattr(mod, 'merged', False):
                    try:
                        mod.unmerge()
                    except Exception:
                        pass
                if isinstance(mod, _LL):
                    try:
                        mod.disable_adapters = False
                    except AttributeError:
                        logger.warning(f"Could not set disable_adapters=False on {type(mod).__name__}, assuming it's active.")
                        pass
            logger.info("LoRA adapters added per official adapter_config targeting; decoder LoRA active and projector trainable")
    
    return model


def setup_model_for_training_legacy(
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
        # Check if model is already a PEFT model (from pretrained Jina model)
        if hasattr(model, 'peft_config'):
            logger.info("Model is already a PEFT model, updating PEFT configuration for training")
            
            # Force enable training mode and disable inference mode
            model.train()
            
            # Critical: Enable all LoRA parameters for training
            for name, param in model.named_parameters():
                if 'lora_' in name or 'adapters' in name:
                    param.requires_grad = True
                    logger.info(f"Enabled gradients for LoRA parameter: {name}")
            
            # Also make sure PEFT config is set to training mode
            for peft_config in model.peft_config.values():
                peft_config.inference_mode = False
                logger.info(f"Set PEFT config inference_mode=False for training")
        else:
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
                inference_mode=False,  # Explicitly set to False for training
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
        
        # # Log which parameters are trainable
        # logger.info("Trainable parameters:")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         logger.info(f"  {name}: {param.shape}")
    
    return model


def nested_detach(tensors):
    """Detach tensors from computation graph"""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()
