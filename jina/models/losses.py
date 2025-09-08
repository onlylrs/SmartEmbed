"""
Loss functions for Jina Embeddings V4 training
This file contains the ACTUAL loss implementations used in training
Migrated from jina_trainer.py to ensure code equivalence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Optional, Union, List, Tuple
import logging

logger = logging.getLogger(__name__)


class JinaPairTraining(nn.Module):
    """
    Pair Training loss implementation for Jina Embeddings V4 (Phase 1)
    
    Implements the contrastive learning approach described in the paper's Phase 1.
    This is a simplified version of the full Ljoint formula, containing:
    - w1 * LNCE(Sdense): Single-vector InfoNCE loss  
    - w2 * LNCE(Slate): Multi-vector late-interaction InfoNCE loss
    - w3 * KL divergence loss
    """
    
    def __init__(
        self,
        temperature: float = 0.02,
    ):
        super().__init__()
        self.temperature = temperature
        
    def _is_dist_initialized(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _concat_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all ranks and concatenate along dim 0.
        Gradients only flow to local chunk (standard behavior for DDP contrastive losses).
        """
        if not self._is_dist_initialized():
            return tensor
        world_size = dist.get_world_size()
        if world_size == 1:
            return tensor
        tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensors_gather, tensor)
        return torch.cat(tensors_gather, dim=0)

    def _gather_concat_keep_local(self, tensor: torch.Tensor) -> torch.Tensor:
        """Like concat(all_gather(tensor)) but preserves autograd for the local rank's slice
        by replacing gathered local copy with the original tensor.
        """
        if not self._is_dist_initialized():
            return tensor
        world_size = dist.get_world_size()
        if world_size == 1:
            return tensor
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        rank = dist.get_rank()
        gathered[rank] = tensor
        return torch.cat(gathered, dim=0)

    def _pad_to_length_2d(self, x: torch.Tensor, target_len: int, pad_value: float = 0.0) -> torch.Tensor:
        # Pad along dim=1 to target_len; x: (B, T) -> (B, target_len)
        if x.size(1) >= target_len:
            return x
        pad_amount = target_len - x.size(1)
        pad = x.new_full((x.size(0), pad_amount), pad_value)
        return torch.cat([x, pad], dim=1)

    def _pad_to_length_3d(self, x: torch.Tensor, target_len: int, pad_value: float = 0.0) -> torch.Tensor:
        # Pad along dim=1 to target_len; x: (B, T, D) -> (B, target_len, D)
        if x.size(1) >= target_len:
            return x
        pad_amount = target_len - x.size(1)
        pad = x.new_full((x.size(0), pad_amount, x.size(2)), pad_value)
        return torch.cat([x, pad], dim=1)

    def _info_nce_from_dense_cosine(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # q: (B, D), p: (B_or_global, D). Both assumed L2-normalized so dot==cosine
        if self._is_dist_initialized() and dist.get_world_size() > 1:
            local_bs = q.size(0)
            # Check equal batch size across ranks; if not equal, fall back to local loss
            bs_t = torch.tensor([local_bs], device=q.device)
            bs_all = [torch.zeros_like(bs_t) for _ in range(dist.get_world_size())]
            dist.all_gather(bs_all, bs_t)
            bs_list = [int(b.item()) for b in bs_all]
            equal_bs = all(b == bs_list[0] for b in bs_list)
            if equal_bs:
                gathered_p = self._gather_concat_keep_local(p)  # (B*world, D)
                logits = (q @ gathered_p.t()) / self.temperature  # (B, B*world)
                rank = dist.get_rank()
                B = bs_list[0]
                labels = torch.arange(local_bs, device=q.device) + rank * B
                return torch.nn.functional.cross_entropy(logits, labels)
            # Fallback to local positives only
        logits = (q @ p.t()) / self.temperature  # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        return torch.nn.functional.cross_entropy(logits, labels)

    def _late_interaction_similarity(self, q_tokens: torch.Tensor, q_mask: torch.Tensor,
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
        
    def compute_single_vector_loss(self, query_single: torch.Tensor, pos_single: torch.Tensor) -> torch.Tensor:
        """Compute single-vector InfoNCE loss"""
        return self._info_nce_from_dense_cosine(query_single, pos_single)
        
    def compute_multi_vector_loss(self, query_multi: torch.Tensor, pos_multi: torch.Tensor, 
                                  q_mask: torch.Tensor, p_mask: torch.Tensor) -> torch.Tensor:
        """Compute multi-vector late-interaction InfoNCE loss"""
        # Late-interaction uses dense token-token sims per pair. For global negatives under DDP,
        # we gather the passage side tokens and masks across ranks when shapes are consistent.
        if self._is_dist_initialized() and dist.get_world_size() > 1:
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

                pos_multi_pad = self._pad_to_length_3d(pos_multi, max_T, 0.0)
                p_mask_pad = self._pad_to_length_2d(p_mask, max_T, 0.0)

                gathered_p = self._gather_concat_keep_local(pos_multi_pad)  # (B*world, T*, D)
                gathered_pm = self._concat_all_gather(p_mask_pad)           # (B*world, T*)
                S_late = self._late_interaction_similarity(query_multi, q_mask, gathered_p, gathered_pm)
                logits_late = S_late / self.temperature
                rank = dist.get_rank()
                B = bs_list[0]
                labels = torch.arange(local_bs, device=logits_late.device) + rank * B
                multi_loss = torch.nn.functional.cross_entropy(logits_late, labels)
            else:
                S_late = self._late_interaction_similarity(query_multi, q_mask, pos_multi, p_mask)
                logits_late = S_late / self.temperature
                labels = torch.arange(logits_late.size(0), device=logits_late.device)
                multi_loss = torch.nn.functional.cross_entropy(logits_late, labels)
        else:
            S_late = self._late_interaction_similarity(query_multi, q_mask, pos_multi, p_mask)
            logits_late = S_late / self.temperature
            labels = torch.arange(logits_late.size(0), device=logits_late.device)
            multi_loss = torch.nn.functional.cross_entropy(logits_late, labels)
        return multi_loss
    
    def compute_kl_divergence_loss(
        self,
        query_single: torch.Tensor,
        pos_single: torch.Tensor,
        query_multi: torch.Tensor,
        pos_multi: torch.Tensor,
        q_mask: torch.Tensor,
        p_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between single-vector and multi-vector similarity distributions.

        LD(B,τ) := DKL(S'_dense(B) || S'_late(B))
        where S'_{i,j} = softmax(S, τ, i, j)
        
        Args:
            query_single: Single-vector query embeddings (B, D)
            pos_single: Single-vector positive embeddings (B, D) 
            query_multi: Multi-vector query embeddings (B, T, D)
            pos_multi: Multi-vector positive embeddings (B, T, D)
            q_mask: Query attention mask (B, T)
            p_mask: Positive attention mask (B, T)
            
        Returns:
            KL divergence loss between the two similarity distributions
        """
        # Compute dense similarity matrix (single-vector)
        # Both tensors are already L2-normalized, so dot product = cosine similarity
        dense_similarities = query_single @ pos_single.t()  # (B, B)
        
        # Compute late-interaction similarity matrix (multi-vector)
        late_similarities = self._late_interaction_similarity(
            query_multi, q_mask, pos_multi, p_mask
        )  # (B, B)
        
        # Apply temperature scaling and softmax normalization
        dense_probs = torch.softmax(dense_similarities / self.temperature, dim=1)  # (B, B)
        late_probs = torch.softmax(late_similarities / self.temperature, dim=1)    # (B, B)
        
        # Compute KL divergence: KL(P || Q) = sum(P * log(P / Q))
        # We use dense as P and late as Q: KL(dense || late)
        # Add small epsilon for numerical stability
        eps = 1e-8
        kl_div = torch.sum(
            dense_probs * torch.log((dense_probs + eps) / (late_probs + eps)), 
            dim=1
        )  # (B,)
        
        # Return mean KL divergence across the batch
        return torch.mean(kl_div)
        
    def forward(
        self,
        query_single: torch.Tensor,
        pos_single: torch.Tensor,
        query_multi: torch.Tensor,
        pos_multi: torch.Tensor,
        q_mask: torch.Tensor,
        p_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the joint loss with KL divergence term.
        
        Following the Jina paper, the joint loss function is:
        L_joint = L_NCE(S_dense) + L_NCE(S_late) + L_D(KL_divergence)
        
        Currently using equal weights (w1 = w2 = w3 = 1) for all three components.
        
        Returns total loss and individual loss components
        """
        # Single-vector InfoNCE (dense cosine matrix)
        single_loss = self.compute_single_vector_loss(query_single, pos_single)
        
        # Multi-vector late-interaction InfoNCE
        multi_loss = self.compute_multi_vector_loss(query_multi, pos_multi, q_mask, p_mask)
        
        # KL divergence between single and multi-vector similarity distributions
        kl_loss = self.compute_kl_divergence_loss(
            query_single, pos_single, query_multi, pos_multi, q_mask, p_mask
        )
        
        # Joint loss with equal weights (as requested)
        total_loss = single_loss + multi_loss + kl_loss
        
        loss_dict = {
            "loss_single": single_loss,
            "loss_multi": multi_loss,
            "loss_kl": kl_loss
        }
        
        return total_loss, loss_dict


# Placeholder classes for future extension (Phase 2 losses)
class JinaMatryoshkaLoss(nn.Module):
    """
    Matryoshka loss for training embeddings at multiple dimensions
    Currently unused but kept for future implementation
    """
    
    def __init__(
        self,
        matryoshka_dims: List[int] = [128, 256, 512, 1024, 2048],
        base_loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.matryoshka_dims = matryoshka_dims
        self.base_loss_fn = base_loss_fn
        logger.warning("JinaMatryoshkaLoss is initialized but not currently used in training")
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Matryoshka loss not yet implemented")


class JinaMultiTaskLoss(nn.Module):
    """
    Multi-task loss function placeholder
    Currently unused but kept for future implementation
    """
    
    def __init__(
        self,
        temperature: float = 0.02,
        margin: float = 0.2,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        logger.warning("JinaMultiTaskLoss is initialized but not currently used in training")
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Multi-task loss not yet implemented")
