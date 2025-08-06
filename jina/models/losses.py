"""
Loss functions for Jina Embeddings V4 training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, List, Tuple
import logging

logger = logging.getLogger(__name__)


class JinaContrastiveLoss(nn.Module):
    """
    Contrastive loss function for Jina Embeddings V4 training
    Supports both single-vector and multi-vector embeddings
    """
    
    def __init__(
        self,
        temperature: float = 0.02,
        margin: float = 0.0,
        distance_metric: str = "cosine",
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.distance_metric = distance_metric
        self.reduction = reduction
        
    def forward(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query_embeddings: [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
            document_embeddings: [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
            labels: [batch_size] optional labels for supervised learning
        """
        
        # Handle multi-vector embeddings by taking mean pooling
        if query_embeddings.dim() == 3:
            query_embeddings = query_embeddings.mean(dim=1)
        if document_embeddings.dim() == 3:
            document_embeddings = document_embeddings.mean(dim=1)
            
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        document_embeddings = F.normalize(document_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        if self.distance_metric == "cosine":
            similarity_matrix = torch.matmul(query_embeddings, document_embeddings.T)
        elif self.distance_metric == "euclidean":
            # For euclidean, we use negative distance as similarity
            similarity_matrix = -torch.cdist(query_embeddings, document_embeddings, p=2)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
            
        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels if not provided (assume diagonal is positive)
        if labels is None:
            batch_size = query_embeddings.size(0)
            labels = torch.arange(batch_size, device=query_embeddings.device)
            
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels, reduction=self.reduction)
        
        return loss


class JinaTripletLoss(nn.Module):
    """
    Triplet loss for Jina Embeddings V4 training
    """
    
    def __init__(
        self,
        margin: float = 0.2,
        distance_metric: str = "cosine",
        reduction: str = "mean",
    ):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.reduction = reduction
        
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor_embeddings: [batch_size, hidden_dim]
            positive_embeddings: [batch_size, hidden_dim]
            negative_embeddings: [batch_size, hidden_dim]
        """
        
        # Handle multi-vector embeddings
        if anchor_embeddings.dim() == 3:
            anchor_embeddings = anchor_embeddings.mean(dim=1)
        if positive_embeddings.dim() == 3:
            positive_embeddings = positive_embeddings.mean(dim=1)
        if negative_embeddings.dim() == 3:
            negative_embeddings = negative_embeddings.mean(dim=1)
            
        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=-1)
        
        if self.distance_metric == "cosine":
            pos_sim = torch.sum(anchor_embeddings * positive_embeddings, dim=-1)
            neg_sim = torch.sum(anchor_embeddings * negative_embeddings, dim=-1)
            # For cosine similarity, higher is better, so we want pos > neg
            loss = torch.clamp(neg_sim - pos_sim + self.margin, min=0.0)
        elif self.distance_metric == "euclidean":
            pos_dist = torch.norm(anchor_embeddings - positive_embeddings, p=2, dim=-1)
            neg_dist = torch.norm(anchor_embeddings - negative_embeddings, p=2, dim=-1)
            # For euclidean distance, lower is better, so we want pos < neg
            loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
            
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class JinaMultiTaskLoss(nn.Module):
    """
    Multi-task loss function for Jina Embeddings V4
    Combines losses from different tasks with task-specific weights
    """
    
    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
        temperature: float = 0.02,
        margin: float = 0.2,
        loss_type: str = "contrastive",  # "contrastive" or "triplet"
    ):
        super().__init__()
        self.task_weights = task_weights or {
            "retrieval": 1.0,
            "text-matching": 1.0, 
            "code": 1.0
        }
        self.temperature = temperature
        self.margin = margin
        self.loss_type = loss_type
        
        if loss_type == "contrastive":
            self.loss_fn = JinaContrastiveLoss(temperature=temperature)
        elif loss_type == "triplet":
            self.loss_fn = JinaTripletLoss(margin=margin)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
            
    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        task_labels: List[str],
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            embeddings: Dict containing embeddings for each task
            task_labels: List of task names for each sample
            labels: Optional labels for supervised learning
            
        Returns:
            total_loss: Combined loss across all tasks
            task_losses: Individual losses for each task
        """
        
        task_losses = {}
        total_loss = 0.0
        
        # Group samples by task
        unique_tasks = list(set(task_labels))
        
        for task in unique_tasks:
            if task not in embeddings:
                continue
                
            task_indices = [i for i, t in enumerate(task_labels) if t == task]
            if len(task_indices) == 0:
                continue
                
            task_embeddings = embeddings[task][task_indices]
            task_labels_subset = labels[task_indices] if labels is not None else None
            
            if self.loss_type == "contrastive":
                # For contrastive loss, split embeddings into queries and documents
                mid_point = len(task_embeddings) // 2
                query_emb = task_embeddings[:mid_point]
                doc_emb = task_embeddings[mid_point:]
                
                if len(query_emb) > 0 and len(doc_emb) > 0:
                    task_loss = self.loss_fn(query_emb, doc_emb, task_labels_subset)
                    task_losses[task] = task_loss
                    total_loss += self.task_weights.get(task, 1.0) * task_loss
                    
            elif self.loss_type == "triplet":
                # For triplet loss, assume embeddings are [anchor, positive, negative]
                if len(task_embeddings) >= 3:
                    anchor_emb = task_embeddings[0::3]
                    positive_emb = task_embeddings[1::3]
                    negative_emb = task_embeddings[2::3]
                    
                    min_len = min(len(anchor_emb), len(positive_emb), len(negative_emb))
                    if min_len > 0:
                        task_loss = self.loss_fn(
                            anchor_emb[:min_len], 
                            positive_emb[:min_len], 
                            negative_emb[:min_len]
                        )
                        task_losses[task] = task_loss
                        total_loss += self.task_weights.get(task, 1.0) * task_loss
        
        return total_loss, task_losses


class JinaMatryoshkaLoss(nn.Module):
    """
    Matryoshka loss for training embeddings at multiple dimensions
    """
    
    def __init__(
        self,
        matryoshka_dims: List[int] = [128, 256, 512, 1024, 2048],
        matryoshka_weights: Optional[List[float]] = None,
        base_loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.matryoshka_dims = matryoshka_dims
        self.matryoshka_weights = matryoshka_weights or [1.0] * len(matryoshka_dims)
        self.base_loss_fn = base_loss_fn or JinaContrastiveLoss()
        
        assert len(self.matryoshka_dims) == len(self.matryoshka_weights), \
            "Number of dimensions and weights must match"
            
    def forward(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Args:
            query_embeddings: [batch_size, full_dim]
            document_embeddings: [batch_size, full_dim]
            labels: Optional labels
            
        Returns:
            total_loss: Combined loss across all dimensions
            dim_losses: Individual losses for each dimension
        """
        
        total_loss = 0.0
        dim_losses = {}
        
        for dim, weight in zip(self.matryoshka_dims, self.matryoshka_weights):
            # Truncate embeddings to current dimension
            query_emb_truncated = query_embeddings[:, :dim]
            doc_emb_truncated = document_embeddings[:, :dim]
            
            # Compute loss for this dimension
            dim_loss = self.base_loss_fn(query_emb_truncated, doc_emb_truncated, labels)
            dim_losses[dim] = dim_loss
            
            # Add weighted loss to total
            total_loss += weight * dim_loss
            
        return total_loss, dim_losses
