"""
Custom data collator for Jina Embeddings V4 training
"""

import torch
from typing import Dict, List, Any
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorMixin


@dataclass
class JinaDataCollator(DataCollatorMixin):
    """
    Custom data collator for Jina Embeddings V4 training
    Handles the specific data format returned by our datasets
    """
    
    tokenizer: PreTrainedTokenizer = None
    padding: bool = True
    max_length: int = None
    pad_to_multiple_of: int = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features into tensors
        """
        if not features:
            return {}
        
        batch = {}
        
        # Handle tensor keys (input_ids, attention_mask, etc.)
        tensor_keys = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        
        for key in tensor_keys:
            if key in features[0]:
                tensors = [f[key] for f in features if key in f]
                if tensors:
                    batch[key] = self._collate_tensors(tensors, key)
        
        # Handle list keys (task_labels)
        if "task_labels" in features[0]:
            # Flatten all task labels into a single list
            all_task_labels = []
            for f in features:
                if isinstance(f["task_labels"], list):
                    all_task_labels.extend(f["task_labels"])
                else:
                    all_task_labels.append(f["task_labels"])
            batch["task_labels"] = all_task_labels
        
        # Handle other keys
        for key in features[0].keys():
            if key not in tensor_keys and key != "task_labels":
                values = [f[key] for f in features if key in f]
                if values:
                    if isinstance(values[0], torch.Tensor):
                        batch[key] = self._collate_tensors(values, key)
                    else:
                        batch[key] = values
        
        return batch
    
    def _collate_tensors(self, tensors: List[torch.Tensor], key: str) -> torch.Tensor:
        """
        Collate a list of tensors with proper padding
        """
        if not tensors:
            return torch.tensor([])
        
        # Check if all tensors have the same shape
        shapes = [t.shape for t in tensors]
        if len(set(shapes)) == 1:
            # All same shape, can stack directly
            return torch.stack(tensors)
        
        # Need to pad tensors to same shape
        if key in ["input_ids", "attention_mask"]:
            return self._pad_sequence_tensors(tensors, key)
        elif key == "pixel_values":
            return self._pad_image_tensors(tensors)
        else:
            # For other keys, try to stack if possible
            try:
                return torch.stack(tensors)
            except RuntimeError:
                # If can't stack, return as list
                return tensors
    
    def _pad_sequence_tensors(self, tensors: List[torch.Tensor], key: str) -> torch.Tensor:
        """
        Pad sequence tensors (input_ids, attention_mask) to same length
        """
        if not tensors:
            return torch.tensor([])
        
        # Find max length
        max_len = max(t.shape[-1] for t in tensors)
        
        # Apply max_length limit if set
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
        
        padded_tensors = []
        for tensor in tensors:
            # Truncate if necessary
            if tensor.shape[-1] > max_len:
                tensor = tensor[..., :max_len]
            
            # Pad if necessary
            if tensor.shape[-1] < max_len:
                pad_length = max_len - tensor.shape[-1]
                
                if key == "input_ids":
                    pad_value = self.tokenizer.pad_token_id if self.tokenizer else 0
                else:  # attention_mask
                    pad_value = 0
                
                # Create padding tensor with same shape except last dimension
                pad_shape = list(tensor.shape)
                pad_shape[-1] = pad_length
                padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype)
                
                tensor = torch.cat([tensor, padding], dim=-1)
            
            padded_tensors.append(tensor)
        
        return torch.stack(padded_tensors)
    
    def _pad_image_tensors(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Handle image tensors (pixel_values)
        """
        if not tensors:
            return torch.tensor([])
        
        # For images, we usually don't pad but stack if shapes match
        try:
            return torch.stack(tensors)
        except RuntimeError:
            # If shapes don't match, return the first one or handle differently
            return tensors[0] if tensors else torch.tensor([])


@dataclass
class JinaContrastiveDataCollator(JinaDataCollator):
    """
    Specialized collator for contrastive learning data
    """
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features for contrastive learning
        Each feature contains separate query and positive data
        """
        if not features:
            return {}
        
        batch = {}
        
        # Handle query and positive inputs separately
        query_keys = ["query_input_ids", "query_attention_mask"]
        positive_keys = ["positive_input_ids", "positive_attention_mask"]
        
        # Collect query inputs
        query_inputs = []
        positive_inputs = []
        
        for feature in features:
            # Extract query data
            query_data = {}
            for key in query_keys:
                if key in feature and feature[key] is not None:
                    query_data[key.replace('query_', '')] = feature[key]
            if query_data:
                query_inputs.append(query_data)
            
            # Extract positive data
            positive_data = {}
            for key in positive_keys:
                if key in feature and feature[key] is not None:
                    positive_data[key.replace('positive_', '')] = feature[key]
            if positive_data:
                positive_inputs.append(positive_data)
        
        # Collate query inputs
        if query_inputs:
            query_batch = super().__call__(query_inputs)
            for key, value in query_batch.items():
                batch[f'query_{key}'] = value
        
        # Collate positive inputs
        if positive_inputs:
            positive_batch = super().__call__(positive_inputs)
            for key, value in positive_batch.items():
                batch[f'positive_{key}'] = value
        
        # Handle task labels
        if "task_labels" in features[0]:
            batch["task_labels"] = [f["task_labels"] for f in features]
        
        # Handle other keys (like pixel_values)
        for key in features[0].keys():
            if key.startswith(('query_', 'positive_')) and key not in batch:
                # Handle image-related keys
                if 'pixel_values' in key or 'image_grid_thw' in key:
                    values = [f[key] for f in features if key in f and f[key] is not None]
                    if values:
                        try:
                            batch[key] = torch.stack(values)
                        except:
                            batch[key] = values[0] if values else None
        
        return batch
