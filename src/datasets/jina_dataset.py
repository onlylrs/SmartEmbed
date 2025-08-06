"""
Dataset classes for Jina Embeddings V4 training
"""

import os
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from PIL import Image
import random

from transformers import PreTrainedTokenizer
from ..models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Processor


@dataclass
class JinaTrainingExample:
    """Single training example for Jina Embeddings V4"""
    query: str
    positive: str
    negative: Optional[str] = None
    task: str = "retrieval"
    query_image: Optional[str] = None  # Path to query image
    positive_image: Optional[str] = None  # Path to positive image
    negative_image: Optional[str] = None  # Path to negative image


class JinaEmbeddingDataset(Dataset):
    """
    Dataset for training Jina Embeddings V4 model
    Supports both text-only and multimodal (text + image) data
    """
    
    def __init__(
        self,
        examples: List[JinaTrainingExample],
        processor: JinaEmbeddingsV4Processor,
        max_length: int = 512,
        negative_sampling: str = "random",  # "random" or "hard" 
        num_negatives: int = 1,
    ):
        self.examples = examples
        self.processor = processor
        self.max_length = max_length
        self.negative_sampling = negative_sampling
        self.num_negatives = num_negatives
        
        # Group examples by task for easier sampling
        self.task_examples = {}
        for i, example in enumerate(examples):
            task = example.task
            if task not in self.task_examples:
                self.task_examples[task] = []
            self.task_examples[task].append(i)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training example with query, positive, and negative samples
        """
        example = self.examples[idx]
        
        # Process query
        query_inputs = self._process_text_and_image(
            text=example.query,
            image_path=example.query_image,
            prefix="Query"
        )
        
        # Process positive
        positive_inputs = self._process_text_and_image(
            text=example.positive,
            image_path=example.positive_image,
            prefix="Passage"
        )
        
        # Sample negative examples
        negatives = []
        if example.negative is not None:
            # Use provided negative
            negative_inputs = self._process_text_and_image(
                text=example.negative,
                image_path=example.negative_image,
                prefix="Passage"
            )
            negatives.append(negative_inputs)
        else:
            # Sample random negatives from same task
            negatives = self._sample_negatives(idx, example.task)
        
        # Combine all inputs
        all_inputs = [query_inputs, positive_inputs] + negatives
        
        # Batch the inputs
        batched_inputs = self._batch_inputs(all_inputs)
        
        # Add task labels
        batched_inputs["task_labels"] = [example.task] * len(all_inputs)
        
        # Add labels (0 for query, 1 for positive, 2+ for negatives)
        labels = [0, 1] + list(range(2, 2 + len(negatives)))
        batched_inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        
        return batched_inputs
    
    def _process_text_and_image(
        self, 
        text: str, 
        image_path: Optional[str] = None,
        prefix: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Process text and optional image"""
        
        if prefix:
            text = f"{prefix}: {text}"
        
        if image_path and os.path.exists(image_path):
            # Load and process image
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = self.processor.process_images([image])
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Fallback to text-only
                inputs = self.processor.process_texts(
                    [text], 
                    max_length=self.max_length
                )
        else:
            # Text-only processing
            inputs = self.processor.process_texts(
                [text], 
                max_length=self.max_length
            )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) if v.dim() > 1 else v for k, v in inputs.items()}
        
        return inputs
    
    def _sample_negatives(self, current_idx: int, task: str) -> List[Dict[str, torch.Tensor]]:
        """Sample negative examples from the same task"""
        
        negatives = []
        task_indices = self.task_examples.get(task, [])
        
        # Remove current example from candidates
        candidates = [i for i in task_indices if i != current_idx]
        
        if len(candidates) == 0:
            return negatives
        
        # Sample negatives
        for _ in range(self.num_negatives):
            if len(candidates) == 0:
                break
                
            if self.negative_sampling == "random":
                neg_idx = random.choice(candidates)
            else:
                # For "hard" negative sampling, we would need similarity scores
                # For now, just use random
                neg_idx = random.choice(candidates)
            
            neg_example = self.examples[neg_idx]
            neg_inputs = self._process_text_and_image(
                text=neg_example.positive,  # Use positive text as negative
                image_path=neg_example.positive_image,
                prefix="Passage"
            )
            negatives.append(neg_inputs)
            
            # Remove sampled candidate to avoid duplicates
            candidates.remove(neg_idx)
        
        return negatives
    
    def _batch_inputs(self, inputs_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Batch multiple inputs together"""
        
        if len(inputs_list) == 0:
            return {}
        
        # Get all keys
        all_keys = set()
        for inputs in inputs_list:
            all_keys.update(inputs.keys())
        
        batched = {}
        for key in all_keys:
            tensors = []
            for inputs in inputs_list:
                if key in inputs:
                    tensors.append(inputs[key])
                else:
                    # Handle missing keys by padding with zeros
                    if len(tensors) > 0:
                        # Use shape of first tensor as reference
                        ref_tensor = tensors[0]
                        zero_tensor = torch.zeros_like(ref_tensor)
                        tensors.append(zero_tensor)
            
            if len(tensors) > 0:
                # Stack tensors
                if tensors[0].dim() == 0:  # Scalar tensors
                    batched[key] = torch.stack(tensors)
                else:
                    # Pad to same length if needed
                    max_len = max(t.size(0) for t in tensors)
                    padded_tensors = []
                    for t in tensors:
                        if t.size(0) < max_len:
                            pad_size = [0] * (t.dim() * 2)
                            pad_size[-1] = max_len - t.size(0)
                            t = torch.nn.functional.pad(t, pad_size)
                        padded_tensors.append(t)
                    batched[key] = torch.stack(padded_tensors)
        
        return batched


class JinaContrastiveDataset(Dataset):
    """
    Simplified dataset for contrastive learning
    Each sample returns a query-document pair
    """
    
    def __init__(
        self,
        examples: List[JinaTrainingExample],
        processor: JinaEmbeddingsV4Processor,
        max_length: int = 512,
    ):
        self.examples = examples
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return query and positive document pair"""
        
        example = self.examples[idx]
        
        # Process query and positive
        query_text = f"Query: {example.query}"
        positive_text = f"Passage: {example.positive}"
        
        # Handle images if present
        if example.query_image and os.path.exists(example.query_image):
            query_image = Image.open(example.query_image).convert("RGB")
            query_inputs = self.processor.process_images([query_image])
        else:
            query_inputs = self.processor.process_texts([query_text], max_length=self.max_length)
        
        if example.positive_image and os.path.exists(example.positive_image):
            positive_image = Image.open(example.positive_image).convert("RGB")
            positive_inputs = self.processor.process_images([positive_image])
        else:
            positive_inputs = self.processor.process_texts([positive_text], max_length=self.max_length)
        
        # Remove batch dimension from inputs
        query_inputs = {k: v.squeeze(0) if v.dim() > 1 else v for k, v in query_inputs.items()}
        positive_inputs = {k: v.squeeze(0) if v.dim() > 1 else v for k, v in positive_inputs.items()}
        
        # Return separate inputs instead of concatenating
        # The data collator will handle batching properly
        result = {
            'query_input_ids': query_inputs.get('input_ids'),
            'query_attention_mask': query_inputs.get('attention_mask'),
            'positive_input_ids': positive_inputs.get('input_ids'),
            'positive_attention_mask': positive_inputs.get('attention_mask'),
            'task_labels': example.task,
        }
        
        # Add other keys if present
        for key in query_inputs:
            if key not in ['input_ids', 'attention_mask']:
                result[f'query_{key}'] = query_inputs[key]
        
        for key in positive_inputs:
            if key not in ['input_ids', 'attention_mask']:
                result[f'positive_{key}'] = positive_inputs[key]
        
        return result


def load_training_data(
    data_path: str,
    data_format: str = "json",
) -> List[JinaTrainingExample]:
    """
    Load training data from file
    
    Args:
        data_path: Path to data file
        data_format: Format of data file ("json", "jsonl", "tsv")
    
    Returns:
        List of JinaTrainingExample objects
    """
    
    examples = []
    
    if data_format == "json":
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data:
            example = JinaTrainingExample(
                query=item.get("query", ""),
                positive=item.get("positive", ""),
                negative=item.get("negative"),
                task=item.get("task", "retrieval"),
                query_image=item.get("query_image"),
                positive_image=item.get("positive_image"),
                negative_image=item.get("negative_image"),
            )
            examples.append(example)
            
    elif data_format == "jsonl":
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                example = JinaTrainingExample(
                    query=item.get("query", ""),
                    positive=item.get("positive", ""),
                    negative=item.get("negative"),
                    task=item.get("task", "retrieval"),
                    query_image=item.get("query_image"),
                    positive_image=item.get("positive_image"),
                    negative_image=item.get("negative_image"),
                )
                examples.append(example)
                
    elif data_format == "tsv":
        import pandas as pd
        df = pd.read_csv(data_path, sep='\t')
        
        for _, row in df.iterrows():
            example = JinaTrainingExample(
                query=row.get("query", ""),
                positive=row.get("positive", ""),
                negative=row.get("negative") if pd.notna(row.get("negative")) else None,
                task=row.get("task", "retrieval"),
                query_image=row.get("query_image") if pd.notna(row.get("query_image")) else None,
                positive_image=row.get("positive_image") if pd.notna(row.get("positive_image")) else None,
                negative_image=row.get("negative_image") if pd.notna(row.get("negative_image")) else None,
            )
            examples.append(example)
    
    else:
        raise ValueError(f"Unsupported data format: {data_format}")
    
    return examples


def create_dataloaders(
    train_examples: List[JinaTrainingExample],
    eval_examples: Optional[List[JinaTrainingExample]],
    processor: JinaEmbeddingsV4Processor,
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 0,
    dataset_type: str = "contrastive",  # "contrastive" or "triplet"
):
    """Create train and eval dataloaders"""
    
    from torch.utils.data import DataLoader
    
    if dataset_type == "contrastive":
        train_dataset = JinaContrastiveDataset(
            train_examples, processor, max_length
        )
        eval_dataset = JinaContrastiveDataset(
            eval_examples, processor, max_length
        ) if eval_examples else None
    else:
        train_dataset = JinaEmbeddingDataset(
            train_examples, processor, max_length
        )
        eval_dataset = JinaEmbeddingDataset(
            eval_examples, processor, max_length
        ) if eval_examples else None
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    ) if eval_dataset else None
    
    return train_dataloader, eval_dataloader
