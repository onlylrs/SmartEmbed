"""
Production multimodal dataset and dataloader for Jina Embeddings v4 training.

This module provides PyTorch Dataset and DataLoader classes that can be directly
integrated into the training pipeline.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .data_collator import JinaContrastiveDataCollator
import numpy as np

# Import Jina components
import sys
sys.path.append('/homes/rliuar/Desktop/FYP/SmartEmbed')
from jina.models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model, JinaEmbeddingsV4Processor


logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for multimodal text-image data using Jina Embeddings v4.
    
    This dataset loads JSONL data, processes text and images, and returns
    batches compatible with the Jina unified decoder.
    """
    
    def __init__(
        self,
        jsonl_path: str,
        processor: JinaEmbeddingsV4Processor,
        text_max_length: int = 128,
        image_max_patches: int = 256,
        task_name: str = "retrieval",
        image_base_dir: Optional[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            jsonl_path: Path to JSONL data file
            processor: Jina processor for text and image processing
            text_max_length: Maximum text sequence length
            image_max_patches: Maximum number of image patches
            task_name: Task name for Jina model (retrieval, text-matching, code)
            image_base_dir: Base directory for resolving relative image paths
        """
        self.processor = processor
        self.text_max_length = text_max_length
        self.image_max_patches = image_max_patches
        self.task_name = task_name
        self.image_base_dir = image_base_dir or "data0/Images"
        
        # Load and process data
        self.data = self._load_data(jsonl_path)
        logger.info(f"Loaded {len(self.data)} examples from {jsonl_path}")
        
    def _load_data(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """Load and preprocess data from JSONL file."""
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        processed_item = self._process_item(item)
                        if processed_item:
                            data.append(processed_item)
                    except Exception as e:
                        logger.warning(f"Failed to process line {line_num}: {e}")
        return data
    
    def _process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single data item to extract text and image path."""
        processed = {}
        
        # Extract image path
        if 'query_image' in item:
            # Use absolute path directly
            processed['image_path'] = item['query_image']
        elif 'image' in item:
            # Resolve relative path
            processed['image_path'] = str(Path(self.image_base_dir) / item['image'])
        else:
            logger.warning(f"No image field found in item: {item}")
            return None
            
        # Extract text
        if 'positive' in item:
            processed['text'] = item['positive']
        elif 'text' in item:
            processed['text'] = item['text']
        else:
            logger.warning(f"No text field found in item: {item}")
            return None
            
        # Extract negative text if available (for contrastive learning)
        if 'negative' in item:
            processed['negative_text'] = item['negative']
            
        # Extract task info
        if 'task' in item:
            processed['task'] = item['task']
        else:
            processed['task'] = self.task_name
            
        return processed
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing processed text and image data
        """
        item = self.data[idx]
        
        try:
            # Process text
            text_encoded = self._process_text(item['text'])
            
            # Process image
            image_encoded = self._process_image(item['image_path'])
            
            # Combine results
            result = {
                # Text (positive branch)
                'text_input_ids': text_encoded['input_ids'].squeeze(0),
                'text_attention_mask': text_encoded['attention_mask'].squeeze(0),
                # Image (query branch)
                'image_input_ids': image_encoded['input_ids'].squeeze(0),
                'image_attention_mask': image_encoded['attention_mask'].squeeze(0),
                'pixel_values': image_encoded['pixel_values'].squeeze(0),
                'task_label': item['task']
            }
            
            # Add image_grid_thw if present
            if 'image_grid_thw' in image_encoded:
                result['image_grid_thw'] = image_encoded['image_grid_thw'].squeeze(0)
            
            # Add negative text if available
            if 'negative_text' in item:
                neg_encoded = self._process_text(item['negative_text'])
                result['negative_input_ids'] = neg_encoded['input_ids'].squeeze(0)
                result['negative_attention_mask'] = neg_encoded['attention_mask'].squeeze(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            # Return a dummy item to avoid breaking the dataloader
            return self._get_dummy_item()
    
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text using Jina processor."""
        with torch.no_grad():
            encoded = self.processor.process_texts(
                [text],
                max_length=self.text_max_length,
                padding="max_length"
            )
        return encoded
    
    def _process_image(self, image_path: str) -> Dict[str, torch.Tensor]:
        """Process image using Jina processor."""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            with torch.no_grad():
                # Process image
                image_inputs = self.processor.process_images([image])
                
                # Limit patches if needed
                if image_inputs['pixel_values'].shape[1] > self.image_max_patches:
                    image_inputs['pixel_values'] = image_inputs['pixel_values'][:, :self.image_max_patches, :]
                
            return image_inputs
            
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Create dummy image
            dummy_image = Image.new('RGB', (224, 224), color='black')
            with torch.no_grad():
                return self.processor.process_images([dummy_image])
    
    def _get_dummy_item(self) -> Dict[str, Any]:
        """Create a dummy item for error cases."""
        dummy_text = self._process_text("dummy text")
        dummy_image = self._process_image("")  # Will create black image
        
        return {
            # Text branch dummy
            'text_input_ids': dummy_text['input_ids'].squeeze(0),
            'text_attention_mask': dummy_text['attention_mask'].squeeze(0),
            # Image branch dummy (no real tokens, but keep shape)
            'image_input_ids': dummy_text['input_ids'].squeeze(0),  # reuse text tokens as placeholder
            'image_attention_mask': dummy_text['attention_mask'].squeeze(0),
            'pixel_values': dummy_image['pixel_values'].squeeze(0),
            'task_label': self.task_name
        }


class MultimodalCollator:
    """
    Custom collate function for multimodal batches.
    
    Handles variable-length sequences and ensures proper tensor shapes
    for the Jina unified decoder.
    """
    
    def __init__(self, pad_token_id: int = 0):
        """
        Initialize collator.
        
        Args:
            pad_token_id: Token ID to use for padding
        """
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batched tensors ready for model input
        """
        # Extract components
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        pixel_values = [item['pixel_values'] for item in batch]
        task_labels = [item['task_label'] for item in batch]
        
        # Stack tensors (assuming they're already padded from dataset)
        batched = {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'pixel_values': torch.stack(pixel_values),
            'task_labels': task_labels  # Keep as list since they're strings
        }
        
        # Add image_grid_thw if present
        if 'image_grid_thw' in batch[0]:
            image_grid_thw = [item['image_grid_thw'] for item in batch]
            batched['image_grid_thw'] = torch.stack(image_grid_thw)
        
        # Add negative samples if present
        if 'negative_input_ids' in batch[0]:
            neg_input_ids = [item['negative_input_ids'] for item in batch]
            neg_attention_masks = [item['negative_attention_mask'] for item in batch]
            batched['negative_input_ids'] = torch.stack(neg_input_ids)
            batched['negative_attention_mask'] = torch.stack(neg_attention_masks)
        
        return batched


def create_multimodal_dataloader(
    jsonl_path: str,
    processor: JinaEmbeddingsV4Processor,
    batch_size: int = 2,
    text_max_length: int = 128,
    image_max_patches: int = 256,
    task_name: str = "retrieval",
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    image_base_dir: Optional[str] = None
) -> DataLoader:
    """
    Create a production-ready DataLoader for multimodal training.
    
    Args:
        jsonl_path: Path to training data
        processor: Jina processor
        batch_size: Batch size
        text_max_length: Maximum text length
        image_max_patches: Maximum image patches
        task_name: Task name for Jina model
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        image_base_dir: Base directory for images
        
    Returns:
        PyTorch DataLoader ready for training
    """
    # Create dataset
    dataset = MultimodalDataset(
        jsonl_path=jsonl_path,
        processor=processor,
        text_max_length=text_max_length,
        image_max_patches=image_max_patches,
        task_name=task_name,
        image_base_dir=image_base_dir
    )
    
    # Create collator (prefixed keys for Trainer)
    collator = JinaContrastiveDataCollator(
        tokenizer=processor.tokenizer,
        max_length=text_max_length,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=True  # Drop incomplete batches for stable training
    )
    
    logger.info(f"Created DataLoader with {len(dataset)} samples, batch_size={batch_size}")
    return dataloader


def load_processor_and_freeze(model_path: str) -> JinaEmbeddingsV4Processor:
    """
    Load Jina processor and freeze model parameters.
    
    Args:
        model_path: Path to Jina model
        
    Returns:
        Loaded and frozen processor
    """
    # Load processor and model
    processor = JinaEmbeddingsV4Processor.from_pretrained(model_path, trust_remote_code=True)
    model = JinaEmbeddingsV4Model.from_pretrained(model_path, trust_remote_code=True)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad_(False)
    
    # Verify all parameters are frozen
    for name, param in model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} still has requires_grad=True"
    
    logger.info("Loaded and froze Jina processor and model")
    return processor


# Factory function for easy integration
def get_training_dataloader(
    data_config: Dict[str, Any],
    model_path: str = "/homes/rliuar/Desktop/FYP/jina-embedding-v4"
) -> DataLoader:
    """
    Factory function to create training dataloader with config.
    
    Args:
        data_config: Configuration dictionary with keys:
            - jsonl_path: Path to training data
            - batch_size: Batch size (default: 2)
            - text_max_length: Max text length (default: 128)  
            - image_max_patches: Max image patches (default: 256)
            - task_name: Task name (default: "retrieval")
            - shuffle: Whether to shuffle (default: True)
            - num_workers: Number of workers (default: 0)
        model_path: Path to Jina model
        
    Returns:
        Configured DataLoader
    """
    # Load processor
    processor = load_processor_and_freeze(model_path)
    
    # Create dataloader with config
    return create_multimodal_dataloader(
        jsonl_path=data_config['jsonl_path'],
        processor=processor,
        batch_size=data_config.get('batch_size', 2),
        text_max_length=data_config.get('text_max_length', 128),
        image_max_patches=data_config.get('image_max_patches', 256),
        task_name=data_config.get('task_name', 'retrieval'),
        shuffle=data_config.get('shuffle', True),
        num_workers=data_config.get('num_workers', 0),
        pin_memory=data_config.get('pin_memory', True),
        image_base_dir=data_config.get('image_base_dir', None)
    )