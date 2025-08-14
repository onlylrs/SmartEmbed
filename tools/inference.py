#!/usr/bin/env python3
"""
Inference utilities for Jina Embeddings V4 model
"""

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from jina.models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model, JinaEmbeddingsV4Processor


def load_model(model_path: str, device: str = "auto") -> Tuple[JinaEmbeddingsV4Model, str]:
    """
    Load Jina Embeddings V4 model and processor
    
    Args:
        model_path: Path to model directory
        device: Device to load model on (auto, cuda, cpu)
        
    Returns:
        model: Loaded model
        device: Device used
    """
    try:
        # Auto-detect device if not specified
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model from: {model_path}")
        model = JinaEmbeddingsV4Model.from_pretrained(model_path, trust_remote_code=True)
        model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, "none"


def encode_texts(
    model: JinaEmbeddingsV4Model, 
    texts: List[str],
    processor: Optional[JinaEmbeddingsV4Processor] = None,
    task: str = "retrieval",
    prompt_name: Optional[str] = None,
    truncate_dim: Optional[int] = None,
    batch_size: int = 8,
) -> List[List[float]]:
    """
    Encode a list of texts to embeddings
    
    Args:
        model: JinaEmbeddingsV4Model
        texts: List of text strings to encode
        processor: Optional processor (will be loaded from model if not provided)
        task: Task type (retrieval, text-matching, code)
        prompt_name: Optional prompt name (query, passage)
        truncate_dim: Optional dimension to truncate embeddings to
        batch_size: Batch size for encoding
        
    Returns:
        List of embeddings (as lists of floats)
    """
    if processor is None:
        processor = JinaEmbeddingsV4Processor.from_pretrained(model.config._name_or_path)
    
    device = next(model.parameters()).device
    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch_texts = texts[i:i+batch_size]
        
        # Apply prompting based on task
        if task == "retrieval" and prompt_name:
            if prompt_name == "query":
                batch_texts = [f"Represent this query for retrieval: {text}" for text in batch_texts]
            elif prompt_name == "passage":
                batch_texts = [f"Represent this passage for retrieval: {text}" for text in batch_texts]
        
        # Prepare inputs
        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        
        # Get embeddings
        with torch.no_grad():
            model_output = model(**inputs, output_hidden_states=True)
            batch_embeddings = model_output.text_embeds.cpu().numpy()
        
        # Truncate dimensions if specified
        if truncate_dim is not None and truncate_dim < batch_embeddings.shape[1]:
            batch_embeddings = batch_embeddings[:, :truncate_dim]
        
        # Normalize embeddings
        batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        
        embeddings.extend(batch_embeddings.tolist())
    
    return embeddings


def encode_images(
    model: JinaEmbeddingsV4Model,
    image_paths: List[str],
    processor: Optional[JinaEmbeddingsV4Processor] = None,
    truncate_dim: Optional[int] = None,
    batch_size: int = 4,
) -> List[List[float]]:
    """
    Encode a list of images to embeddings
    
    Args:
        model: JinaEmbeddingsV4Model
        image_paths: List of paths to image files
        processor: Optional processor (will be loaded from model if not provided)
        truncate_dim: Optional dimension to truncate embeddings to
        batch_size: Batch size for encoding
        
    Returns:
        List of embeddings (as lists of floats)
    """
    if processor is None:
        processor = JinaEmbeddingsV4Processor.from_pretrained(model.config._name_or_path)
    
    device = next(model.parameters()).device
    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i+batch_size]
        
        # Load images
        batch_images = [processor.image_processor.load_image(path) for path in batch_paths]
        
        # Prepare inputs
        inputs = processor(
            images=batch_images,
            return_tensors="pt",
            padding=True,
        ).to(device)
        
        # Get embeddings
        with torch.no_grad():
            model_output = model(**inputs, output_hidden_states=True)
            batch_embeddings = model_output.image_embeds.cpu().numpy()
        
        # Truncate dimensions if specified
        if truncate_dim is not None and truncate_dim < batch_embeddings.shape[1]:
            batch_embeddings = batch_embeddings[:, :truncate_dim]
        
        # Normalize embeddings
        batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        
        embeddings.extend(batch_embeddings.tolist())
    
    return embeddings
