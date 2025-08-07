"""
Multimodal tokenization demo pipeline.

This module provides functionality to tokenize text and encode images using
the official Jina Embeddings v4 components, with configurable parameters
and artefact saving capabilities.
"""

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModel


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_run_folder(base_dir: str = "intermediate") -> str:
    """
    Create a timestamped subdirectory under the base directory for this run.
    
    Args:
        base_dir: Base directory where the run folder will be created
        
    Returns:
        Path to the created run folder
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_folder = Path(base_dir) / f"tokens_{timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created run folder: {run_folder}")
    return str(run_folder)


def load_jsonl_dataset(path: str) -> List[Dict[str, str]]:
    """
    Load dataset from JSONL file.
    
    Args:
        path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing text and image fields
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(data)} examples from {path}")
    return data


def sample_dataset(data: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    """
    Randomly sample n examples from the dataset deterministically.
    
    Args:
        data: List of data examples
        n: Number of examples to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled examples
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if n >= len(data):
        logger.warning(f"Requested {n} samples but only {len(data)} available. Using all data.")
        return data
    
    sampled = random.sample(data, n)
    logger.info(f"Sampled {len(sampled)} examples with seed {seed}")
    return sampled


def resolve_image_paths(data: List[Dict], image_base_dir: str = "data0/Images") -> List[Dict]:
    """
    Resolve image file paths and extract text from the dataset format.
    
    Args:
        data: List of data examples
        image_base_dir: Base directory for images
        
    Returns:
        List of examples with resolved image paths and text
    """
    resolved_data = []
    for example in data:
        resolved_example = {}
        
        # Handle different possible image field names
        if 'query_image' in example:
            # Use the absolute path directly since it's already absolute in the dataset
            resolved_example['image_path'] = example['query_image']
        elif 'image' in example:
            image_path = Path(image_base_dir) / example['image']
            resolved_example['image_path'] = str(image_path)
        else:
            logger.warning(f"No image field found in example: {example}")
            continue
            
        # Extract text - use 'positive' field as the description
        if 'positive' in example:
            resolved_example['text'] = example['positive']
        elif 'text' in example:
            resolved_example['text'] = example['text']
        else:
            logger.warning(f"No text field found in example: {example}")
            continue
            
        resolved_data.append(resolved_example)
    
    logger.info(f"Resolved image paths and text for {len(resolved_data)} examples")
    return resolved_data


def freeze_model_parameters(model) -> None:
    """
    Freeze all parameters in a model by setting requires_grad=False.
    
    Args:
        model: PyTorch model to freeze
    """
    for param in model.parameters():
        param.requires_grad_(False)
    
    # Assert all parameters are frozen
    for name, param in model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} still has requires_grad=True"
    
    logger.info(f"Frozen all parameters in model")


def load_jina_components(model_name: str = "/homes/rliuar/Desktop/FYP/jina-embedding-v4") -> Tuple[Any, Any, Any]:
    """
    Load Jina tokenizer, processor, and model components.
    
    Args:
        model_name: Name/path of the Jina model
        
    Returns:
        Tuple of (tokenizer, processor, model)
    """
    logger.info(f"Loading Jina components from {model_name}")
    
    try:
        # Import the custom Jina classes
        import sys
        sys.path.append('/homes/rliuar/Desktop/FYP/SmartEmbed')
        from jina.models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model, JinaEmbeddingsV4Processor
        
        processor = JinaEmbeddingsV4Processor.from_pretrained(model_name, trust_remote_code=True)
        model = JinaEmbeddingsV4Model.from_pretrained(model_name, trust_remote_code=True)
        
        # The tokenizer is part of the processor in Jina v4
        tokenizer = processor.tokenizer
        
        # Freeze all model parameters
        freeze_model_parameters(model)
        
        logger.info("Successfully loaded and froze Jina components")
        return tokenizer, processor, model
        
    except Exception as e:
        logger.error(f"Failed to load Jina components: {e}")
        raise


def tokenize_texts(
    texts: List[str], 
    processor, 
    max_length: int = 128
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a list of texts using the Jina processor.
    
    Args:
        texts: List of text strings to tokenize
        processor: Jina processor
        max_length: Maximum sequence length
        
    Returns:
        Dictionary containing tokenized inputs
    """
    with torch.no_grad():
        encoded = processor.process_texts(
            texts,
            max_length=max_length,
            padding="longest"
        )
    
    logger.info(f"Tokenized {len(texts)} texts with max_length={max_length}")
    logger.info(f"Text input_ids shape: {encoded['input_ids'].shape}")
    
    return encoded


def encode_images(
    image_paths: List[str], 
    processor, 
    model, 
    max_patches: int = 256
) -> Dict[str, torch.Tensor]:
    """
    Encode images using the Jina image processor and model.
    
    Args:
        image_paths: List of paths to image files
        processor: Jina processor
        model: Jina model
        max_patches: Maximum number of image patches to keep
        
    Returns:
        Dictionary containing image inputs for the model
    """
    images = []
    for path in image_paths:
        try:
            image = Image.open(path).convert('RGB')
            images.append(image)
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            # Create a dummy image
            images.append(Image.new('RGB', (224, 224), color='black'))
    
    with torch.no_grad():
        # Process images using Jina processor
        image_inputs = processor.process_images(images)
        
        # Limit pixel_values if needed (this is a rough approximation)
        if image_inputs['pixel_values'].shape[1] > max_patches:
            image_inputs['pixel_values'] = image_inputs['pixel_values'][:, :max_patches, :]
    
    logger.info(f"Encoded {len(images)} images with max_patches={max_patches}")
    logger.info(f"Image pixel_values shape: {image_inputs['pixel_values'].shape}")
    
    return image_inputs


def process_batch(
    batch_data: List[Dict],
    processor, 
    model,
    text_max_len: int = 128,
    image_max_len: int = 256
) -> Dict[str, torch.Tensor]:
    """
    Process a batch of multimodal data.
    
    Args:
        batch_data: List of examples with text and image_path fields
        processor: Jina processor
        model: Jina model
        text_max_len: Maximum text sequence length
        image_max_len: Maximum number of image patches
        
    Returns:
        Dictionary containing processed batch tensors
    """
    texts = [example.get('text', '') for example in batch_data]
    image_paths = [example.get('image_path', '') for example in batch_data]
    
    # Tokenize texts
    text_encoded = tokenize_texts(texts, processor, text_max_len)
    
    # Encode images
    image_encoded = encode_images(image_paths, processor, model, image_max_len)
    
    return {
        'input_ids': text_encoded['input_ids'],
        'attention_mask': text_encoded['attention_mask'],
        'pixel_values': image_encoded['pixel_values'],
        'image_grid_thw': image_encoded.get('image_grid_thw', None)
    }


def prepare_decoder_inputs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Prepare batch data in the format expected by the unified decoder.
    
    This function takes the tokenized text and encoded image data and formats
    it according to the interface expected by Jina's unified decoder.
    
    Args:
        batch: Dictionary containing tokenized inputs
            - input_ids: Text token IDs (B, L_t)
            - attention_mask: Text attention mask (B, L_t) 
            - pixel_values: Image patch embeddings (B, L_i, D)
            - image_grid_thw: Image grid dimensions (optional)
            
    Returns:
        Dictionary with keys expected by unified decoder
    """
    decoder_inputs = {
        'input_ids': batch['input_ids'],
        'pixel_values': batch['pixel_values'], 
        'attention_mask': batch['attention_mask']
    }
    
    # Add image_grid_thw if present
    if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
        decoder_inputs['image_grid_thw'] = batch['image_grid_thw']
    
    logger.info("Prepared decoder inputs with keys: " + ", ".join(decoder_inputs.keys()))
    return decoder_inputs


def save_artifacts(
    batch: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    run_folder: str
) -> None:
    """
    Save tokenized artifacts to disk.
    
    Args:
        batch: Dictionary containing tokenized inputs
        config: Configuration dictionary
        run_folder: Path to save artifacts
    """
    run_path = Path(run_folder)
    
    # Save tensors as numpy arrays
    np.save(run_path / "text_input_ids.npy", batch['input_ids'].numpy(), allow_pickle=False)
    np.save(run_path / "img_tokens.npy", batch['pixel_values'].numpy(), allow_pickle=False)
    
    # Create manifest
    manifest = {
        "shapes": {
            "text_input_ids": list(batch['input_ids'].shape),
            "img_tokens": list(batch['pixel_values'].shape)
        },
        "files": {
            "text_input_ids": "text_input_ids.npy",
            "img_tokens": "img_tokens.npy",
            "config": "config.json"
        },
        "config": config,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save config and manifest
    with open(run_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    with open(run_path / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Saved artifacts to {run_folder}")


def run(
    sample_size: int = 50,
    text_max_len: int = 128,
    image_max_len: int = 256,
    batch_size: int = 2,
    save_tensors: bool = True,
    seed: int = 42,
    dataset_path: str = "data0/train.jsonl",
    model_name: str = "/homes/rliuar/Desktop/FYP/jina-embedding-v4"
) -> Dict[str, torch.Tensor]:
    """
    Main function to run the tokenization demo.
    
    Args:
        sample_size: Number of examples to sample
        text_max_len: Maximum text sequence length
        image_max_len: Maximum number of image patches
        batch_size: Batch size for processing
        save_tensors: Whether to save artifacts to disk
        seed: Random seed for reproducibility
        dataset_path: Path to the dataset JSONL file
        model_name: Name/path of the Jina model
        
    Returns:
        Dictionary containing processed batch data
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.info("Starting multimodal tokenization demo")
    logger.info(f"Config: sample_size={sample_size}, text_max_len={text_max_len}, "
                f"image_max_len={image_max_len}, batch_size={batch_size}")
    
    # Create run folder if saving
    run_folder = None
    if save_tensors:
        run_folder = get_run_folder()
    
    # Load and sample dataset
    data = load_jsonl_dataset(dataset_path)
    sampled_data = sample_dataset(data, sample_size, seed)
    resolved_data = resolve_image_paths(sampled_data)
    
    # Load Jina components
    tokenizer, processor, model = load_jina_components(model_name)
    
    # Process all data (for demo, we'll process all at once)
    # In practice, you might want to batch this for memory efficiency
    all_results = process_batch(
        resolved_data, 
        processor, 
        model,
        text_max_len,
        image_max_len
    )
    
    # Print shapes
    logger.info("=== TOKEN SHAPES ===")
    for key, tensor in all_results.items():
        if tensor is not None:
            logger.info(f"{key}: {tensor.shape}")
            if key == 'input_ids':
                # Calculate text lengths (non-zero tokens)
                lengths = (tensor != 0).sum(dim=1).tolist()
                logger.info(f"Text token lengths per sample: {lengths}")
    
    # Prepare decoder inputs
    decoder_inputs = prepare_decoder_inputs(all_results)
    
    # Save artifacts if requested
    if save_tensors and run_folder:
        config = {
            'sample_size': sample_size,
            'text_max_len': text_max_len,
            'image_max_len': image_max_len,
            'batch_size': batch_size,
            'save_tensors': save_tensors,
            'seed': seed,
            'dataset_path': dataset_path,
            'model_name': model_name
        }
        save_artifacts(all_results, config, run_folder)
    
    logger.info("Tokenization demo completed successfully")
    return decoder_inputs