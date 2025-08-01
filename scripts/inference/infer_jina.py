#!/usr/bin/env python3
"""
Inference script for Jina Embeddings V4 model
"""

import os
import json
import argparse
from typing import List, Union, Optional
import torch
import numpy as np
from PIL import Image

from src.models.modeling_qwen2_5_vl import JinaEmbeddingsV4Model, JinaEmbeddingsV4Processor
from src.models.configuration_jina_embeddings_v4 import JinaEmbeddingsV4Config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run inference with Jina Embeddings V4 model")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--texts", nargs="+", help="List of texts to encode")
    parser.add_argument("--images", nargs="+", help="List of image paths to encode")
    parser.add_argument("--text_file", type=str, help="File containing texts to encode (one per line)")
    parser.add_argument("--image_dir", type=str, help="Directory containing images to encode")
    
    parser.add_argument("--task", type=str, default="retrieval", 
                       choices=["retrieval", "text-matching", "code"],
                       help="Task type for encoding")
    parser.add_argument("--prompt_name", type=str, choices=["query", "passage"],
                       help="Prompt type for text encoding")
    parser.add_argument("--truncate_dim", type=int, choices=[128, 256, 512, 1024, 2048],
                       help="Dimension to truncate embeddings to")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for encoding")
    
    parser.add_argument("--output_file", type=str, help="File to save embeddings to")
    parser.add_argument("--output_format", type=str, default="numpy", 
                       choices=["numpy", "json", "tensor"],
                       help="Output format for embeddings")
    
    parser.add_argument("--return_multivector", action="store_true", 
                       help="Return multi-vector embeddings instead of single-vector")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on")
    
    return parser.parse_args()


def load_model(model_path: str, device: str = "auto"):
    """Load the trained model"""
    
    print(f"Loading model from {model_path}")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = JinaEmbeddingsV4Model.from_pretrained(
            model_path,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        print("Loaded Jina Embeddings V4 model")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    model.to(device)
    model.eval()
    
    return model, device


def load_texts(texts: Optional[List[str]], text_file: Optional[str]) -> List[str]:
    """Load texts from arguments or file"""
    
    all_texts = []
    
    if texts:
        all_texts.extend(texts)
    
    if text_file and os.path.exists(text_file):
        with open(text_file, 'r', encoding='utf-8') as f:
            file_texts = [line.strip() for line in f if line.strip()]
            all_texts.extend(file_texts)
    
    return all_texts


def load_images(images: Optional[List[str]], image_dir: Optional[str]) -> List[str]:
    """Load image paths from arguments or directory"""
    
    all_images = []
    
    if images:
        all_images.extend(images)
    
    if image_dir and os.path.exists(image_dir):
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        for filename in os.listdir(image_dir):
            if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                all_images.append(os.path.join(image_dir, filename))
    
    return all_images


def encode_texts(
    model: JinaEmbeddingsV4Model,
    texts: List[str],
    task: str = "retrieval",
    prompt_name: Optional[str] = None,
    truncate_dim: Optional[int] = None,
    max_length: int = 512,
    batch_size: int = 8,
    return_multivector: bool = False,
):
    """Encode texts using the model"""
    
    if not texts:
        return []
    
    print(f"Encoding {len(texts)} texts...")
    
    model.task = task
    
    embeddings = model.encode_text(
        texts=texts,
        task=task,
        max_length=max_length,
        batch_size=batch_size,
        return_multivector=return_multivector,
        return_numpy=True,
        truncate_dim=truncate_dim,
        prompt_name=prompt_name,
    )
    
    return embeddings


def encode_images(
    model: JinaEmbeddingsV4Model,
    image_paths: List[str],
    task: str = "retrieval",
    truncate_dim: Optional[int] = None,
    batch_size: int = 8,
    return_multivector: bool = False,
):
    """Encode images using the model"""
    
    if not image_paths:
        return []
    
    print(f"Encoding {len(image_paths)} images...")
    
    model.task = task
    
    embeddings = model.encode_image(
        images=image_paths,
        task=task,
        batch_size=batch_size,
        return_multivector=return_multivector,
        return_numpy=True,
        truncate_dim=truncate_dim,
    )
    
    return embeddings


def save_embeddings(
    embeddings: Union[np.ndarray, List[torch.Tensor]],
    output_file: str,
    output_format: str = "numpy",
    metadata: Optional[dict] = None,
):
    """Save embeddings to file"""
    
    print(f"Saving embeddings to {output_file}")
    
    if output_format == "numpy":
        if isinstance(embeddings, list):
            # Convert list of tensors to numpy array
            embeddings = np.array([emb.numpy() if hasattr(emb, 'numpy') else emb for emb in embeddings])
        np.save(output_file, embeddings)
        
        if metadata:
            metadata_file = output_file.replace('.npy', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
    elif output_format == "json":
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        elif isinstance(embeddings, list) and hasattr(embeddings[0], 'numpy'):
            embeddings = [emb.numpy().tolist() for emb in embeddings]
        
        output_data = {
            "embeddings": embeddings,
            "metadata": metadata or {}
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
    elif output_format == "tensor":
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        elif isinstance(embeddings, list):
            embeddings = torch.stack(embeddings)
        
        torch.save({
            "embeddings": embeddings,
            "metadata": metadata or {}
        }, output_file)
    
    print(f"Embeddings saved successfully")


def main():
    """Main inference function"""
    
    args = parse_args()
    
    # Load model
    model, device = load_model(args.model_path, args.device)
    if model is None:
        print("Failed to load model")
        return
    
    print(f"Model loaded on device: {device}")
    print(f"Task: {args.task}")
    
    # Load data
    texts = load_texts(args.texts, args.text_file)
    images = load_images(args.images, args.image_dir)
    
    if not texts and not images:
        print("No texts or images provided for encoding")
        return
    
    print(f"Found {len(texts)} texts and {len(images)} images")
    
    # Encode data
    all_embeddings = []
    all_sources = []
    
    if texts:
        text_embeddings = encode_texts(
            model=model,
            texts=texts,
            task=args.task,
            prompt_name=args.prompt_name,
            truncate_dim=args.truncate_dim,
            max_length=args.max_length,
            batch_size=args.batch_size,
            return_multivector=args.return_multivector,
        )
        all_embeddings.extend(text_embeddings)
        all_sources.extend([{"type": "text", "content": text} for text in texts])
    
    if images:
        image_embeddings = encode_images(
            model=model,
            image_paths=images,
            task=args.task,
            truncate_dim=args.truncate_dim,
            batch_size=args.batch_size,
            return_multivector=args.return_multivector,
        )
        all_embeddings.extend(image_embeddings)
        all_sources.extend([{"type": "image", "path": path} for path in images])
    
    print(f"Generated {len(all_embeddings)} embeddings")
    
    # Print embedding info
    if len(all_embeddings) > 0:
        if isinstance(all_embeddings[0], np.ndarray):
            emb_shape = all_embeddings[0].shape
            print(f"Embedding shape: {emb_shape}")
        elif hasattr(all_embeddings[0], 'shape'):
            emb_shape = all_embeddings[0].shape
            print(f"Embedding shape: {emb_shape}")
    
    # Save embeddings if output file specified
    if args.output_file:
        metadata = {
            "model_path": args.model_path,
            "task": args.task,
            "prompt_name": args.prompt_name,
            "truncate_dim": args.truncate_dim,
            "max_length": args.max_length,
            "return_multivector": args.return_multivector,
            "num_embeddings": len(all_embeddings),
            "sources": all_sources,
        }
        
        save_embeddings(
            embeddings=all_embeddings,
            output_file=args.output_file,
            output_format=args.output_format,
            metadata=metadata,
        )
    else:
        print("No output file specified. Embeddings not saved.")
    
    # Print sample embeddings
    if len(all_embeddings) > 0:
        print("\nSample embeddings:")
        for i, emb in enumerate(all_embeddings[:3]):
            if isinstance(emb, np.ndarray):
                print(f"  {i}: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
            elif hasattr(emb, 'shape'):
                print(f"  {i}: shape={emb.shape}, norm={torch.norm(emb).item():.4f}")
            else:
                print(f"  {i}: {type(emb)}")


if __name__ == "__main__":
    main()
