#!/usr/bin/env python3
"""
Evaluation script for Jina Embeddings V4 model
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from jina.models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model
from inference import load_model, encode_texts, encode_images


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate Jina Embeddings V4 model")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to evaluation data")
    parser.add_argument("--eval_type", type=str, default="retrieval", 
                       choices=["retrieval", "similarity", "classification"],
                       help="Type of evaluation")
    
    parser.add_argument("--task", type=str, default="retrieval", 
                       choices=["retrieval", "text-matching", "code"],
                       help="Task type for encoding")
    parser.add_argument("--truncate_dim", type=int, choices=[128, 256, 512, 1024, 2048],
                       help="Dimension to truncate embeddings to")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for encoding")
    
    parser.add_argument("--output_file", type=str, help="File to save evaluation results")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on")
    
    return parser.parse_args()


def load_eval_data(eval_data_path: str, eval_type: str) -> List[Dict]:
    """Load evaluation data"""
    
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        if eval_data_path.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
    
    print(f"Loaded {len(data)} evaluation examples")
    return data


def evaluate_retrieval(
    model: JinaEmbeddingsV4Model,
    eval_data: List[Dict],
    task: str = "retrieval",
    truncate_dim: int = None,
    batch_size: int = 8,
) -> Dict[str, float]:
    """Evaluate retrieval performance"""
    
    print("Evaluating retrieval performance...")
    
    queries = [item["query"] for item in eval_data]
    positives = [item["positive"] for item in eval_data]
    negatives = [item.get("negative", "") for item in eval_data if item.get("negative")]
    
    # Encode queries and documents
    query_embeddings = encode_texts(
        model, queries, task=task, prompt_name="query", 
        truncate_dim=truncate_dim, batch_size=batch_size
    )
    
    positive_embeddings = encode_texts(
        model, positives, task=task, prompt_name="passage",
        truncate_dim=truncate_dim, batch_size=batch_size
    )
    
    if negatives:
        negative_embeddings = encode_texts(
            model, negatives, task=task, prompt_name="passage",
            truncate_dim=truncate_dim, batch_size=batch_size
        )
    else:
        # Use other positives as negatives
        negative_embeddings = positive_embeddings
    
    # Convert to numpy arrays
    query_embeddings = np.array(query_embeddings)
    positive_embeddings = np.array(positive_embeddings)
    negative_embeddings = np.array(negative_embeddings)
    
    # Compute similarities
    positive_similarities = np.diag(cosine_similarity(query_embeddings, positive_embeddings))
    
    # For each query, compute similarity with all negatives
    all_similarities = cosine_similarity(query_embeddings, negative_embeddings)
    
    # Compute metrics
    total_correct = 0
    mrr_scores = []
    
    for i in range(len(queries)):
        pos_sim = positive_similarities[i]
        neg_sims = all_similarities[i]
        
        # Check if positive is ranked higher than all negatives
        rank = 1 + np.sum(neg_sims > pos_sim)
        
        if rank == 1:
            total_correct += 1
        
        mrr_scores.append(1.0 / rank)
    
    accuracy = total_correct / len(queries)
    mrr = np.mean(mrr_scores)
    
    # Compute additional metrics
    avg_pos_sim = np.mean(positive_similarities)
    avg_neg_sim = np.mean(all_similarities)
    
    results = {
        "accuracy@1": accuracy,
        "mrr": mrr,
        "avg_positive_similarity": avg_pos_sim,
        "avg_negative_similarity": avg_neg_sim,
        "num_queries": len(queries),
    }
    
    return results


def evaluate_similarity(
    model: JinaEmbeddingsV4Model,
    eval_data: List[Dict],
    task: str = "text-matching",
    truncate_dim: int = None,
    batch_size: int = 8,
) -> Dict[str, float]:
    """Evaluate similarity/matching performance"""
    
    print("Evaluating similarity performance...")
    
    text1_list = [item["text1"] for item in eval_data]
    text2_list = [item["text2"] for item in eval_data]
    labels = [item["label"] for item in eval_data]  # 1 for similar, 0 for dissimilar
    
    # Encode texts
    embeddings1 = encode_texts(
        model, text1_list, task=task, truncate_dim=truncate_dim, batch_size=batch_size
    )
    embeddings2 = encode_texts(
        model, text2_list, task=task, truncate_dim=truncate_dim, batch_size=batch_size
    )
    
    # Convert to numpy arrays
    embeddings1 = np.array(embeddings1)
    embeddings2 = np.array(embeddings2)
    
    # Compute similarities
    similarities = np.diag(cosine_similarity(embeddings1, embeddings2))
    
    # Find optimal threshold
    thresholds = np.linspace(0, 1, 100)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        predictions = (similarities > threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Compute final metrics with best threshold
    predictions = (similarities > best_threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_threshold": best_threshold,
        "avg_similarity": np.mean(similarities),
        "num_pairs": len(eval_data),
    }
    
    return results


def evaluate_classification(
    model: JinaEmbeddingsV4Model,
    eval_data: List[Dict],
    task: str = "retrieval",
    truncate_dim: int = None,
    batch_size: int = 8,
) -> Dict[str, float]:
    """Evaluate classification performance using nearest neighbor"""
    
    print("Evaluating classification performance...")
    
    # Split data into train and test
    split_idx = int(0.8 * len(eval_data))
    train_data = eval_data[:split_idx]
    test_data = eval_data[split_idx:]
    
    # Extract texts and labels
    train_texts = [item["text"] for item in train_data]
    train_labels = [item["label"] for item in train_data]
    test_texts = [item["text"] for item in test_data]
    test_labels = [item["label"] for item in test_data]
    
    # Encode texts
    train_embeddings = encode_texts(
        model, train_texts, task=task, truncate_dim=truncate_dim, batch_size=batch_size
    )
    test_embeddings = encode_texts(
        model, test_texts, task=task, truncate_dim=truncate_dim, batch_size=batch_size
    )
    
    # Convert to numpy arrays
    train_embeddings = np.array(train_embeddings)
    test_embeddings = np.array(test_embeddings)
    
    # Nearest neighbor classification
    similarities = cosine_similarity(test_embeddings, train_embeddings)
    nearest_indices = np.argmax(similarities, axis=1)
    predictions = [train_labels[idx] for idx in nearest_indices]
    
    # Compute metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='weighted'
    )
    
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_train": len(train_data),
        "num_test": len(test_data),
    }
    
    return results


def main():
    """Main evaluation function"""
    
    args = parse_args()
    
    # Load model
    model, device = load_model(args.model_path, args.device)
    if model is None:
        print("Failed to load model")
        return
    
    print(f"Model loaded on device: {device}")
    
    # Load evaluation data
    eval_data = load_eval_data(args.eval_data, args.eval_type)
    
    # Run evaluation
    if args.eval_type == "retrieval":
        results = evaluate_retrieval(
            model, eval_data, args.task, args.truncate_dim, args.batch_size
        )
    elif args.eval_type == "similarity":
        results = evaluate_similarity(
            model, eval_data, args.task, args.truncate_dim, args.batch_size
        )
    elif args.eval_type == "classification":
        results = evaluate_classification(
            model, eval_data, args.task, args.truncate_dim, args.batch_size
        )
    else:
        print(f"Unknown evaluation type: {args.eval_type}")
        return
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Save results
    if args.output_file:
        output_data = {
            "model_path": args.model_path,
            "eval_data": args.eval_data,
            "eval_type": args.eval_type,
            "task": args.task,
            "truncate_dim": args.truncate_dim,
            "results": results,
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
