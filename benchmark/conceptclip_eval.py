#!/usr/bin/env python3
"""
ConceptCLIP Zero-shot Cross-modal Retrieval Evaluation

Usage:
    python conceptclip_eval.py
    python conceptclip_eval.py --data_jsonl /path/to/eval.jsonl --batch_size 64
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm

# Default paths
DATA_PATH = "/home/shebd/4_Collaboration/FYP2526/data/eval_full_path.jsonl"
SAVE_DIR = "/home/shebd/4_Collaboration/FYP2526/SmartEmbed_liam/output/benchmark"


def load_eval_data(jsonl_path: str) -> Tuple[List[str], List[str], Dict[int, set], List[int]]:
    """Load evaluation data from JSONL file, same format as cross_retrieval_eval.py"""
    images: List[str] = []
    texts: List[str] = []
    img_index: Dict[str, int] = {}
    img_to_text_idxs: Dict[int, set] = defaultdict(set)
    text_to_img_idx: List[int] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            img_path = item.get("query_image") or item.get("image")
            txt = item.get("positive") or item.get("text")
            if not img_path or not txt:
                continue
            if img_path not in img_index:
                img_index[img_path] = len(images)
                images.append(img_path)
            i = img_index[img_path]
            texts.append(txt)
            t = len(texts) - 1
            img_to_text_idxs[i].add(t)
            text_to_img_idx.append(i)

    return images, texts, img_to_text_idxs, text_to_img_idx


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float((ranks < k).mean()) if ranks.size else 0.0


def compute_directional_recalls(
    sim: np.ndarray,
    img_to_text_idxs=None,
    text_to_img_idx=None,
    direction: str = "I2T",
    ks: Tuple[int, ...] = (1, 5, 10, 30, 50),
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if direction == "I2T":
        ranks = []
        for i in range(sim.shape[0]):
            order = np.argsort(-sim[i])
            if not img_to_text_idxs[i]:
                continue
            pos = min(order.tolist().index(t) for t in img_to_text_idxs[i])
            ranks.append(pos)
        ranks = np.array(ranks)
    else:  # T2I
        ranks = []
        sim_t = sim.T
        for t in range(sim_t.shape[0]):
            order = np.argsort(-sim_t[t])
            pos = order.tolist().index(text_to_img_idx[t])
            ranks.append(pos)
        ranks = np.array(ranks)

    for k in ks:
        metrics[f"R@{k}"] = recall_at_k(ranks, k)
    return metrics


@torch.inference_mode()
def evaluate_conceptclip(jsonl_path: str, batch_size: int, device: str) -> Dict[str, Dict[str, float]]:
    """Evaluate ConceptCLIP on cross-modal retrieval task"""
    
    print("Loading ConceptCLIP model...")
    # Load ConceptCLIP model and processor
    model = AutoModel.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
    processor = AutoProcessor.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
    
    model.to(device)
    model.eval()
    
    print("Loading evaluation data...")
    images, texts, img_to_text_idxs, text_to_img_idx = load_eval_data(jsonl_path)
    if len(images) == 0 or len(texts) == 0:
        raise ValueError("No valid images/texts found in dataset")
    
    print(f"Found {len(images)} unique images and {len(texts)} texts")
    
    # Encode images
    print("Encoding images...")
    img_features_list = []
    for i in tqdm(range(0, len(images), batch_size), desc="Image batches"):
        batch_images = images[i:i+batch_size]
        batch_imgs = []
        for img_path in batch_images:
            if not os.path.exists(img_path):
                project_root = Path(__file__).resolve().parents[2]
                img_path = str(project_root / img_path)
            try:
                img = Image.open(img_path).convert('RGB')
                batch_imgs.append(img)
            except:
                continue
        
        if batch_imgs:
            # Process images only
            inputs = processor(images=batch_imgs, return_tensors='pt', padding=True).to(device)
            outputs = model(**inputs)
            features = outputs['image_features']
            # Normalize features for cosine similarity
            features = features / features.norm(dim=-1, keepdim=True)
            img_features_list.append(features.cpu().numpy())
    
    img_embeddings = np.concatenate(img_features_list, axis=0)
    
    # Encode texts
    print("Encoding texts...")
    txt_features_list = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Text batches"):
        batch_texts = texts[i:i+batch_size]
        # Process texts only
        inputs = processor(text=batch_texts, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        features = outputs['text_features']
        # Normalize features for cosine similarity
        features = features / features.norm(dim=-1, keepdim=True)
        txt_features_list.append(features.cpu().numpy())
    
    txt_embeddings = np.concatenate(txt_features_list, axis=0)
    
    print("Computing similarity matrix...")
    # Compute cosine similarity matrix (embeddings are already normalized)
    sim = img_embeddings @ txt_embeddings.T
    
    print("Computing retrieval metrics...")
    print("- Computing I2T metrics...")
    i2t = compute_directional_recalls(sim, img_to_text_idxs=img_to_text_idxs, direction="I2T")
    print("- Computing T2I metrics...")
    t2i = compute_directional_recalls(sim, text_to_img_idx=text_to_img_idx, direction="T2I")
    
    mean = {f"mean_{k}": (i2t[k] + t2i[k]) / 2.0 for k in i2t.keys()}
    
    return {"I2T": i2t, "T2I": t2i, "mean": mean}


def main():
    parser = argparse.ArgumentParser(description="ConceptCLIP cross-modal retrieval evaluation")
    parser.add_argument("--data_jsonl", default=DATA_PATH, help="Path to eval data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--save_dir", default=SAVE_DIR, help="Save directory")
    args = parser.parse_args()
    
    print(f"Device: {args.device}, Batch: {args.batch_size}, Data: {args.data_jsonl}")
    results = evaluate_conceptclip(args.data_jsonl, args.batch_size, args.device)
    
    # Print results
    print("\nConceptCLIP Zero-shot Retrieval Results:")
    for section in ["I2T", "T2I", "mean"]:
        metrics = results[section]
        keys = [f"mean_{k}" for k in ["R@1", "R@5", "R@10", "R@30", "R@50"]] if section == "mean" else ["R@1", "R@5", "R@10", "R@30", "R@50"]
        values = ", ".join([f"{k}={metrics[k]:.4f}" for k in keys if k in metrics])
        print(f"{section}: {values}")
    
    # Save results
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = Path(args.save_dir) / f"conceptclip_{ts}.json"
    json.dump({"model": "ConceptCLIP", "data": args.data_jsonl, "results": results}, out_file.open("w"), indent=2)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()