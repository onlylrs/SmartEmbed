import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Suppress verbose logging from transformers and PEFT during model loading
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("safetensors").setLevel(logging.ERROR)

# Also suppress the root logger temporarily during imports
logging.getLogger().setLevel(logging.ERROR)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Ensure project root is on sys.path so `import jina` works when running as a script
_ROOT = _project_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from jina.models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model
from jina.utils.local_paths import get_path
from peft import PeftModel


def _default_model_path() -> str:
    # Priority 1: local_paths.yaml key
    p = get_path("finetuned_model_path")
    if p and os.path.isdir(p):
        return p
    # Priority 2: outputs/models/finetuned under project root
    candidate = _project_root() / "outputs" / "models" / "finetuned"
    return str(candidate)


def _find_default_data() -> str:
    # Try a few common locations in this repo
    candidates = [
        _project_root() / "fyp25_hc2" / "example_data" / "eval.jsonl",
        _project_root() / "fyp25_hc2" / "example_data" / "val.jsonl",
        _project_root() / "fyp25_hc2" / "example_data" / "train.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


def load_eval_data(jsonl_path: str) -> Tuple[List[str], List[str], Dict[int, set], List[int]]:
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
def evaluate(model_path: str, base_model_path: str | None, jsonl_path: str, batch_size: int, device: str) -> Dict[str, Dict[str, float]]:
    try:
        print("Loading model...")
        # Set ultra-strict logging during model loading to suppress weight info
        original_loggers = {}
        logger_names = [
            "",  # root logger
            "transformers", "peft", "huggingface_hub", "safetensors",
            "transformers.modeling_utils", "transformers.configuration_utils", 
            "transformers.tokenization_utils_base", "transformers.generation_utils"
        ]
        
        for name in logger_names:
            logger = logging.getLogger(name)
            original_loggers[name] = logger.getEffectiveLevel()
            logger.setLevel(logging.CRITICAL)  # Only show critical errors
        
        try:
            # Use stdout/stderr redirection to suppress all weight loading messages
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                # Determine whether model_path already contains full model (config.json)
                if os.path.isfile(os.path.join(model_path, "config.json")):
                    model = JinaEmbeddingsV4Model.from_pretrained(model_path, trust_remote_code=True).to(device)
                else:
                    if base_model_path is None:
                        raise ValueError("base_model_path must be provided when model_path does not contain a full model")
                    base = JinaEmbeddingsV4Model.from_pretrained(base_model_path, trust_remote_code=True).to(device)
                    # adapter_dir = adapters subdir if exists else model_path
                    adapter_dir = os.path.join(model_path, "adapters") if os.path.isdir(os.path.join(model_path, "adapters")) else model_path
                    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
                model.task = "retrieval"
        finally:
            # Restore original logging levels
            for name, level in original_loggers.items():
                logging.getLogger(name).setLevel(level)
        
        print("Model loaded successfully!")
    finally:
        pass # No explicit cleanup needed for model, it's managed by torch.inference_mode

    images, texts, img_to_text_idxs, text_to_img_idx = load_eval_data(jsonl_path)
    if len(images) == 0 or len(texts) == 0:
        raise ValueError("No valid images/texts found in dataset")

    img_emb = model.encode_image(images, task="retrieval", batch_size=batch_size, return_numpy=True)
    txt_emb = model.encode_text(texts, task="retrieval", batch_size=batch_size, prompt_name="query", return_numpy=True)

    # cosine similarities (embeddings are normalized)
    sim = img_emb @ txt_emb.T

    i2t = compute_directional_recalls(sim, img_to_text_idxs=img_to_text_idxs, direction="I2T")
    t2i = compute_directional_recalls(sim, text_to_img_idx=text_to_img_idx, direction="T2I")

    mean = {f"mean_{k}": (i2t[k] + t2i[k]) / 2.0 for k in i2t.keys()}
    return {"I2T": i2t, "T2I": t2i, "mean": mean}


def main():
    parser = argparse.ArgumentParser(description="Cross-modal retrieval evaluation (I2T & T2I) with R@K metrics")
    parser.add_argument("--data_jsonl", type=str, default=_find_default_data(), help="Path to JSONL eval data")
    parser.add_argument("--model_path", type=str, default=_default_model_path(), help="Path to finetuned model directory")
    parser.add_argument("--base_model_path", type=str, default=get_path("base_model_path"), help="Path to full base model (needed when --model_path only contains LoRA adapters)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default=str(_project_root() / "outputs" / "eval"))
    args = parser.parse_args()

    if not args.data_jsonl or not os.path.exists(args.data_jsonl):
        raise FileNotFoundError(f"data_jsonl not found: {args.data_jsonl}")
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"model_path directory not found: {args.model_path}")

    results = evaluate(args.model_path, args.base_model_path, args.data_jsonl, args.batch_size, args.device)

    # Pretty print
    print("\nCross-Retrieval Evaluation (R@1/5/10/30/50)")
    print(f"Model: {args.model_path}")
    print(f"Data:  {args.data_jsonl}\n")
    for section in ["I2T", "T2I", "mean"]:
        metrics = results[section]
        ordered_keys = [k for k in ["R@1", "R@5", "R@10", "R@30", "R@50"] if k in metrics] if section != "mean" else [f"mean_{k}" for k in ["R@1", "R@5", "R@10", "R@30", "R@50"]]
        kv = ", ".join([f"{k}={metrics[k]:.4f}" for k in ordered_keys])
        print(f"{section}: {kv}")

    # Save to file
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.save_dir) / f"cross_retrieval_metrics_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": args.model_path,
            "data_jsonl": args.data_jsonl,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved metrics to: {out_path}")


if __name__ == "__main__":
    main()


