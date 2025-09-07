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
import torch.distributed as dist

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
from jina.utils.config_manager import load_config
from peft import PeftModel, LoraConfig


def _default_model_path() -> str:
    """Get default model path from unified config system"""
    try:
        config = load_config()
        # Try to get from training output directory
        output_dir = config.get('training', {}).get('output_dir', 'outputs/models')
        candidate = _project_root() / output_dir / "finetuned"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    
    # Fallback to default location
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


def _find_adapter_dir(path: str) -> str | None:
    """Heuristically detect a PEFT/LoRA adapter directory within or equal to `path`.
    Returns the directory that contains adapter files, or None if not found.
    """
    def has_adapter_files(d: str) -> bool:
        has_cfg = os.path.isfile(os.path.join(d, "adapter_config.json"))
        has_weight = any(
            os.path.isfile(os.path.join(d, fn))
            for fn in ("adapter_model.safetensors", "adapter_model.bin")
        )
        return has_cfg or has_weight

    # Direct folder
    if os.path.isdir(path) and has_adapter_files(path):
        return path

    # Common subfolders used by PEFT trainers
    for sub in ["adapters", "adapters/default", "adapter", "lora", "peft"]:
        d = os.path.join(path, sub)
        if os.path.isdir(d) and has_adapter_files(d):
            return d

    return None


def _is_full_model_dir(path: str) -> bool:
    """Detect if `path` looks like a full HF model directory (has actual weights)."""
    names = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    return any(os.path.isfile(os.path.join(path, n)) for n in names)


def _adapter_fingerprint(adapter_dir: str) -> str:
    """Return a lightweight fingerprint for an adapter: size + sha256 of first 1MB."""
    import hashlib
    for fn in ("adapter_model.safetensors", "adapter_model.bin"):
        p = os.path.join(adapter_dir, fn)
        if os.path.isfile(p):
            st = os.stat(p)
            size = st.st_size
            h = hashlib.sha256()
            with open(p, "rb") as f:
                chunk = f.read(1024 * 1024)
                h.update(chunk)
            return f"{fn}:{size}:{h.hexdigest()[:12]}"
    return "<missing-adapter-file>"


def load_eval_data(jsonl_path: str) -> Tuple[List[str], List[str], Dict[int, set], List[int]]:
    """
    Load evaluation data from a JSONL file and organize it for cross-modal retrieval tasks.
    Each line in the input file should be a JSON object containing at least an image path
    (under the key "query_image" or "image") and a text (under the key "positive" or "text").
    Args:
        jsonl_path (str): Path to the JSONL file containing evaluation data.
    Returns:
        Tuple containing:
            images (List[str]): List of unique image paths, where each index corresponds to an image ID.
            texts (List[str]): List of all texts, where each index corresponds to a text ID.
            img_to_text_idxs (Dict[int, set]): Mapping from image index to a set of text indices associated with that image.
            text_to_img_idx (List[int]): List mapping each text index to its corresponding image index.
    """
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


def _is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_dist_avail_and_initialized() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_dist_avail_and_initialized() else 1


def _partition_indices(n: int, world_size: int, rank: int) -> Tuple[int, int]:
    # Even partition [start, end) for this rank
    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def _init_distributed_if_needed(device: str | None = None) -> str:
    """Initialize torch.distributed if launched with torchrun. Returns device string for this rank."""
    if dist.is_available() and ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
        if backend == "nccl":
            dev = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        else:
            dev = device or ("cpu" if not torch.cuda.is_available() else "cuda")
        return dev
    # Fallback single-process
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _gather_numpy_by_rank(local_array: np.ndarray) -> np.ndarray | None:
    """Gather numpy arrays from all ranks to rank 0 by concatenation along axis 0.
    Returns concatenated array on rank 0, and None on other ranks.
    """
    world_size = _get_world_size()
    if world_size == 1:
        return local_array
    # Use all_gather_object for variable-sized parts
    gathered: List[np.ndarray] = [None for _ in range(world_size)]  # type: ignore
    dist.all_gather_object(gathered, local_array)
    if _get_rank() == 0:
        return np.concatenate(gathered, axis=0) if len(gathered) > 0 else None
    return None


@torch.inference_mode()
def evaluate(model_path: str, base_model_path: str | None, jsonl_path: str, batch_size: int, device: str) -> Dict[str, Dict[str, float]]:
    """
    Evaluates a cross-modal retrieval model on a given dataset and computes retrieval metrics.
    Args:
        model_path (str): Path to the trained model checkpoint to be evaluated.
        base_model_path (str | None): Path to the base model (if using PEFT/LoRA), or None if not applicable.
        jsonl_path (str): Path to the evaluation data in JSONL format.
        batch_size (int): Batch size to use during evaluation.
        device (str): Device identifier (e.g., 'cpu', 'cuda', 'cuda:0') on which to run the evaluation.
    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing retrieval metrics for each direction (e.g., 'I2T', 'T2I').
            Each sub-dictionary contains metrics such as recall at various values of k (e.g., 'R@1', 'R@5', etc.).
    The function loads the specified model, processes the evaluation dataset, computes similarity scores between
    images and texts, and calculates retrieval metrics (such as recall@k) for both image-to-text (I2T) and
    text-to-image (T2I) retrieval directions.
    """
    # Initialize distributed (if torchrun is used). Device may be overridden per-rank.
    device = _init_distributed_if_needed(device)
    rank = _get_rank()
    world_size = _get_world_size()

    try:
        # Pre-compute loading mode and log it (outside suppressed stdout)
        is_full = _is_full_model_dir(model_path)
        adapter_dir = None if is_full else _find_adapter_dir(model_path)
        if rank == 0:
            print("Loading model...")
            if is_full:
                print(f"- Detected full model directory: {model_path}")
            else:
                if adapter_dir:
                    print(f"- Using base model: {base_model_path}")
                    print(f"- Applying adapters from: {adapter_dir}")
                    print(f"- Adapter fingerprint: {_adapter_fingerprint(adapter_dir)}")
                else:
                    print(f"- No adapter files found under: {model_path}")

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
                if is_full:
                    model = JinaEmbeddingsV4Model.from_pretrained(model_path, trust_remote_code=True).to(device)
                else:
                    if adapter_dir is None:
                        raise ValueError(
                            f"`model_path` does not contain model weights and no adapter files were found.\n"
                            f"Checked: {model_path}\n"
                            f"Expect one of ['adapter_model.safetensors', 'adapter_model.bin'] and 'adapter_config.json'"
                        )
                    if not base_model_path:
                        raise ValueError("base_model_path must be provided when using LoRA/PEFT adapters")
                    base = JinaEmbeddingsV4Model.from_pretrained(base_model_path, trust_remote_code=True).to(device)
                    # If base already comes as a PeftModel (it will, per Jina's implementation),
                    # load our adapter into it and activate explicitly. Otherwise, wrap with PEFT.
                    if isinstance(base, PeftModel):
                        adapter_name = "eval_adapter"
                        base.load_adapter(adapter_dir, adapter_name=adapter_name)
                        base.set_adapter(adapter_name)
                        model = base
                    else:
                        lora_cfg = None
                        cfg_path = os.path.join(adapter_dir, "adapter_config.json")
                        if os.path.isfile(cfg_path):
                            lora_cfg = LoraConfig.from_pretrained(adapter_dir)
                        model = PeftModel.from_pretrained(base, adapter_dir, config=lora_cfg).to(device)
                model.task = "retrieval"
        finally:
            # Restore original logging levels
            for name, level in original_loggers.items():
                logging.getLogger(name).setLevel(level)
        
        if rank == 0:
            try:
                name_or_path = getattr(getattr(model, "config", object()), "_name_or_path", "<unknown>")
            except Exception:
                name_or_path = "<unknown>"
            is_peft = hasattr(model, "peft_config")
            active_adapter = None
            available_adapters = []
            try:
                if is_peft:
                    # peft_config is a dict name->cfg
                    available_adapters = list(getattr(model, "peft_config", {}).keys())
                    # active adapter name is stored in model.active_adapter for LoRA
                    active_adapter = getattr(model, "active_adapter", None)
            except Exception:
                pass
            print(
                f"Model loaded successfully! name_or_path={name_or_path}, peft={is_peft}, "
                f"active_adapter={active_adapter}, adapters={available_adapters}"
            )
    finally:
        pass # No explicit cleanup needed for model, it's managed by torch.inference_mode

    images, texts, img_to_text_idxs, text_to_img_idx = load_eval_data(jsonl_path)
    if len(images) == 0 or len(texts) == 0:
        raise ValueError("No valid images/texts found in dataset")

    # Partition work across ranks
    img_s, img_e = _partition_indices(len(images), world_size, rank)
    txt_s, txt_e = _partition_indices(len(texts), world_size, rank)

    local_images = images[img_s:img_e]
    local_texts = texts[txt_s:txt_e]

    # Compute local embeddings
    img_emb_local = model.encode_image(local_images, task="retrieval", batch_size=batch_size, return_numpy=True)
    txt_emb_local = model.encode_text(local_texts, task="retrieval", batch_size=batch_size, prompt_name="query", return_numpy=True)

    # Gather to rank 0
    img_emb = _gather_numpy_by_rank(img_emb_local)
    txt_emb = _gather_numpy_by_rank(txt_emb_local)

    # Only rank 0 computes metrics and returns
    if rank != 0:
        # Ensure all ranks sync before exit
        if _is_dist_avail_and_initialized():
            dist.barrier()
        return {}

    assert img_emb is not None and txt_emb is not None

    # cosine similarities (embeddings are normalized)
    sim = img_emb @ txt_emb.T

    i2t = compute_directional_recalls(sim, img_to_text_idxs=img_to_text_idxs, direction="I2T")
    t2i = compute_directional_recalls(sim, text_to_img_idx=text_to_img_idx, direction="T2I")

    mean = {f"mean_{k}": (i2t[k] + t2i[k]) / 2.0 for k in i2t.keys()}

    # Sync before returning
    if _is_dist_avail_and_initialized():
        dist.barrier()

    return {"I2T": i2t, "T2I": t2i, "mean": mean}


def main():
    parser = argparse.ArgumentParser(description="Cross-modal retrieval evaluation (I2T & T2I) with R@K metrics")
    parser.add_argument("--data_jsonl", type=str, default=_find_default_data(), help="Path to JSONL eval data")
    parser.add_argument("--model_path", type=str, default=_default_model_path(), help="Path to finetuned model directory")
    def get_base_model_path():
        try:
            config = load_config()
            return config.get('model', {}).get('base_model_path')
        except Exception:
            return None
    
    parser.add_argument("--base_model_path", type=str, default=get_base_model_path(), help="Path to full base model (needed when --model_path only contains LoRA adapters)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default=str(_project_root() / "outputs" / "eval"))
    args = parser.parse_args()

    if not args.data_jsonl or not os.path.exists(args.data_jsonl):
        raise FileNotFoundError(f"data_jsonl not found: {args.data_jsonl}")
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"model_path directory not found: {args.model_path}")

    results = evaluate(args.model_path, args.base_model_path, args.data_jsonl, args.batch_size, args.device)

    # Only rank 0 prints/saves
    if _get_rank() == 0 and results:
        print("\nCross-Retrieval Evaluation (R@1/5/10/30/50)")
        print(f"Model: {args.model_path}")
        print(f"Data:  {args.data_jsonl}\n")
        for section in ["I2T", "T2I", "mean"]:
            metrics = results[section]
            ordered_keys = [k for k in ["R@1", "R@5", "R@10", "R@30", "R@50"] if k in metrics] if section != "mean" else [f"mean_{k}" for k in ["R@1", "R@5", "R@10", "R@30", "R@50"]]
            kv = ", ".join([f"{k}={metrics[k]:.4f}" for k in ordered_keys])
            print(f"{section}: {kv}")

    # Save to file (rank 0)
    if _get_rank() == 0 and results:
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

    # Cleanup distributed
    if _is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

