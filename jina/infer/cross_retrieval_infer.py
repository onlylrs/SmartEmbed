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
from jina.utils.local_paths import get_path
from peft import PeftModel


def _default_model_path() -> str:
    p = get_path("finetuned_model_path")
    if p and os.path.isdir(p):
        return p
    candidate = _project_root() / "outputs" / "models" / "finetuned"
    return str(candidate)


def load_infer_data(jsonl_path: str) -> Tuple[List[str], List[str]]:
    """
    Load inference data from a JSONL file and return ordered lists of images and texts.
    Accepts keys: image path under "query_image" or "image", text under "positive" or "text".
    """
    images: List[str] = []
    texts: List[str] = []
    img_index: Dict[str, int] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            img_path = item.get("query_image") or item.get("image")
            txt = item.get("positive") or item.get("text")
            if not img_path and not txt:
                continue
            if img_path:
                if img_path not in img_index:
                    img_index[img_path] = len(images)
                    images.append(img_path)
            if txt:
                texts.append(txt)

    return images, texts


def _is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_dist_avail_and_initialized() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_dist_avail_and_initialized() else 1


def _partition_indices(n: int, world_size: int, rank: int) -> Tuple[int, int]:
    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def _init_distributed_if_needed(device: str | None = None) -> str:
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
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _gather_numpy_by_rank(local_array: np.ndarray) -> np.ndarray | None:
    world_size = _get_world_size()
    if world_size == 1:
        return local_array
    gathered: List[np.ndarray] = [None for _ in range(world_size)]  # type: ignore
    dist.all_gather_object(gathered, local_array)
    if _get_rank() == 0:
        return np.concatenate(gathered, axis=0) if len(gathered) > 0 else None
    return None


@torch.inference_mode()
def run_inference(
    model_path: str,
    base_model_path: str | None,
    jsonl_path: str,
    batch_size: int,
    device: str,
    save_dir: str,
    save_topk: bool = False,
    topk: int = 10,
    prompt_name: str = "query",
) -> Dict[str, str]:
    """
    Compute embeddings for images and texts and save them to disk. Optionally produce top-k retrieval.
    Returns a dict of output file paths (rank 0 only). Other ranks return empty dicts.
    """
    device = _init_distributed_if_needed(device)
    rank = _get_rank()
    world_size = _get_world_size()

    try:
        if rank == 0:
            print("Loading model...")
        original_loggers = {}
        logger_names = [
            "",
            "transformers",
            "peft",
            "huggingface_hub",
            "safetensors",
            "transformers.modeling_utils",
            "transformers.configuration_utils",
            "transformers.tokenization_utils_base",
            "transformers.generation_utils",
        ]
        for name in logger_names:
            logger = logging.getLogger(name)
            original_loggers[name] = logger.getEffectiveLevel()
            logger.setLevel(logging.CRITICAL)
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                if os.path.isfile(os.path.join(model_path, "config.json")):
                    model = JinaEmbeddingsV4Model.from_pretrained(model_path, trust_remote_code=True).to(device)
                else:
                    if base_model_path is None:
                        raise ValueError("base_model_path must be provided when model_path does not contain a full model")
                    base = JinaEmbeddingsV4Model.from_pretrained(base_model_path, trust_remote_code=True).to(device)
                    adapter_dir = os.path.join(model_path, "adapters") if os.path.isdir(os.path.join(model_path, "adapters")) else model_path
                    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
                model.task = "retrieval"
        finally:
            for name, level in original_loggers.items():
                logging.getLogger(name).setLevel(level)
        if rank == 0:
            print("Model loaded successfully!")
    finally:
        pass

    images, texts = load_infer_data(jsonl_path)
    if len(images) == 0 and len(texts) == 0:
        raise ValueError("No valid images or texts found in dataset")

    img_s, img_e = _partition_indices(len(images), world_size, rank)
    txt_s, txt_e = _partition_indices(len(texts), world_size, rank)
    local_images = images[img_s:img_e]
    local_texts = texts[txt_s:txt_e]

    img_emb_local = None
    txt_emb_local = None
    if len(local_images) > 0:
        img_emb_local = model.encode_image(local_images, task="retrieval", batch_size=batch_size, return_numpy=True)
    else:
        img_emb_local = np.zeros((0, model.config.projection_dim), dtype=np.float32)
    if len(local_texts) > 0:
        txt_emb_local = model.encode_text(local_texts, task="retrieval", batch_size=batch_size, prompt_name=prompt_name, return_numpy=True)
    else:
        txt_emb_local = np.zeros((0, model.config.projection_dim), dtype=np.float32)

    img_emb = _gather_numpy_by_rank(img_emb_local)
    txt_emb = _gather_numpy_by_rank(txt_emb_local)

    if rank != 0:
        if _is_dist_avail_and_initialized():
            dist.barrier()
        return {}

    out: Dict[str, str] = {}
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if img_emb is not None:
        img_out = Path(save_dir) / f"image_embeddings_{ts}.npy"
        np.save(img_out, img_emb)
        out["image_embeddings"] = str(img_out)
    if txt_emb is not None:
        txt_out = Path(save_dir) / f"text_embeddings_{ts}.npy"
        np.save(txt_out, txt_emb)
        out["text_embeddings"] = str(txt_out)

    meta = {
        "model_path": model_path,
        "data_jsonl": jsonl_path,
        "num_images": len(images),
        "num_texts": len(texts),
        "images": images,
        "texts": texts,
    }
    meta_out = Path(save_dir) / f"metadata_{ts}.json"
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    out["metadata"] = str(meta_out)

    if save_topk and img_emb is not None and txt_emb is not None and topk > 0:
        sim = img_emb @ txt_emb.T
        # Top-k for each image to texts
        k = min(topk, sim.shape[1])
        topk_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        # sort topk per row
        row_indices = np.arange(sim.shape[0])[:, None]
        sorted_order = np.argsort(-sim[row_indices, topk_idx])
        topk_sorted = topk_idx[row_indices, sorted_order]
        topk_scores = sim[row_indices, topk_sorted]

        results = []
        for i in range(sim.shape[0]):
            results.append({
                "image_index": i,
                "image_path": images[i],
                "topk": [
                    {
                        "text_index": int(t_idx),
                        "text": texts[int(t_idx)],
                        "score": float(topk_scores[i, j]),
                    }
                    for j, t_idx in enumerate(topk_sorted[i])
                ],
            })

        topk_out = Path(save_dir) / f"topk_predictions_{ts}.json"
        with open(topk_out, "w", encoding="utf-8") as f:
            json.dump({"I2T_topk": results}, f, indent=2)
        out["topk_predictions"] = str(topk_out)

    if _is_dist_avail_and_initialized():
        dist.barrier()

    return out


def main():
    parser = argparse.ArgumentParser(description="Cross-modal inference: compute embeddings and optional top-k results")
    parser.add_argument("--data_jsonl", type=str, required=True, help="Path to JSONL inference data")
    parser.add_argument("--model_path", type=str, default=_default_model_path(), help="Path to finetuned model directory")
    parser.add_argument("--base_model_path", type=str, default=get_path("base_model_path"), help="Path to full base model (needed when --model_path only contains LoRA adapters)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default=str(_project_root() / "outputs" / "infer"))
    parser.add_argument("--save_topk", action="store_true", help="If set, compute and save top-k I2T predictions")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--prompt_name", type=str, default="query")
    args = parser.parse_args()

    if not args.data_jsonl or not os.path.exists(args.data_jsonl):
        raise FileNotFoundError(f"data_jsonl not found: {args.data_jsonl}")
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"model_path directory not found: {args.model_path}")

    outputs = run_inference(
        args.model_path,
        args.base_model_path,
        args.data_jsonl,
        args.batch_size,
        args.device,
        args.save_dir,
        args.save_topk,
        args.topk,
        args.prompt_name,
    )

    if _get_rank() == 0 and outputs:
        print("\nSaved inference outputs:")
        for k, v in outputs.items():
            print(f"- {k}: {v}")

    if _is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


