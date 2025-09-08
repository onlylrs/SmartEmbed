# Copied from assistant suggestion: utility to validate data pipeline end-to-end
import os
import json
import tempfile
from pathlib import Path

import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jina.models.modeling_jina_embeddings_v4 import (
    JinaEmbeddingsV4Processor,
    JinaEmbeddingsV4Model,
)
from jina.data.multimodal_dataset import MultimodalDataset
from jina.data.data_collator import JinaContrastiveDataCollator


def _create_dummy_image(path: Path):
    """Create a 64×64 RGB dummy image and save it to *path*."""
    img = Image.new("RGB", (64, 64), color=(123, 222, 64))
    img.save(path)


def _create_dummy_jsonl(image_path: Path, jsonl_path: Path):
    """Write a single-line JSONL file containing the dummy sample."""
    record = {
        "query_image": str(image_path),  # 供 Dataset 读取
        "positive": "A dummy caption for the green square.",
        "task": "retrieval",
    }
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main(base_model_path: str):
    print("🔍 Running end-to-end data-pipeline validation…")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_file = tmpdir / "dummy.jpg"
        jsonl_file = tmpdir / "sample.jsonl"

        _create_dummy_image(img_file)
        _create_dummy_jsonl(img_file, jsonl_file)

        # Load processor & model
        print("⚙️  Loading processor and model… (can take a while)")
        processor = JinaEmbeddingsV4Processor.from_pretrained(
            base_model_path, trust_remote_code=True, use_fast=True
        )
        model = JinaEmbeddingsV4Model.from_pretrained(
            base_model_path, trust_remote_code=True, torch_dtype="auto"
        )
        model.eval()

        # Build dataset / dataloader components
        dataset = MultimodalDataset(
            jsonl_path=str(jsonl_file),
            processor=processor,
            text_max_length=128,
            image_max_patches=256,
            task_name="retrieval",
        )
        collator = JinaContrastiveDataCollator()

        sample = dataset[0]

        # --------------------------------------------------
        # Create image branch inputs WITH text tokens
        # --------------------------------------------------
        # Reload the dummy image to make sure we have full inputs
        dummy_img = Image.open(img_file).convert("RGB")
        img_features = processor.process_images([dummy_img])

        image_inputs = {
            "input_ids": img_features["input_ids"],
            "attention_mask": img_features["attention_mask"],
            "pixel_values": img_features["pixel_values"],
            "image_grid_thw": img_features["image_grid_thw"],
        }

        # --------------------------------------------------
        # Create text branch inputs from dataset sample
        # --------------------------------------------------
        text_inputs = {
            "input_ids": sample["text_input_ids"].unsqueeze(0),
            "attention_mask": sample["text_attention_mask"].unsqueeze(0),
        }

        with torch.no_grad():
            img_out = model(task_label="retrieval", **{k: v.to(model.device) for k, v in image_inputs.items()})
            txt_out = model(task_label="retrieval", **{k: v.to(model.device) for k, v in text_inputs.items()})

        print("✅ Forward on image branch OK – single_vec_emb shape:", img_out.single_vec_emb.shape)
        print("✅ Forward on text branch  OK – single_vec_emb shape:", txt_out.single_vec_emb.shape)

    print("🎉 Pipeline validation finished – everything wired correctly.")


if __name__ == "__main__":
    # Try to locate base_model_path from project_config.yaml, fallback to env var
    default_path = None
    config_yaml = Path(__file__).resolve().parents[1] / "project_config.yaml"
    if config_yaml.exists():
        import yaml

        with open(config_yaml, "r") as f:
            cfg = yaml.safe_load(f)
            default_path = cfg.get("base_model_path")

    base_path = os.environ.get("BASE_MODEL_PATH", default_path)
    if not base_path or not Path(base_path).exists():
        raise FileNotFoundError(
            "Base model path not found – set BASE_MODEL_PATH env or update project_config.yaml"
        )

    main(base_path) 