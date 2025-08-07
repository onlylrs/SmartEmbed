# Helper script: Inspect MultimodalDataset and DataCollator output
import os
from pathlib import Path
import json
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jina.data.multimodal_dataset import MultimodalDataset, get_training_dataloader
from jina.data.data_collator import JinaContrastiveDataCollator
from jina.models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Processor

BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "/csproject/fyp25_hc2/jina-embeddings-v4")
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "train.jsonl"

print("🔍 Debugging dataset at", DATA_FILE)

# Load processor
processor = JinaEmbeddingsV4Processor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

# Load dataset only
dataset = MultimodalDataset(
    jsonl_path=str(DATA_FILE),
    processor=processor,
    text_max_length=128,
    image_max_patches=256,
)
print("Dataset size:", len(dataset))
print("First 3 samples, pixel_values length:")
for i in range(3):
    sample = dataset[i]
    pv_len = sample["pixel_values"].size(0)
    print(f"  idx {i}: patch len = {pv_len}, keys = {list(sample.keys())[:5]}…")

# Test collator on first 4 samples
batch_samples = [dataset[i] for i in range(4)]
collator = JinaContrastiveDataCollator(tokenizer=dataset.processor.tokenizer)
try:
    batch = collator(batch_samples)
    print("✅ Collator succeeded. Batch keys:", batch.keys())
    print("   query_pixel_values shape:", batch["query_pixel_values"].shape)
except Exception as e:
    print("❌ Collator failed:", e)
    # Show lengths causing mismatch
    lengths = [s["pixel_values"].size(0) for s in batch_samples]
    print("Patch lengths:", lengths) 