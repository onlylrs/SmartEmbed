#!/usr/bin/env python3
"""
逐 batch 检查 DataLoader 输出，如果缺 query_pixel_values
就把整个 batch 打印出来并退出。
"""

import sys, os, json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from jina.data.multimodal_dataset import create_multimodal_dataloader
from jina.models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Processor

BASE_MODEL = os.getenv("BASE_MODEL_PATH", "/csproject/fyp25_hc2/jina-embeddings-v4")
JSONL      = root / "data" / "train.jsonl"

processor   = JinaEmbeddingsV4Processor.from_pretrained(BASE_MODEL, trust_remote_code=True)
loader      = create_multimodal_dataloader(
    jsonl_path=str(JSONL),
    processor=processor,
    batch_size=2,
    num_workers=0,
    shuffle=True,
)

for step, batch in enumerate(loader):
    if "query_pixel_values" not in batch:
        print(f"\n⚠️  step={step}: 没有 query_pixel_values ！")
        print("batch keys:", batch.keys())
        sys.exit(0)

print("✔ 所有 batch 都包含 query_pixel_values")