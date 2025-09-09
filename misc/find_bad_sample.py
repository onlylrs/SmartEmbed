import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jina.data.multimodal_dataset import MultimodalDataset
from jina.models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Processor

BASE = os.environ.get('BASE_MODEL_PATH', '/csproject/fyp25_hc2/jina-embeddings-v4')
processor = JinaEmbeddingsV4Processor.from_pretrained(BASE, trust_remote_code=True)

DATA = Path(__file__).resolve().parents[1]/'data'/'train.jsonl'

ds = MultimodalDataset(str(DATA), processor, text_max_length=128, image_max_patches=256)

for idx in range(len(ds)):
    sample = ds[idx]
    if 'pixel_values' not in sample:
        print('Found missing pixel at', idx)
        print(sample.keys())
        break
else:
    print('All samples ok') 