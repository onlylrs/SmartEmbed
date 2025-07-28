# Jina Embeddings v4 Reproduction

## Features

- Based on Qwen2.5-VL-3B-Instruct architecture
- Support for text, image, and multimodal embeddings
- Task-specific adapters for retrieval, text-matching, and code
- Matryoshka embedding support (128, 256, 512, 1024, 2048 dimensions)
- Mean pooling strategy

## Project Structure (待定）

```
SmartEmbed/
├── src/
│   ├── models/              # Model definitions
│   ├── datasets/            # Dataset classes
│   ├── trainer/             # Custom trainers
│   └── utils/               # Utility functions
├── configs/                 # Configuration files
├── data/                    # Data directory
├── scripts/
│   ├── train/               # Training scripts
│   ├── inference/           # Inference scripts
│   └── evaluation/          # Evaluation scripts
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
# Optional: Install flash attention for better performance
pip install flash-attn --no-build-isolation
```

### 2. Prepare data

Create your dataset in JSONL format:

```json
{"text": "A beautiful sunset over the beach"}
{"text": "The ocean waves crashing against the shore"}
{"image": "path/to/image1.jpg"}
{"image": "path/to/image2.jpg"}
```

Place it in \`data/processed/\` directory.

### 3. Train the model

```bash
python scripts/train/train.py \\
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \\
  --data_path data/processed/train.jsonl \\
  --output_dir output/jina-embeddings \\
  --num_train_epochs 3 \\
  --per_device_train_batch_size 4 \\
  --learning_rate 2e-5 \\
  --bf16 True
```

### 4. Generate embeddings

```bash
python scripts/inference/embed.py \\
  --model output/jina-embeddings \\
  --task retrieval \\
  --input data/processed/test.jsonl \\
  --output embeddings.jsonl \\
  --truncate_dim 1024
```

### 5. Evaluate the model

```bash
python scripts/evaluation/evaluate.py
```

## Advanced Usage

### Using Matryoshka Embeddings

```python
from src.models.jina_model import JinaEmbeddingModel

model = JinaEmbeddingModel.from_pretrained("output/jina-embeddings")

# Get embeddings at different dimensions
embeddings = model.encode_text(["A beautiful sunset"], task="retrieval")
multivector = model.get_matryoshka_embeddings(embeddings)

for dim, emb in multivector.items():
    print(f"Dimension {dim}: shape {emb.shape}")
```
