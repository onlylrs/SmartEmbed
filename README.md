# Jina Embeddings v4 Reproduction

## Features

- Based on Qwen2.5-VL-3B-Instruct architecture
- Support for text, image, and multimodal embeddings
- Task-specific adapters for retrieval, text-matching, and code
- Matryoshka embedding support (128, 256, 512, 1024, 2048 dimensions)
- Mean pooling strategy

## Project Structure 

```
待定
```

## Setup Instructions

1. Copy `config.yaml.example` to `config.yaml`:
  ```bash
   cp config.yaml.example config.yaml
   ```
2. Edit config.yaml to set your model_path (e.g., /homes/your_username/Desktop/FYP/jina-embedding-v4). Ensure the model directory exists and contains the required files.
3. Run the training script:
  ```bash
  ./run_training.sh
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
