# 🚀 Multimodal DataLoader Quick Start

## Single Command Usage

### See Input/Output Format
```bash
python dataloader_guide.py --show-io
```
**Shows:** What the dataloader takes as input and what it produces as output

### See All Config Options
```bash
python dataloader_guide.py --show-config  
```
**Shows:** Every parameter you can customize, with types, defaults, and descriptions

### See Usage Examples
```bash
python dataloader_guide.py --show-examples
```
**Shows:** 5 practical examples from minimal to high-performance setups

### Test with Real Data
```bash
python dataloader_guide.py --test
```
**Shows:** Live test with your actual data, showing real tensor shapes and values

### See Everything
```bash
python dataloader_guide.py --all
```
**Shows:** Complete guide with all information

## Ultra-Quick Usage

```python
from src.datasets.multimodal_dataset import get_training_dataloader

# Minimal - just provide data path
dataloader = get_training_dataloader({'jsonl_path': 'data0/train.jsonl'})

# Use like any PyTorch dataloader
for batch in dataloader:
    outputs = model(
        task_label=batch['task_labels'],
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        pixel_values=batch['pixel_values'],
        image_grid_thw=batch['image_grid_thw']
    )
```

## What You Get

**Input:** Simple config dict  
**Output:** PyTorch DataLoader with batches containing:
- `input_ids`: Text tokens (batch_size, seq_len)
- `pixel_values`: Image patches (batch_size, patches, embed_dim)  
- `attention_mask`: Text attention mask
- `image_grid_thw`: Image grid dimensions
- `task_labels`: Task names for each sample
- `negative_input_ids/negative_attention_mask`: For contrastive learning

**Ready for:** Direct use with Jina Embeddings v4 model training!