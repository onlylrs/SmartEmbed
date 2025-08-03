# Baseline Model - Jina Embeddings V4

A reproduction and training framework for Jina Embeddings V4 model, supporting LoRA fine-tuning with optimized storage management.

## Features

- 🚀 **Based on Qwen2.5-VL-3B-Instruct** architecture
- 🎯 **LoRA Fine-tuning** with configurable parameters (reduces model size from 686MB to ~388MB)
- 📊 **Multiple Task Support** for retrieval, text-matching, and code tasks
- 💾 **Smart Storage Management** with automatic cleanup and space optimization
- 📝 **Comprehensive Logging** with detailed training output and progress tracking
- 🔧 **Flexible Configuration** via YAML and JSON config files

## Project Structure

```
SmartEmbed/
├── configs/                    # Training configuration files
│   ├── jina_training_config.json  # Main training parameters
│   └── ...                     # Other model configs
├── src/                        # Source code
│   ├── models/                 # Model implementations
│   ├── trainer/                # Training logic
│   ├── datasets/               # Data loading utilities
│   └── ...
├── scripts/                    # Training and evaluation scripts
│   ├── train/                  # Training scripts
│   │   └── train_jina.py       # Main training script
│   └── ...
├── data0/                      # Training data
│   ├── train.jsonl             # Training examples
│   └── eval.jsonl              # Evaluation examples
├── config.yaml                 # User-specific configuration (model paths)
├── run_training.sh             # Main training script with detailed output
└── requirements.txt            # Python dependencies
```

## Quick Start

### 1. Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configuration

1. **Copy configuration template:**
   ```bash
   cp config.yaml.example config.yaml
   ```

2. **Edit config.yaml** to set your model path:
   ```yaml
   model_path: "/your/path/to/jina-embedding-v4"  # Update with your actual path
   ```
   
   Make sure the model directory exists and contains the Jina Embeddings V4 model files.

### 3. Training

Run the training script:
```bash
./run_training.sh
```

The script will:
- ✅ Display detailed training progress in real-time
- ✅ Save complete logs to `training_output.log`
- ✅ Use LoRA for efficient fine-tuning
- ✅ Automatically clean up old checkpoints to save space
- ✅ Save the final adapter to `/project/fyp25_hc2/results/jina_test_run/`

## Training Configuration

### Default Parameters
- **Epochs:** 1
- **Batch Size:** 1
- **Learning Rate:** 2e-5
- **Sequence Length:** 256
- **LoRA Rank:** 16
- **LoRA Alpha:** 16
- **Precision:** FP16

### Custom Configuration

Modify `configs/jina_training_config.json` for advanced settings:
```json
{
  "num_train_epochs": 3,
  "per_device_train_batch_size": 2,
  "learning_rate": 5e-5,
  "use_lora": true,
  "lora_r": 32,
  "lora_alpha": 32
}
```

## Training Data Format

The training data should be in JSONL format. Example (`data0/train.jsonl`):
```json
{"task": "retrieval", "query": "What is machine learning?", "positive": "Machine learning is a subset of artificial intelligence...", "negative": "..."}
{"task": "text-matching", "query": "Python programming", "positive": "Python is a high-level programming language...", "negative": "..."}
```

## Output and Results

### Training Output
- **Real-time progress:** Displayed in terminal with progress bars
- **Complete logs:** Saved to `training_output.log`
- **Training metrics:** Loss, gradient norm, learning rate progression

### Model Artifacts
- **LoRA Adapter:** `adapter_model.safetensors` (~388MB)
- **Configuration:** `adapter_config.json`
- **Training Results:** `train_results.json`

### Storage Optimization
- ✅ Only LoRA adapters are saved (not full model)
- ✅ Automatic cleanup of old checkpoints
- ✅ Single checkpoint retention policy
- ✅ Uses shared storage space efficiently

## Advanced Usage

### Custom Training Arguments
```bash
python scripts/train/train_jina.py \
    --train_data your_data.jsonl \
    --config_file configs/jina_training_config.json \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --output_dir /your/output/path \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 32
```

### Loading Trained Model
```python
from peft import PeftModel
from src.models.modeling_qwen2_5_vl import JinaEmbeddingsV4Model

# Load base model
base_model = JinaEmbeddingsV4Model.from_pretrained("/path/to/base/model")

# Load trained adapter
model = PeftModel.from_pretrained(base_model, "/path/to/adapter")

# Use for inference
embeddings = model.encode(["Your text here"])
```

## Troubleshooting

### Common Issues

1. **Disk space errors:** The framework automatically uses `/project/fyp25_hc2/results/` for storage
2. **Memory issues:** Reduce batch size or enable gradient checkpointing
3. **Configuration errors:** Ensure `config.yaml` has the correct model path

## Requirements

- Python 3.8+
- PyTorch 2.6.0+
- Transformers 4.52.0+
- PEFT 0.15.2+
- See `requirements.txt` for complete list
