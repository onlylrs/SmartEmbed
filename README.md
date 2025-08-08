# SmartEmbed (Jina Embeddings V4 Fine‑Tuning)

Internal research repo for adapting Jina Embeddings V4 (multi‑modal: text + image) with LoRA. Audience: project contributors only (not an end‑user package yet).

## 1. Quick Overview
- Base model: Jina Embeddings V4 (Qwen2.5-VL backbone + projection heads)
- Adaptation: Parameter‑efficient via LoRA only (base frozen)
- Tasks targeted (initial): retrieval (image ↔ text), can extend to text matching / code later
- Data format: JSONL (each line holds an image path + positive text [+ optional negative])

## 2. Key Files
```
project_config.yaml      High‑level editable training config (paths, epochs, LR, LoRA flags)
local_paths.yaml         (gitignored) Developer local absolute paths (base_model_path)
train.py                 Orchestrates config → dataloader → trainer → save
jina/training/jina_trainer.py   Custom Trainer (loss wiring, PEFT handling)
jina/data/multimodal_dataset.py Data loading + processor freezing
jina/data/data_collator.py      Maps dataset sample → contrastive batch (query_/positive_)
jina/models/*                  Model, config, losses, optional custom LoRA hooks
```

## 3. Local Path Handling
Create `local_paths.yaml` (copy from `local_paths.example.yaml`):
```
base_model_path: /abs/path/to/jina-embeddings-v4-base
```
Nothing else is required unless you add more path keys; code only reads `base_model_path`.

## 4. Branch Workflow
Current naming convention for personal work:
```
main        Stable / integrated
dev/fred    Fred's active development
dev/liam    Liam's active development
```
Create feature slices off `main` (e.g. `feature/xyz`) if scope is narrow.

## 5. Data Expectations
`train.py` expects (by default):
```
data/train.jsonl    # sample lines like:
{"query_image": "/abs/path/to/image.jpg", "positive": "a cat on a sofa"}
```
If images are relative, pass `image_base_dir` via `data_config` or extend loader.
All images are resized uniformly (currently 448x448) in the dataset for stable token counts.

## 6. Training Flow (Simplified)
```
project_config.yaml → create JinaTrainingConfig
↓
load & freeze base model + processor
↓
MultimodalDataset (text + image) → DataLoader (contrastive batch)
↓
Forward (query=image branch, positive=text branch)
↓
Contrastive loss (optionally Matryoshka later)
↓
Backward (LoRA parameters only)
↓
Checkpoint save (model + processor)
```

## 7. Run
```
# Activate your environment first
python train.py
```
Logs will show: samples loaded, LoRA param count, loss progression.

## 8. Verifying Real Training
Indicators training is genuine:
- Logged `contrastive_loss` changes over steps
- Non‑zero `grad_norm` early in training
- Only LoRA + projector parameters require grad (small trainable count)

## 9. Extending
| Goal | Where to touch |
|------|----------------|
| Add task label routing | `multimodal_dataset.py` (set task_label) + trainer logic |
| Add hard negatives | Modify dataset to include `negative_text` and adapt collator/trainer |
| Change image resolution | Dataset `_process_image` (keep consistent sizing) |
| Add eval phase | Provide eval JSONL + plug into `train.py` (currently skipped) |
| Switch loss variant | `jina/models/losses.py` |

## 10. Common Pitfalls
- Missing `local_paths.yaml` → model path error
- Batch size too small (2) → loss may collapse quickly (add gradient accumulation or increase device batch if memory allows)
- Relative image paths failing → supply `image_base_dir`

## 11. TODO (Short Horizon)
- Add evaluation loop hook
- Optional Matryoshka loss toggle & metrics
- Better in‑batch negative strategy
- Lightweight inference script (current `inference.py` removed; re‑introduce later)

## 12. Minimal Inference Sketch (Future)
```python
from jina.models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model, JinaEmbeddingsV4Processor
proc = JinaEmbeddingsV4Processor.from_pretrained("outputs/models/finetuned")
model = JinaEmbeddingsV4Model.from_pretrained("outputs/models/finetuned")
emb = model.get_single_vector_embeddings(texts=["hello world"], task_label="retrieval")
```

## 13. Environment
See `requirements.txt`. Flash‑Attention is listed; ensure GPU build compatibility. Set CUDA device externally if needed (e.g. `setenv CUDA_VISIBLE_DEVICES 0`).

---
Lean internal doc. Expand only when we stabilise external interface.
