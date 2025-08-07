# jina/data/data_collator.py
"""Data collator that bridges `src.datasets.multimodal_dataset.MultimodalDataset`
with `jina.training.jina_trainer.JinaEmbeddingTrainer`.

The trainer expects each batch dict to contain keys prefixed with
`query_` and `positive_`, where the *suffix* must match the argument
names accepted by `JinaEmbeddingsV4Model.forward`, e.g. `input_ids`,
`attention_mask`, `pixel_values` …

Our `MultimodalDataset` already provides **both** modalities for a
single sample (an image *and* its paired caption).  Therefore the
conversion rule is:

    query   -> image branch  (pixel_values [+ optional image_grid_thw])
    positive -> text branch  (input_ids, attention_mask)

If your task uses text→text retrieval, feel free to adjust the mapping
logic accordingly.
"""

from typing import List, Dict, Any

import torch
from torch.nn.utils.rnn import pad_sequence


class JinaContrastiveDataCollator:
    """Collate multimodal retrieval samples into a contrastive batch."""

    def __init__(self, tokenizer=None, padding: bool = True, max_length: int = 128, return_tensors: str = "pt"):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.return_tensors = return_tensors

    # ---------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------
    def _pad_text_batch(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Pad a list of 1-D or 2-D tensors (token ids / masks)."""
        if not sequences:
            return None
        # All sequences already right-padded in MultimodalDataset, stack directly
        return torch.stack(sequences)

    # ------------------------------------------------------------------

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Split into separate modality lists
        pixel_values = [item["pixel_values"] for item in batch]
        image_grid_thw = [item.get("image_grid_thw") for item in batch if "image_grid_thw" in item]

        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]

        task_labels = [item.get("task_label", "retrieval") for item in batch]

        # Assemble prefixed batch dict
        collated: Dict[str, Any] = {}

        # Query branch (image)
        collated["query_pixel_values"] = torch.stack(pixel_values)
        if image_grid_thw:
            collated["query_image_grid_thw"] = torch.stack(image_grid_thw)

        # Positive branch (text)
        collated["positive_input_ids"] = self._pad_text_batch(input_ids)
        collated["positive_attention_mask"] = self._pad_text_batch(attention_masks)

        # Meta
        collated["task_labels"] = task_labels

        return collated 