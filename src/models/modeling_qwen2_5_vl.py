# Jina Embeddings V4 Model implementation was inspired by the ColPali codebase:
# https://github.com/illuin-tech/colpali

import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from io import BytesIO
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union, cast

import numpy as np
import requests
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchFeature
from transformers.utils import is_flash_attn_2_available

from .configuration_jina_embeddings_v4 import JinaEmbeddingsV4Config
from .custom_lora_module import MultiAdapterLinear
from .qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor


class PromptType(str, Enum):
    query = "query"
    passage = "passage"


PREFIX_DICT = {"query": "Query", "passage": "Passage"}


class JinaEmbeddingsV4Processor(Qwen2_5_VLProcessor):
    def __init__(self, *args, **kwargs) -> None:
        Qwen2_5_VLProcessor.__init__(self, *args, **kwargs)
        self.assistant_prefix_len = 58
        self.text_max_length = 32768

    def process_images(
        self,
        images: Union[List[Image.Image], List[List[Image.Image]]],
    ) -> BatchFeature:

        if isinstance(images[0], list):
            images = cast(List[List[Image.Image]], images)
            text_doc = []
            for i in range(len(images)):
                conversation = [
                    {"role": "user", "content": [{"type": "image"}] * len(images[i])}
                ]
                template = self.apply_chat_template(
                    conversation, add_generation_prompt=False
                )
                text_doc.append(template[self.assistant_prefix_len :])

        else:
            images = cast(List[Image.Image], images)
            text_doc = [
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n"
            ] * len(images)

        # The following code is a hack to make sure the scatter in DDP is done correctly when training on multiple GPUs
        batch_doc = self(text=text_doc, images=images, padding="longest", return_tensors="pt")  # type: ignore
        # Separate pixel_values for each image
        offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]
        # Pad pixel_values to the same length to be able to make it into a tensor
        pixel_values = torch.split(batch_doc["pixel_values"], offsets.tolist())

        max_length = max([len(pv) for pv in pixel_values])

        pixel_values = [
            torch.cat(
                [
                    pv,
                    torch.zeros(
                        (max_length - len(pv), pv.shape[1]),
                        dtype=pv.dtype,
                        device=pv.device,
                    ),
                ]
            )
            for pv in pixel_values
        ]

        batch_doc["pixel_values"] = torch.stack(pixel_values)
        return batch_doc

    def process_texts(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        prefix: Optional[str] = None,
        padding: Optional[str] = None,
    ) -> BatchFeature:

        max_length = (
            self.text_max_length
            if max_length is None
            else min(max_length, self.text_max_length)
        )
        padded_texts: List[str] = []

        for text in texts:
            if prefix:
                text = f"{prefix}: {text}"
            padded_texts.append(text)

        text_batch = self(
            text=padded_texts,
            return_tensors="pt",
            padding=padding or "longest",
            max_length=max_length,
            truncation=True,
        )

        return text_batch


@dataclass
class JinaEmbeddingsV4ModelOutput:
    """
    Base class for the Hybrid Model outputs.
    Args:
        vlm_last_hidden_states (torch.Tensor, optional): Last hidden states of the VLM.
        single_vec_emb (torch.Tensor, optional): Single-vector embeddings.
        multi_vec_emb (torch.Tensor, optional): Multi-vector embeddings.
    """

    vlm_last_hidden_states: Optional[torch.Tensor] = None
    single_vec_emb: Optional[torch.Tensor] = None
    multi_vec_emb: Optional[torch.Tensor] = None


class JinaEmbeddingsV4Model(Qwen2_5_VLForConditionalGeneration):
    config_class = JinaEmbeddingsV4Config
    main_input_name: ClassVar[str] = "doc_input_ids"

    def __init__(self, config: JinaEmbeddingsV4Config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        self._init_projection_layer(config)
        self.post_init()
        self.processor = JinaEmbeddingsV4Processor.from_pretrained(
            self.name_or_path, trust_remote_code=True, use_fast=True
        )
        self.multi_vector_projector_dim = config.multi_vector_projector_dim
        self.verbosity = config.verbosity
        self._task = None

    @property
    def task(self) -> Optional[str]:
        """Get the current task set for the model."""
        return self._task

    @task.setter
    def task(self, task: str):
        """
        Set the task for the model.

        Args:
            task (str): The task name. Must be one of ['retrieval', 'text-matching', 'code']
        """
        if task not in self.config.task_names:
            raise ValueError(
                f"Invalid task: {task}. Must be one of {self.config.task_names}."
            )
        self._task = task

    def get_last_hidden_states(
        self,
        task_label: Union[str, List[str]],
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            kwargs["pixel_values"] = torch.cat(
                [pv[:o] for pv, o in zip(kwargs["pixel_values"], offsets)], dim=0
            )
        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=kwargs.get("image_grid_thw", None),
            attention_mask=attention_mask,
        )

        kwargs["output_hidden_states"] = True
        outputs = super().forward(
            task_label=task_label,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            use_cache=False,
        )

        hidden_states = outputs.hidden_states
        if not hidden_states:
            raise ValueError("Hidden states not found in model output")

        return hidden_states[-1]

    def _init_projection_layer(self, config) -> None:
        """
        Initializes projection layers.
        """
        self.config.multi_vector_projector_dim = config.multi_vector_projector_dim

        self.multi_vector_projector = nn.Linear(
            in_features=self.config.text_config.hidden_size,
            out_features=self.config.multi_vector_projector_dim,
        )

    def get_single_vector_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Get the single-vector embeddings from the hidden states.
        """
        if self._input_has_image(input_ids[0]):  # got document image
            img_start_positions = torch.where(
                input_ids == self.config.vision_start_token_id
            )[1]
            img_end_positions = torch.where(
                input_ids == self.config.vision_end_token_id
            )[1]

            batch_size, seq_len = input_ids.shape
            position_indices = torch.arange(seq_len, device=input_ids.device).expand(
                batch_size, -1
            )
            image_mask = (position_indices >= img_start_positions.unsqueeze(1)) & (
                position_indices <= img_end_positions.unsqueeze(1)
            )

            masked_hidden_states = hidden_states * image_mask.unsqueeze(-1)
            pooled_output = masked_hidden_states.sum(dim=1) / image_mask.sum(
                dim=1, keepdim=True
            )
        else:  # got query text
            pooled_output = torch.sum(
                hidden_states * attention_mask.unsqueeze(-1), dim=1
            ) / torch.sum(attention_mask, dim=1, keepdim=True)

        return torch.nn.functional.normalize(pooled_output, dim=-1)

    def get_multi_vector_embeddings(
        self,
        task_label: Union[str, List[str]],
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project the hidden states to multi-vector embeddings.
        """
        multi_vec_emb = self.multi_vector_projector(
            hidden_states, task_label=task_label
        )
        multi_vec_emb = torch.nn.functional.normalize(multi_vec_emb, dim=-1)
        return multi_vec_emb * attention_mask.unsqueeze(-1)

    def _input_has_image(self, input_ids):
        return self.config.vision_start_token_id in input_ids

    def forward(
        self,
        task_label: Union[str, List[str]],
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        output_vlm_last_hidden_states: bool = False,
        **kwargs,
    ) -> JinaEmbeddingsV4ModelOutput:
        """
        Forward pass through the model. Returns both single-vector and multi-vector embeddings.
        Args:
            input_ids (torch.Tensor): The input tokens tensor.
            attention_mask (torch.Tensor): The attention mask tensor.
        Returns:
            JinaEmbeddingsV4ModelOutput:
                vlm_last_hidden_states (torch.Tensor, optional): Last hidden states of the VLM.
                single_vec_emb (torch.Tensor, optional): Single-vector embeddings.
                multi_vec_emb (torch.Tensor, optional): Multi-vector embeddings.
        """
        # Forward pass through the VLM
        hidden_states = self.get_last_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_label=task_label,
            **kwargs,
        )  # (batch_size, seq_length, hidden_size)
        # Compute the embeddings
        single_vec_emb = self.get_single_vector_embeddings(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            input_ids=input_ids,
        )
        multi_vec_emb = self.get_multi_vector_embeddings(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            task_label=task_label,
        )

        return JinaEmbeddingsV4ModelOutput(
            vlm_last_hidden_states=(
                hidden_states if output_vlm_last_hidden_states else None
            ),
            single_vec_emb=single_vec_emb,
            multi_vec_emb=multi_vec_emb,
        )

    def _process_batches(
        self,
        data: List[Union[str, Image.Image]],
        task_label: Union[str, List[str]],
        processor_fn: Callable,
        desc: str,
        return_multivector: bool = False,
        return_numpy: bool = False,
        batch_size: int = 32,
        truncate_dim: Optional[int] = None,
    ) -> Union[np.ndarray, List[torch.Tensor]]:
        dataloader = DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=processor_fn,
        )
        if return_multivector and len(data) > 1:
            assert (
                not return_numpy
            ), "`return_numpy` is not supported when `return_multivector=True` and more than one data is encoded"
        results = []
        self.eval()
        for batch in tqdm(dataloader, desc=desc, disable=self.verbosity == 0):
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.autocast(
                    device_type=torch.device(self.device).type, dtype=torch.bfloat16
                ):
                    embeddings = self(**batch, task_label=task_label)
                    if not return_multivector:
                        embeddings = embeddings.single_vec_emb
                        if truncate_dim is not None:
                            embeddings = embeddings[:, :truncate_dim]
                            embeddings = torch.nn.functional.normalize(
                                embeddings, p=2, dim=-1
                            )
                    else:
                        embeddings = embeddings.multi_vec_emb

                    if return_multivector and not return_numpy:
                        valid_tokens = batch["attention_mask"].bool()
                        embeddings = [
                            emb[mask] for emb, mask in zip(embeddings, valid_tokens)
                        ]
                        results.append(embeddings)
                    else:
                        results.append(
                            embeddings.cpu()
                            if return_numpy
                            else list(torch.unbind(embeddings))
                        )
        if return_numpy:
            return np.concatenate([result.numpy() for result in results], axis=0)
        return [item for sublist in results for item in sublist]

    def _validate_encoding_params(
        self,
        truncate_dim: Optional[int] = None,
        prompt_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        encode_kwargs = {}
        if prompt_name is not None:
            if prompt_name not in PREFIX_DICT:
                raise ValueError(
                    f"Invalid prompt_name: {prompt_name}. Must be one of {list(PREFIX_DICT.keys())}."
                )
            else:
                encode_kwargs["prefix"] = (
                    PREFIX_DICT[prompt_name]
                    if self.task != "text-matching"
                    else PREFIX_DICT["query"]
                )

        truncate_dim = truncate_dim or self.config.truncate_dim
        if truncate_dim is not None and truncate_dim not in self.config.matryoshka_dims:
            raise ValueError(
                f"Invalid truncate_dim: {truncate_dim}. Must be one of {self.config.matryoshka_dims}."
            )
        else:
            encode_kwargs["truncate_dim"] = truncate_dim

        return encode_kwargs

    def _validate_task(self, task: Optional[str] = None) -> str:
        if task is None:
            if self.task is None:
                raise ValueError(
                    "Task must be specified before encoding data. You can set it either as a model property "
                    "(e.g., model.task = 'retrieval') or pass it as an argument to the encode method."
                )
            task = self.task
        else:
            if task not in self.config.task_names:
                raise ValueError(
                    f"Invalid task: {task}. Must be one of {self.config.task_names}."
                )
        return task

    def encode_text(
        self,
        texts: Union[str, List[str]],
        task: Optional[str] = None,
        max_length: int = 32768,
        batch_size: int = 8,
        return_multivector: bool = False,
        return_numpy: bool = False,
        truncate_dim: Optional[int] = None,
        prompt_name: Optional[str] = None,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Encodes a list of texts into embeddings.

        Args:
            texts: text or list of text strings to encode
            max_length: Maximum token length for text processing
            batch_size: Number of texts to process at once
            return_multivector: Whether to return multi-vector embeddings instead of single-vector embeddings
            return_numpy: Whether to return numpy arrays instead of torch tensors
            truncate_dim: Dimension to truncate embeddings to (128, 256, 512, or 1024)
            prompt_name: Type of text being encoded ('query' or 'passage')

        Returns:
            List of text embeddings as tensors or numpy arrays when encoding multiple texts, or single text embedding as tensor when encoding a single text
        """
        prompt_name = prompt_name or "query"
        encode_kwargs = self._validate_encoding_params(
            truncate_dim=truncate_dim, prompt_name=prompt_name
        )

        task = self._validate_task(task)

        processor_fn = partial(
            self.processor.process_texts,
            max_length=max_length,
            prefix=encode_kwargs.pop("prefix"),
        )

        return_list = isinstance(texts, list)

        # If return_multivector is True and encoding multiple texts, ignore return_numpy
        if return_multivector and return_list and len(texts) > 1:
            if return_numpy:
                print(
                    "Warning: `return_numpy` is ignored when `return_multivector=True` and `len(texts) > 1`"
                )
            return_numpy = False

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._process_batches(
            data=texts,
            processor_fn=processor_fn,
            desc="Encoding texts...",
            task_label=task,
            return_multivector=return_multivector,
            return_numpy=return_numpy,
            batch_size=batch_size,
            **encode_kwargs,
        )

        return embeddings if return_list else embeddings[0]

    def _load_images_if_needed(
        self, images: List[Union[str, Image.Image]]
    ) -> List[Image.Image]:
        loaded_images = []
        for image in images:
            if isinstance(image, str):
                if image.startswith("http"):
                    response = requests.get(image)
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    image = Image.open(image).convert("RGB")
            loaded_images.append(image)
        return loaded_images

    def encode_image(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        task: Optional[str] = None,
        batch_size: int = 8,
        return_multivector: bool = False,
        return_numpy: bool = False,
        truncate_dim: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Encodes a list of images or a single image into embedding(s).

        Args:
            images: image(s) to encode, can be PIL Image(s), URL(s), or local file path(s)
            batch_size: Number of images to process at once
            return_multivector: Whether to return multi-vector embeddings instead of single-vector embeddings
            return_numpy: Whether to return numpy arrays instead of torch tensors. If `return_multivector` is `True` and more than one image is encoded, this parameter is ignored.
            truncate_dim: Dimension to truncate embeddings to (128, 256, 512, or 1024)
            max_pixels: Maximum number of pixels to process per image

        Returns:
            List of image embeddings as tensors or numpy arrays when encoding multiple images, or single image embedding as tensor when encoding a single image
        """
        if max_pixels:
            default_max_pixels = self.processor.image_processor.max_pixels
            self.processor.image_processor.max_pixels = (
                max_pixels  # change during encoding
            )
        encode_kwargs = self._validate_encoding_params(truncate_dim=truncate_dim)
        task = self._validate_task(task)

        return_list = isinstance(images, list)

        # If return_multivector is True and encoding multiple images, ignore return_numpy
        if return_multivector and return_list and len(images) > 1:
            if return_numpy:
                print(
                    "Warning: `return_numpy` is ignored when `return_multivector=True` and `len(images) > 1`"
                )
            return_numpy = False

        # Convert single image to list
        if isinstance(images, (str, Image.Image)):
            images = [images]

        images = self._load_images_if_needed(images)
        embeddings = self._process_batches(
            data=images,
            processor_fn=self.processor.process_images,
            desc="Encoding images...",
            task_label=task,
            batch_size=batch_size,
            return_multivector=return_multivector,
            return_numpy=return_numpy,
            **encode_kwargs,
        )

        if max_pixels:
            self.processor.image_processor.max_pixels = default_max_pixels

        return embeddings if return_list else embeddings[0]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        """
        Loads a pretrained model and configures it with the appropriate task adapter (`retrieval` by default).
        """
        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = "auto"

        kwargs["key_mapping"] = super()._checkpoint_conversion_mapping
        if not is_flash_attn_2_available():
            kwargs["attn_implementation"] = "sdpa"

        base_model = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

        # Configure adapter directory
        if os.path.isdir(base_model.name_or_path):
            adapter_dir = os.path.join(base_model.name_or_path, "adapters")
        else:
            adapter_cache_path = snapshot_download(
                repo_id=base_model.name_or_path, allow_patterns=["adapters/*"]
            )
            adapter_dir = os.path.join(adapter_cache_path, "adapters")

        lora_config = LoraConfig.from_pretrained(adapter_dir)
        lora_config._custom_modules = {
            torch.nn.modules.linear.Linear: partial(
                MultiAdapterLinear,
                task_names=base_model.config.task_names,
            )
        }
        peft_model = PeftModel.from_pretrained(
            model=base_model,
            model_id=adapter_dir,
            config=lora_config,
        )

        def task_getter(self):
            return self.model.task

        def task_setter(self, value):
            self.model.task = value

        peft_model.__class__.task = property(task_getter, task_setter)

        return peft_model
