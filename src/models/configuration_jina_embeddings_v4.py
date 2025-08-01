from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig
from typing import Optional


class JinaEmbeddingsV4Config(Qwen2_5_VLConfig):
    """
    Configuration for the JinaEmbeddingsV4 model.
    """

    def __init__(
        self,
        single_vector_pool_strategy: str = "mean",
        multi_vector_projector_dim: int = 128,
        pretrained_peft_model_name_or_path: Optional[str] = None,
        verbosity: int = 1,
        task_names: list = None,
        matryoshka_dims: list = None,
        truncate_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.single_vector_pool_strategy = single_vector_pool_strategy
        self.multi_vector_projector_dim = multi_vector_projector_dim
        self.pretrained_peft_model_name_or_path = pretrained_peft_model_name_or_path
        self.verbosity = verbosity
        self.task_names = task_names or ["retrieval", "text-matching", "code"]
        self.matryoshka_dims = matryoshka_dims or [128, 256, 512, 1024, 2048]
        self.truncate_dim = truncate_dim
