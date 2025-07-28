# src/models/modeling.py
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import Qwen2Model, Qwen2Config
from transformers.image_processing_utils import select_best_resolution

logger = logging.get_logger(__name__)

class Qwen25VLConfig(Qwen2Config):
    model_type = "qwen2_5_vl"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=3072,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=24,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        vision_config=None,
        **kwargs,
    ):
        self.vision_config = vision_config if vision_config is not None else {}
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            **kwargs,
        )

class QwenVLModel(PreTrainedModel):
    config_class = Qwen25VLConfig
    
    def __init__(self, config: Qwen25VLConfig):
        super().__init__(config)
        
        # 文本模型
        self.text_model = Qwen2Model(config)
        
        # 视觉模型（简化版）
        self.vision_tower = nn.Sequential(
            nn.Conv2d(3, 1024, kernel_size=14, stride=14, padding=0),
            nn.GELU(),
            nn.Linear(1024, config.hidden_size)
        )
        
        # 视觉到文本的投影
        self.mm_projector = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.text_model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.text_model.embed_tokens = value
    
    def _merge_vision_tokens(self, text_embeds, vision_embeds, image_grid_thw):
        """
        合并文本和视觉特征
        """
        batch_size = text_embeds.shape[0]
        merged_embeds = []
        
        for i in range(batch_size):
            # 简化版：假设我们有一个图像嵌入
            if vision_embeds is not None:
                # 将视觉特征投影到文本空间
                vision_features = self.mm_projector(vision_embeds[i])
                # 合并文本和视觉特征
                merged = torch.cat([text_embeds[i], vision_features], dim=0)
                merged_embeds.append(merged)
            else:
                merged_embeds.append(text_embeds[i])
        
        return torch.stack(merged_embeds)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.text_model.embed_tokens(input_ids)

        # 处理视觉输入
        vision_embeds = None
        if pixel_values is not None:
            # 简化版视觉特征提取
            batch_size, num_channels, height, width = pixel_values.shape
            vision_embeds = self.vision_tower(pixel_values)
            vision_embeds = vision_embeds.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # 合并文本和视觉特征
        inputs_embeds = self._merge_vision_tokens(inputs_embeds, vision_embeds, image_grid_thw)
        
        # 获取文本模型输出
        outputs = self.text_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        return outputs
