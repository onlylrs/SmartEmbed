from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any

# 从我们创建的文件中导入
from src.models.modeling import QwenVLModel
from src.utils.qwen_utils.configuration import Qwen25VLConfig

class JinaEmbeddingModel(PreTrainedModel):
    """
    Jina Embeddings v4 模型的实现，基于 Qwen2.5-VL-3B-Instruct
    """
    config_class = Qwen25VLConfig
    
    def __init__(self, config, **kwargs):
        super().__init__(config)
        # 使用 Qwen2.5-VL 作为基础模型
        self.qwen_vl = QwenVLModel(config)
        self.pooling_strategy = kwargs.get('pooling_strategy', 'mean')
        self.matryoshka_dims = kwargs.get('matryoshka_dims', [128, 256, 512, 1024, 2048])
        
        # 任务适配器
        self.adapters = nn.ModuleDict({
            "retrieval": nn.Identity(),
            "text-matching": nn.Identity(),
            "code": nn.Identity()
        })
    
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        # 可以添加自定义的权重初始化
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """从预训练模型加载"""
        # 加载配置
        config = kwargs.pop("config", None)
        if config is None:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, 
                trust_remote_code=True,
                **kwargs
            )
        
        # 创建实例
        instance = cls(config, **kwargs)
        
        # 加载 Qwen2.5-VL 模型权重
        qwen_vl = QwenVLModel.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            trust_remote_code=True,
            **kwargs
        )
        
        # 替换内部模型
        instance.qwen_vl = qwen_vl
        
        return instance
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> BaseModelOutputWithPooling:
        """
        前向传播，返回嵌入向量
        """
        # 获取 Qwen2.5-VL 的输出
        outputs = self.qwen_vl(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
            **kwargs
        )
        
        # 提取最后一层隐藏状态
        last_hidden_state = outputs.last_hidden_state
        
        # 应用池化策略（Jina Embeddings v4 使用 mean pooling）
        if self.pooling_strategy == "mean":
            # Mean pooling (忽略 padding)
            if attention_mask is not None:
                masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1)
                embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                embeddings = last_hidden_state.mean(dim=1)
        elif self.pooling_strategy == "cls":
            embeddings = last_hidden_state[:, 0]
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
        
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
    
    def encode_text(
        self,
        texts: List[str],
        task: str = "retrieval",
        prompt_name: Optional[str] = None,
        truncate_dim: int = 2048,
        return_multivector: bool = False
    ) -> torch.Tensor:
        """
        编码文本为嵌入向量
        
        Args:
            texts: 文本列表
            task: 任务类型 (retrieval, text-matching, code)
            prompt_name: 提示名称 (query, passage)
            truncate_dim: 截断维度
            return_multivector: 是否返回多向量
            
        Returns:
            嵌入向量张量
        """
        # 设置为评估模式
        self.eval()
        
        # 准备输入
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config._name_or_path, trust_remote_code=True)
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=32768
        ).to(self.device)
        
        # 获取嵌入
        with torch.no_grad():
            outputs = self(**inputs)
            embeddings = outputs.pooler_output
            
            # 应用任务适配器
            if task in self.adapters:
                embeddings = self.adapters[task](embeddings)
            
            # 截断到指定维度（Matryoshka）
            if truncate_dim < embeddings.size(-1):
                embeddings = embeddings[:, :truncate_dim]
        
        return embeddings
    
    def encode_image(
        self,
        images: List[str],
        task: str = "retrieval",
        truncate_dim: int = 2048,
        return_multivector: bool = False
    ) -> torch.Tensor:
        """
        编码图像为嵌入向量
        
        Args:
            images: 图像路径或URL列表
            task: 任务类型
            truncate_dim: 截断维度
            return_multivector: 是否返回多向量
            
        Returns:
            嵌入向量张量
        """
        # 设置为评估模式
        self.eval()
        
        # 加载图像处理器
        from src.utils.qwen_utils.image_processing import Qwen25VLImageProcessor
        image_processor = Qwen25VLImageProcessor.from_pretrained(self.config._name_or_path)
        
        # 处理图像
        processed_images = []
        for img in images:
            image = load_image(img)
            image_inputs = image_processor(images=image, return_tensors="pt")
            processed_images.append(image_inputs)
        
        # 批量处理
        pixel_values = torch.cat([img['pixel_values'] for img in processed_images], dim=0).to(self.device)
        # 简化：假设所有图像网格为1x1x1
        image_grid_thw = torch.ones(len(images), 3).long().to(self.device)
        
        # 获取嵌入
        with torch.no_grad():
            outputs = self(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw
            )
            embeddings = outputs.pooler_output
            
            # 应用任务适配器
            if task in self.adapters:
                embeddings = self.adapters[task](embeddings)
            
            # 截断到指定维度
            if truncate_dim < embeddings.size(-1):
                embeddings = embeddings[:, :truncate_dim]
        
        return embeddings
    
    def get_matryoshka_embeddings(self, embeddings: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        生成不同维度的 Matryoshka 嵌入
        """
        results = {}
        for dim in self.matryoshka_dims:
            if dim <= embeddings.size(-1):
                results[dim] = embeddings[:, :dim]
        return results
