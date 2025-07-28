import math
from typing import List, Optional, Union, Dict, Any

import numpy as np
from PIL import Image

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import (
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    to_numpy_array,
    valid_images,
)
from transformers.utils import TensorType, is_vision_available, logging

logger = logging.get_logger(__name__)

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')
    
    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - (downscaled_width * downscaled_height)
        
        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)
    
    return best_fit

class Qwen25VLImageProcessor(BaseImageProcessor):
    """
    Processor for Qwen2.5-VL images.
    """
    
    def __init__(
        self,
        image_size: int = 448,
        patch_size: int = 14,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std if image_std is not None else [0.26862954, 0.26130258, 0.27577711]
        
        self.crop_size = {
            "width": image_size,
            "height": image_size
        }
        self.size = {
            "shortest_edge": image_size
        }
    
    def preprocess(self, images, return_tensors: Optional[str] = None, **kwargs) -> BatchFeature:
        """
        Preprocess an image or batch of images.
        """
        if not isinstance(images, (list, tuple)):
            images = [images]
        
        images = [self._preprocess_image(image) for image in images]
        
        data = {"pixel_values": images}
        
        return BatchFeature(data=data, tensor_type=return_tensors)
    
    def _preprocess_image(self, image: ImageInput) -> np.ndarray:
        """
        Preprocess a single image.
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # 调整图像大小
        width, height = image.size
        aspect_ratio = width / height
        
        # 计算新的尺寸
        if aspect_ratio > 1:
            new_width = self.image_size
            new_height = int(self.image_size / aspect_ratio)
        else:
            new_height = self.image_size
            new_width = int(self.image_size * aspect_ratio)
        
        # 调整大小
        image = image.resize((new_width, new_height), resample=PILImageResampling.BICUBIC)
        
        # 创建空白图像并粘贴调整后的图像
        new_image = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        paste_x = (self.image_size - new_width) // 2
        paste_y = (self.image_size - new_height) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        # 转换为numpy数组并归一化
        image_array = np.array(new_image).astype(np.float32) / 255.0
        
        # 归一化
        if self.do_normalize:
            mean = np.array(self.image_mean)
            std = np.array(self.image_std)
            image_array = (image_array - mean) / std
        
        # 转换为CHW格式
        image_array = np.transpose(image_array, (2, 0, 1))
        
        return image_array
    
    def postprocess(self, image_features, output_type: str = "pil"):
        """
        Postprocess image features back to images.
        """
        # 简化版：直接返回特征
        return image_features
