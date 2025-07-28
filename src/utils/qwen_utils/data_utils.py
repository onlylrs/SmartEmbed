import json
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO

def load_image(image_file):
    """加载图像文件或URL"""
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

class MultimodalDataset(Dataset):
    """多模态数据集，用于训练嵌入模型"""
    
    def __init__(self, json_file, tokenizer, image_processor, max_length=512):
        with open(json_file, 'r') as f:
            self.data = [json.loads(line) for line in f]
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')
        image_path = item.get('image', None)
        
        # 处理文本
        text_inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 移除batch维度
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        
        # 处理图像
        pixel_values = None
        image_grid_thw = None
        
        if image_path:
            try:
                image = load_image(image_path)
                image_inputs = self.image_processor(images=image, return_tensors='pt')
                pixel_values = image_inputs['pixel_values'].squeeze(0)
                # 简化：假设图像网格为1x1x1
                image_grid_thw = torch.tensor([1, 1, 1])
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        
        # 返回数据项
        return {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'labels': idx  # 用于对比学习
        }

def contrastive_collate_fn(batch):
    """对比学习的批处理函数"""
    batch = [b for b in batch if b is not None]
    
    # 处理文本输入
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    
    # 处理图像输入
    pixel_values = None
    image_grid_thw = None
    
    if any(b['pixel_values'] is not None for b in batch):
        pixel_values = torch.stack([b['pixel_values'] for b in batch if b['pixel_values'] is not None])
        image_grid_thw = torch.stack([b['image_grid_thw'] for b in batch if b['image_grid_thw'] is not None])
    
    # 处理标签
    labels = torch.tensor([b['labels'] for b in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'labels': labels
    }
