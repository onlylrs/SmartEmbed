# Jina Embeddings V4 Research Project

## 🎯 项目概述

本项目基于 Jina Embeddings V4 进行多任务嵌入模型的微调研究，使用 LoRA (Low-Rank Adaptation) 技术进行高效的参数更新。

## 📁 项目结构

### 核心配置文件
```
project_config.yaml          # 🔧 用户配置接口 (你和Liam编辑这个)
├─ 基础模型路径
├─ 训练超参数 (epochs, batch_size, learning_rate)
├─ LoRA配置
└─ 系统设置

train.py                     # 🚀 主训练脚本
├─ 读取 project_config.yaml
├─ 转换为 JinaTrainingConfig
└─ 启动训练流程
```

### Jina 模型架构 (jina/models/)

#### 📋 文件分类

**从 Base Model 复制的文件：**
- `configuration_jina_embeddings_v4.py` - 模型配置类
- `modeling_jina_embeddings_v4.py` - 主模型实现
- `qwen2_5_vl.py` - Qwen2.5-VL 骨干网络

**自定义实现的文件：**
- `losses.py` - 对比学习损失函数
- `custom_lora_module.py` - 任务特定的 LoRA 模块

## 🔄 模型数据流链条

### 链条 1: 配置流
```
project_config.yaml 
    ↓ (train.py 读取)
JinaTrainingConfig 对象
    ↓ (传递给)
JinaEmbeddingTrainer
    ↓ (初始化)
损失函数 + 训练参数
```

### 链条 2: 模型初始化流
```
1. JinaEmbeddingsV4Config (配置参数)
    ↓
2. Qwen2_5_VLModel (基础Transformer)
    ↓
3. JinaEmbeddingsV4Model (包装 + 投影层)
    ↓
4. LoRA 适配器 (任务特定参数)
    ↓
5. JinaEmbeddingTrainer (训练控制)
```

### 链条 3: 训练数据流
```
输入文本/图像
    ↓
JinaEmbeddingsV4Processor (预处理)
    ↓
Qwen2_5_VLModel (编码器)
    ↓
投影层 (single_vector_projector/multi_vector_projector)
    ↓
嵌入向量
    ↓
损失函数 (JinaContrastiveLoss)
    ↓
梯度更新 (仅更新 LoRA 参数)
```

## 🧩 模型组件详解

### 1. 配置层 (`configuration_jina_embeddings_v4.py`)
```python
JinaEmbeddingsV4Config
├─ 继承自 Qwen2_5_VLConfig
├─ 定义模型架构参数
└─ 设置任务特定配置
```

### 2. 骨干网络 (`qwen2_5_vl.py`)
```python
Qwen2_5_VLModel
├─ 多模态 Transformer 编码器
├─ 支持文本和图像输入
├─ 集成 LoRA 适配器
└─ 输出：隐藏状态 → 投影层
```

### 3. 主模型 (`modeling_jina_embeddings_v4.py`)
```python
JinaEmbeddingsV4Model
├─ 包装 Qwen2_5_VLModel
├─ 添加投影层 (single_vector, multi_vector)
├─ 任务路由逻辑
└─ 输出：任务特定嵌入向量

JinaEmbeddingsV4Processor
├─ 文本/图像预处理
├─ Tokenization
└─ 数据格式转换
```

### 4. LoRA 模块 (`custom_lora_module.py`)
```python
TaskSpecificLoRAModule
├─ 任务特定的 LoRA 适配器
├─ 动态参数路由
└─ 高效参数更新
```

### 5. 损失函数 (`losses.py`)
```python
JinaContrastiveLoss
├─ 对比学习损失
├─ 温度缩放 (temperature)
└─ 负样本挖掘

JinaMultiTaskLoss
├─ 多任务联合损失
└─ 任务权重平衡

JinaMatryoshkaLoss
├─ 多维度嵌入损失
└─ 维度递进训练
```

## 🚀 训练流程

### 1. 初始化阶段
```bash
python train.py
├─ 加载 project_config.yaml
├─ 创建 JinaTrainingConfig
├─ 初始化模型和处理器
└─ 设置 LoRA 适配器
```

### 2. 训练阶段
```
每个训练步骤：
1. 数据加载 → JinaEmbeddingDataset
2. 预处理 → JinaEmbeddingsV4Processor  
3. 模型前向 → JinaEmbeddingsV4Model
4. 损失计算 → JinaContrastiveLoss
5. 反向传播 → 仅更新 LoRA 参数
6. 参数更新 → AdamW 优化器
```

## 🔧 开发工作分工

### Fred (模型与训练专家)
- ✅ 训练流程优化 (`train.py`, `jina_trainer.py`)
- ✅ 模型架构调试 (`modeling_jina_embeddings_v4.py`)
- ✅ 损失函数实验 (`losses.py`)
- ✅ 推理接口开发 (`inference.py`)

### Liam (数据与评估专家)  
- ✅ 数据处理管道 (`jina_dataset.py`, `preprocess.py`)
- ✅ 评估指标实现 (`evaluate.py`)
- ✅ 数据集扩展和质量控制
- ✅ 多任务数据准备

## 📊 快速开始

```bash
# 1. 编辑配置
vim project_config.yaml

# 2. 开始训练
python train.py

# 3. 推理测试
python inference.py --model_path outputs/models/finetuned --texts "Hello world"

# 4. 评估模型
python evaluate.py --model_path outputs/models/finetuned
```

## 🎯 当前状态

- ✅ 配置系统完整打通
- ✅ 训练流程验证通过
- ✅ 小数据集测试就绪 (8条训练样本)
- 🔄 等待大规模数据集准备
- 🔄 推理系统优化进行中
