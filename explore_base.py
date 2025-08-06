#!/usr/bin/env python3
"""
Base Model Explorer - 用于探索和测试base模型的API
"""

import sys
import os
from pathlib import Path

# 添加base模型路径
base_model_path = Path("/csproject/fyp25_hc2/jina-embeddings-v4")
sys.path.insert(0, str(base_model_path))

def explore_base_model():
    """探索base模型的API"""
    print("=== Jina Embeddings V4 Base Model Explorer ===")
    print(f"Base model path: {base_model_path}")
    
    try:
        # 导入主要组件
        from modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model, JinaEmbeddingsV4Processor
        from configuration_jina_embeddings_v4 import JinaEmbeddingsV4Config
        
        print("✓ 成功导入base模型组件")
        
        # 查看模型配置
        print("\n=== 模型配置 ===")
        config = JinaEmbeddingsV4Config()
        print(f"Config attributes: {[attr for attr in dir(config) if not attr.startswith('_')]}")
        
        # 查看处理器
        print("\n=== 处理器信息 ===") 
        print(f"Processor class: {JinaEmbeddingsV4Processor}")
        print(f"Processor methods: {[method for method in dir(JinaEmbeddingsV4Processor) if not method.startswith('_')]}")
        
        # 查看模型
        print("\n=== 模型信息 ===")
        print(f"Model class: {JinaEmbeddingsV4Model}")
        print(f"Model methods: {[method for method in dir(JinaEmbeddingsV4Model) if not method.startswith('_') and not method.startswith('load')][:10]}...")
        
        print("\n=== 使用示例 ===")
        print("""
        # 加载模型和处理器
        model = JinaEmbeddingsV4Model.from_pretrained(base_model_path)
        processor = JinaEmbeddingsV4Processor.from_pretrained(base_model_path)
        
        # 处理文本
        texts = ["Your text here"]
        inputs = processor(texts, return_tensors="pt")
        
        # 获取嵌入
        with torch.no_grad():
            outputs = model(**inputs)
        """)
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请检查base模型路径是否正确")
    except Exception as e:
        print(f"❌ 其他错误: {e}")

def check_base_dependencies():
    """检查base模型的依赖"""
    print("\n=== 检查base模型依赖 ===")
    base_files = list(base_model_path.glob("*.py"))
    for file in base_files:
        print(f"📄 {file.name}")
        
    # 检查关键文件
    key_files = [
        "modeling_jina_embeddings_v4.py",
        "configuration_jina_embeddings_v4.py", 
        "qwen2_5_vl.py",
        "custom_lora_module.py"
    ]
    
    print("\n=== 关键文件检查 ===")
    for file in key_files:
        file_path = base_model_path / file
        if file_path.exists():
            print(f"✓ {file}")
        else:
            print(f"❌ {file} - 缺失")

if __name__ == "__main__":
    check_base_dependencies()
    explore_base_model()
