#!/usr/bin/env python3
"""
Main training script for Jina Embeddings V4 fine-tuning

简化说明：
1. 直接从项目中导入Jina模型（已复制到src/models/）
2. 不需要复杂的动态导入
3. 模型路径直接指向base预训练权重目录
4. 简单直接，易于理解和维护
"""

import os
import sys
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main training function"""
    
    # Load configuration
    config_path = project_root / "project_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("=== Jina Embeddings V4 Fine-Tuning ===")
    print(f"Project root: {project_root}")
    print(f"Base model path: {config['base_model_path']}")
    
    # Validate base model path
    base_model_path = Path(config['base_model_path'])
    if not base_model_path.exists():
        print(f"❌ Base model path does not exist: {base_model_path}")
        return
    
    print("✅ Base model path exists")
    
    # Test imports
    try:
        print("\n� Testing imports...")
        from src.models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model, JinaEmbeddingsV4Processor, JinaEmbeddingsV4Config
        print("✅ Successfully imported Jina model classes")
        
        from src.datasets.jina_dataset import JinaEmbeddingDataset, JinaTrainingExample, load_training_data
        print("✅ Successfully imported dataset classes")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Test data loading
    try:
        print("\n� Testing data loading...")
        data_path = project_root / "data" / "examples" / "test_data.jsonl"
        
        if not data_path.exists():
            print(f"❌ Test data not found: {data_path}")
            return
            
        examples = load_training_data(str(data_path), data_format="jsonl")
        print(f"✅ Loaded {len(examples)} training examples")
        
        # Display first example
        if examples:
            ex = examples[0]
            print(f"  Sample - Query: {ex.query[:50]}...")
            print(f"  Sample - Positive: {ex.positive[:50]}...")
            print(f"  Sample - Task: {ex.task}")
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return
    
    # Test model initialization
    try:
        print("\n🤖 Testing model initialization...")
        
        # Load processor
        processor = JinaEmbeddingsV4Processor.from_pretrained(str(base_model_path))
        print("✅ Processor loaded successfully")
        
        # Test text processing
        test_texts = ["Query: What is AI?", "Passage: AI is artificial intelligence"]
        inputs = processor.process_texts(test_texts, max_length=128)
        print(f"✅ Text processing successful - Input shape: {inputs['input_ids'].shape}")
        
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return
    
    print("\n🎉 All tests passed! Ready for training.")
    print("\nNext steps:")
    print("- Run full training with: python train.py")
    print("- Check training configuration in project_config.yaml")

if __name__ == "__main__":
    main()
