#!/usr/bin/env python3
"""
扩展的验证脚本 - 验证完整训练流程的各个组件

该文件功能：
✅ 基础模型路径存在
✅ 测试模块导入 (jina.models, jina.data, jina.training)
✅ 测试数据加载 (4个测试样本)
✅ 测试处理器初始化
✅ 测试文本处理功能
✅ 测试模型加载和LoRA设置 (新增)
✅ 测试数据加载器创建 (新增)
✅ 测试训练器初始化 (新增)
❌ 不进行实际训练
❌ 不保存模型

验证范围：
✅ 导入路径正确
✅ 基础模型路径有效
✅ 数据格式兼容
✅ 处理器能正常工作
✅ 基本的文本处理功能正常
✅ LoRA配置正确 (新增)
✅ 数据加载器工作正常 (新增)
✅ 训练器可以初始化 (新增)
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
        print("\n🔧 Testing imports...")
        from jina.models.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model, JinaEmbeddingsV4Processor, JinaEmbeddingsV4Config
        print("✅ Successfully imported Jina model classes")
        
        from jina.data.jina_dataset import JinaEmbeddingDataset, JinaTrainingExample, load_training_data, create_dataloaders
        print("✅ Successfully imported dataset classes")
        
        from jina.training.jina_trainer import JinaEmbeddingTrainer, setup_model_for_training
        print("✅ Successfully imported training classes")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Test data loading
    try:
        print("\n📊 Testing data loading...")
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
        
        # Load model
        model = JinaEmbeddingsV4Model.from_pretrained(str(base_model_path))
        print("✅ Base model loaded successfully")
        
        # Test text processing
        test_texts = ["Query: What is AI?", "Passage: AI is artificial intelligence"]
        inputs = processor.process_texts(test_texts, max_length=128)
        print(f"✅ Text processing successful - Input shape: {inputs['input_ids'].shape}")
        
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return

    # Test LoRA setup (new)
    try:
        print("\n⚙️ Testing LoRA configuration...")
        
        # Test LoRA setup with the loaded model
        model_with_lora = setup_model_for_training(
            model,
            use_lora=True,
            lora_r=16,
            lora_alpha=16,
            lora_dropout=0.1
        )
        
        print("✅ LoRA setup successful")
        print(f"✅ Model type: {type(model_with_lora).__name__}")
        
        # Check if LoRA was applied
        if hasattr(model_with_lora, 'peft_config'):
            print("✅ PEFT/LoRA adapters detected")
        else:
            print("⚠️ No PEFT adapters detected (model may already have them)")
            
        # Update model reference for subsequent tests
        model = model_with_lora
        
    except Exception as e:
        print(f"❌ LoRA setup error: {e}")
        print(f"   Error details: {type(e).__name__}: {str(e)}")
        return

    # Test data loader creation (new)
    try:
        print("\n📊 Testing data loader creation...")
        
        # Create small dataloaders with test data
        train_dataloader, eval_dataloader = create_dataloaders(
            train_examples=examples,
            eval_examples=examples[:2],  # Use first 2 as eval
            processor=processor,
            batch_size=1,
            max_length=128,
            dataset_type="contrastive"
        )
        
        print(f"✅ Train dataloader created - {len(train_dataloader.dataset)} samples")
        print(f"✅ Eval dataloader created - {len(eval_dataloader.dataset)} samples")
        
        # Test one batch
        batch = next(iter(train_dataloader))
        print(f"✅ Batch creation successful - keys: {list(batch.keys())}")
        
    except Exception as e:
        print(f"❌ Data loader creation error: {e}")
        return

    # Test trainer initialization (new)
    try:
        print("\n🏋️ Testing trainer initialization...")
        
        from transformers import TrainingArguments
        from jina.training.training_config import JinaTrainingConfig
        
        # Create Jina training configuration (required for trainer)
        training_config = JinaTrainingConfig(
            model_name_or_path=str(base_model_path),
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            use_lora=True,
            lora_r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            temperature=0.02,  # Required for loss functions
            margin=0.0,
            matryoshka_dims=[128, 256, 512],
        )
        
        # Create minimal training arguments  
        training_args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            logging_steps=1,
            save_steps=100,
            eval_steps=100,
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
            label_names=[],  # Explicit empty label_names for feature extraction
        )
        
        # Initialize trainer with both configs
        trainer = JinaEmbeddingTrainer(
            model=model,
            training_config=training_config,  # This was missing!
            training_args=training_args,
            tokenizer=processor,  # Pass processor as tokenizer
            train_dataset=train_dataloader.dataset,
            eval_dataset=eval_dataloader.dataset,
        )
        
        print("✅ Trainer initialized successfully")
        
    except Exception as e:
        print(f"❌ Trainer initialization error: {e}")
        print(f"   Error details: {type(e).__name__}: {str(e)}")
        return

    print("\n🎉 All extended tests passed! Training pipeline is ready.")
    print("\nValidated components:")
    print("- ✅ Model loading and initialization")
    print("- ✅ Data processing and loading")  
    print("- ✅ LoRA configuration availability")
    print("- ✅ Trainer initialization")
    print("\nNext steps:")
    print("- Run actual training with: python train.py")
    print("- Check training configuration in project_config.yaml")

if __name__ == "__main__":
    main()
