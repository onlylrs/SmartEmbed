#!/usr/bin/env python3
"""
Base Model Explorer - 用于探索和测试base模型的API

这个脚本的作用：
1. 🔍 探索base模型提供的API接口
2. ✅ 检查必要的文件是否存在
3. 📚 提供使用base模型的示例代码
4. 🛠️ 帮助开发时理解base模型的结构

为什么需要这个脚本：
- base模型是一个完整的Jina实现，我们需要了解它的接口
- 避免在开发时盲目猜测API的使用方法
- 提供开发参考和调试信息
"""

import sys
import os
from pathlib import Path

# 添加base模型路径
base_model_path = Path("/csproject/fyp25_hc2/jina-embeddings-v4")

# 为了处理相对导入，我们需要将base路径添加到sys.path
# 并且将当前工作目录切换到base模型目录
original_cwd = os.getcwd()
original_path = sys.path.copy()

def setup_base_import():
    """设置base模型的导入环境"""
    # 切换到base模型目录（这样相对导入就能工作）
    os.chdir(str(base_model_path))
    # 将base路径添加到Python路径
    if str(base_model_path) not in sys.path:
        sys.path.insert(0, str(base_model_path))

def cleanup_import():
    """清理导入环境"""
    os.chdir(original_cwd)
    sys.path[:] = original_path

def explore_base_model():
    """探索base模型的API
    
    修改说明：
    1. 不直接导入base模型（因为相对导入问题）
    2. 改为读取和分析源代码文件
    3. 提供实际可用的导入方式示例
    4. 展示base模型的结构和API信息
    """
    print("=== Jina Embeddings V4 Base Model Explorer ===")
    print(f"Base model path: {base_model_path}")
    print("\n🎯 这个脚本的目的：")
    print("   - 了解base模型提供的API接口")
    print("   - 查看可用的类和方法")
    print("   - 获取开发时的参考信息")
    
    # 由于相对导入问题，我们改为分析源代码文件
    print("\n=== � 分析base模型源代码... ===")
    
    try:
        # 读取主要模型文件
        modeling_file = base_model_path / "modeling_jina_embeddings_v4.py"
        config_file = base_model_path / "configuration_jina_embeddings_v4.py"
        
        if modeling_file.exists():
            with open(modeling_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 分析类定义
            print("🤖 在 modeling_jina_embeddings_v4.py 中发现的类：")
            import re
            class_matches = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
            for class_name in class_matches:
                print(f"   � {class_name}")
            
            # 分析主要函数
            function_matches = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
            print(f"\n�️ 发现 {len(function_matches)} 个函数定义")
            
            # 查找from_pretrained方法
            if 'from_pretrained' in content:
                print("✅ 支持 from_pretrained 方法")
            
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print("\n⚙️ 在 configuration_jina_embeddings_v4.py 中发现的类：")
            class_matches = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
            for class_name in class_matches:
                print(f"   📋 {class_name}")
        
        # 展示正确的使用方式
        print("\n=== 📚 正确的导入和使用方式 ===")
        print("""
💡 由于相对导入问题，在您的训练脚本中需要这样处理：

🔧 方法1：修改base模型文件（临时解决）
```python
# 在base模型文件中，将相对导入改为绝对导入
# 例如：from .configuration_jina_embeddings_v4 import JinaEmbeddingsV4Config
# 改为：from configuration_jina_embeddings_v4 import JinaEmbeddingsV4Config
```

🔧 方法2：使用模块导入（推荐）
```python
import sys
import os
import importlib.util

# 设置路径
base_path = "/csproject/fyp25_hc2/jina-embeddings-v4"
sys.path.insert(0, base_path)
os.chdir(base_path)

# 动态导入模块
spec = importlib.util.spec_from_file_location(
    "modeling_jina", 
    os.path.join(base_path, "modeling_jina_embeddings_v4.py")
)
modeling_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modeling_module)

# 使用导入的类
JinaModel = modeling_module.JinaEmbeddingsV4Model
```

🔧 方法3：复制核心文件到您的项目中
```bash
# 将需要的文件复制到您的src/models/目录
cp /csproject/fyp25_hc2/jina-embeddings-v4/modeling_jina_embeddings_v4.py src/models/
cp /csproject/fyp25_hc2/jina-embeddings-v4/configuration_jina_embeddings_v4.py src/models/
# 然后修改其中的相对导入为绝对导入
```
        """)
        
        print("✅ 源码分析完成！")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        print("💡 建议：检查文件权限和路径")
    
    finally:
        cleanup_import()

def check_base_dependencies():
    """检查base模型的依赖
    
    这个函数会：
    1. 列出base模型目录中的所有Python文件
    2. 检查关键文件是否存在
    3. 验证base模型的完整性
    """
    print("\n=== 📁 检查base模型依赖 ===")
    base_files = list(base_model_path.glob("*.py"))
    print(f"🔍 发现 {len(base_files)} 个Python文件：")
    for file in base_files:
        print(f"   📄 {file.name}")
        
    # 检查关键文件
    key_files = [
        "modeling_jina_embeddings_v4.py",    # 核心模型实现
        "configuration_jina_embeddings_v4.py", # 模型配置
        "qwen2_5_vl.py",                     # 基础架构
        "custom_lora_module.py"              # LoRA适配器
    ]
    
    print(f"\n=== ✅ 关键文件检查 ===")
    all_present = True
    for file in key_files:
        file_path = base_model_path / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} - 缺失")
            all_present = False
    
    if all_present:
        print("🎉 所有关键文件都存在！")
    else:
        print("⚠️ 某些关键文件缺失，可能影响功能")
    
    return all_present


if __name__ == "__main__":
    print("🚀 启动Base Model Explorer...")
    print("=" * 60)
    
    # 检查文件完整性
    files_ok = check_base_dependencies()
    
    if files_ok:
        # 探索API
        explore_base_model()
    else:
        print("❌ 由于文件缺失，跳过API探索")
    
    print("=" * 60)
    print("🏁 探索完成！")
