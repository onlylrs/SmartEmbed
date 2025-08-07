#!/usr/bin/env python3
"""
Comprehensive guide to using the multimodal dataloader.

Run with different commands to see:
- What inputs/outputs look like
- What config options are available
- How to customize settings
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datasets.multimodal_dataset import get_training_dataloader


def show_input_output():
    """Show what the dataloader takes as input and produces as output."""
    print("=" * 60)
    print("📥 DATALOADER INPUT/OUTPUT GUIDE")
    print("=" * 60)
    
    print("\n🔧 CREATING DATALOADER:")
    print("from src.datasets.multimodal_dataset import get_training_dataloader")
    print("dataloader = get_training_dataloader(config)")
    
    print("\n📥 INPUT (config dict):")
    config = {
        'jsonl_path': 'data0/train.jsonl',
        'batch_size': 2,
        'text_max_length': 128,
        'image_max_patches': 256
    }
    
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n🏗️  CREATING DATALOADER...")
    try:
        dataloader = get_training_dataloader(config)
        print(f"✅ DataLoader created: {len(dataloader.dataset)} samples, {len(dataloader)} batches")
        
        print("\n📤 OUTPUT (batch dict):")
        batch = next(iter(dataloader))
        
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
        
        print("\n💡 USAGE IN TRAINING:")
        print("for batch in dataloader:")
        print("    outputs = model(")
        print("        task_label=batch['task_labels'],")
        print("        input_ids=batch['input_ids'],")
        print("        attention_mask=batch['attention_mask'],")
        print("        pixel_values=batch['pixel_values'],")
        print("        image_grid_thw=batch['image_grid_thw']")
        print("    )")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def show_config_options():
    """Show all available configuration options."""
    print("=" * 60)
    print("⚙️  DATALOADER CONFIGURATION OPTIONS")
    print("=" * 60)
    
    config_options = {
        '📁 Data Settings': {
            'jsonl_path': {
                'type': 'str',
                'required': True,
                'default': 'N/A',
                'description': 'Path to JSONL training data file'
            },
            'image_base_dir': {
                'type': 'str',
                'required': False,
                'default': 'data0/Images',
                'description': 'Base directory for resolving image paths'
            }
        },
        '🔢 Batch Settings': {
            'batch_size': {
                'type': 'int',
                'required': False,
                'default': 2,
                'description': 'Number of samples per batch'
            },
            'shuffle': {
                'type': 'bool',
                'required': False,
                'default': True,
                'description': 'Whether to shuffle the data'
            },
            'num_workers': {
                'type': 'int',
                'required': False,
                'default': 0,
                'description': 'Number of worker processes for data loading'
            },
            'pin_memory': {
                'type': 'bool',
                'required': False,
                'default': True,
                'description': 'Whether to pin memory for GPU transfer'
            }
        },
        '📝 Text Processing': {
            'text_max_length': {
                'type': 'int',
                'required': False,
                'default': 128,
                'description': 'Maximum text sequence length (tokens)'
            }
        },
        '🖼️  Image Processing': {
            'image_max_patches': {
                'type': 'int',
                'required': False,
                'default': 256,
                'description': 'Maximum number of image patches to keep'
            }
        },
        '🎯 Model Settings': {
            'task_name': {
                'type': 'str',
                'required': False,
                'default': 'retrieval',
                'description': 'Task name for Jina model (retrieval, text-matching, code)'
            }
        }
    }
    
    for category, options in config_options.items():
        print(f"\n{category}")
        print("-" * 40)
        
        for param, info in options.items():
            required = "REQUIRED" if info['required'] else "optional"
            print(f"  {param}:")
            print(f"    Type: {info['type']}")
            print(f"    Required: {required}")
            print(f"    Default: {info['default']}")
            print(f"    Description: {info['description']}")
            print()


def show_examples():
    """Show practical usage examples."""
    print("=" * 60)
    print("💡 PRACTICAL USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        {
            'name': 'Minimal Usage',
            'description': 'Just provide the data path, use all defaults',
            'config': {
                'jsonl_path': 'data0/train.jsonl'
            }
        },
        {
            'name': 'Custom Batch Size',
            'description': 'Larger batches for faster training',
            'config': {
                'jsonl_path': 'data0/train.jsonl',
                'batch_size': 8
            }
        },
        {
            'name': 'Memory Efficient',
            'description': 'Smaller sequences and patches to save memory',
            'config': {
                'jsonl_path': 'data0/train.jsonl',
                'batch_size': 4,
                'text_max_length': 64,
                'image_max_patches': 128
            }
        },
        {
            'name': 'High Performance',
            'description': 'Multiple workers and larger batches',
            'config': {
                'jsonl_path': 'data0/train.jsonl',
                'batch_size': 16,
                'num_workers': 4,
                'pin_memory': True,
                'shuffle': True
            }
        },
        {
            'name': 'Different Task',
            'description': 'Configure for text-matching instead of retrieval',
            'config': {
                'jsonl_path': 'data0/train.jsonl',
                'task_name': 'text-matching',
                'batch_size': 4
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   {example['description']}")
        print("   " + "-" * 50)
        print("   config = {")
        for key, value in example['config'].items():
            if isinstance(value, str):
                print(f"       '{key}': '{value}',")
            else:
                print(f"       '{key}': {value},")
        print("   }")
        print("   dataloader = get_training_dataloader(config)")


def test_dataloader():
    """Test the dataloader with a small example."""
    print("=" * 60)
    print("🧪 TESTING DATALOADER")
    print("=" * 60)
    
    print("\n🔧 Creating test dataloader...")
    config = {
        'jsonl_path': 'data0/train.jsonl',
        'batch_size': 2,
        'text_max_length': 64,
        'image_max_patches': 128,
        'shuffle': False  # For consistent testing
    }
    
    try:
        dataloader = get_training_dataloader(config)
        print(f"✅ Created dataloader with {len(dataloader.dataset)} samples")
        
        print("\n📦 Testing first batch:")
        batch = next(iter(dataloader))
        
        # Show detailed batch info
        print(f"  Batch size: {batch['input_ids'].shape[0]}")
        print(f"  Text tokens: {batch['input_ids'].shape}")
        print(f"  Image patches: {batch['pixel_values'].shape}")
        print(f"  Task labels: {batch['task_labels']}")
        
        # Show actual data sample
        print(f"\n🔍 Sample data from first item:")
        print(f"  First 10 token IDs: {batch['input_ids'][0][:10].tolist()}")
        print(f"  Attention mask: {batch['attention_mask'][0][:10].tolist()}")
        print(f"  Image grid info: {batch['image_grid_thw'][0].tolist()}")
        print(f"  Pixel value range: [{batch['pixel_values'].min():.3f}, {batch['pixel_values'].max():.3f}]")
        
        print("\n✅ Dataloader test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Multimodal DataLoader Guide",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataloader_guide.py --show-io          # Show input/output format
  python dataloader_guide.py --show-config      # Show all config options  
  python dataloader_guide.py --show-examples    # Show usage examples
  python dataloader_guide.py --test             # Test the dataloader
  python dataloader_guide.py --all              # Show everything
        """
    )
    
    parser.add_argument('--show-io', action='store_true', 
                       help='Show input/output format')
    parser.add_argument('--show-config', action='store_true',
                       help='Show all configuration options')
    parser.add_argument('--show-examples', action='store_true',
                       help='Show practical usage examples')
    parser.add_argument('--test', action='store_true',
                       help='Test the dataloader with real data')
    parser.add_argument('--all', action='store_true',
                       help='Show everything')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    if args.all or args.show_io:
        show_input_output()
    
    if args.all or args.show_config:
        show_config_options()
    
    if args.all or args.show_examples:
        show_examples()
    
    if args.all or args.test:
        test_dataloader()


if __name__ == "__main__":
    main()