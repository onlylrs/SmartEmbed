#!/usr/bin/env python3
"""
CLI script for multimodal tokenization demo.

This script provides a command-line interface to the tokenization demo pipeline,
with configurable parameters and optional config file loading.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path so we can import our pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.tokenize_demo import run


def load_config(config_path: str) -> dict:
    """Load configuration from JSON or YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    elif config_path.suffix.lower() in ['.yaml', '.yml']:
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML config files. Install with: pip install PyYAML")
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Multimodal tokenization demo using Jina Embeddings v4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core parameters
    parser.add_argument(
        '--sample-size', 
        type=int, 
        default=50,
        help='Number of examples to randomly sample from dataset'
    )
    
    parser.add_argument(
        '--text-max-length',
        type=int,
        default=128, 
        help='Maximum text sequence length for tokenization'
    )
    
    parser.add_argument(
        '--image-max-length',
        type=int,
        default=256,
        help='Maximum number of image patches to keep'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size for processing'
    )
    
    parser.add_argument(
        '--save-tensors',
        action='store_true',
        default=True,
        help='Save tokenized tensors and metadata to disk'
    )
    
    parser.add_argument(
        '--no-save-tensors',
        dest='save_tensors',
        action='store_false',
        help='Do not save tensors to disk'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Data and model paths
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='data0/train.jsonl',
        help='Path to the dataset JSONL file'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='/homes/rliuar/Desktop/FYP/jina-embedding-v4',
        help='Name or path of the Jina model to use'
    )
    
    # Config file option
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON or YAML config file to load defaults from'
    )
    
    return parser


def merge_config_and_args(args: argparse.Namespace, config: dict = None) -> dict:
    """
    Merge configuration from file with command line arguments.
    Command line arguments take precedence over config file values.
    """
    # Start with defaults from config file (if provided)
    merged_config = config.copy() if config else {}
    
    # Override with command line arguments (only non-None values)
    args_dict = vars(args)
    for key, value in args_dict.items():
        if key != 'config' and value is not None:
            # Convert kebab-case to snake_case for function parameters
            config_key = key.replace('-', '_')
            # Handle specific parameter name mappings
            if config_key == 'text_max_length':
                config_key = 'text_max_len'
            elif config_key == 'image_max_length':
                config_key = 'image_max_len'
            merged_config[config_key] = value
    
    return merged_config


def main():
    """Main entry point for the CLI script."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Load config file if provided
    config = {}
    if args.config:
        try:
            config = load_config(args.config)
            print(f"Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"Error loading config file: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Merge config and command line arguments
    final_config = merge_config_and_args(args, config)
    
    # Print final configuration
    print("Running with configuration:")
    for key, value in sorted(final_config.items()):
        print(f"  {key}: {value}")
    print()
    
    try:
        # Run the tokenization demo
        result = run(**final_config)
        
        print(f"\n✅ Demo completed successfully!")
        print(f"Processed batch shapes:")
        for key, tensor in result.items():
            print(f"  {key}: {tensor.shape}")
            
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()