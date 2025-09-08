#!/usr/bin/env python3
"""
Script to split a JSONL file into train and eval sets with 9:1 ratio.
Usage: python split_jsonl.py <input_file.jsonl> [output_dir]
"""

import json
import random
import argparse
import os
from pathlib import Path


def split_jsonl(input_file, output_dir=None, train_ratio=0.9, seed=42):
    """
    Split a JSONL file into train and eval sets.
    
    Args:
        input_file (str): Path to input JSONL file
        output_dir (str): Output directory (default: same as input file)
        train_ratio (float): Ratio for training set (default: 0.9)
        seed (int): Random seed for reproducibility (default: 42)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read all records from input file
    records = []
    print(f"Reading records from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
    
    print(f"Total records read: {len(records)}")
    
    # Shuffle records randomly
    random.shuffle(records)
    
    # Calculate split point
    split_point = int(len(records) * train_ratio)
    train_records = records[:split_point]
    eval_records = records[split_point:]
    
    print(f"Train records: {len(train_records)}")
    print(f"Eval records: {len(eval_records)}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Write train file
    train_file = os.path.join(output_dir, 'train.jsonl')
    print(f"Writing train set to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for record in train_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # Write eval file
    eval_file = os.path.join(output_dir, 'eval.jsonl')
    print(f"Writing eval set to {eval_file}...")
    with open(eval_file, 'w', encoding='utf-8') as f:
        for record in eval_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print("Split completed successfully!")
    print(f"Train set: {train_file} ({len(train_records)} records)")
    print(f"Eval set: {eval_file} ({len(eval_records)} records)")


def main():
    parser = argparse.ArgumentParser(description='Split JSONL file into train and eval sets')
    parser.add_argument('input_file', help='Input JSONL file path')
    parser.add_argument('--output_dir', '-o', help='Output directory (default: same as input file)')
    parser.add_argument('--train_ratio', '-r', type=float, default=0.9, 
                        help='Training set ratio (default: 0.9)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return 1
    
    try:
        split_jsonl(args.input_file, args.output_dir, args.train_ratio, args.seed)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
