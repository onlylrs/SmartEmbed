#!/usr/bin/env python3
"""
OpenI Dataset Validation Script

This script validates the OpenI dataset by:
1. Checking if the images root directory exists
2. Analyzing the structure and statistics of the JSONL metadata file  
3. Checking if image files referenced in metadata actually exist on disk

The script uses a hard-coded images root directory (/scratch/medimgfmod/Generalist/medical)
and constructs full paths by combining this root with the 'image' field from each JSONL record.

Usage:
    python validate_openi_dataset.py [--verbose]
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Any
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Dataset paths
IMAGES_ROOT = "/scratch/medimgfmod/Generalist/medical"
METADATA_JSONL = "/project/fyp25_hc2/new_data/openi_data_generation_parsed.jsonl"

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def check_images_root_exists(images_root: str) -> bool:
    """Check if the images root directory exists."""
    if not os.path.exists(images_root):
        logging.error(f"Images root directory not found: {images_root}")
        return False
    
    if not os.path.isdir(images_root):
        logging.error(f"Images root path is not a directory: {images_root}")
        return False
    
    logging.info(f"Images root directory exists: {images_root}")
    return True

def load_and_analyze_metadata(jsonl_path: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load JSONL metadata and analyze its structure.
    
    Returns:
        tuple: (list of records, analysis dictionary)
    """
    if not os.path.exists(jsonl_path):
        logging.error(f"Metadata file not found: {jsonl_path}")
        return [], {}
    
    records = []
    all_keys = Counter()
    key_examples = defaultdict(set)
    key_types = defaultdict(Counter)
    empty_counts = defaultdict(int)
    text_lengths = []
    
    logging.info(f"Loading and analyzing metadata from {jsonl_path}")
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    records.append(record)
                    
                    # Analyze keys
                    for key, value in record.items():
                        all_keys[key] += 1
                        
                        # Store examples (limit to avoid memory issues)
                        if len(key_examples[key]) < 5:
                            if isinstance(value, str) and len(value) < 100:
                                key_examples[key].add(value)
                            elif not isinstance(value, str):
                                key_examples[key].add(str(value)[:100])
                        
                        # Track types
                        key_types[key][type(value).__name__] += 1
                        
                        # Check for empty values
                        if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
                            empty_counts[key] += 1
                        
                        # Track text lengths if it's a text field
                        if key in ['text', 'caption', 'title'] and isinstance(value, str):
                            text_lengths.append(len(value))
                
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse JSON at line {line_num}: {e}")
                    continue
                
                # Progress logging for large files
                if line_num % 10000 == 0:
                    logging.debug(f"Processed {line_num} lines...")
    
    except Exception as e:
        logging.error(f"Error reading metadata file: {e}")
        return [], {}
    
    # Compile analysis
    analysis = {
        'total_records': len(records),
        'all_keys': dict(all_keys),
        'key_examples': {k: list(v) for k, v in key_examples.items()},
        'key_types': {k: dict(v) for k, v in key_types.items()},
        'empty_counts': dict(empty_counts),
        'text_length_stats': {
            'count': len(text_lengths),
            'min': min(text_lengths) if text_lengths else 0,
            'max': max(text_lengths) if text_lengths else 0,
            'avg': sum(text_lengths) / len(text_lengths) if text_lengths else 0
        } if text_lengths else None
    }
    
    logging.info(f"Loaded {len(records)} records from metadata file")
    return records, analysis

def check_image_metadata_matching(records: List[Dict], images_root: str) -> Dict[str, Any]:
    """
    Check how many images referenced in metadata actually exist using images_root + image path.
    
    Args:
        records: List of metadata records
        images_root: Root directory path for images
    
    Returns:
        Dict with matching statistics
    """
    logging.info(f"Checking image existence using root: {images_root}")
    
    missing_image_key_count = 0
    invalid_image_values = []
    existing_files = []
    missing_files = []
    
    for i, record in enumerate(records):
        # Look for 'image' key specifically
        if 'image' not in record:
            missing_image_key_count += 1
            logging.debug(f"Record {i} missing 'image' key. Available keys: {list(record.keys())}")
            continue
        
        image_path = record['image']
        
        # Validate image path
        if not isinstance(image_path, str) or not image_path.strip():
            invalid_image_values.append({
                'record_index': i,
                'value': image_path,
                'type': type(image_path).__name__
            })
            logging.debug(f"Record {i} has invalid 'image' value: {image_path} (type: {type(image_path)})")
            continue
        
        # Construct full path
        image_path = image_path.strip()
        # Remove leading slash if present to avoid double slashes
        if image_path.startswith('/'):
            image_path = image_path[1:]
        
        full_path = os.path.join(images_root, image_path)
        
        # Check if file exists
        if os.path.exists(full_path) and os.path.isfile(full_path):
            existing_files.append({
                'record_index': i,
                'image_path': image_path,
                'full_path': full_path
            })
            logging.debug(f"Record {i}: Found image at {full_path}")
        else:
            missing_files.append({
                'record_index': i,
                'image_path': image_path,
                'full_path': full_path
            })
            logging.debug(f"Record {i}: Missing image at {full_path}")
    
    # Calculate statistics
    total_valid_refs = len(existing_files) + len(missing_files)
    existence_percentage = (len(existing_files) / total_valid_refs * 100) if total_valid_refs > 0 else 0
    
    matching_stats = {
        'total_records': len(records),
        'missing_image_key_count': missing_image_key_count,
        'invalid_image_values_count': len(invalid_image_values),
        'invalid_image_values_sample': invalid_image_values[:5],
        'total_valid_image_refs': total_valid_refs,
        'existing_files_count': len(existing_files),
        'missing_files_count': len(missing_files),
        'existence_percentage': existence_percentage,
        'existing_files_sample': [f['image_path'] for f in existing_files[:5]],
        'missing_files_sample': [f['image_path'] for f in missing_files[:10]],
        'missing_full_paths_sample': [f['full_path'] for f in missing_files[:5]]
    }
    
    return matching_stats

def print_analysis_report(analysis: Dict[str, Any], matching_stats: Dict[str, Any]):
    """Print a comprehensive analysis report."""
    print("\n" + "="*80)
    print("OPENI DATASET VALIDATION REPORT")
    print("="*80)
    
    print(f"\n📊 DATASET OVERVIEW")
    print(f"Images root directory: {IMAGES_ROOT}")
    print(f"Metadata file: {METADATA_JSONL}")
    print(f"Total metadata records: {analysis['total_records']:,}")
    
    print(f"\n🔑 METADATA STRUCTURE ANALYSIS")
    print(f"Total unique keys found: {len(analysis['all_keys'])}")
    print("\nKey frequency and types:")
    for key, count in sorted(analysis['all_keys'].items()):
        types_info = analysis['key_types'].get(key, {})
        empty_count = analysis['empty_counts'].get(key, 0)
        empty_pct = (empty_count / analysis['total_records']) * 100
        
        print(f"  {key:20} | Count: {count:8,} | Types: {types_info} | Empty: {empty_count:6,} ({empty_pct:.1f}%)")
    
    print(f"\n📝 KEY EXAMPLES (first few values for each key):")
    for key, examples in analysis['key_examples'].items():
        print(f"  {key:20} | Examples: {examples}")
    
    if analysis['text_length_stats']:
        stats = analysis['text_length_stats']
        print(f"\n📏 TEXT LENGTH STATISTICS")
        print(f"Text fields analyzed: {stats['count']:,} values")
        print(f"Length range: {stats['min']} - {stats['max']} characters")
        print(f"Average length: {stats['avg']:.1f} characters")
    
    print(f"\n🔍 IMAGE FILE EXISTENCE ANALYSIS")
    print(f"Images root directory: {IMAGES_ROOT}")
    print(f"Total metadata records: {matching_stats['total_records']:,}")
    print(f"Records missing 'image' key: {matching_stats['missing_image_key_count']:,}")
    print(f"Records with invalid 'image' values: {matching_stats['invalid_image_values_count']:,}")
    print(f"Valid image references to check: {matching_stats['total_valid_image_refs']:,}")
    
    if matching_stats['invalid_image_values_sample']:
        print(f"\n⚠️  SAMPLE INVALID 'image' VALUES:")
        for invalid in matching_stats['invalid_image_values_sample']:
            print(f"  Record {invalid['record_index']}: {invalid['value']} (type: {invalid['type']})")
    
    print(f"\n📊 FILE EXISTENCE RESULTS")
    print(f"Existing files (found): {matching_stats['existing_files_count']:,}")
    print(f"Missing files (not found): {matching_stats['missing_files_count']:,}")
    print(f"Existence percentage: {matching_stats['existence_percentage']:.1f}% (found / valid references)")
    
    # Show samples of existing files
    if matching_stats['existing_files_sample']:
        print(f"\n✅ SAMPLE EXISTING FILES:")
        for file_path in matching_stats['existing_files_sample']:
            print(f"  - {file_path}")
    
    # Show samples of missing files
    if matching_stats['missing_files_sample']:
        print(f"\n❌ SAMPLE MISSING FILES (image paths from metadata):")
        for file_path in matching_stats['missing_files_sample']:
            print(f"  - {file_path}")
    
    if matching_stats['missing_full_paths_sample']:
        print(f"\n🔍 SAMPLE MISSING FULL PATHS (for debugging):")
        for full_path in matching_stats['missing_full_paths_sample']:
            print(f"  - {full_path}")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Validate OpenI dataset structure and image file existence")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    print("Starting OpenI dataset validation...")
    
    # Step 1: Check images root directory exists
    logging.info("Step 1: Checking images root directory...")
    if not check_images_root_exists(IMAGES_ROOT):
        logging.error("Images root directory check failed. Exiting.")
        return
    
    # Step 2: Load and analyze metadata
    logging.info("Step 2: Loading and analyzing metadata...")
    records, analysis = load_and_analyze_metadata(METADATA_JSONL)
    
    if not records:
        logging.error("No records loaded from metadata file. Exiting.")
        return
    
    # Step 3: Check image file existence
    logging.info("Step 3: Checking image file existence...")
    matching_stats = check_image_metadata_matching(records, IMAGES_ROOT)
    
    # Step 4: Print comprehensive report
    print_analysis_report(analysis, matching_stats)
    
    print("\n✅ Dataset validation completed!")

if __name__ == "__main__":
    main()
