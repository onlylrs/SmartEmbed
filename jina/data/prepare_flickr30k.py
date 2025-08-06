#!/usr/bin/env python3
"""
Convert Flickr30K captions data to Jina training format
"""

import json
import csv
import random
import os
from typing import List, Dict, Tuple
from pathlib import Path


def load_captions(captions_file: str) -> Dict[str, List[str]]:
    """Load captions and group by image"""
    image_captions = {}
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 2:
                image_name, caption = row[0].strip(), row[1].strip()
                if image_name not in image_captions:
                    image_captions[image_name] = []
                image_captions[image_name].append(caption)
    
    return image_captions


def create_training_pairs(image_captions: Dict[str, List[str]], 
                         images_dir: str,
                         num_negatives: int = 3) -> List[Dict]:
    """Create training pairs for contrastive learning"""
    training_examples = []
    
    # Get list of all images and captions for negative sampling
    all_images = list(image_captions.keys())
    all_captions = []
    for captions in image_captions.values():
        all_captions.extend(captions)
    
    for image_name, captions in image_captions.items():
        image_path = os.path.join(images_dir, image_name)
        
        # For each image, create pairs with its captions as positives
        for i, query_caption in enumerate(captions):
            # Use other captions of the same image as positives
            positive_captions = [cap for j, cap in enumerate(captions) if j != i]
            
            if positive_captions:
                positive_caption = random.choice(positive_captions)
            else:
                # If only one caption, use it as positive (self-pairing)
                positive_caption = query_caption
            
            # Sample negative captions from other images
            negative_captions = []
            while len(negative_captions) < num_negatives:
                neg_image = random.choice(all_images)
                if neg_image != image_name:  # Ensure it's from a different image
                    neg_caption = random.choice(image_captions[neg_image])
                    if neg_caption not in negative_captions:
                        negative_captions.append(neg_caption)
            
            # Create text-to-text examples (retrieval task)
            for neg_caption in negative_captions:
                training_examples.append({
                    "task": "retrieval",
                    "query": query_caption,
                    "positive": positive_caption,
                    "negative": neg_caption
                })
            
            # Create image-to-text examples (multimodal retrieval)
            if os.path.exists(image_path):
                for neg_caption in negative_captions:
                    training_examples.append({
                        "task": "retrieval", 
                        "query": "Describe this image",
                        "positive": query_caption,
                        "negative": neg_caption,
                        "query_image": image_path
                    })
    
    return training_examples


def save_jsonl(examples: List[Dict], output_file: str):
    """Save examples in JSONL format"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')


def main():
    # Configuration
    captions_file = "/project/fyp25_hc2/data/captions.txt"
    images_dir = "/project/fyp25_hc2/data/Images"  # Assume images are here
    output_dir = "/project/fyp25_hc2/data"
    
    # Create output directory if not exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Loading captions...")
    image_captions = load_captions(captions_file)
    print(f"Loaded {len(image_captions)} images with captions")
    
    print("Creating training pairs...")
    training_examples = create_training_pairs(image_captions, images_dir)
    print(f"Created {len(training_examples)} training examples")
    
    # Split into train/eval (90/10)
    random.shuffle(training_examples)
    split_idx = int(0.9 * len(training_examples))
    
    train_examples = training_examples[:split_idx]
    eval_examples = training_examples[split_idx:]
    
    # Save training and evaluation data
    train_file = os.path.join(output_dir, "train.jsonl")
    eval_file = os.path.join(output_dir, "eval.jsonl")
    
    print(f"Saving {len(train_examples)} training examples to {train_file}")
    save_jsonl(train_examples, train_file)
    
    print(f"Saving {len(eval_examples)} evaluation examples to {eval_file}")
    save_jsonl(eval_examples, eval_file)
    
    print("Data preparation completed!")
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Total images: {len(image_captions)}")
    print(f"Total training examples: {len(train_examples)}")
    print(f"Total evaluation examples: {len(eval_examples)}")
    
    # Print sample examples
    print(f"\nSample training examples:")
    for i, example in enumerate(train_examples[:3]):
        print(f"Example {i+1}:")
        print(f"  Task: {example['task']}")
        print(f"  Query: {example['query'][:100]}...")
        print(f"  Positive: {example['positive'][:100]}...")
        print(f"  Negative: {example['negative'][:100]}...")
        if 'query_image' in example:
            print(f"  Query Image: {example['query_image']}")
        print()


if __name__ == "__main__":
    main()
