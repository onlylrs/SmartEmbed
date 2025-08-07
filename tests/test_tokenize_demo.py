"""
Unit tests for the tokenization demo pipeline.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import numpy as np

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.tokenize_demo import (
    get_run_folder,
    load_jsonl_dataset,
    sample_dataset,
    resolve_image_paths,
    prepare_decoder_inputs,
    save_artifacts,
)


class TestTokenizeDemo(unittest.TestCase):
    """Test cases for tokenization demo functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_run_folder(self):
        """Test run folder creation."""
        run_folder = get_run_folder(self.temp_dir)
        
        # Check that folder exists
        self.assertTrue(os.path.exists(run_folder))
        
        # Check that folder name contains timestamp
        folder_name = os.path.basename(run_folder)
        self.assertTrue(folder_name.startswith("tokens_"))
        self.assertTrue(len(folder_name) > 10)  # tokens_ + timestamp
    
    def test_load_jsonl_dataset(self):
        """Test JSONL dataset loading."""
        # Create test JSONL file
        test_data = [
            {"positive": "A dog in the park", "query_image": "/path/to/img1.jpg"},
            {"positive": "A cat on a tree", "query_image": "/path/to/img2.jpg"}
        ]
        
        test_file = os.path.join(self.temp_dir, "test.jsonl")
        with open(test_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        # Test loading
        loaded_data = load_jsonl_dataset(test_file)
        
        self.assertEqual(len(loaded_data), 2)
        self.assertEqual(loaded_data[0]["positive"], "A dog in the park")
        self.assertEqual(loaded_data[1]["positive"], "A cat on a tree")
    
    def test_sample_dataset(self):
        """Test deterministic dataset sampling."""
        data = [{"id": i} for i in range(100)]
        
        # Test sampling
        sampled1 = sample_dataset(data, 10, seed=42)
        sampled2 = sample_dataset(data, 10, seed=42)
        sampled3 = sample_dataset(data, 10, seed=123)
        
        # Same seed should give same results
        self.assertEqual(sampled1, sampled2)
        
        # Different seed should give different results
        self.assertNotEqual(sampled1, sampled3)
        
        # Check sample size
        self.assertEqual(len(sampled1), 10)
    
    def test_resolve_image_paths(self):
        """Test image path resolution and text extraction."""
        data = [
            {"positive": "A dog in the park", "query_image": "/abs/path/to/img1.jpg"},
            {"positive": "A cat on a tree", "image": "relative/img2.jpg"},
            {"text": "Alternative text field", "query_image": "/abs/path/to/img3.jpg"}
        ]
        
        resolved = resolve_image_paths(data, image_base_dir="data/Images")
        
        self.assertEqual(len(resolved), 3)
        
        # Test absolute path handling
        self.assertEqual(resolved[0]['image_path'], "/abs/path/to/img1.jpg")
        self.assertEqual(resolved[0]['text'], "A dog in the park")
        
        # Test relative path handling
        self.assertEqual(resolved[1]['image_path'], "data/Images/relative/img2.jpg")
        self.assertEqual(resolved[1]['text'], "A cat on a tree")
        
        # Test alternative text field
        self.assertEqual(resolved[2]['text'], "Alternative text field")
    
    def test_prepare_decoder_inputs(self):
        """Test decoder input preparation."""
        # Create mock batch data
        batch = {
            'input_ids': torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=torch.long),
            'attention_mask': torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.long),
            'pixel_values': torch.randn(2, 256, 768),
            'image_grid_thw': torch.tensor([[1, 14, 14], [1, 14, 14]], dtype=torch.long)
        }
        
        decoder_inputs = prepare_decoder_inputs(batch)
        
        # Check required keys exist
        required_keys = ['input_ids', 'pixel_values', 'attention_mask']
        for key in required_keys:
            self.assertIn(key, decoder_inputs)
        
        # Check optional key
        self.assertIn('image_grid_thw', decoder_inputs)
        
        # Check shapes are preserved
        self.assertEqual(decoder_inputs['input_ids'].shape, batch['input_ids'].shape)
        self.assertEqual(decoder_inputs['pixel_values'].shape, batch['pixel_values'].shape)
    
    def test_save_artifacts(self):
        """Test artifact saving functionality."""
        # Create mock batch data
        batch = {
            'input_ids': torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=torch.long),
            'pixel_values': torch.randn(2, 256, 768),
            'attention_mask': torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.long)
        }
        
        config = {
            'sample_size': 2,
            'text_max_len': 128,
            'image_max_len': 256,
            'seed': 42
        }
        
        # Test saving
        save_artifacts(batch, config, self.temp_dir)
        
        # Check files exist
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "text_input_ids.npy")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "img_tokens.npy")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "manifest.json")))
        
        # Check file contents
        loaded_input_ids = np.load(os.path.join(self.temp_dir, "text_input_ids.npy"))
        loaded_img_tokens = np.load(os.path.join(self.temp_dir, "img_tokens.npy"))
        
        np.testing.assert_array_equal(loaded_input_ids, batch['input_ids'].numpy())
        np.testing.assert_array_equal(loaded_img_tokens, batch['pixel_values'].numpy())
        
        # Check manifest
        with open(os.path.join(self.temp_dir, "manifest.json"), 'r') as f:
            manifest = json.load(f)
        
        self.assertIn("shapes", manifest)
        self.assertIn("config", manifest)
        self.assertEqual(manifest['config']['sample_size'], 2)


if __name__ == '__main__':
    unittest.main()