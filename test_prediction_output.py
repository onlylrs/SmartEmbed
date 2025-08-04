#!/usr/bin/env python3

from transformers.trainer_utils import PredictionOutput
import inspect

print("=== Testing PredictionOutput ===")

# Check the class signature
print("Checking PredictionOutput signature...")
try:
    sig = inspect.signature(PredictionOutput.__init__)
    print(f"Signature: {sig}")
    for param_name, param in sig.parameters.items():
        if param_name != 'self':
            print(f"  {param_name}: {param}")
except Exception as e:
    print(f"Failed to get signature: {e}")

print("\n=== Testing parameter combinations ===")

# Test 1: Basic parameters
print("Test 1: Basic parameters (predictions, label_ids, metrics)")
try:
    output1 = PredictionOutput(predictions=None, label_ids=None, metrics={})
    print("✅ SUCCESS: Basic parameters work")
    print(f"   Has num_samples attr: {hasattr(output1, 'num_samples')}")
    if hasattr(output1, 'num_samples'):
        print(f"   num_samples value: {output1.num_samples}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 2: With num_samples
print("\nTest 2: With num_samples parameter")
try:
    output2 = PredictionOutput(predictions=None, label_ids=None, metrics={}, num_samples=10)
    print("✅ SUCCESS: num_samples parameter works")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 3: Check what fields the object actually has
print("\nTest 3: Checking actual fields of PredictionOutput")
try:
    output3 = PredictionOutput(predictions=None, label_ids=None, metrics={})
    print("Fields in PredictionOutput object:")
    for attr in dir(output3):
        if not attr.startswith('_'):
            print(f"  {attr}: {getattr(output3, attr, 'N/A')}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n=== Checking transformers version ===")
import transformers
print(f"Transformers version: {transformers.__version__}")
