#!/bin/tcsh

# Get script directory
set SCRIPT_DIR = `dirname "$0"`
set PROJECT_ROOT = `dirname "$SCRIPT_DIR"`  # Get the parent directory of tools
cd "$PROJECT_ROOT"

# wandb configuration
setenv WANDB_ENTITY "smart-search-fyp"
setenv WANDB_PROJECT "jina-embeddings-test"

echo "Running test_wandb.py..."
python tools/test_wandb.py
