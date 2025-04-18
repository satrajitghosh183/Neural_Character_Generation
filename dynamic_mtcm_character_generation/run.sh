#!/bin/bash

# Clear CUDA cache before running
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU if you have multiple
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # Helps with memory fragmentation

# Set environment variables to help with CUDA memory management
export CUDA_LAUNCH_BLOCKING=1  # Better error reporting for CUDA errors

# Run with reduced batch size and more conservative memory usage
python ./scripts/train_consistency.py