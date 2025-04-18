#!/usr/bin/env python3
"""
Prepare FDNeRF-compatible dataset directory.

This script creates a symlinked dataset folder for training FDNeRF using
preprocessed identities like `E057_all` or `E061_all` from a larger prepared directory.
"""

import os
import argparse
import shutil
from pathlib import Path

def prepare_fdnerf_dataset(prepared_dir, output_dir, identity):
    """
    Creates a new directory for FDNeRF training that links or copies required files from the prepared dataset.

    Args:
        prepared_dir (str): Path to the folder where preprocessed identities live (e.g., fdnerf_prepared).
        output_dir (str): Path to output dataset folder for FDNeRF training.
        identity (str): Identity folder name to prepare (e.g., 'E057_all').
    """
    src = Path(prepared_dir) / identity
    dst = Path(output_dir)

    if not src.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src}")

    print(f"Creating FDNeRF dataset at: {dst}")

    # Recreate the folder structure
    dst.mkdir(parents=True, exist_ok=True)

    # Symlink or copy meta.json
    meta_src = src / "meta.json"
    meta_dst = dst / "meta.json"
    if not meta_dst.exists():
        os.symlink(meta_src.resolve(), meta_dst)

    # Symlink or copy all frames
    for file in src.iterdir():
        if file.name == "meta.json":
            continue
        dst_file = dst / file.name
        if not dst_file.exists():
            os.symlink(file.resolve(), dst_file)

    print(f"✅ Dataset ready: {identity} → {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FDNeRF dataset from preprocessed identity folder.")
    parser.add_argument("--prepared_dir", type=str, required=True,
                        help="Path to preprocessed folder (e.g., fdnerf_prepared)")
    parser.add_argument("--identity", type=str, required=True,
                        help="Merged identity folder name (e.g., E057_all or E061_all)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output path for FDNeRF training dataset (e.g., datasets/E057_fdnerf)")
    args = parser.parse_args()

    prepare_fdnerf_dataset(args.prepared_dir, args.output_dir, args.identity)
