#!/usr/bin/env python3
"""
run_pipeline.py: Script to run the complete pipeline from feature extraction to 
pose prediction.
"""

import os
import subprocess
import argparse
import time
from pathlib import Path

def run_command(command, desc=None):
    """Run a shell command and print output."""
    if desc:
        print(f"\n=== {desc} ===")
    
    print(f"Running: {' '.join(command)}\n")
    start_time = time.time()
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    end_time = time.time()
    
    if process.returncode != 0:
        print(f"\nCommand failed with return code {process.returncode}")
        exit(process.returncode)
    
    print(f"\nCommand completed in {end_time - start_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Run the complete 3D reconstruction pipeline')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Base directory for data')
    parser.add_argument('--colmap_path', type=str, default='colmap',
                        help='Path to COLMAP executable')
    parser.add_argument('--skip_make_pairs', action='store_true',
                        help='Skip the make_pairs step')
    parser.add_argument('--skip_colmap', action='store_true',
                        help='Skip the COLMAP step')
    parser.add_argument('--skip_mtcm', action='store_true',
                        help='Skip the MTCM step')
    parser.add_argument('--skip_pose_prediction', action='store_true',
                        help='Skip the pose prediction step')
    
    args = parser.parse_args()
    
    # Define paths
    data_dir = Path(args.data_dir)
    raw_images_dir = data_dir / "raw_images"
    masks_dir = data_dir / "masks"
    metadata_dir = data_dir / "metadata"
    colmap_dir = data_dir / "colmap"
    
    # Create directories if they don't exist
    os.makedirs(raw_images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(colmap_dir, exist_ok=True)
    
    # Define paths for intermediate files
    exif_path = metadata_dir / "exif.json"
    embeddings_dir = data_dir / "embeddings"
    pair_list_path = metadata_dir / "pair_list.txt"
    database_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    poses_json_path = metadata_dir / "poses.json"
    mtcm_feats_dir = metadata_dir / "mtcm_feats"
    pose_prediction_dir = metadata_dir / "pose_prediction"
    
    # Step 1: Make pairs
    if not args.skip_make_pairs:
        run_command(
            ["python", "make_pairs.py",
             "--raw_images", str(raw_images_dir),
             "--embeddings", str(embeddings_dir),
             "--masks", str(masks_dir),
             "--exif", str(exif_path),
             "--output", str(pair_list_path)],
            desc="Step 1: Make pairs for COLMAP guided matching"
        )
    
    # Step 2: Run COLMAP
    if not args.skip_colmap:
        run_command(
            ["python", "run_colmap.py",
             "--colmap_path", args.colmap_path,
             "--database_path", str(database_path),
             "--image_dir", str(raw_images_dir),
             "--pair_list_path", str(pair_list_path),
             "--sparse_dir", str(sparse_dir),
             "--output_json", str(poses_json_path)],
            desc="Step 2: Run COLMAP feature matching and sparse reconstruction"
        )
    
    # Step 3: Train MTCM Transformer (optional)
    if not args.skip_mtcm:
        run_command(
            ["python", "mtcm_transformer.py",
             "--embeddings_dir", str(embeddings_dir),
             "--poses_json", str(poses_json_path),
             "--output_dir", str(mtcm_feats_dir),
             "--num_epochs", "50"],
            desc="Step 3: Train MTCM Transformer and generate refined features"
        )
    
    # Step 4: Train pose prediction MLP
    if not args.skip_pose_prediction:
        # Use refined features if available, otherwise use original embeddings
        features_dir = str(mtcm_feats_dir / "refined_features") if os.path.exists(mtcm_feats_dir / "refined_features") else str(embeddings_dir)
        
        run_command(
            ["python", "pose_prediction.py",
             "--features_dir", features_dir,
             "--poses_json", str(poses_json_path),
             "--output_dir", str(pose_prediction_dir),
             "--num_epochs", "100"],
            desc="Step 4: Train pose prediction MLP"
        )
    
    print("\n=== Pipeline completed successfully! ===")
    print(f"Predicted poses saved to: {pose_prediction_dir}/predicted_poses.json")

if __name__ == "__main__":
    main()