#!/usr/bin/env python3
# Multiface Dataset Preprocessing for FDNeRF
# This script processes the Multiface dataset to create a format compatible with FDNeRF training

import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
import shutil
from pathlib import Path
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("multiface_preprocessing.log")
    ]
)
logger = logging.getLogger("multiface-prep")

# Constants
PERSON_CLASS_ID = 15  # COCO class ID for "person"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Multiface dataset for FDNeRF training")
    parser.add_argument("--datadir", type=str, required=True, help="Path to the Multiface dataset")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for preprocessed data")
    parser.add_argument("--identities", type=str, default=None, help="Comma-separated list of identities to process (e.g. 'E057,E058')")
    parser.add_argument("--expressions", type=str, default=None, help="Comma-separated list of expressions to process (e.g. 'Cheeks_Puffed,Lips_Puffed')")
    parser.add_argument("--merge_expressions", action="store_true", help="Merge multiple expressions per identity")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process per sequence")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with minimal processing")
    return parser.parse_args()

def load_segmentation_model():
    """Load DeepLabV3 model with ResNet-101 backbone pre-trained on COCO."""
    logger.info(f"Loading DeepLabV3 segmentation model (device: {DEVICE})...")
    model = deeplabv3_resnet101(pretrained=True)
    model.eval()
    model.to(DEVICE)
    return model

def get_transform():
    """Get preprocessing transforms for DeepLabV3 model."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def extract_segmentation_mask(model, image_path, transform):
    """
    Extract person segmentation mask from an image using DeepLabV3.
    
    Args:
        model: DeepLabV3 model
        image_path: Path to input image
        transform: Preprocessing transforms
        
    Returns:
        Binary mask for the person class
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return None
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    
    # Get the person mask (class 15 in COCO)
    mask = output.argmax(0).cpu().numpy()
    person_mask = (mask == PERSON_CLASS_ID).astype(np.uint8) * 255
    
    return person_mask

def parse_transform_file(transform_file):
    """
    Parse a transform.txt file into a 4x4 camera pose matrix.
    
    Args:
        transform_file: Path to transform file
        
    Returns:
        4x4 numpy array representing the camera pose
    """
    try:
        with open(transform_file, 'r') as f:
            lines = f.readlines()
        
        # Parse matrix rows
        matrix = []
        for line in lines:
            row = [float(x) for x in line.strip().split()]
            if row:  # Skip empty lines
                matrix.append(row)
        
        # Convert to numpy array
        matrix = np.array(matrix, dtype=np.float32)
        
        # Ensure it's a 4x4 matrix
        if matrix.shape != (4, 4):
            logger.warning(f"Transform matrix in {transform_file} has shape {matrix.shape}, expected (4, 4)")
            # If it's 3x4, add a homogeneous row
            if matrix.shape == (3, 4):
                matrix = np.vstack([matrix, [0, 0, 0, 1]])
        
        return matrix
    
    except Exception as e:
        logger.error(f"Error parsing transform file {transform_file}: {e}")
        return None

def map_transform_to_image(transform_file, image_dir):
    """
    Map transform file to corresponding image based on frame ID.
    
    Args:
        transform_file: Path to transform file (e.g., "021897_transform.txt")
        image_dir: Directory containing image frames (subdirectories)
        
    Returns:
        Path to corresponding image file and sequence ID
    """
    # Extract frame ID from transform file name
    transform_frame_id = os.path.basename(transform_file).split('_')[0]
    transform_frame_num = int(transform_frame_id)
    
    # List sequence subdirectories in image_dir (these are sequence IDs)
    sequence_dirs = [d for d in os.listdir(image_dir) 
                    if os.path.isdir(os.path.join(image_dir, d))]
    
    # This is now a mapping problem. In the Multiface dataset, the transform frame IDs
    # don't directly match the image sequence directories, but there's likely a pattern.
    # We'll try a few strategies:
    
    # Strategy 1: Simple indexing - assume frames are in order
    # Sort sequence directories numerically
    sequence_dirs.sort(key=lambda x: int(x))
    
    # If we have the same number of transform frames as image sequences,
    # we can map by position (first transform to first sequence, etc.)
    # For this example, we'll just take the first image in each sequence
    if 0 <= transform_frame_num % len(sequence_dirs) < len(sequence_dirs):
        seq_id = sequence_dirs[transform_frame_num % len(sequence_dirs)]
        seq_path = os.path.join(image_dir, seq_id)
        
        # Get first image in the sequence
        img_files = sorted([f for f in os.listdir(seq_path) 
                           if f.endswith(('.jpg', '.png'))])
        if img_files:
            return os.path.join(seq_path, img_files[0]), seq_id
    
    # Strategy 2: Try all sequences and take a frame with similar index
    for seq_dir in sequence_dirs:
        seq_path = os.path.join(image_dir, seq_dir)
        img_files = sorted([f for f in os.listdir(seq_path) 
                           if f.endswith(('.jpg', '.png'))])
        
        # If we have images in this sequence
        if img_files:
            # Try to find a matching frame by index
            frame_index = transform_frame_num % len(img_files)
            return os.path.join(seq_path, img_files[frame_index]), seq_dir
    
    # Strategy 3: Fixed mapping - assume each transform corresponds to a specific sequence
    # In the real dataset, we might need to derive the exact mapping
    if sequence_dirs:
        # Use modulo to ensure we stay within the available sequences
        seq_index = transform_frame_num % len(sequence_dirs)
        seq_id = sequence_dirs[seq_index]
        seq_path = os.path.join(image_dir, seq_id)
        
        # Get first image in the sequence as fallback
        img_files = sorted([f for f in os.listdir(seq_path) 
                           if f.endswith(('.jpg', '.png'))])
        if img_files:
            return os.path.join(seq_path, img_files[0]), seq_id
    
    logger.warning(f"No matching image found for transform file: {transform_file}")
    return None, None

def process_identity_expression(data_dir, output_dir, identity_expression, seg_model, transform, max_frames=None, debug=False):
    """
    Process a single identity-expression pair.
    
    Args:
        data_dir: Base directory of Multiface dataset
        output_dir: Output directory for preprocessed data
        identity_expression: Identity-expression pair (e.g., "E057_Cheeks_Puffed")
        seg_model: Segmentation model
        transform: Image transform for segmentation
        max_frames: Maximum number of frames to process
        debug: Enable debug mode
        
    Returns:
        Dictionary with metadata about processed frames
    """
    identity, expression = identity_expression.split('_', 1)
    logger.info(f"Processing {identity_expression}...")
    
    # Create output directory
    output_identity_dir = os.path.join(output_dir, identity_expression)
    os.makedirs(output_identity_dir, exist_ok=True)
    
    # Get paths
    image_dir = os.path.join(data_dir, "images", identity_expression)
    mesh_dir = os.path.join(data_dir, "tracked_mesh", identity_expression)
    
    # Check if directories exist
    if not os.path.exists(image_dir):
        logger.error(f"Image directory not found: {image_dir}")
        return {}
    if not os.path.exists(mesh_dir):
        logger.error(f"Mesh directory not found: {mesh_dir}")
        return {}
    
    # Get transform files
    transform_files = [f for f in os.listdir(mesh_dir) if f.endswith('_transform.txt')]
    if debug:
        transform_files = transform_files[:min(5, len(transform_files))]
    elif max_frames is not None:
        transform_files = transform_files[:min(max_frames, len(transform_files))]
    
    if not transform_files:
        logger.warning(f"No transform files found in {mesh_dir}")
        return {}
    
    # Get all sequence directories for this identity-expression
    sequence_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    sequence_dirs.sort(key=lambda x: int(x))
    
    # Create a mapping of sequence_dir -> images
    sequence_images = {}
    for seq_dir in sequence_dirs:
        seq_path = os.path.join(image_dir, seq_dir)
        img_files = sorted([f for f in os.listdir(seq_path) if f.endswith(('.jpg', '.png'))])
        sequence_images[seq_dir] = img_files
    
    # Process each transform file
    metadata = {}
    frame_counter = 0
    
    # Create a deterministic mapping between transform files and sequence directories
    # For simplicity, we'll distribute transform files across sequence directories
    transform_files.sort()  # Sort transform files to ensure deterministic behavior
    
    for i, tf_file in enumerate(tqdm(transform_files, desc=f"Processing {identity_expression}")):
        tf_path = os.path.join(mesh_dir, tf_file)
        
        # Parse transform matrix
        pose_matrix = parse_transform_file(tf_path)
        if pose_matrix is None:
            continue
        
        # Select a sequence directory and image
        seq_dir = sequence_dirs[i % len(sequence_dirs)]
        img_files = sequence_images[seq_dir]
        
        if not img_files:
            continue
        
        # Use modulo to cycle through available images in the sequence
        img_file = img_files[frame_counter % len(img_files)]
        img_path = os.path.join(image_dir, seq_dir, img_file)
        
        # Extract frame ID from image path
        frame_id = os.path.splitext(os.path.basename(img_path))[0]
        
        # Output paths
        output_img_path = os.path.join(output_identity_dir, f"{seq_dir}_{frame_id}.jpg")
        output_mask_path = os.path.join(output_identity_dir, f"{seq_dir}_{frame_id}_mask.png")
        output_pose_path = os.path.join(output_identity_dir, f"{seq_dir}_{frame_id}_pose.npy")
        
        # Copy the image
        shutil.copy(img_path, output_img_path)
        
        # Generate and save segmentation mask
        mask = extract_segmentation_mask(seg_model, img_path, transform)
        if mask is not None:
            cv2.imwrite(output_mask_path, mask)
        
        # Save pose matrix
        np.save(output_pose_path, pose_matrix)
        
        # Add to metadata
        metadata[f"{seq_dir}_{frame_id}"] = {
            "image": output_img_path,
            "mask": output_mask_path,
            "pose": output_pose_path,
            "identity": identity,
            "expression": expression,
            "sequence": seq_dir
        }
        
        frame_counter += 1
    
    # Save metadata
    with open(os.path.join(output_identity_dir, "meta.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Processed {len(metadata)} frames for {identity_expression}")
    return metadata

def merge_expressions(output_dir, identity):
    """
    Merge multiple expressions for a single identity.
    
    Args:
        output_dir: Output directory
        identity: Identity to merge (e.g., "E057")
        
    Returns:
        Path to merged directory
    """
    logger.info(f"Merging expressions for identity {identity}...")
    
    # Find all directories for this identity
    identity_expr_dirs = [d for d in os.listdir(output_dir) 
                         if os.path.isdir(os.path.join(output_dir, d)) and d.startswith(f"{identity}_")]
    
    if not identity_expr_dirs:
        logger.warning(f"No expression directories found for identity {identity}")
        return None
    
    # Create merged directory
    merged_dir = os.path.join(output_dir, f"{identity}_all")
    os.makedirs(merged_dir, exist_ok=True)
    
    # Process each expression directory
    merged_metadata = {}
    for expr_dir in tqdm(identity_expr_dirs, desc=f"Merging expressions for {identity}"):
        # Load metadata
        meta_path = os.path.join(output_dir, expr_dir, "meta.json")
        if not os.path.exists(meta_path):
            logger.warning(f"Metadata file not found: {meta_path}")
            continue
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract expression from directory name
        expression = expr_dir.split('_', 1)[1]
        
        # Copy files and update metadata
        for frame_id, frame_data in metadata.items():
            # Create a unique frame ID that includes the expression
            unique_frame_id = f"{expression}_{frame_id}"
            
            # Output paths
            output_img_path = os.path.join(merged_dir, f"{unique_frame_id}.jpg")
            output_mask_path = os.path.join(merged_dir, f"{unique_frame_id}_mask.png")
            output_pose_path = os.path.join(merged_dir, f"{unique_frame_id}_pose.npy")
            
            # Copy files
            shutil.copy(frame_data["image"], output_img_path)
            shutil.copy(frame_data["mask"], output_mask_path)
            shutil.copy(frame_data["pose"], output_pose_path)
            
            # Add to merged metadata
            merged_metadata[unique_frame_id] = {
                "image": output_img_path,
                "mask": output_mask_path,
                "pose": output_pose_path,
                "identity": identity,
                "expression": expression,
                "sequence": frame_data.get("sequence", "unknown")
            }
    
    # Save merged metadata
    with open(os.path.join(merged_dir, "meta.json"), 'w') as f:
        json.dump(merged_metadata, f, indent=2)
    
    logger.info(f"Merged {len(merged_metadata)} frames for identity {identity}")
    return merged_dir

def main():
    args = parse_args()
    
    # Validate input directory
    if not os.path.exists(args.datadir):
        logger.error(f"Input directory does not exist: {args.datadir}")
        return
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load segmentation model
    seg_model = load_segmentation_model()
    transform = get_transform()
    
    # Get identities and expressions to process
    identity_expressions = []
    
    if args.debug:
        # In debug mode, just process one identity-expression pair
        for subdir in os.listdir(os.path.join(args.datadir, "images")):
            if os.path.isdir(os.path.join(args.datadir, "images", subdir)):
                identity_expressions.append(subdir)
                break
    else:
        # Get all identity-expression pairs
        image_dir = os.path.join(args.datadir, "images")
        identity_expressions = [subdir for subdir in os.listdir(image_dir)
                              if os.path.isdir(os.path.join(image_dir, subdir))]
        
        # Filter by identity if specified
        if args.identities:
            identities = args.identities.split(',')
            identity_expressions = [ie for ie in identity_expressions
                                 if any(ie.startswith(f"{identity}_") for identity in identities)]
        
        # Filter by expression if specified
        if args.expressions:
            expressions = args.expressions.split(',')
            identity_expressions = [ie for ie in identity_expressions
                                 if any(f"_{expression}" in ie for expression in expressions)]
    
    if not identity_expressions:
        logger.error("No identity-expression pairs found to process")
        return
    
    logger.info(f"Processing {len(identity_expressions)} identity-expression pairs")
    
    # Process each identity-expression pair
    processed_identities = set()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {}
        for ie in identity_expressions:
            identity = ie.split('_', 1)[0]
            processed_identities.add(identity)
            
            future = executor.submit(
                process_identity_expression,
                args.datadir,
                args.outdir,
                ie,
                seg_model,
                transform,
                args.max_frames,
                args.debug
            )
            futures[future] = ie
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing identity-expressions"):
            ie = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing {ie}: {e}")
    
    # Merge expressions if requested
    if args.merge_expressions:
        logger.info("Merging expressions by identity...")
        for identity in tqdm(processed_identities, desc="Merging expressions"):
            merge_expressions(args.outdir, identity)
    
    logger.info(f"Preprocessing complete. Output directory: {args.outdir}")
    
    # Print example FDNeRF training command
    if args.merge_expressions and processed_identities:
        identity = next(iter(processed_identities))
        print("\nExample FDNeRF training command:")
        print(f"""
python train/train_fdnerf.py \\
    --resume \\
    --batch_size 4 \\
    --gpu_id 0 \\
    --datadir {args.outdir} \\
    --dataset_prefix {identity}_all \\
    --name fdnerf_multiface_test \\
    --conf conf/exp/fp_mixexp_2D_implicit.conf \\
    --chunk_size 4000
        """)

if __name__ == "__main__":
    main()