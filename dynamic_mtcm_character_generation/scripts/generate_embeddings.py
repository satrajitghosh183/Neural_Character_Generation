#!/usr/bin/env python3
"""
generate_embeddings.py: Script to generate embeddings for all images using a pre-trained model.
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import glob

def extract_dino_embeddings(model, image_path, transform):
    """Extract DINO embeddings for a single image."""
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        # Move to device
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        # Generate embedding
        with torch.no_grad():
            embedding = model(img_tensor).cpu().numpy()[0]
        
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for all images")
    parser.add_argument('--images_dir', type=str, default='data/raw_images',
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='data/embeddings',
                        help='Directory to save embeddings')
    parser.add_argument('--model', type=str, default='dinov2',
                        choices=['dinov2', 'resnet50', 'clip'],
                        help='Feature extraction model')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    if args.model == 'dinov2':
        try:
            print("Loading DINOv2 model...")
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"Error loading DINOv2: {e}")
            print("Falling back to ResNet50...")
            args.model = 'resnet50'
    
    if args.model == 'resnet50':
        print("Loading ResNet50 model...")
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        # Remove the final classification layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif args.model == 'clip':
        print("Loading CLIP model...")
        try:
            import clip
            model, transform = clip.load('ViT-B/32', device=device)
        except ImportError:
            print("CLIP not installed. Please install with: pip install clip")
            print("Falling back to ResNet50...")
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            # Remove the final classification layer
            model = torch.nn.Sequential(*list(model.children())[:-1])
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Find all image files
    all_image_files = []
    
    # Check if the folder structure includes person subfolders
    has_subfolders = False
    for item in os.listdir(args.images_dir):
        if os.path.isdir(os.path.join(args.images_dir, item)):
            has_subfolders = True
            break
    
    if has_subfolders:
        # Process images in subfolders
        for person_dir in os.listdir(args.images_dir):
            person_path = os.path.join(args.images_dir, person_dir)
            if os.path.isdir(person_path):
                for img_file in glob.glob(os.path.join(person_path, "*.jpg")):
                    all_image_files.append(img_file)
    else:
        # Process images directly in the images directory
        all_image_files = glob.glob(os.path.join(args.images_dir, "*.jpg"))
    
    print(f"Found {len(all_image_files)} image files")
    
    # Process all images
    for img_path in tqdm(all_image_files, desc="Generating embeddings"):
        # Extract image ID from path
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        
        # Skip if embedding already exists
        output_path = os.path.join(args.output_dir, f"{img_id}.npy")
        if os.path.exists(output_path):
            continue
        
        # Generate embedding
        if args.model == 'clip':
            # Special handling for CLIP
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(img_tensor).cpu().numpy()[0]
        else:
            # For DINOv2 or ResNet
            embedding = extract_dino_embeddings(model, img_path, transform)
        
        if embedding is not None:
            # Save embedding
            np.save(output_path, embedding)
    
    print(f"Embeddings saved to {args.output_dir}")

if __name__ == "__main__":
    main()