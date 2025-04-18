#!/usr/bin/env python3
"""
make_pairs.py: Computes image pairs for guided matching in COLMAP based on 
feature similarity, masks, and temporal proximity.
"""
import os
import glob
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import argparse

def load_embeddings(embeddings_dir):
    """Recursively load all .npy embeddings, using relative paths (no extension) as IDs."""
    embeddings = {}
    pattern = os.path.join(embeddings_dir, "**", "*.npy")
    for file_path in tqdm(glob.glob(pattern, recursive=True), desc="Loading embeddings"):
        rel = os.path.relpath(file_path, embeddings_dir)
        # Normalize to POSIX style
        rel = rel.replace(os.sep, "/")
        img_id = os.path.splitext(rel)[0]
        embeddings[img_id] = np.load(file_path)
    return embeddings

def load_masks(masks_dir):
    """Recursively load all mask images, using relative paths (no extension) as IDs."""
    masks = {}
    pattern = os.path.join(masks_dir, "**", "*.png")
    for file_path in tqdm(glob.glob(pattern, recursive=True), desc="Loading masks"):
        rel = os.path.relpath(file_path, masks_dir)
        rel = rel.replace(os.sep, "/")
        img_id = os.path.splitext(rel)[0]
        mask = np.array(Image.open(file_path).convert("L")) / 255.0
        masks[img_id] = mask
    return masks

def load_exif_timestamps(exif_path):
    """Load timestamps from EXIF JSON, strip extensions to match IDs."""
    with open(exif_path, 'r') as f:
        exif_data = json.load(f)
    timestamps = {}
    for key, meta in exif_data.items():
        # key like "ClassName/0001.jpg" -> strip extension
        id_no_ext = os.path.splitext(key)[0].replace('\\', '/')
        if 'DateTimeOriginal' in meta:
            timestamps[id_no_ext] = meta['DateTimeOriginal']
        elif 'DateTime' in meta:
            timestamps[id_no_ext] = meta['DateTime']
        else:
            timestamps[id_no_ext] = id_no_ext
    return timestamps

def compute_similarity_matrix(embeddings, masks, timestamps, temporal_weight=0.2):
    image_ids = list(embeddings.keys())
    n = len(image_ids)
    if n == 0:
        raise RuntimeError("No embeddings found to compute pairs.")
    # Embedding tensor
    emb_stack = np.stack([embeddings[i] for i in image_ids])
    emb_tensor = torch.tensor(emb_stack, dtype=torch.float32)
    emb_tensor = emb_tensor / torch.norm(emb_tensor, dim=1, keepdim=True)
    feature_sim = torch.mm(emb_tensor, emb_tensor.T).cpu().numpy()
    # Temporal sim
    vals = []
    for i in image_ids:
        t = timestamps.get(i, None)
        vals.append(str(t))
    temporal_sim = np.ones((n,n))
    for i in range(n):
        for j in range(i+1, n):
            if vals[i] == vals[j]:
                ts = 1.0
            else:
                ts = 1.0 / (1.0 + abs(ord(vals[i][0]) - ord(vals[j][0])))
            temporal_sim[i,j] = ts
            temporal_sim[j,i] = ts
    # Mask IoU
    mask_sim = np.ones((n,n))
    for i in range(n):
        m1 = masks.get(image_ids[i])
        if m1 is None: continue
        for j in range(i+1, n):
            m2 = masks.get(image_ids[j])
            if m2 is None: continue
            inter = np.sum(m1 * m2)
            union = np.sum(np.clip(m1 + m2, 0, 1))
            iou = inter / union if union>0 else 0
            mask_sim[i,j] = iou
            mask_sim[j,i] = iou
    # Combined
    sim = feature_sim * mask_sim * (1 + temporal_weight * temporal_sim)
    np.fill_diagonal(sim, 0)
    return image_ids, sim

def get_top_k_pairs(image_ids, sim_matrix, k=10):
    pairs = []
    for i, img in enumerate(image_ids):
        idx = np.argsort(sim_matrix[i])[::-1][:k]
        for j in idx:
            if i!=j:
                pairs.append((img, image_ids[j]))
    return pairs
def write_pair_list(pairs, output_path):
    with open(output_path, 'w') as f:
        for a, b in pairs:
            f.write(f"{a}.jpg {b}.jpg\n")  # <-- Add extension explicitly

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw_images', type=str, required=True)
    p.add_argument('--embeddings', type=str, required=True)
    p.add_argument('--masks', type=str, required=True)
    p.add_argument('--exif', type=str,   required=True)
    p.add_argument('--output', type=str, required=True)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--temporal_weight', type=float, default=0.2)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print("Loading embeddings...")
    emb = load_embeddings(args.embeddings)
    print("Loading masks...")
    msk = load_masks(args.masks)
    print("Loading EXIF data...")
    ts  = load_exif_timestamps(args.exif)
    print("Computing similarity matrix...")
    ids, sim = compute_similarity_matrix(emb, msk, ts, args.temporal_weight)
    print(f"Finding top-{args.k} pairs per image...")
    pairs = get_top_k_pairs(ids, sim, args.k)
    print(f"Writing {len(pairs)} pairs to {args.output}...")
    write_pair_list(pairs, args.output)
    print("Done.")

if __name__ == '__main__':
    main()
