#!/usr/bin/env python3
"""
run_colmap.py: Script to run COLMAP with guided matching using your pair_list.txt
and convert the sparse model to a JSON poses file.
"""
import os
import subprocess
import json
import argparse
import numpy as np
from pathlib import Path
def run_feature_matching(colmap_path, database_path, image_dir, pair_list_path, gpu=True):
    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    if os.path.exists(database_path):
        os.remove(database_path)

    # 1. Feature extraction
    print("→ Running COLMAP feature_extractor…")
    feat_cmd = [
        colmap_path, "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--SiftExtraction.use_gpu", "1" if gpu else "0",
    ]
    subprocess.run(feat_cmd, check=True)

    # 2. Matches importer with your pair list
    print("→ Importing matches from pair list…")
    match_cmd = [
        colmap_path, "matches_importer",
        "--database_path", database_path,
        "--match_list_path", pair_list_path,
        "--match_type", "pairs",
    ]
    subprocess.run(match_cmd, check=True)

    # ❌ No geometric verification (COLMAP will do it during mapping)


def run_sparse_reconstruction(colmap_path, database_path, image_dir, sparse_dir):
    os.makedirs(sparse_dir, exist_ok=True)
    print("→ Running COLMAP mapper for sparse reconstruction…")
    mapper_cmd = [
        colmap_path, "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir,
        "--Mapper.num_threads", "1",
        "--Mapper.init_min_tri_angle", "2.0",
        "--Mapper.abs_pose_min_num_inliers", "10",
        "--Mapper.filter_max_reproj_error", "4.0",
        "--Mapper.ba_refine_focal_length", "0",  # Don't optimize focal (it’s a guess)
        "--Mapper.ba_refine_principal_point", "0",
    ]
    subprocess.run(mapper_cmd, check=True)

def convert_and_parse(colmap_path, sparse_dir, output_json):
    """
    Convert the binary model to TXT and then parse cameras.txt + images.txt
    into a simple JSON with intrinsics + extrinsics.
    """
    # find the numeric subfolder (usually '0')
    subdirs = [d for d in os.listdir(sparse_dir) if d.isdigit()]
    if not subdirs:
        raise RuntimeError(f"No numeric COLMAP models in {sparse_dir}")
    model_dir = os.path.join(sparse_dir, sorted(subdirs, key=int)[-1])

    # convert .bin → .txt
    txt_dir = os.path.join(model_dir, "txt")
    os.makedirs(txt_dir, exist_ok=True)
    print(f"→ Converting {model_dir}/*.bin → {txt_dir}/*.txt…")
    subprocess.run([
        colmap_path, "model_converter",
        "--input_path", model_dir,
        "--output_path", txt_dir,
        "--output_type", "TXT"
    ], check=True)

    # now parse it
    from parse_colmap import parse_colmap_to_json
    parse_colmap_to_json(txt_dir, output_json)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_path',     type=str, default='colmap')
    parser.add_argument('--database_path',   type=str, default='data/colmap/database.db')
    parser.add_argument('--image_dir',       type=str, default='data/raw_images')
    parser.add_argument('--pair_list_path',  type=str, default='data/metadata/pair_list.txt')
    parser.add_argument('--sparse_dir',      type=str, default='data/colmap/sparse')
    parser.add_argument('--output_json',     type=str, default='data/metadata/poses.json')
    parser.add_argument('--gpu',             action='store_true', default=True)
    args = parser.parse_args()

    # 1) features + guided exhaustive matching
    run_feature_matching(
        args.colmap_path,
        args.database_path,
        args.image_dir,
        args.pair_list_path,
        gpu=args.gpu
    )

    # 2) sparse reconstruction
    run_sparse_reconstruction(
        args.colmap_path,
        args.database_path,
        args.image_dir,
        args.sparse_dir
    )

    # 3) convert to TXT & parse JSON
    convert_and_parse(args.colmap_path, args.sparse_dir, args.output_json)

    print("✅ COLMAP processing complete!")
    print(f"--> Poses JSON written to {args.output_json}")

if __name__ == "__main__":
    main()
