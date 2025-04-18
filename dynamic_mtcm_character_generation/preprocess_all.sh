#!/bin/bash

IDENTITY_DIR="data/raw_images"

for name in "$IDENTITY_DIR"/*; do
    base=$(basename "$name")
    echo ""
    echo "──────────────────────────────────────────────"
    echo "🧠 Processing Identity: $base"
    echo "──────────────────────────────────────────────"

    # 1. Segmentation
    python scripts/segment.py \
        --input_dir "data/raw_images/$base" \
        --mask_dir "data/masks/$base"

    # 2. Embedding generation
    python scripts/generate_embeddings.py \
        --images_dir "data/raw_images/$base" \
        --output_dir "data/embeddings/$base" \
        --model dinov2

    # 3. EXIF / metadata ingestion
    python scripts/ingest.py \
        --input_dir "data/raw_images/$base" \
        --output_dir "data/metadata/$base"

    # 4. Make pair list
    python scripts/make_pairs.py \
        --raw_images "data/raw_images/$base" \
        --embeddings "data/embeddings/$base" \
        --masks "data/masks/$base" \
        --exif "data/metadata/$base/exif.json" \
        --output "data/metadata/$base/pair_list.txt"

    # 5. Run COLMAP
    python scripts/run_colmap.py \
        --colmap_path colmap \
        --image_dir "data/raw_images/$base" \
        --pair_list_path "data/metadata/$base/pair_list.txt" \
        --database_path "data/colmap/$base/database.db" \
        --sparse_dir "data/colmap/$base/sparse" \
        --output_json "data/metadata/$base/poses.json"

done
