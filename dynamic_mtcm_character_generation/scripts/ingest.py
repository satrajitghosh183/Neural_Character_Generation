#!/usr/bin/env python3
import os
import argparse
import json
import exifread
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser(
        description="Recursively extract EXIF timestamps and focal lengths"
    )
    p.add_argument('--input_dir',  required=True,
                   help="Top‑level folder of images (will recurse into subfolders)")
    p.add_argument('--output_dir', required=True,
                   help="Where to write metadata/exif.json")
    return p.parse_args()

def get_exif_data(path):
    with open(path, 'rb') as f:
        tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal")
    # Timestamp
    dto = tags.get('EXIF DateTimeOriginal') or tags.get('Image DateTime')
    if dto:
        try:
            dt = datetime.strptime(str(dto), "%Y:%m:%d %H:%M:%S")
            ts = dt.timestamp()
        except:
            ts = None
    else:
        ts = None
    # Focal length (in mm)
    fl = tags.get('EXIF FocalLength')
    if fl:
        try:
            num, den = fl.values[0].num, fl.values[0].den
            fl_val = float(num) / float(den)
        except:
            fl_val = None
    else:
        fl_val = None
    return ts, fl_val

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Recursively collect image files (relative paths)
    files = []
    for root, _, fnames in os.walk(args.input_dir):
        for fn in fnames:
            if fn.lower().endswith(('.jpg','jpeg','.png')):
                abs_path = os.path.join(root, fn)
                rel_path = os.path.relpath(abs_path, args.input_dir)
                files.append(rel_path)
    files.sort()

    metadata = {}
    for idx, rel in enumerate(files):
        abs_path = os.path.join(args.input_dir, rel)
        ts, fl = get_exif_data(abs_path)
        if ts is None:
            ts = float(idx)          # fallback: file‐order index
        if fl is None:
            fl = 50.0                # fallback: 50 mm equiv.
        metadata[rel] = {'timestamp': ts, 'focal_length': fl}

    out_path = os.path.join(args.output_dir, 'exif.json')
    with open(out_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[✔] Wrote EXIF metadata for {len(files)} images to {out_path}")

if __name__ == '__main__':
    main()
