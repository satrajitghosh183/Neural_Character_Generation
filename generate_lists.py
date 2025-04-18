import os
import json
import random
import argparse
import glob

def write_list_file(file_path, sample_keys):
    with open(file_path, "w") as f:
        for key in sample_keys:
            f.write(key + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test lists for FDNeRF.")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to FDNeRF dataset directory where list files should be saved")
    parser.add_argument("--meta_json_path", type=str, required=True, 
                        help="Full path to the meta.json file")
    args = parser.parse_args()

    meta_path = args.meta_json_path
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found at {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Get all keys from meta.json
    all_keys = list(meta.keys())
    
    print(f"Found {len(all_keys)} total samples in meta.json")
    
    if len(all_keys) == 0:
        raise ValueError(f"No data samples found in {meta_path}")
        
    random.shuffle(all_keys)

    # 80-10-10 split
    n_total = len(all_keys)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    train_keys = all_keys[:n_train]
    val_keys = all_keys[n_train:n_train + n_val]
    test_keys = all_keys[n_train + n_val:]

    # Extract identity name from meta_json_path
    identity = os.path.basename(os.path.dirname(meta_path))
    
    # Save directly to the data directory
    write_list_file(os.path.join(args.data_dir, f"{identity}_train.lst"), train_keys)
    write_list_file(os.path.join(args.data_dir, f"{identity}_val.lst"), val_keys)
    write_list_file(os.path.join(args.data_dir, f"{identity}_test.lst"), test_keys)

    print(f"✅ Lists generated successfully:")
    print(f"Train: {len(train_keys)} | Val: {len(val_keys)} | Test: {len(test_keys)}")
    print(f"Written to {args.data_dir}")

if __name__ == "__main__":
    main()