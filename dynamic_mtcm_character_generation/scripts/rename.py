import os
import json

embeddings_dir = "data/embeddings/Angelina Jolie"
poses_path = "data/metadata/Angelina Jolie/poses.json"

with open(poses_path, "r") as f:
    poses = json.load(f)

# Create a map of existing embedding files
existing_files = {f[: -4]: f for f in os.listdir(embeddings_dir) if f.endswith(".npy")}

renamed = 0
for pose_key in poses:
    stem = pose_key.replace(".jpg", "")  # e.g., "070_abc123"
    
    # Try matching by order
    if renamed >= len(existing_files):
        break

    wrong_name = list(existing_files.values())[renamed]
    wrong_path = os.path.join(embeddings_dir, wrong_name)
    correct_path = os.path.join(embeddings_dir, f"{stem}.npy")

    print(f"Renaming {wrong_name} → {stem}.npy")
    os.rename(wrong_path, correct_path)
    renamed += 1
