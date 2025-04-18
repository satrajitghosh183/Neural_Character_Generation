# Filename: scripts/generate_pose_descriptions.py

import os
import glob
import numpy as np
import json
import torch
import open_clip
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import math

# === CONFIGURATION ===
logs_dir = "logs"
output_dir = os.path.join(logs_dir, "pose_analysis")
output_json = os.path.join(output_dir, "pose_descriptions.json")
filtered_images_path = os.path.join(output_dir, "filtered_images.txt")
num_clusters = 15
coverage_threshold = 0.7
device = "cuda" if torch.cuda.is_available() else "cpu"
grid_size = (4, 4)
generate_grid_previews = True

# === Load CLIP ===
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()

# === Poses ===
comprehensive_pose_prompts = [
    "a person looking straight ahead", "a person looking up", "a person looking slightly up", "a person looking down",
    "a person looking down at the ground", "a person facing front", "a person from left side", "a person from right side",
    "a person at 45 degree angle", "a person turning back", "a person from behind", "a person in profile",
    "a person tilting head left", "a person tilting head right", "a close-up of a person's face", "a full body shot",
    "a person with chin up", "a person with chin down", "a person sitting", "a person leaning forward"
]

# === Directories ===
image_dir = None
for candidate in [
    "raw_images/Angelina Jolie",
    "data/raw_images/Angelina Jolie"
]:
    if os.path.exists(candidate) and len(glob.glob(os.path.join(candidate, "*.jpg"))) > 0:
        image_dir = candidate
        break
if image_dir is None:
    raise FileNotFoundError("Could not locate raw_images/Angelina Jolie")

os.makedirs(output_dir, exist_ok=True)

# === Step 1: Load pose embeddings ===
print("[1/8] Loading pose embeddings...")
pose_files = sorted(glob.glob(os.path.join(logs_dir, "pose_embedding_epoch*.npy")))
poses, epochs = [], []
for f in pose_files:
    vec = np.load(f)
    if vec.ndim == 2:
        poses.append(vec[0])
        epochs.append(int(os.path.basename(f).split("epoch")[1].split(".")[0]))
poses = np.array(poses)

# === Step 2: Clustering ===
print("[2/8] Clustering pose embeddings...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(poses)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# === Step 3: Match images to embeddings ===
image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
image_files = [f for f in image_files if int(os.path.basename(f).split("_")[0]) in epochs]
assert len(image_files) == len(poses), f"Image count ({len(image_files)}) != pose count ({len(poses)})"

# === Step 4: Dimensionality reduction ===
print("[3/8] Reducing dimensionality...")
pose_tsne = TSNE(n_components=2, random_state=42, perplexity=10).fit_transform(poses)
plt.scatter(pose_tsne[:, 0], pose_tsne[:, 1], c=labels, cmap='tab10', alpha=0.6)
plt.title("t-SNE of Pose Embeddings")
plt.savefig(os.path.join(output_dir, "pose_space_visualization.png"))
plt.close()

# === Step 5: CLIP captioning ===
print("[4/8] Captioning representative images...")
caption_results = []
filtered_diverse_images = []
clusters_with_samples = {}
cluster_to_description = {}
cluster_members = {i: np.where(labels == i)[0].tolist() for i in range(num_clusters)}

for cluster_id, member_indices in cluster_members.items():
    best_idx = member_indices[np.argmin([np.linalg.norm(poses[i] - centroids[cluster_id]) for i in member_indices])]
    image_path = image_files[best_idx]
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    except:
        continue
    with torch.no_grad():
        text_tokens = tokenizer(comprehensive_pose_prompts).to(device)
        image_feat = model.encode_image(image)
        text_feat = model.encode_text(text_tokens)
        sims = torch.nn.functional.cosine_similarity(image_feat, text_feat)
        top_idx = torch.topk(sims, 3).indices
        top_desc = [comprehensive_pose_prompts[i] for i in top_idx]
        top_scores = [sims[i].item() for i in top_idx]

    caption_results.append({
        "epoch": epochs[best_idx],
        "image": os.path.basename(image_path),
        "cluster_id": cluster_id,
        "description": top_desc[0],
        "similarity_scores": top_scores
    })
    filtered_diverse_images.append(os.path.basename(image_path))
    clusters_with_samples[cluster_id] = {"idx": best_idx}
    cluster_to_description[cluster_id] = top_desc[0]

# === Step 6: Missing pose coverage ===
print("[5/8] Checking missing pose categories...")
missing_poses = []
with torch.no_grad():
    text_tokens = tokenizer(comprehensive_pose_prompts).to(device)
    text_feats = model.encode_text(text_tokens)
for i, desc in enumerate(comprehensive_pose_prompts):
    best_sim = max(
        torch.nn.functional.cosine_similarity(text_feats[i:i+1], model.encode_image(preprocess(Image.open(image_files[clusters_with_samples[cid]["idx"]]).convert("RGB")).unsqueeze(0).to(device))).item()
        for cid in clusters_with_samples
    )
    if best_sim < coverage_threshold:
        missing_poses.append({"pose_description": desc, "similarity": best_sim})

# === Step 7: Save results ===
with open(output_json, "w") as f:
    json.dump({
        "cluster_descriptions": caption_results,
        "missing_poses": missing_poses
    }, f, indent=2)
with open(filtered_images_path, "w") as f:
    for img in filtered_diverse_images:
        f.write(img + "\n")

print(f"[✓] Saved to {output_dir}")
print(f"[✓] Filtered {len(filtered_diverse_images)} representative images")
print(f"[✓] Found {len(missing_poses)} missing pose categories")
