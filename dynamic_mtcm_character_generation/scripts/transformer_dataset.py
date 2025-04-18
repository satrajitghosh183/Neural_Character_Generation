import os, json, numpy as np, torch
from torch.utils.data import Dataset
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AngelinaDataset")

class ConsistencySequenceDataset(Dataset):
    def __init__(self, root_dir, embedding_dim=384, max_seq_len=100, mask_ratio=0.3):
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio
        self.identity = "Angelina Jolie"
        
        self.emb_dir = os.path.join(root_dir, "embeddings", self.identity)
        self.mask_dir = os.path.join(root_dir, "masks", self.identity)
        self.pose_path = os.path.join(root_dir, "metadata", self.identity, "poses.json")
        self.exif_path = os.path.join(root_dir, "metadata", self.identity, "exif.json")

        # Load metadata
        with open(self.pose_path, "r") as f:
            poses = json.load(f)
            self.pose_data = poses["images"] if "images" in poses else poses
        with open(self.exif_path, "r") as f:
            self.exif_data = json.load(f)

        # Filter valid samples
        self.image_names = sorted([
            f for f in os.listdir(self.emb_dir)
            if f.endswith(".npy") and not f.startswith("cameras") and not f.startswith("images")
        ])
        self.valid_images = self._filter_valid_images()
        logger.info(f"✅ Found {len(self.valid_images)} valid image entries.")

    def _filter_valid_images(self):
        valid = []
        for fname in self.image_names:
            stem = fname.replace(".npy", "")
            jpg_key = stem + ".jpg"
            pose = self.pose_data.get(jpg_key)
            if not pose or "quaternion" not in pose or ("position" not in pose and "translation" not in pose):
                continue
            emb_path = os.path.join(self.emb_dir, fname)
            if not os.path.exists(emb_path):
                continue
            try:
                emb = np.load(emb_path)
                if emb.shape[0] != self.embedding_dim:
                    continue
            except:
                continue
            valid.append(stem)
        return valid

    def __len__(self):
        return max(1, len(self.valid_images))

    def __getitem__(self, idx):
        if not self.valid_images:
            return self._empty_batch()
        tokens = []
        for stem in self.valid_images:
            try:
                emb = np.load(os.path.join(self.emb_dir, stem + ".npy"))
                pose = self.pose_data.get(stem + ".jpg")
                position = pose.get("position") or pose.get("translation")
                quat = pose["quaternion"]
                exif = self.exif_data.get(stem, {})
                timestamp = exif.get("timestamp", 0.0)
                focal = exif.get("focal", 1.0)
                mask_path = os.path.join(self.mask_dir, stem + ".png")
                mask_area = 0.0
                if os.path.exists(mask_path):
                    try:
                        mask_area = np.array(Image.open(mask_path).convert("L")).mean() / 255.0
                    except: pass
                token = np.concatenate([emb, position, quat, [timestamp, mask_area, focal]])
                tokens.append(token)
            except Exception as e:
                logger.warning(f"Token failed for {stem}: {e}")
                continue
        if not tokens:
            return self._empty_batch()
        tokens = tokens[:self.max_seq_len]
        token_tensor = torch.tensor(tokens, dtype=torch.float32)
        seq_len = token_tensor.shape[0]
        num_mask = max(1, int(seq_len * self.mask_ratio))
        mask_indices = np.random.choice(seq_len, num_mask, replace=False)
        masked_tokens = token_tensor.clone()
        masked_tokens[mask_indices] = 0.0
        return {
            "input_tokens": masked_tokens,
            "target_tokens": token_tensor,
            "mask_indices": torch.tensor(mask_indices, dtype=torch.long),
        }

    def _empty_batch(self):
        D = self.embedding_dim + 10
        return {
            "input_tokens": torch.zeros((0, D), dtype=torch.float32),
            "target_tokens": torch.zeros((0, D), dtype=torch.float32),
            "mask_indices": torch.zeros((0,), dtype=torch.long),
        }
