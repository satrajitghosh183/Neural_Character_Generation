Here's a complete and structured `README.md` summarizing the **math, methodology, training progress, and technical stack** of your project focused on **Transformer-based Temporal-Spatial Consistency Modeling** for view selection in 3D neural reconstructions of **Angelina Jolie**:

---

# 🧠 Temporal-Spatial Consistency Transformer for View Selection in NeRFs

## 🧾 Project Goal

This project develops a **Transformer-based Multi-Modal Temporal Consistency Module (MTCM)** to:
- Filter unstructured image sequences for NeRF training.
- Select optimal views by modeling **temporal smoothness**, **pose diversity**, and **appearance consistency**.
- Predict **missing views** for better 3D avatar reconstruction.
- Export filtered images and pose embeddings to drive NeRF pipelines.

---

## 📊 Mathematical Foundation

### 1. **Token Structure**
Each frame is encoded as a token:

\[
\mathbf{t}_i = \left[ \mathbf{e}_i, \mathbf{p}_i, s_i \right] \in \mathbb{R}^{394}
\]

Where:
- \(\mathbf{e}_i \in \mathbb{R}^{384}\): Embedding vector (e.g., CLIP)
- \(\mathbf{p}_i = [x, y, z, q_w, q_x, q_y, q_z] \in \mathbb{R}^7\): 3D camera pose
- \(s_i = [\text{timestamp}, \text{mask area}, \text{focal length}] \in \mathbb{R}^3\)

---

### 2. **Transformer Architecture**
We use a BERT-style masked Transformer encoder \(T\):

\[
\hat{\mathbf{t}}_i = T(\{\mathbf{t}_1, \dots, \mathbf{t}_N\})
\]

- Inputs are projected via a linear layer into \(d_{\text{model}} = 256\) space.
- Positional encodings \(\mathbf{PE}\) are added.
- Masked token prediction: MSE loss over masked entries.
- Pose suggestion head: Predicts a plausible next pose.

---

### 3. **Loss Functions**

#### a) Masked Token Reconstruction Loss
\[
\mathcal{L}_{\text{recon}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left\| \hat{\mathbf{t}}_i - \mathbf{t}_i \right\|_2^2
\]

#### b) Pose Suggestion Loss (optional)
\[
\mathcal{L}_{\text{pose}} = \left\| \hat{\mathbf{p}} - \mathbf{p}_{\text{last}} \right\|_1
\]

---

## 🧪 Dataset Used

Currently trained only on:

```
data/
├── embeddings/Angelina Jolie/*.npy      ← Embeddings (384D)
├── raw_images/Angelina Jolie/*.jpg      ← Input images
├── masks/Angelina Jolie/*.png           ← Segmentation masks
├── metadata/Angelina Jolie/
│   ├── poses.json                        ← COLMAP format (with 'images' and 'cameras')
│   └── exif.json                         ← Timestamps, focal length
```

Pose matching is handled via `poses.json['images'][filename]` using image keys like `059_xxx.jpg`.

---

## 🚧 Current Progress

### ✅ Fully Working:
- Consistency transformer pipeline
- Angelina Jolie-only training
- Masked token reconstruction & pose supervision
- Embedding token generation
- COLMAP pose parsing support
- Robust dataset loader with fallback for missing entries
- Attention heatmap visualization
- Pose error deltas and reconstruction difference plots
- View selection via similarity & embedding proximity
- Export of optimal views (`filtered_images.txt`)
- Support for both t-SNE and UMAP visualization (when `umap-learn` is installed)

### 🛠️ In Progress / TODO:
- Integrate with NeRF directly for differentiable fine-tuning
- Generate **textual pose labels** (e.g., “side view”, “look down”)
- Add cross-identity training and regularization
- Evaluate pose diversity after filtering
- Export optimized pose embeddings as `.npy` for NeRF sampling

---

## 🧯 Key Files

| File | Description |
|------|-------------|
| `transformer_model.py` | Transformer encoder with masked token and pose heads |
| `transformer_dataset.py` | Dataset with embedding, mask, pose, and EXIF handling |
| `train_consistency.py` | Trainer script with logging, loss handling, and checkpointing |
| `analysis_utils.py` | Helper functions for attention heatmaps, pose deltas, similarity matrix |
| `visualize_embeddings.py` | Runs UMAP and t-SNE on tokens and pose predictions |

---

## 📌 Training Tips

- Batch size = 1 due to per-identity sampling.
- Each token is a full image frame.
- Mask ratio = 0.3 (BERT-style prediction).
- Use `train_consistency.py` and ensure `Angelina Jolie` data is complete.

---

## 📤 Outputs

During training:
- `logs/attention_heatmap_epochX.png`
- `logs/pose_deltas_epochX.png`
- `logs/recon_deltas_epochX.png`
- `logs/tsne_embeddings.png`, `logs/umap_embeddings.png`
- `logs/similarity_matrix_epochX.npy`
- `logs/filtered_views_epochX.json`
- `logs/pose_embeddings_epochX.npy`

