# # import matplotlib.pyplot as plt
# # import numpy as np
# # import pandas as pd
# # import json
# # from sklearn.metrics.pairwise import cosine_similarity
# # from sklearn.manifold import TSNE

# # def plot_attention(attn, image_names, save_path):
# #     plt.figure(figsize=(8, 6))
# #     plt.imshow(attn.cpu().numpy(), cmap='hot', interpolation='nearest')
# #     plt.colorbar()
# #     plt.xticks(ticks=np.arange(len(image_names)), labels=image_names, rotation=90, fontsize=6)
# #     plt.yticks(ticks=np.arange(len(image_names)), labels=image_names, fontsize=6)
# #     plt.title("Attention Heatmap")
# #     plt.tight_layout()
# #     plt.savefig(save_path)
# #     plt.close()

# # def plot_recon_errors(pred, gt, mask_indices, save_path):
# #     errors = (pred - gt).pow(2).mean(dim=-1).detach().cpu().numpy()
# #     plt.figure(figsize=(10, 3))
# #     plt.plot(errors.T)
# #     plt.title("Reconstruction Error per Masked Token")
# #     plt.xlabel("Token Index")
# #     plt.ylabel("MSE")
# #     plt.grid(True)
# #     plt.tight_layout()
# #     plt.savefig(save_path)
# #     plt.close()

# # def plot_pose_diff(pred_pose, gt_pose, save_path):
# #     diff = (pred_pose - gt_pose).abs().detach().cpu().numpy()
# #     plt.figure(figsize=(8, 2))
# #     plt.bar(np.arange(len(diff[0])), diff[0])
# #     plt.title("Pose Difference")
# #     plt.xlabel("Pose Vector Index (x, y, z, qw, qx, qy, qz)")
# #     plt.tight_layout()
# #     plt.savefig(save_path)
# #     plt.close()
# # def save_optimal_views(tokens, image_names, save_path_prefix):
# #     cos_sim = cosine_similarity(tokens)
# #     avg_sim = np.mean(cos_sim, axis=1)
# #     ranking = np.argsort(-avg_sim)

# #     view_data = []
# #     for i in ranking:
# #         view_data.append({
# #             "image": image_names[i],
# #             "avg_similarity": float(avg_sim[i])
# #         })

# #     with open(f"{save_path_prefix}.json", "w") as f:
# #         json.dump(view_data, f, indent=2)
# #     pd.DataFrame(view_data).to_csv(f"{save_path_prefix}.csv", index=False)


# # def plot_temporal_similarity(tokens, save_path):
# #     cos_sim = cosine_similarity(tokens)
# #     sims = [cos_sim[i, i-1] for i in range(1, len(tokens))]
# #     plt.figure(figsize=(10, 3))
# #     plt.plot(sims, marker='o')
# #     plt.title("Temporal Similarity Between Frames")
# #     plt.xlabel("Frame Index")
# #     plt.ylabel("Cosine Similarity")
# #     plt.grid(True)
# #     plt.tight_layout()
# #     plt.savefig(save_path)
# #     plt.close()

# # def plot_tsne(tokens, save_path):
# #     embeddings = TSNE(n_components=2, random_state=42).fit_transform(tokens)
# #     plt.figure(figsize=(6, 5))
# #     plt.scatter(embeddings[:, 0], embeddings[:, 1], c=np.arange(len(tokens)), cmap='viridis')
# #     plt.title("t-SNE on Embeddings")
# #     plt.tight_layout()
# #     plt.savefig(save_path)
# #     plt.close()
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import json
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.manifold import TSNE

# def plot_attention(attn, image_names, save_path):
#     attn = attn.detach().cpu().numpy() if hasattr(attn, 'detach') else attn
#     plt.figure(figsize=(8, 6))
#     plt.imshow(attn, cmap='hot', interpolation='nearest')
#     plt.colorbar()
#     plt.xticks(ticks=np.arange(len(image_names)), labels=image_names, rotation=90, fontsize=6)
#     plt.yticks(ticks=np.arange(len(image_names)), labels=image_names, fontsize=6)
#     plt.title("Attention Heatmap")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     print(f"[✔] Saved attention heatmap: {save_path}")
#     plt.close()

# def plot_recon_errors(pred, gt, mask_indices, save_path):
#     errors = (pred - gt).pow(2).mean(dim=-1).detach().cpu().numpy()
#     plt.figure(figsize=(10, 3))
#     plt.plot(errors.T)
#     plt.title("Reconstruction Error per Masked Token")
#     plt.xlabel("Token Index")
#     plt.ylabel("MSE")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     print(f"[✔] Saved reconstruction error plot: {save_path}")
#     plt.close()

# def plot_pose_diff(pred_pose, gt_pose, save_path):
#     diff = (pred_pose - gt_pose).abs().detach().cpu().numpy()
#     plt.figure(figsize=(8, 2))
#     plt.bar(np.arange(len(diff[0])), diff[0])
#     plt.title("Pose Difference")
#     plt.xlabel("Pose Vector Index (x, y, z, qw, qx, qy, qz)")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     print(f"[✔] Saved pose diff bar chart: {save_path}")
#     plt.close()

# def save_optimal_views(tokens, image_names, save_prefix):
#     tokens = tokens.detach().cpu().numpy() if hasattr(tokens, 'detach') else tokens
#     cos_sim = cosine_similarity(tokens)
#     avg_sim = np.mean(cos_sim, axis=1)
#     ranking = np.argsort(-avg_sim)

#     view_data = [{"image": image_names[i], "avg_similarity": float(avg_sim[i])} for i in ranking]

#     json_path = f"{save_prefix}.json"
#     csv_path = f"{save_prefix}.csv"
#     with open(json_path, "w") as f:
#         json.dump(view_data, f, indent=2)
#     pd.DataFrame(view_data).to_csv(csv_path, index=False)

#     print(f"[✔] Saved optimal views:\n    ↳ JSON: {json_path}\n    ↳ CSV: {csv_path}")

# def plot_temporal_similarity(tokens, save_path):
#     tokens = tokens.detach().cpu().numpy() if hasattr(tokens, 'detach') else tokens
#     cos_sim = cosine_similarity(tokens)
#     sims = [cos_sim[i, i-1] for i in range(1, len(tokens))]
#     plt.figure(figsize=(10, 3))
#     plt.plot(sims, marker='o')
#     plt.title("Temporal Similarity Between Frames")
#     plt.xlabel("Frame Index")
#     plt.ylabel("Cosine Similarity")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     print(f"[✔] Saved temporal similarity plot: {save_path}")
#     plt.close()

# def plot_tsne(tokens, save_path):
#     tokens = tokens.detach().cpu().numpy() if hasattr(tokens, 'detach') else tokens
#     embeddings = TSNE(n_components=2, random_state=42).fit_transform(tokens)
#     plt.figure(figsize=(6, 5))
#     plt.scatter(embeddings[:, 0], embeddings[:, 1], c=np.arange(len(tokens)), cmap='viridis')
#     plt.title("t-SNE on Embeddings")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     print(f"[✔] Saved t-SNE visualization: {save_path}")
#     plt.close()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

def plot_attention(attn, image_names, save_path):
    attn = attn.detach().cpu().numpy() if hasattr(attn, 'detach') else attn
    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(image_names)), labels=image_names, rotation=90, fontsize=6)
    plt.yticks(ticks=np.arange(len(image_names)), labels=image_names, fontsize=6)
    plt.title("Attention Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_recon_errors(pred, gt, mask_indices, save_path):
    errors = (pred - gt).pow(2).mean(dim=-1).detach().cpu().numpy()
    plt.figure(figsize=(10, 3))
    plt.plot(errors.T)
    plt.title("Reconstruction Error per Masked Token")
    plt.xlabel("Token Index")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_pose_diff(pred_pose, gt_pose, save_path):
    diff = (pred_pose - gt_pose).abs().detach().cpu().numpy()
    plt.figure(figsize=(8, 2))
    plt.bar(np.arange(len(diff[0])), diff[0])
    plt.title("Pose Difference")
    plt.xlabel("Pose Vector Index (x, y, z, qw, qx, qy, qz)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_optimal_views(tokens, image_names, save_prefix):
    tokens = tokens.detach().cpu().numpy() if hasattr(tokens, 'detach') else tokens
    cos_sim = cosine_similarity(tokens)
    avg_sim = np.mean(cos_sim, axis=1)
    ranking = np.argsort(-avg_sim)

    view_data = [{"image": image_names[i], "avg_similarity": float(avg_sim[i])} for i in ranking]

    json_path = f"{save_prefix}.json"
    csv_path = f"{save_prefix}.csv"
    txt_path = f"{save_prefix}_filtered.txt"

    with open(json_path, "w") as f:
        json.dump(view_data, f, indent=2)
    pd.DataFrame(view_data).to_csv(csv_path, index=False)

    top_k = int(len(view_data) * 0.7)  # keep top 70%
    with open(txt_path, "w") as f:
        for i in range(top_k):
            f.write(view_data[i]["image"] + "\n")

def plot_temporal_similarity(tokens, save_path):
    tokens = tokens.detach().cpu().numpy() if hasattr(tokens, 'detach') else tokens
    cos_sim = cosine_similarity(tokens)
    sims = [cos_sim[i, i - 1] for i in range(1, len(tokens))]
    plt.figure(figsize=(10, 3))
    plt.plot(sims, marker='o')
    plt.title("Temporal Similarity Between Frames")
    plt.xlabel("Frame Index")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_tsne(tokens, save_path):
    tokens = tokens.detach().cpu().numpy() if hasattr(tokens, 'detach') else tokens
    embeddings = TSNE(n_components=2, random_state=42).fit_transform(tokens)
    plt.figure(figsize=(6, 5))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=np.arange(len(tokens)), cmap='viridis')
    plt.title("t-SNE on Embeddings")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
