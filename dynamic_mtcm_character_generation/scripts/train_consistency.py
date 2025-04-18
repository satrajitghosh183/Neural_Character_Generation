# import os
# import torch
# import torch.nn as nn
# import numpy as np
# import logging
# from torch.utils.data import DataLoader
# from transformer_model import ConsistencyTransformer
# from transformer_dataset import ConsistencySequenceDataset
# from analysis_utils import (
#     plot_attention, plot_recon_errors, plot_pose_diff,
#     plot_temporal_similarity, plot_tsne, save_optimal_views
# )
# import os
# import json
# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger("train_consistency")

# def train():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")

#     # Config
#     root_dir = "data"
#     identity = "Angelina Jolie"
#     embedding_dim = 384
#     token_dim = 394  # 384 (embedding) + 7 (pose) + 3 (timestamp, mask_area, focal)
#     save_dir = "logs"
#     checkpoint_dir = "checkpoints"
#     os.makedirs(save_dir, exist_ok=True)
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     save_path = os.path.join(checkpoint_dir, f"{identity.replace(' ', '_')}_transformer.pth")

#     # Dataset
#     dataset = ConsistencySequenceDataset(
#         root_dir=root_dir,
#         embedding_dim=384,  # ✅ use 384 since your .npy embeddings are 384D
#         max_seq_len=100,
#         mask_ratio=0.3
# )

#     if len(dataset.valid_images) == 0:
#         logger.error("❌ No valid images in dataset. Aborting training.")
#         return

#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

#     # Model
#     model = ConsistencyTransformer(
#         input_dim=token_dim,
#         model_dim=256,
#         num_layers=4,
#         num_heads=4,
#         use_pose_suggestion=True
#     ).to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
#     recon_loss_fn = nn.MSELoss()
#     pose_loss_fn = nn.L1Loss()

#     for epoch in range(100):
#         model.train()
#         total_recon_loss = 0
#         total_pose_loss = 0
#         total_batches = 0

#         for i, batch in enumerate(dataloader):
#             input_tokens = batch["input_tokens"].to(device)     # (1, N, D)
#             target_tokens = batch["target_tokens"].to(device)   # (1, N, D)
#             mask_indices = batch["mask_indices"].to(device)     # (1, M)

#             if input_tokens.size(1) == 0:
#                 logger.warning(f"Skipping batch {i} due to empty input.")
#                 continue

#             output = model(input_tokens, mask_indices, return_attention=(i == 0 and epoch % 5 == 0))
#             recon_pred = output["reconstructed_tokens"]

#             # Gather GT tokens at masked positions
#             target_masked = torch.stack([
#                 target_tokens[b, mask_indices[b]] for b in range(input_tokens.size(0))
#             ])

#             recon_loss = recon_loss_fn(recon_pred, target_masked)

#             if output["pose_suggestion"] is not None:
#                 true_poses = torch.stack([
#                     target_tokens[b, -1, -10:-3] for b in range(input_tokens.size(0))
#                 ]).to(device)
#                 pose_loss = pose_loss_fn(output["pose_suggestion"], true_poses)
#             else:
#                 pose_loss = torch.tensor(0.0).to(device)

#             loss = recon_loss + pose_loss

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_recon_loss += recon_loss.item()
#             total_pose_loss += pose_loss.item()
#             total_batches += 1

#             logger.info(f"[Epoch {epoch+1}, Batch {i}] Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, Pose: {pose_loss.item():.4f})")

#         torch.save(model.state_dict(), save_path)
#         logger.info(f"[Epoch {epoch+1}] ✅ Model saved to {save_path}")

#         # Optional analysis and visualization
#         if epoch % 5 == 0:
#             try:
#                 input_tokens_np = input_tokens[0].cpu().numpy()
#                 target_tokens_np = target_tokens[0].cpu().numpy()
#                 image_names = dataset.valid_images[:input_tokens_np.shape[0]]

#                 if output["attention"] is not None:
#                     attn = output["attention"][0].mean(0)
#                     plot_attention(attn, image_names, f"{save_dir}/attention_epoch{epoch+1}.png")

#                 plot_recon_errors(recon_pred, target_masked, mask_indices, f"{save_dir}/recon_epoch{epoch+1}.png")
#                 if output["pose_suggestion"] is not None:
#                     plot_pose_diff(output["pose_suggestion"], true_poses, f"{save_dir}/pose_diff_epoch{epoch+1}.png")
#                     np.save(f"{save_dir}/pose_embedding_epoch{epoch+1}.npy", output["pose_suggestion"].detach().cpu().numpy())


#                 plot_temporal_similarity(target_tokens_np, f"{save_dir}/temporal_sim_epoch{epoch+1}.png")
#                 plot_tsne(target_tokens_np, f"{save_dir}/tsne_epoch{epoch+1}.png")
#                 save_optimal_views(target_tokens, image_names, f"{save_dir}/optimal_views_epoch{epoch+1}")

#             except Exception as e:
#                 logger.error(f"Visualization failed at epoch {epoch+1}: {e}")

# if __name__ == "__main__":
#     train()
import os
import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader
from transformer_model import ConsistencyTransformer
from transformer_dataset import ConsistencySequenceDataset
from analysis_utils import (
    plot_attention, plot_recon_errors, plot_pose_diff,
    plot_temporal_similarity, plot_tsne, save_optimal_views
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_consistency")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    root_dir = "data"
    identity = "Angelina Jolie"
    embedding_dim = 384
    token_dim = 394  # 384 + 7 + 3
    save_dir = "logs"
    checkpoint_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"{identity.replace(' ', '_')}_transformer.pth")

    dataset = ConsistencySequenceDataset(
        root_dir=root_dir,
        embedding_dim=embedding_dim,
        max_seq_len=100,
        mask_ratio=0.3
    )

    if len(dataset.valid_images) == 0:
        logger.error("No valid images in dataset. Aborting training.")
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = ConsistencyTransformer(
        input_dim=token_dim,
        model_dim=256,
        num_layers=4,
        num_heads=4,
        use_pose_suggestion=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    recon_loss_fn = nn.MSELoss()
    pose_loss_fn = nn.L1Loss()

    for epoch in range(100):
        model.train()
        total_recon_loss = 0
        total_pose_loss = 0
        total_batches = 0

        for i, batch in enumerate(dataloader):
            input_tokens = batch["input_tokens"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            mask_indices = batch["mask_indices"].to(device)

            if input_tokens.size(1) == 0:
                continue

            output = model(input_tokens, mask_indices, return_attention=(i == 0 and epoch % 5 == 0))
            recon_pred = output["reconstructed_tokens"]

            target_masked = torch.stack([
                target_tokens[b, mask_indices[b]] for b in range(input_tokens.size(0))
            ])

            recon_loss = recon_loss_fn(recon_pred, target_masked)

            if output["pose_suggestion"] is not None:
                true_poses = torch.stack([
                    target_tokens[b, -1, -10:-3] for b in range(input_tokens.size(0))
                ]).to(device)
                pose_loss = pose_loss_fn(output["pose_suggestion"], true_poses)
            else:
                pose_loss = torch.tensor(0.0).to(device)

            loss = recon_loss + pose_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_pose_loss += pose_loss.item()
            total_batches += 1

            logger.info(f"Epoch {epoch+1}, Batch {i}: Loss={loss.item():.4f}, Recon={recon_loss.item():.4f}, Pose={pose_loss.item():.4f}")

        torch.save(model.state_dict(), save_path)
        logger.info(f"Saved model to {save_path}")

        if epoch % 5 == 0:
            try:
                input_tokens_np = input_tokens[0].cpu().numpy()
                target_tokens_np = target_tokens[0].cpu().numpy()
                image_names = dataset.valid_images[:input_tokens_np.shape[0]]

                if output["attention"] is not None:
                    attn = output["attention"][0].mean(0)
                    plot_attention(attn, image_names, f"{save_dir}/attention_epoch{epoch+1}.png")

                plot_recon_errors(recon_pred, target_masked, mask_indices, f"{save_dir}/recon_epoch{epoch+1}.png")

                if output["pose_suggestion"] is not None:
                    plot_pose_diff(output["pose_suggestion"], true_poses, f"{save_dir}/pose_diff_epoch{epoch+1}.png")
                    np.save(f"{save_dir}/pose_embedding_epoch{epoch+1}.npy", output["pose_suggestion"].detach().cpu().numpy())

                plot_temporal_similarity(target_tokens_np, f"{save_dir}/temporal_sim_epoch{epoch+1}.png")
                plot_tsne(target_tokens_np, f"{save_dir}/tsne_epoch{epoch+1}.png")
                save_optimal_views(target_tokens_np, image_names, f"{save_dir}/optimal_views_epoch{epoch+1}")

            except Exception as e:
                logger.error(f"Visualization failed at epoch {epoch+1}: {e}")

if __name__ == "__main__":
    train()
