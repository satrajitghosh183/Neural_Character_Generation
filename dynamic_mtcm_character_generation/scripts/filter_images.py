import os
import torch
import numpy as np
import json
from transformer_dataset import ConsistencySequenceDataset
from transformer_model import ConsistencyTransformer

def compute_per_token_recon_error(model, dataset, device):
    model.eval()
    batch = dataset[0]
    input_tokens = batch["input_tokens"].unsqueeze(0).to(device)   # (1, N, D)
    target_tokens = batch["target_tokens"].unsqueeze(0).to(device) # (1, N, D)

    seq_len = input_tokens.shape[1]
    recon_errors = []

    for i in range(seq_len):
        masked = input_tokens.clone()
        masked[0, i] = 0.0  # mask the i-th token

        with torch.no_grad():
            outputs = model(masked, torch.tensor([[i]]).to(device))
            pred = outputs["reconstructed_tokens"][0, 0]  # (D,)
            true = target_tokens[0, i]
            error = torch.nn.functional.mse_loss(pred, true).item()
            recon_errors.append(error)

    return recon_errors

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    root_dir = "data"
    identity = "Angelina Jolie"
    embedding_dim = 384
    token_dim = 394  # 384 emb + 7 pose + 3 scalar
    checkpoint_path = f"checkpoints/{identity.replace(' ', '_')}_transformer.pth"
    output_path = os.path.join(root_dir, "metadata", identity, "filtered_images.txt")
    keep_ratio = 0.7  # top 70% consistency

    # Load dataset
    dataset = ConsistencySequenceDataset(
        root_dir=root_dir,
        identity_name=identity,
        embedding_dim=embedding_dim,
        max_seq_len=100,
        mask_ratio=0.3  # used only for training, ignored here
    )

    # Load model
    model = ConsistencyTransformer(
        input_dim=token_dim,
        model_dim=256,
        num_layers=4,
        num_heads=4,
        use_pose_suggestion=True
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    # Compute reconstruction consistency per token
    recon_errors = compute_per_token_recon_error(model, dataset, device)
    image_names = dataset.image_names[:len(recon_errors)]

    # Rank and keep top-k
    sorted_pairs = sorted(zip(image_names, recon_errors), key=lambda x: x[1])  # low error = high quality
    top_k = int(len(sorted_pairs) * keep_ratio)
    filtered_names = [name for name, _ in sorted_pairs[:top_k]]

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for name in filtered_names:
            f.write(name + "\n")

    print(f"[✔] Filtered {len(filtered_names)} / {len(image_names)} images")
    print(f"[→] Saved to: {output_path}")

if __name__ == "__main__":
    main()
