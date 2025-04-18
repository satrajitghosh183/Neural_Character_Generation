#!/usr/bin/env python3
"""
pose_prediction.py: MLP model for predicting camera extrinsics based on 
refined image features.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
import argparse
from pathlib import Path

class PoseMLP(nn.Module):
    """
    MLP for predicting camera poses from image features.
    Predicts position (x, y, z) and rotation quaternion (qw, qx, qy, qz).
    """
    def __init__(self, feature_dim=512, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Feature projection
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Position prediction head
        self.position_head = nn.Linear(hidden_dim // 2, 3)
        
        # Quaternion prediction head
        self.quaternion_head = nn.Linear(hidden_dim // 2, 4)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, features):
        """
        Args:
            features: Image features [batch_size, feature_dim]
        
        Returns:
            position: Predicted position [batch_size, 3]
            quaternion: Predicted quaternion [batch_size, 4]
        """
        # Extract features
        x = self.mlp(features)
        
        # Predict position
        position = self.position_head(x)
        
        # Predict quaternion
        quaternion = self.quaternion_head(x)
        
        # Normalize quaternion
        quaternion = F.normalize(quaternion, p=2, dim=1)
        
        return position, quaternion

class PoseDataset(Dataset):
    """
    Dataset for training the pose prediction MLP.
    """
    def __init__(self, features_dir, poses_json_path, use_refined_features=True):
        super().__init__()
        self.features_dir = features_dir
        
        # Load poses
        with open(poses_json_path, 'r') as f:
            poses_data = json.load(f)
        
        self.images = []
        self.positions = []
        self.quaternions = []
        
        # Extract image IDs and poses
        for image_name, image_data in poses_data["images"].items():
            # Get image ID from filename (without extension)
            image_id = os.path.splitext(image_name)[0]
            
            # Check if feature exists
            feature_path = os.path.join(features_dir, f"{image_id}.npy")
            if os.path.exists(feature_path):
                # Extract camera pose (position and quaternion)
                position = image_data["position"]
                quaternion = image_data["quaternion"]
                
                self.images.append(image_id)
                self.positions.append(position)
                self.quaternions.append(quaternion)
        
        print(f"Loaded {len(self.images)} images with valid poses and features")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_id = self.images[idx]
        position = self.positions[idx]
        quaternion = self.quaternions[idx]
        
        # Load feature
        feature_path = os.path.join(self.features_dir, f"{image_id}.npy")
        feature = np.load(feature_path).astype(np.float32)
        
        return {
            "image_id": image_id,
            "feature": torch.tensor(feature, dtype=torch.float32),
            "position": torch.tensor(position, dtype=torch.float32),
            "quaternion": torch.tensor(quaternion, dtype=torch.float32)
        }

def train_pose_mlp(features_dir, poses_json_path, output_dir, 
                  batch_size=32, num_epochs=100, learning_rate=1e-4,
                  feature_dim=512, hidden_dim=1024, dropout=0.1,
                  val_split=0.1):
    """
    Train the pose prediction MLP.
    
    Args:
        features_dir: Directory containing image features
        poses_json_path: Path to camera poses JSON file
        output_dir: Directory to save model and predictions
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        feature_dim: Dimension of image features
        hidden_dim: Hidden dimension of MLP
        dropout: Dropout rate
        val_split: Validation split ratio
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    dataset = PoseDataset(features_dir, poses_json_path)
    
    # Split dataset into train and validation
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    print(f"Training on {train_size} samples, validating on {val_size} samples")
    
    # Create model
    model = PoseMLP(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_pos_loss = 0.0
        train_quat_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            features = batch["feature"].to(device)
            target_pos = batch["position"].to(device)
            target_quat = batch["quaternion"].to(device)
            
            # Forward pass
            pred_pos, pred_quat = model(features)
            
            # Compute losses
            pos_loss = F.mse_loss(pred_pos, target_pos)
            quat_loss = 1.0 - torch.sum(pred_quat * target_quat, dim=1).mean()  # Quaternion distance
            
            # Total loss (weighted sum)
            loss = pos_loss + 0.5 * quat_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pos_loss += pos_loss.item()
            train_quat_loss += quat_loss.item()
            
            progress_bar.set_postfix({
                "loss": loss.item(),
                "pos_loss": pos_loss.item(),
                "quat_loss": quat_loss.item()
            })
        
        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_pos_loss = train_pos_loss / len(train_dataloader)
        avg_train_quat_loss = train_quat_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_pos_loss = 0.0
        val_quat_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                features = batch["feature"].to(device)
                target_pos = batch["position"].to(device)
                target_quat = batch["quaternion"].to(device)
                
                # Forward pass
                pred_pos, pred_quat = model(features)
                
                # Compute losses
                pos_loss = F.mse_loss(pred_pos, target_pos)
                quat_loss = 1.0 - torch.sum(pred_quat * target_quat, dim=1).mean()
                
                # Total loss
                loss = pos_loss + 0.5 * quat_loss
                
                val_loss += loss.item()
                val_pos_loss += pos_loss.item()
                val_quat_loss += quat_loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_pos_loss = val_pos_loss / len(val_dataloader)
        avg_val_quat_loss = val_quat_loss / len(val_dataloader)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f} (Pos: {avg_train_pos_loss:.4f}, Quat: {avg_train_quat_loss:.4f}), "
              f"Val Loss: {avg_val_loss:.4f} (Pos: {avg_val_pos_loss:.4f}, Quat: {avg_val_quat_loss:.4f})")
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss
            }, os.path.join(output_dir, "pose_mlp_best.pt"))
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss
            }, os.path.join(output_dir, f"pose_mlp_epoch{epoch+1}.pt"))
    
    # Generate predictions for all images
    model.load_state_dict(torch.load(os.path.join(output_dir, "pose_mlp_best.pt"))['model_state_dict'])
    generate_pose_predictions(model, dataset, output_dir, device, batch_size)
    
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

def generate_pose_predictions(model, dataset, output_dir, device, batch_size=32):
    """
    Generate pose predictions for all images and save to JSON.
    
    Args:
        model: Trained pose prediction model
        dataset: Dataset containing images and features
        output_dir: Directory to save predictions
        device: Device to run the model on
        batch_size: Batch size for inference
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create dataloader for inference
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    # Dictionary to store predictions
    predictions = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating pose predictions"):
            image_ids = batch["image_id"]
            features = batch["feature"].to(device)
            
            # Generate predictions
            pred_positions, pred_quaternions = model(features)
            
            # Convert to numpy
            pred_positions = pred_positions.cpu().numpy()
            pred_quaternions = pred_quaternions.cpu().numpy()
            
            # Store predictions
            for i, image_id in enumerate(image_ids):
                predictions[image_id] = {
                    "position": pred_positions[i].tolist(),
                    "quaternion": pred_quaternions[i].tolist()
                }
    
    # Save predictions to JSON
    output_path = os.path.join(output_dir, "predicted_poses.json")
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Pose predictions saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Train pose prediction MLP')
    parser.add_argument('--features_dir', type=str, default='data/metadata/mtcm_feats/refined_features',
                        help='Directory containing image features')
    parser.add_argument('--poses_json', type=str, default='data/metadata/poses.json',
                        help='Path to camera poses JSON file')
    parser.add_argument('--output_dir', type=str, default='data/metadata/pose_prediction',
                        help='Directory to save model and predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--feature_dim', type=int, default=512,
                        help='Dimension of image features')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                        help='Hidden dimension of MLP')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    train_pose_mlp(
        args.features_dir,
        args.poses_json,
        args.output_dir,
        args.batch_size,
        args.num_epochs,
        args.learning_rate,
        args.feature_dim,
        args.hidden_dim,
        args.dropout,
        args.val_split
    )

if __name__ == "__main__":
    main()