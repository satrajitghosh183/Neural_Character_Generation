import json
import os

import numpy as np
import torch

from PIL import Image

class NeRFDataset(torch.utils.data.Dataset):
    
    def __init__(self, preprocessed_dir, landmark_dir):
        
        self.rgba_paths = [...]
        self.landmark_paths = [...] # to be filled
        
    def __getitem__(self, index):
        rgba = Image.open(self.rgba_paths[index])
        image = torch.from_numpy(np.array(rgba)[..., :3]/255.0) # RGBA -> RGB [H, W, 3] (to be tested, permutation needed?)
        
        alpha = torch.from_numpy(np.array(rgba)[:, :, 3]/255.0) # [H, W] (question, /255.0?)
        
        with open(self.landmark_paths[index], 'r') as f:
            landmarks = json.load(f)
            
        weight_map = self.generate_weight_map(landmarks, alpha)
        
        return {
            'image': image.permute(2, 0, 1),
            'pose': self.parse_pose(landmarks),
            'weight_map': weight_map
        }
    def generate_weight_map(self, landmarks, alpha):
        # increase weights around landmarks
        # to be implemented
        return alpha
    
    def parse_pose(self, landmarks):
        # to be implemented
        return None