#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
def parse_args():
    p = argparse.ArgumentParser(
        description="Recursively segment 'person' using DeepLabV3"
    )
    p.add_argument('--input_dir', required=True,
                   help="Top‑level folder of images (will recurse into subfolders)")
    p.add_argument('--mask_dir',  required=True,
                   help="Where to write binary masks (preserves subfolder structure)")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.mask_dir, exist_ok=True)

    # Load DeepLabV3 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   

    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(device).eval()


    # Preprocessing pipeline
    tf = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    # Recursively collect image files
    files = []
    for root, _, fnames in os.walk(args.input_dir):
        for fn in fnames:
            if fn.lower().endswith(('.jpg','jpeg','.png')):
                rel = os.path.relpath(os.path.join(root, fn), args.input_dir)
                files.append(rel)
    files.sort()

    for rel in files:
        in_path  = os.path.join(args.input_dir,  rel)
        out_path = os.path.join(args.mask_dir,     rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        img = Image.open(in_path).convert('RGB')
        inp = tf(img).unsqueeze(0).to(device)              # [1,3,H,W]
        with torch.no_grad():
            out = model(inp)['out'][0]                     # [C, H, W]
            mask = out.argmax(0).cpu().numpy() == 15       # class 15 = person

        # Save binary mask
        mask_img = Image.fromarray((mask*255).astype(np.uint8))
        # ensure .png extension
        base, _ = os.path.splitext(out_path)
        mask_img.save(base + '.png')
        print(f"[✔] Segmented {rel} → {base + '.png'}")

if __name__ == '__main__':
    main()
