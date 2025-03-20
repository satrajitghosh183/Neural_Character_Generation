import os
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm

# Load DeepLabV3+ model
model = torchvision.models.segmentation.deeplabv3_resnet101(weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
model.eval()

PERSON_CLASS = 15  # COCO person class

def segment_and_save(image_np, save_path, preview=False):
    # Convert to PIL Image
    image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    orig_w, orig_h = image.size

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

    mask = (output_predictions == PERSON_CLASS).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # Add alpha to original
    rgba = cv2.cvtColor(image_np, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    cv2.imwrite(save_path, rgba)

    if preview:
        checkerboard = create_checkerboard(orig_h, orig_w)
        rgba_preview = blend_on_background(rgba, checkerboard)
        combined = np.hstack((image_np, rgba_preview))
        cv2.imshow("Original | Segmented", combined)
        cv2.waitKey(50)


def process_poses(subject_dir, output_base, preview):
    pose_dir = os.path.join(subject_dir, "poses")
    for pose in os.listdir(pose_dir):
        pose_folder = os.path.join(pose_dir, pose)
        save_pose_dir = os.path.join(output_base, "poses", pose)
        os.makedirs(save_pose_dir, exist_ok=True)

        images = [f for f in os.listdir(pose_folder) if f.endswith(".jpg")]
        for img_file in tqdm(images, desc=f"Processing pose: {pose}"):
            img_path = os.path.join(pose_folder, img_file)
            image_np = cv2.imread(img_path)
            save_path = os.path.join(save_pose_dir, img_file.replace(".jpg", ".png"))
            segment_and_save(image_np, save_path, preview)


def process_actions(subject_dir, output_base, preview):
    actions_dir = os.path.join(subject_dir, "actions")
    actions = [f for f in os.listdir(actions_dir) if f.endswith(".mp4")]

    for action_video in actions:
        video_path = os.path.join(actions_dir, action_video)
        action_name = action_video.replace(".mp4", "")
        save_action_dir = os.path.join(output_base, "actions", action_name)
        os.makedirs(save_action_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Processing action: {action_name}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            save_path = os.path.join(save_action_dir, f"{frame_idx:04d}.png")
            segment_and_save(frame, save_path, preview)
            frame_idx += 1
            pbar.update(1)
        cap.release()
        pbar.close()


def create_checkerboard(height, width, square_size=20):
    rows = height // square_size + 1
    cols = width // square_size + 1
    checkerboard = np.zeros((rows * square_size, cols * square_size, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            color = 200 if (i + j) % 2 == 0 else 255
            checkerboard[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = color
    return checkerboard[:height, :width, :]


def blend_on_background(rgba_image, background):
    alpha = rgba_image[:, :, 3:] / 255.0
    rgb = rgba_image[:, :, :3].astype(np.float32)
    bg = background.astype(np.float32)
    out = rgb * alpha + bg * (1 - alpha)
    return out.astype(np.uint8)


def batch_process(subject_id="subject_01", input_base="data/dataset", output_base="data/preprocessed", preview=False):
    subject_dir = os.path.join(input_base, subject_id)
    os.makedirs(os.path.join(output_base, "poses"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "actions"), exist_ok=True)

    process_poses(subject_dir, os.path.join(output_base, subject_id), preview)
    process_actions(subject_dir, os.path.join(output_base, subject_id), preview)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepLabV3+ Segmentation for both poses and action videos")
    parser.add_argument('--subject_id', type=str, required=True, help="Subject ID (e.g., subject_01)")
    parser.add_argument('--preview', action='store_true', help="Enable preview window (optional)")
    args = parser.parse_args()

    batch_process(subject_id=args.subject_id, preview=args.preview)
