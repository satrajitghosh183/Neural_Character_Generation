import argparse
import os
import sys

# === Import internal modules ===
from capture_multi_view import capture_multi_view, capture_action_videos
from segmentation_pipeline import batch_process as run_segmentation
from landmark_pipeline import process_subject as extract_landmarks

# === Optional: Add WildAvatar structuring utility ===
def process_wildavatar(subject_id, wildavatar_base="WildAvatar_Toolbox/data/WildAvatar", output_base="data/dataset"):
    """
    Convert WildAvatar metadata folders to match the format expected by our pipeline.
    This function should map the existing structure (e.g., frames, segmentations) into
    poses/actions folder layout and rename files accordingly.
    """
    import shutil

    src_path = os.path.join(wildavatar_base, subject_id)
    dst_path = os.path.join(output_base, subject_id)

    if not os.path.exists(src_path):
        print(f"❌ WildAvatar subject folder not found: {src_path}")
        return

    os.makedirs(dst_path, exist_ok=True)

    # Example: map smpl_masks to poses/front (simplified)
    pose_target = os.path.join(dst_path, "poses", "front")
    os.makedirs(pose_target, exist_ok=True)

    mask_folder = os.path.join(src_path, "smpl_masks")
    if os.path.exists(mask_folder):
        for i, file in enumerate(sorted(os.listdir(mask_folder))):
            if file.endswith(".png"):
                shutil.copy(os.path.join(mask_folder, file), os.path.join(pose_target, f"{i:03d}.png"))
        print(f"✅ WildAvatar subject '{subject_id}' mapped to pose folder: {pose_target}")
    else:
        print("⚠ SMPL masks not found — skipping.")


def run_pipeline(subject_id, mode, from_wildavatar=False, preview=False):
    """
    Unified entrypoint to run the full avatar data processing pipeline.
    """
    if from_wildavatar:
        print(f"📦 Structuring WildAvatar data for subject: {subject_id}")
        process_wildavatar(subject_id)

    # 1. Capture data (if not from WildAvatar)
    if not from_wildavatar:
        if mode in ['all', 'poses']:
            capture_multi_view(subject_id=subject_id)
        if mode in ['all', 'actions']:
            capture_action_videos(subject_id=subject_id)

    # 2. Segment all captured frames (DeepLabV3)
    print(f"🎯 Running segmentation for {subject_id}")
    run_segmentation(subject_id=subject_id, preview=preview)

    # 3. Extract 2D landmarks using MediaPipe
    print(f"📍 Extracting landmarks for {subject_id}")
    extract_landmarks(subject_id=subject_id)

    print("✅ Full pipeline completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Avatar Data Collection + Processing Pipeline")
    parser.add_argument("--subject_id", type=str, required=True, help="ID for subject (e.g. subject_01)")
    parser.add_argument("--mode", choices=["all", "poses", "actions"], default="all", help="Which data to collect")
    parser.add_argument("--from_wildavatar", action="store_true", help="Use WildAvatar data as input instead of webcam")
    parser.add_argument("--preview", action="store_true", help="Show segmentation previews (optional)")

    args = parser.parse_args()

    run_pipeline(
        subject_id=args.subject_id,
        mode=args.mode,
        from_wildavatar=args.from_wildavatar,
        preview=args.preview
    )
