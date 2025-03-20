import os
import cv2
import mediapipe as mp
import json
from tqdm import tqdm
import traceback

# MediaPipe initialization
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
face_mesh_detector = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def extract_landmarks(image_np):
    """Extract body and face landmarks from an image"""
    try:
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Process with pose detector
        results_pose = pose_detector.process(image_rgb)
        
        # Process with face mesh detector
        results_face = face_mesh_detector.process(image_rgb)

        data = {}

        if results_pose.pose_landmarks:
            data['body'] = [{'x': l.x, 'y': l.y, 'z': l.z, 'visibility': l.visibility} 
                          for l in results_pose.pose_landmarks.landmark]
        else:
            data['body'] = None

        if results_face.multi_face_landmarks and len(results_face.multi_face_landmarks) > 0:
            face_landmarks = results_face.multi_face_landmarks[0]
            data['face'] = [{'x': l.x, 'y': l.y, 'z': l.z} 
                          for l in face_landmarks.landmark]
        else:
            data['face'] = None

        return data, results_pose, results_face
    except Exception as e:
        print(f"⚠ Error extracting landmarks: {str(e)}")
        traceback.print_exc()
        return {'body': None, 'face': None}, None, None


def draw_landmarks(image, results_pose, results_face):
    """Draw landmarks on image for visualization"""
    try:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        annotated = image.copy()
        
        # Draw pose landmarks
        if results_pose and results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated, 
                results_pose.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style())
        
        # Draw face landmarks
        if results_face and results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    annotated, 
                    face_landmarks, 
                    mp_face.FACEMESH_TESSELATION,
                    mp_drawing_styles.get_default_face_mesh_tesselation_style())
        
        return annotated
    except Exception as e:
        print(f"⚠ Error drawing landmarks: {str(e)}")
        traceback.print_exc()
        return image


def process_pose_images(subject_dir, output_base):
    """Process pose images from specific folders"""
    # Normalize paths for consistency
    subject_dir = os.path.normpath(subject_dir)
    output_base = os.path.normpath(output_base)
    
    pose_folders = ["front", "left", "right", "up", "down"]
    processed_count = 0
    
    for pose in pose_folders:
        pose_folder = os.path.join(subject_dir, pose)
        print(f"\n➡ Processing pose folder: {pose_folder}")
        
        if not os.path.exists(pose_folder):
            print(f"❌ Folder not found: {pose_folder}")
            continue
            
        print(f"✓ Found folder. Looking for PNG images...")
        
        save_pose_dir = os.path.join(output_base, "poses", pose)
        os.makedirs(save_pose_dir, exist_ok=True)
        print(f"✓ Created output directory: {save_pose_dir}")

        images = [f for f in os.listdir(pose_folder) if f.lower().endswith(".png")]
        if not images:
            print(f"⚠ No PNG images found in: {pose_folder}")
            continue
            
        print(f"✓ Found {len(images)} PNG images")

        for img_file in tqdm(images, desc=f"Landmarking pose: {pose}"):
            try:
                img_path = os.path.join(pose_folder, img_file)
                print(f"  Processing: {img_path}")
                
                image_np = cv2.imread(img_path)
                if image_np is None:
                    print(f"⚠ Failed to load image: {img_path}")
                    continue

                # Get image dimensions for debug
                h, w = image_np.shape[:2]
                print(f"  Image dimensions: {w}x{h}")

                landmarks, res_pose, res_face = extract_landmarks(image_np)

                if landmarks['body'] is None and landmarks['face'] is None:
                    print(f"⚠ No landmarks detected in: {img_path}")
                else:
                    processed_count += 1
                    print(f"  ✓ Landmarks detected")

                # Save JSON output
                save_path = os.path.join(save_pose_dir, img_file.replace(".png", ".json"))
                with open(save_path, "w") as f:
                    json.dump(landmarks, f, indent=4)
                print(f"  ✓ Saved to: {save_path}")

                # GUI preview
                if res_pose is not None or res_face is not None:
                    preview = draw_landmarks(image_np, res_pose, res_face)
                    cv2.imshow("Landmark Preview", preview)
                    key = cv2.waitKey(100)  # Increased wait time
                    if key == 27:  # ESC key
                        print("🛑 ESC pressed, exiting preview.")
                        cv2.destroyAllWindows()
                        return processed_count
            except Exception as e:
                print(f"⚠ Error processing {img_file}: {str(e)}")
                traceback.print_exc()
                continue
                
    return processed_count


def process_action_videos(subject_dir, output_base):
    """Process action/animation frames"""
    # Normalize paths for consistency
    subject_dir = os.path.normpath(subject_dir)
    output_base = os.path.normpath(output_base)
    
    # Check for either actions or poses directory
    actions_dir = os.path.join(subject_dir, "actions")
    poses_dir = os.path.join(subject_dir, "poses")
    
    target_dir = None
    dir_type = None
    
    if os.path.exists(actions_dir) and os.path.isdir(actions_dir):
        target_dir = actions_dir
        dir_type = "actions"
    elif os.path.exists(poses_dir) and os.path.isdir(poses_dir):
        target_dir = poses_dir
        dir_type = "poses"
    
    if target_dir is None:
        print(f"\n⚠ Neither 'actions' nor 'poses' directory found in: {subject_dir}")
        return 0
    
    print(f"\n➡ Processing {dir_type} directory: {target_dir}")
    
    # Check if there are direct subfolders
    action_subfolders = [f for f in os.listdir(target_dir) 
                        if os.path.isdir(os.path.join(target_dir, f))]
    
    if not action_subfolders:
        print(f"⚠ No {dir_type} subfolders found in: {target_dir}")
        return 0
        
    print(f"✓ Found {len(action_subfolders)} {dir_type} folders: {', '.join(action_subfolders)}")
    
    processed_count = 0

    for action in action_subfolders:
        action_folder = os.path.join(target_dir, action)
        print(f"➡ Processing {dir_type} folder: {action_folder}")
        
        save_action_dir = os.path.join(output_base, dir_type, action)
        os.makedirs(save_action_dir, exist_ok=True)
        print(f"✓ Created output directory: {save_action_dir}")

        frames = [f for f in os.listdir(action_folder) if f.lower().endswith(".png")]
        if not frames:
            print(f"⚠ No frames found in: {action_folder}")
            continue
            
        print(f"✓ Found {len(frames)} frames")

        for frame_file in tqdm(frames, desc=f"Landmarking {dir_type}: {action}"):
            try:
                frame_path = os.path.join(action_folder, frame_file)
                print(f"  Processing: {frame_path}")
                
                image_np = cv2.imread(frame_path)
                if image_np is None:
                    print(f"⚠ Failed to load frame: {frame_path}")
                    continue

                # Get image dimensions for debug
                h, w = image_np.shape[:2]
                print(f"  Image dimensions: {w}x{h}")

                landmarks, res_pose, res_face = extract_landmarks(image_np)
                
                if landmarks['body'] is not None or landmarks['face'] is not None:
                    processed_count += 1
                    print(f"  ✓ Landmarks detected")
                else:
                    print(f"  ⚠ No landmarks detected")

                # Save JSON output
                save_path = os.path.join(save_action_dir, frame_file.replace(".png", ".json"))
                with open(save_path, "w") as f:
                    json.dump(landmarks, f, indent=4)
                print(f"  ✓ Saved to: {save_path}")

                # GUI preview
                if res_pose is not None or res_face is not None:
                    preview = draw_landmarks(image_np, res_pose, res_face)
                    cv2.imshow("Landmark Preview", preview)
                    key = cv2.waitKey(100)  # Increased wait time
                    if key == 27:  # ESC key
                        print("🛑 ESC pressed, exiting preview.")
                        cv2.destroyAllWindows()
                        return processed_count
            except Exception as e:
                print(f"⚠ Error processing {frame_file}: {str(e)}")
                traceback.print_exc()
                continue
                
    return processed_count


def batch_landmark(subject_id="subject_01", input_base="data/preprocessed", output_base="data/landmarks"):
    """Main processing function to handle landmarking of images"""
    print("\n" + "="*80)
    print("LANDMARK EXTRACTION PIPELINE")
    print("="*80)
    
    try:
        # Handle both absolute and relative paths
        if os.path.isabs(subject_id):
            subject_dir = subject_id
        else:
            subject_dir = os.path.join(input_base, subject_id)
        
        # Normalize path for consistent display
        subject_dir = os.path.normpath(subject_dir)
        
        print(f"\n🔍 Looking for subject directory: {subject_dir}")
        if not os.path.exists(subject_dir):
            print(f"❌ Subject directory not found: {subject_dir}")
            return False
            
        print(f"✓ Found subject directory")
        contents = os.listdir(subject_dir)
        print(f"📂 Contents: {', '.join(contents[:10])}" + 
              (f" and {len(contents)-10} more..." if len(contents) > 10 else ""))
        
        # Create output directory with subject name from input path
        subject_name = os.path.basename(os.path.normpath(subject_dir))
        save_dir = os.path.join(output_base, subject_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"✓ Created output directory: {save_dir}")

        # Process images
        pose_count = process_pose_images(subject_dir, save_dir)
        action_count = process_action_videos(subject_dir, save_dir)
        
        total_processed = pose_count + action_count
        
        print("\n" + "="*80)
        print(f"✅ PROCESSING COMPLETE")
        print(f"  - Total images processed: {total_processed}")
        print(f"  - Pose images: {pose_count}")
        print(f"  - Action frames: {action_count}")
        print(f"  - Results saved to: {save_dir}")
        print("="*80)
        
        return total_processed > 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pose + Face Landmark Extraction with GUI Preview")
    parser.add_argument('--subject_id', type=str, required=True, 
                        help="Subject ID (e.g., subject_01) or path to subject directory")
    parser.add_argument('--input_base', type=str, default="data/preprocessed",
                        help="Base directory containing subject folders (default: data/preprocessed)")
    parser.add_argument('--output_base', type=str, default="data/landmarks",
                        help="Base directory for output (default: data/landmarks)")
    parser.add_argument('--no_gui', action='store_true',
                        help="Disable GUI preview")
    args = parser.parse_args()

    # Modify visualization based on no_gui flag
    if args.no_gui:
        def cv2_modified_imshow(*args, **kwargs):
            pass
        cv2.imshow = cv2_modified_imshow
        wait_time = 1
    else:
        wait_time = 100
    
    try:
        success = batch_landmark(
            subject_id=args.subject_id,
            input_base=args.input_base,
            output_base=args.output_base
        )
        
        cv2.destroyAllWindows()
        
        if not success:
            print("\n❌ Processing failed. Please check your directory structure and file paths.")
            print("Expected structure:")
            print("  - <input_base>/<subject_id>/{front,left,right,up,down}/*.png")
            print("  - <input_base>/<subject_id>/actions/*/*.png")
            print("  - <input_base>/<subject_id>/poses/*/*.png")
            exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user")
        cv2.destroyAllWindows()
        exit(0)
    except Exception as e:
        print(f"\n❌ Unhandled error: {str(e)}")
        traceback.print_exc()
        cv2.destroyAllWindows()
        exit(1)