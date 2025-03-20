import os
import cv2
import mediapipe as mp
import json
from tqdm import tqdm

# MediaPipe initialization
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose_detector = mp_pose.Pose(static_image_mode=True)
face_mesh_detector = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

def extract_landmarks(image_np):
    """Extract body and face landmarks from an image"""
    results_pose = pose_detector.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    results_face = face_mesh_detector.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    data = {}

    if results_pose.pose_landmarks:
        data['body'] = [{'x': l.x, 'y': l.y, 'z': l.z, 'visibility': l.visibility} for l in results_pose.pose_landmarks.landmark]
    else:
        data['body'] = None

    if results_face.multi_face_landmarks:
        face_landmarks = results_face.multi_face_landmarks[0]
        data['face'] = [{'x': l.x, 'y': l.y, 'z': l.z} for l in face_landmarks.landmark]
    else:
        data['face'] = None

    return data, results_pose, results_face


def draw_landmarks(image, results_pose, results_face):
    mp_drawing = mp.solutions.drawing_utils
    annotated = image.copy()
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                annotated, face_landmarks, mp_face.FACEMESH_TESSELATION)
    return annotated


def process_pose_images(subject_dir, output_base):
    pose_folders = ["front", "left", "right", "up", "down"]
    for pose in pose_folders:
        pose_folder = os.path.join(subject_dir, pose)
        save_pose_dir = os.path.join(output_base, "poses", pose)
        os.makedirs(save_pose_dir, exist_ok=True)

        images = [f for f in os.listdir(pose_folder) if f.endswith(".png")]
        for img_file in tqdm(images, desc=f"Landmarking pose: {pose}"):
            img_path = os.path.join(pose_folder, img_file)
            image_np = cv2.imread(img_path)
            if image_np is None:
                print(f"⚠ Failed to load image: {img_path}")
                continue
            landmarks, res_pose, res_face = extract_landmarks(image_np)
            if landmarks['body'] is None and landmarks['face'] is None:
                print(f"⚠ No landmarks detected in: {img_path}")
            save_path = os.path.join(save_pose_dir, img_file.replace(".png", ".json"))
            with open(save_path, "w") as f:
                json.dump(landmarks, f, indent=4)

            # GUI preview
            preview = draw_landmarks(image_np, res_pose, res_face)
            cv2.imshow("Landmark Preview", preview)
            if cv2.waitKey(50) == 27:  # ESC to quit early
                cv2.destroyAllWindows()
                return


def process_action_videos(subject_dir, output_base):
    actions_dir = os.path.join(subject_dir, "actions")
    for action in os.listdir(actions_dir):
        action_folder = os.path.join(actions_dir, action)
        save_action_dir = os.path.join(output_base, "actions", action)
        os.makedirs(save_action_dir, exist_ok=True)

        frames = [f for f in os.listdir(action_folder) if f.endswith(".png")]
        for frame_file in tqdm(frames, desc=f"Landmarking action: {action}"):
            frame_path = os.path.join(action_folder, frame_file)
            image_np = cv2.imread(frame_path)
            landmarks, res_pose, res_face = extract_landmarks(image_np)
            save_path = os.path.join(save_action_dir, frame_file.replace(".png", ".json"))
            with open(save_path, "w") as f:
                json.dump(landmarks, f, indent=4)

            # GUI preview
            preview = draw_landmarks(image_np, res_pose, res_face)
            cv2.imshow("Landmark Preview", preview)
            if cv2.waitKey(50) == 27:
                cv2.destroyAllWindows()
                return


def batch_landmark(subject_id="subject_01", input_base="data/preprocessed", output_base="data/landmarks"):
    subject_dir = os.path.join(input_base, subject_id)
    save_dir = os.path.join(output_base, subject_id)
    os.makedirs(save_dir, exist_ok=True)

    process_pose_images(subject_dir, save_dir)
    process_action_videos(subject_dir, save_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pose + Face Landmark Extraction with GUI Preview")
    parser.add_argument('--subject_id', type=str, required=True, help="Subject ID (e.g., subject_01)")
    args = parser.parse_args()

    batch_landmark(subject_id=args.subject_id)
    cv2.destroyAllWindows()