import os
import json
import numpy as np
import argparse

def parse_colmap_to_json(sparse_dir, output_json_path):
    """
    Parse COLMAP sparse reconstruction to a simpler JSON format with camera poses.
    
    Args:
        sparse_dir: Directory containing COLMAP sparse reconstruction in text format
        output_json_path: Output path for JSON file
    """
    # Check if the input directory exists
    if not os.path.exists(sparse_dir):
        raise ValueError(f"Directory {sparse_dir} does not exist")
    
    # Check if we're directly in a text model directory (with cameras.txt and images.txt)
    if os.path.exists(os.path.join(sparse_dir, "cameras.txt")) and os.path.exists(os.path.join(sparse_dir, "images.txt")):
        model_dir = sparse_dir
        print(f"Using model in {model_dir}")
    else:
        # Try to find subdirectories with numeric names (traditional COLMAP structure)
        models = [d for d in os.listdir(sparse_dir) if d.isdigit()]
        if not models:
            raise ValueError(f"No COLMAP models found in {sparse_dir} (no numeric subdirectories and no cameras.txt/images.txt files)")
        
        # Use the model with the most images
        largest_model = None
        max_images = 0
        for model in models:
            images_txt = os.path.join(sparse_dir, model, "images.txt")
            with open(images_txt, "r") as f:
                lines = f.readlines()
            num_images = sum(1 for line in lines if line.strip() and not line.startswith("#"))
            num_images = num_images // 2  # Each image takes up 2 lines in the images.txt file
            if num_images > max_images:
                max_images = num_images
                largest_model = model
        
        if largest_model is None:
            largest_model = models[0]  # Fallback to first model if counting fails
        
        model_dir = os.path.join(sparse_dir, largest_model)
        print(f"Using model {largest_model} in {model_dir}")
    
    # Parse cameras.txt
    cameras = {}
    cameras_txt = os.path.join(model_dir, "cameras.txt")
    with open(cameras_txt, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            camera_id = int(parts[0])
            camera_model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            
            cameras[camera_id] = {
                "model": camera_model,
                "width": width,
                "height": height,
                "params": params
            }
    
    # Parse images.txt
    images = {}
    images_txt = os.path.join(model_dir, "images.txt")
    with open(images_txt, "r") as f:
        image_lines = []
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            image_lines.append(line.strip())
        
        # Each image has two lines
        for i in range(0, len(image_lines), 2):
            if i + 1 >= len(image_lines):
                break
                
            # Parse first line with pose information
            parts = image_lines[i].split()
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            image_name = parts[9]
            
            # Convert quaternion to rotation matrix
            R = np.zeros((3, 3))
            
            # Quaternion to rotation matrix conversion
            R[0, 0] = 1 - 2 * qy * qy - 2 * qz * qz
            R[0, 1] = 2 * qx * qy - 2 * qz * qw
            R[0, 2] = 2 * qx * qz + 2 * qy * qw
            
            R[1, 0] = 2 * qx * qy + 2 * qz * qw
            R[1, 1] = 1 - 2 * qx * qx - 2 * qz * qz
            R[1, 2] = 2 * qy * qz - 2 * qx * qw
            
            R[2, 0] = 2 * qx * qz - 2 * qy * qw
            R[2, 1] = 2 * qy * qz + 2 * qx * qw
            R[2, 2] = 1 - 2 * qx * qx - 2 * qy * qy  # Fixed from a1 to 1
            
            # Create 4x4 transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = np.array([tx, ty, tz])
            
            # Extract position (camera center)
            position = -R.T @ np.array([tx, ty, tz])
            
            images[image_name] = {
                "camera_id": camera_id,
                "quaternion": [qw, qx, qy, qz],
                "translation": [tx, ty, tz],
                "position": position.tolist(),
                "transformation_matrix": T.tolist()
            }
    
    # Create the final JSON structure
    output_data = {
        "cameras": cameras,
        "images": images
    }
    
    # Write to JSON file
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved poses for {len(images)} images to {output_json_path}")
    return output_data

def main():
    parser = argparse.ArgumentParser(description='Parse COLMAP output to JSON')
    parser.add_argument('--sparse_dir', required=True, 
                        help='COLMAP sparse reconstruction directory (text format)')
    parser.add_argument('--output_json', required=True,
                        help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Parse COLMAP output to JSON
    parse_colmap_to_json(args.sparse_dir, args.output_json)
    
    print("COLMAP parsing complete!")

if __name__ == "__main__":
    main()