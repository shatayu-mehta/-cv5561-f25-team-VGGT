"""
Method A: YOLO + SAM Segmentation with 3D Lifting

This script implements object detection and segmentation using:
1. YOLO-World for open-vocabulary object detection
2. SAM (Segment Anything Model) for precise mask generation
3. 3D lifting using VGGT depth estimates and camera poses

Workflow:
- Detect objects of interest in each frame using YOLO-World
- Generate precise segmentation masks with SAM
- Unproject masked pixels to 3D using depth maps
- Apply depth filtering to remove background leakage
- Merge multi-view point clouds with statistical cleaning

Usage:
    python Yolo_SAM.py
    
Input: RGB images in my_datasets/<scene>/images/
Output: Segmented 3D point clouds and debug visualizations
"""

import os
import re
import json
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from torchvision import transforms as TF
from ultralytics import YOLOWorld, SAM
from tqdm import tqdm
import glob
from collections import Counter

# Configuration parameters
IGNORE_CLASSES = ["person", "man", "woman", "handbag", "backpack"]  # Classes to exclude from detection
JSON_CONFIG_PATH = "scene_priors.json"  # Scene-specific object categories

# ---------------------------------------------------------
# 1. Helper Functions (Loading Priors)
# ---------------------------------------------------------
def load_scene_priors(json_path):
    """
    Load scene-specific object categories from JSON configuration.
    
    The scene priors map scene types (e.g., "office", "kitchen") to lists
    of objects commonly found in those scenes. This guides YOLO-World to
    detect relevant objects.
    
    Args:
        json_path: Path to scene_priors.json configuration file
        
    Returns:
        Dictionary mapping scene categories to object lists
    """
    if os.path.exists(json_path):
        print(f" [i] Loading scene priors from: {json_path}")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if "__comment__" in data:
                    del data["__comment__"]
                return data
        except Exception as e:
            print(f" [!] Error loading JSON: {e}. Using fallback.")
    else:
        print(f" [!] Warning: '{json_path}' not found. Using minimal fallback.")
    
    return {
        "office": ["chair", "desk", "monitor", "table"],
        "apple": ["apple", "fruit"],
        "dishes": ["plate", "bowl", "cup"],
        "excavator": ["excavator", "toy excavator"]
    }

SCENE_PRIORS = load_scene_priors(JSON_CONFIG_PATH)

def preprocess_image_vggt_style(image_path, target_size=518):
    """
    Preprocess images to match VGGT's expected format.
    
    Ensures images are:
    - RGB format (handles RGBA by compositing on white background)
    - Resized to target width with height as multiple of 14
    - Center-cropped if taller than target size
    
    Args:
        image_path: Path to input image
        target_size: Target width/height in pixels (default: 518)
        
    Returns:
        Numpy array (H, W, 3) in RGB format, uint8
    """
    try:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        
        to_tensor = TF.ToTensor()
        img_tensor = to_tensor(img) 

        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            img_tensor = img_tensor[:, start_y : start_y + target_size, :]

        img_numpy = img_tensor.permute(1, 2, 0).numpy()
        img_numpy = (img_numpy * 255).astype(np.uint8)
        return img_numpy
    except Exception as e:
        return None

def process_dataset(dataset_root_dir):
    """
    Scan dataset directory and load all images.
    
    Walks through the directory structure looking for 'images' folders,
    preprocesses all image files, and creates a mapping of datasets to
    their image paths.
    
    Args:
        dataset_root_dir: Root directory containing scene subdirectories
        
    Returns:
        Tuple of (processed_images, dataset_map) where:
        - processed_images: dict mapping image paths to numpy arrays
        - dataset_map: dict mapping dataset directories to lists of image paths
    """
    processed_images = {}
    dataset_map = {} 
    
    print(f"Scanning '{dataset_root_dir}'...")
    for root, dirs, files in os.walk(dataset_root_dir):
        if os.path.basename(root) == 'images':
            dataset_dir = os.path.dirname(root)
            dataset_map[dataset_dir] = []
            
            for file in sorted(files):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    full_path = os.path.join(root, file)
                    img_array = preprocess_image_vggt_style(full_path)
                    if img_array is not None:
                        processed_images[full_path] = img_array
                        dataset_map[dataset_dir].append(full_path)
                        
    return processed_images, dataset_map

# Scene Identification: Automatically determine target objects from folder names

def clean_object_name(obj_str):
    """
    Clean object name by removing brackets, numbers, and special characters.
    
    Args:
        obj_str: Raw object name string
        
    Returns:
        Cleaned lowercase string
    """
    obj_str = re.sub(r"[\(\[].*?[\)\]]", "", obj_str)
    obj_str = re.sub(r"\d+", "", obj_str)
    obj_str = obj_str.replace(":", "").replace("_", " ")
    return obj_str.strip().lower()

def build_reverse_index(priors_dict):
    """
    Build reverse lookup from object names to scene categories.
    
    Args:
        priors_dict: Scene priors dictionary
        
    Returns:
        Dictionary mapping object names to list of parent categories
    """
    index = {}
    for category, items in priors_dict.items():
        if category not in index: index[category] = []
        index[category].append(category)
        for item in items:
            item_clean = item.lower().strip()
            if item_clean not in index:
                index[item_clean] = []
            if category not in index[item_clean]:
                index[item_clean].append(category)
    return index

def get_scene_objects(dataset_dir):
    """
    Automatically identify target objects for segmentation.
    
    Uses two strategies:
    1. Folder name analysis: Match tokens against scene_priors.json
    2. Detection log analysis: Find frequently detected objects
    
    Args:
        dataset_dir: Path to dataset directory
        
    Returns:
        List of target object names for YOLO-World detection
    """
    ds_name = os.path.basename(dataset_dir).lower()
    tokens = re.split(r'[_\-\s\.]+', ds_name)
    reverse_index = build_reverse_index(SCENE_PRIORS)
    
    detected_targets = []
    triggered_categories = set()
    
    print(f" [?] Analyzing Folder Name: '{ds_name}'")

    for token in tokens:
        check_tokens = [token]
        if token.endswith('s'): check_tokens.append(token[:-1])
        
        for t in check_tokens:
            if t in reverse_index:
                categories = reverse_index[t]
                for cat in categories:
                    if cat not in triggered_categories:
                        print(f"  -> Match found: '{t}' triggers category '{cat}'")
                        detected_targets.extend(SCENE_PRIORS[cat])
                        triggered_categories.add(cat)

    log_path = os.path.join(dataset_dir, "detected_objects.txt")
    if os.path.exists(log_path):
        object_counts = Counter()
        with open(log_path, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) > 1:
                    objs = parts[1].split(',')
                    for obj in objs:
                        clean_obj = clean_object_name(obj)
                        if clean_obj and clean_obj not in IGNORE_CLASSES and len(clean_obj) > 2:
                            object_counts[clean_obj] += 1
        
        min_freq = 15 if detected_targets else 3
        for obj, count in object_counts.most_common():
            if count >= min_freq and obj not in detected_targets:
                print(f" [2] Log Discovery: Adding frequent object '{obj}' ({count} detections)")
                detected_targets.append(obj)

    if detected_targets:
        unique_targets = list(set(detected_targets))
        print(f" [✓] Final Target List ({len(unique_targets)} items)")
        return unique_targets

    print(f" [WARN] No match found in folder name or logs. Using generic fallback.")
    return ["chair", "table", "box", "door", "floor"]

# Mask Generation: YOLO detection + SAM segmentation

def generate_scene_masks(dataset_dir, processed_images, target_objects):
    """
    Generate segmentation masks for target objects in all frames.
    
    Pipeline:
    1. Run YOLO-World detection with confidence threshold (0.11)
    2. For each detection, generate precise mask using SAM
    3. Combine masks from all detections into single binary mask
    4. Save cutout visualizations and detection logs
    
    Args:
        dataset_dir: Dataset directory path
        processed_images: Dictionary of preprocessed images
        target_objects: List of object classes to detect
        
    Returns:
        Dictionary mapping image paths to binary masks (H, W) uint8
    """
    masks_dict = {}
    if not processed_images or not target_objects:
        return {}

    print(f"Loading models to isolate: {target_objects}")
    try:
        detector = YOLOWorld('yolov8l-world.pt') 
        segmenter = SAM('sam_b.pt') 
        detector.set_classes(target_objects)
    except Exception as e:
        print(f"Model load failed: {e}")
        return {}

    sorted_paths = sorted(processed_images.keys())
    
    log_file_path = os.path.join(dataset_dir, "debug_frame_detections.txt")
    cutout_dir = os.path.join(dataset_dir, "cutouts")
    os.makedirs(cutout_dir, exist_ok=True)
    
    log_lines = []
    
    for filepath in tqdm(sorted_paths, desc="Extracting & Saving Cutouts"):
        img_array = processed_images[filepath]
        h, w = img_array.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        filename = os.path.basename(filepath)
        
        # --- MODIFICATION 1: CONFIDENCE SCORE FILTERING ---
        # Explicitly raised to 0.11 to reject low-probability garbage
        det = detector.predict(img_array, verbose=False, conf=0.11)[0]
        
        detected_items = []
        if det.boxes:
            for i, cls_id in enumerate(det.boxes.cls):
                cls_name = detector.names[int(cls_id)]
                conf = float(det.boxes.conf[i])
                detected_items.append(f"{cls_name} ({conf:.2f})")

            # Run Segmentation
            sam = segmenter(img_array, bboxes=det.boxes.xyxy, verbose=False)[0]
            if sam.masks:
                for m in sam.masks.data.cpu().numpy():
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
                    combined_mask = np.maximum(combined_mask, (m > 0.5).astype(np.uint8))
        else:
             detected_items.append("(no detection)")
        
        log_lines.append(f"{filename}: {', '.join(detected_items)}")
        masks_dict[filepath] = combined_mask

        # VISUALIZATION
        if np.sum(combined_mask) > 0:
            masked_img = cv2.bitwise_and(img_array, img_array, mask=combined_mask)
            save_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)
        else:
            save_img = np.zeros((h, w, 3), dtype=np.uint8)
            
        cv2.imwrite(os.path.join(cutout_dir, filename), save_img)
    
    with open(log_file_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f" [i] Logs: {log_file_path}")
    print(f" [i] Cutouts: {cutout_dir}")
    
    return masks_dict

# 3D Lifting: Unproject masked pixels to world coordinates

def get_vggt_files(dataset_dir):
    """
    Locate VGGT output files (depth maps, poses, intrinsics).
    
    Args:
        dataset_dir: Dataset directory path
        
    Returns:
        Tuple of (depth_files, pose_files, intrinsic_files) sorted lists
    """
    out_dir = os.path.join(dataset_dir, "outputs")
    depths = sorted(glob.glob(os.path.join(out_dir, "depths", "*.npy")))
    if not depths:
        depths = sorted(glob.glob(os.path.join(out_dir, "depths", "*.png")))
    poses = sorted(glob.glob(os.path.join(out_dir, "poses", "*extrinsic.txt")))
    intrinsics = sorted(glob.glob(os.path.join(out_dir, "poses", "*intrinsic.txt")))
    return depths, poses, intrinsics

def clean_point_cloud(pcd):
    """
    Remove statistical outliers from point cloud.
    
    Args:
        pcd: Open3D point cloud
        
    Returns:
        Cleaned point cloud with outliers removed
    """
    print("  > Cleaning noise (Statistical Outlier Removal)...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=75, std_ratio=1.0)
    pcd_clean = pcd.select_by_index(ind)
    print(f"  > Removed {len(pcd.points) - len(pcd_clean.points)} noise points.")
    return pcd_clean

def lift_single_dataset(dataset_dir, image_paths, masks_dict, processed_images):
    """
    Lift 2D segmentation masks to 3D point clouds using depth and poses.
    
    Key features:
    - Depth filtering: Only keeps points near object's mean depth (within 0.5m)
    - This prevents background pixels from leaking through holes in masks
    - Merges points from all frames into single point cloud
    - Applies statistical outlier removal for final cleaning
    
    Args:
        dataset_dir: Dataset directory path
        image_paths: List of image file paths
        masks_dict: Dictionary mapping images to segmentation masks
        processed_images: Dictionary of preprocessed image arrays
        
    Returns:
        Open3D point cloud with cleaned 3D points and colors, or None if failed
    """
    sorted_image_paths = sorted(image_paths)
    depth_files, pose_files, intrin_files = get_vggt_files(dataset_dir)
    
    if not depth_files:
        return None

    num_frames = min(len(sorted_image_paths), len(depth_files), len(pose_files))
    points = []
    colors = []

    for i in range(num_frames):
        img_path = sorted_image_paths[i]
        mask = masks_dict.get(img_path, None)
        
        if mask is None or np.sum(mask) < 50: continue 
        
        img_rgb = processed_images[img_path]
        
        if depth_files[i].endswith('.npy'):
            depth = np.load(depth_files[i])
        else:
            depth = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED).astype(float) / 1000.0
            if depth.max() > 100: depth /= 1000.0 
            
        pose = np.loadtxt(pose_files[i])
        K = np.loadtxt(intrin_files[i]) if i < len(intrin_files) else np.loadtxt(os.path.join(dataset_dir, "intrinsics.txt"))

        h, w = depth.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if img_rgb.shape[:2] != (h, w):
            img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        # Apply base validity checks: must be masked, within sensor range
        valid_base = (mask > 0) & (depth > 0.1) & (depth < 10.0)
        
        # Depth filtering: Remove background pixels seen through holes in mask
        # Calculate mean depth of object in this frame
        object_depths = depth[valid_base]
        
        if len(object_depths) < 10:
            continue

        mean_z = np.mean(object_depths)
        tolerance = 0.5  # 50cm tolerance around object depth
        
        # Only keep pixels close to object's mean depth plane
        # This filters out walls/floors visible through segmentation errors
        valid = valid_base & (depth > mean_z - tolerance) & (depth < mean_z + tolerance)

        if np.sum(valid) < 10:
            continue

        ys, xs = np.where(valid)
        z = depth[ys, xs]
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        
        # Unproject to camera space using pinhole camera model
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        points_cam = np.stack([x, y, z], axis=-1)
        
        # Ensure pose is 4x4 transformation matrix
        if pose.shape == (3, 4):
            pose = np.vstack([pose, [0,0,0,1]])
            
        try:
            # Transform from camera to world coordinates
            c2w = np.linalg.inv(pose)
            world_pts = (points_cam @ c2w[:3, :3].T) + c2w[:3, 3]
            cols = img_rgb[ys, xs] / 255.0
            points.append(world_pts)
            colors.append(cols)
        except:
            continue

    if not points: return None
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(points, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors, axis=0))
    
    return clean_point_cloud(pcd)

# Main execution

def main():
    """
    Main entry point for YOLO+SAM segmentation pipeline.
    
    Processes all datasets in my_datasets/ directory:
    1. Auto-detect target objects from folder names
    2. Generate segmentation masks using YOLO+SAM
    3. Lift masks to 3D using VGGT depth estimates
    4. Save cleaned point clouds as PLY files
    
    Output files:
    - <dataset>/<dataset>_full_scene_cleaned.ply: Final 3D reconstruction
    - <dataset>/cutouts/: Segmentation visualizations
    - <dataset>/debug_frame_detections.txt: Detection log
    """
    MY_DATASET_PATH = "/projects/standard/csci5561/shared/G11/my_datasets" 
    if not os.path.exists(MY_DATASET_PATH):
        return print("Path not found")

    output_data, dataset_map = process_dataset(MY_DATASET_PATH)
    if not output_data: return print("No images loaded")

    print("\n--- Processing Datasets ---")
    for ds_path, img_paths in dataset_map.items():
        ds_name = os.path.basename(ds_path)
        print(f"\nDataset: {ds_name}")
        
        target_objs = get_scene_objects(ds_path)
        
        if not target_objs:
            print("No valid objects found to lift.")
            continue
        
        ds_imgs = {k: output_data[k] for k in img_paths}
        masks = generate_scene_masks(ds_path, ds_imgs, target_objs)
        
        pcd = lift_single_dataset(ds_path, img_paths, masks, output_data)
        
        if pcd is not None:
            save_name = f"{ds_name}_full_scene_cleaned.ply"
            save_path = os.path.join(ds_path, save_name)
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"Saved Scene: {save_path}")
        else:
            print(f"Failed to extract scene from {ds_name}")

if __name__ == "__main__":
    main()