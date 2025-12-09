import os
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from torchvision import transforms as TF
from ultralytics import YOLOWorld, SAM
from tqdm import tqdm
import glob
from collections import Counter

# --- CONFIGURATION ---
# Objects to NEVER include, even if frequent
IGNORE_CLASSES = ["person", "man", "woman", "handbag", "backpack"]
# Minimum occurrences to be considered part of the scene (to filter random flickering detections)
MIN_FREQUENCY_THRESHOLD = 3 

# ---------------------------------------------------------
# 1. Dataset Helpers
# ---------------------------------------------------------
def preprocess_image_vggt_style(image_path, target_size=518):
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

# ---------------------------------------------------------
# 2. Smart Scene Identification
# ---------------------------------------------------------
def get_scene_objects(dataset_dir):
    """
    Reads detected_objects.txt to find ALL consistent objects.
    """
    log_path = os.path.join(dataset_dir, "detected_objects.txt")
    if not os.path.exists(log_path):
        print(f"[WARN] No log found for {os.path.basename(dataset_dir)}. Defaulting to generic office items.")
        return ["chair", "table", "desk", "cabinet", "monitor", "keyboard", "shelf"]

    object_counts = Counter()
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(':')
            if len(parts) > 1:
                objs = parts[1].split(',')
                for obj in objs:
                    obj = obj.strip().lower()
                    if obj and obj != "(no detection)":
                        object_counts[obj] += 1

    # Filter the list
    keep_list = []
    print(f"\nObject Analysis for {os.path.basename(dataset_dir)}:")
    for obj, count in object_counts.most_common():
        if obj in IGNORE_CLASSES:
            print(f"  [X] Ignored: {obj} ({count} frames)")
            continue
            
        if count >= MIN_FREQUENCY_THRESHOLD:
            print(f"  [O] Keeping: {obj} ({count} frames)")
            keep_list.append(obj)
        else:
            print(f"  [ ] Dropped: {obj} (only {count} frames - noise)")
            
    return keep_list

# ---------------------------------------------------------
# 3. Multi-Class Mask Generation
# ---------------------------------------------------------
def generate_scene_masks(processed_images, target_objects):
    masks_dict = {}
    if not processed_images or not target_objects: return {}

    print(f"Loading models to isolate: {target_objects}")
    try:
        detector = YOLOWorld('yolov8l-world.pt') 
        segmenter = SAM('sam_b.pt') 
        detector.set_classes(target_objects)
    except:
        return {}

    sorted_paths = sorted(processed_images.keys())
    
    for filepath in tqdm(sorted_paths, desc="Extracting Scene"):
        img_array = processed_images[filepath]
        h, w = img_array.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Run detection on specific targets
        # conf=0.05 catches hard-to-see objects
        det = detector.predict(img_array, verbose=False, conf=0.05)[0]
        
        if det.boxes:
            sam = segmenter(img_array, bboxes=det.boxes.xyxy, verbose=False)[0]
            if sam.masks:
                for m in sam.masks.data.cpu().numpy():
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
                    combined_mask = np.maximum(combined_mask, (m > 0.5).astype(np.uint8))
        
        masks_dict[filepath] = combined_mask
    return masks_dict

# ---------------------------------------------------------
# 4. 3D Lifting
# ---------------------------------------------------------
def get_vggt_files(dataset_dir):
    out_dir = os.path.join(dataset_dir, "outputs")
    depths = sorted(glob.glob(os.path.join(out_dir, "depths", "*.npy")))
    if not depths: depths = sorted(glob.glob(os.path.join(out_dir, "depths", "*.png")))
    poses = sorted(glob.glob(os.path.join(out_dir, "poses", "*extrinsic.txt")))
    intrinsics = sorted(glob.glob(os.path.join(out_dir, "poses", "*intrinsic.txt")))
    return depths, poses, intrinsics

def clean_point_cloud(pcd):
    print("  > Cleaning noise (Statistical Outlier Removal)...")
    # Slightly relaxed filter to keep thin objects like chair legs
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
    pcd_clean = pcd.select_by_index(ind)
    print(f"  > Removed {len(pcd.points) - len(pcd_clean.points)} noise points.")
    return pcd_clean

def lift_single_dataset(dataset_dir, image_paths, masks_dict, processed_images):
    sorted_image_paths = sorted(image_paths)
    depth_files, pose_files, intrin_files = get_vggt_files(dataset_dir)
    
    if not depth_files: return None

    num_frames = min(len(sorted_image_paths), len(depth_files), len(pose_files))
    points = []
    colors = []

    for i in range(num_frames):
        img_path = sorted_image_paths[i]
        mask = masks_dict.get(img_path, None)
        
        # Skip empty frames
        if mask is None or np.sum(mask) < 50: continue 
        
        img_rgb = processed_images[img_path]
        
        # Load VGGT
        if depth_files[i].endswith('.npy'):
            depth = np.load(depth_files[i])
        else:
            depth = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED).astype(float) / 1000.0
            if depth.max() > 100: depth /= 1000.0 
            
        pose = np.loadtxt(pose_files[i])
        K = np.loadtxt(intrin_files[i]) if i < len(intrin_files) else np.loadtxt(os.path.join(dataset_dir, "intrinsics.txt"))

        # Resize
        h, w = depth.shape[:2]
        if mask.shape[:2] != (h, w): mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if img_rgb.shape[:2] != (h, w): img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        # Valid Mask
        valid = (mask > 0) & (depth > 0.1) & (depth < 10.0)
        if np.sum(valid) < 10: continue

        # Project
        ys, xs = np.where(valid)
        z = depth[ys, xs]
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        points_cam = np.stack([x, y, z], axis=-1)
        
        # Transform
        if pose.shape == (3, 4): pose = np.vstack([pose, [0,0,0,1]])
        try:
            c2w = np.linalg.inv(pose)
            world_pts = (points_cam @ c2w[:3, :3].T) + c2w[:3, 3]
            cols = img_rgb[ys, xs] / 255.0
            points.append(world_pts)
            colors.append(cols)
        except: continue

    if not points: return None
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(points, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors, axis=0))
    
    return clean_point_cloud(pcd)

# ---------------------------------------------------------
# 5. Main
# ---------------------------------------------------------
def main():
    MY_DATASET_PATH = "/projects/standard/csci5561/shared/G11/my_datasets" 
    if not os.path.exists(MY_DATASET_PATH): return print("Path not found")

    # 1. Load
    output_data, dataset_map = process_dataset(MY_DATASET_PATH)
    if not output_data: return print("No images loaded")

    print("\n--- Processing Datasets ---")
    for ds_path, img_paths in dataset_map.items():
        ds_name = os.path.basename(ds_path)
        print(f"\nDataset: {ds_name}")
        
        # A. Get List of Frequent Objects
        target_objs = get_scene_objects(ds_path)
        if not target_objs:
            print("No valid objects found to lift.")
            continue
        
        # B. Generate Masks for those objects
        ds_imgs = {k: output_data[k] for k in img_paths}
        masks = generate_scene_masks(ds_imgs, target_objs)
        
        # C. Lift
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