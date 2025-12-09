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

# --- CONFIGURATION ---
IGNORE_CLASSES = ["person", "man", "woman", "handbag", "backpack"]
JSON_CONFIG_PATH = "scene_priors.json"

# ---------------------------------------------------------
# 1. Helper Functions (Loading Priors)
# ---------------------------------------------------------
def load_scene_priors(json_path):
    if os.path.exists(json_path):
        print(f" [i] Loading scene priors from: {json_path}")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                # Remove comments if any
                if "__comment__" in data: del data["__comment__"]
                return data
        except Exception as e:
            print(f" [!] Error loading JSON: {e}. Using fallback.")
    else:
        print(f" [!] Warning: '{json_path}' not found. Using minimal fallback.")
    
    # Minimal Fallback if JSON is missing
    return {
        "office": ["chair", "desk", "monitor", "table"],
        "apple": ["apple", "fruit"],
        "dishes": ["plate", "bowl", "cup"],
        "excavator": ["excavator", "toy excavator"]
    }

# Load Priors Globally
SCENE_PRIORS = load_scene_priors(JSON_CONFIG_PATH)

def clean_object_name(obj_str):
    """Cleans garbage like 'bowl (5)' into just 'bowl'."""
    obj_str = re.sub(r"[\(\[].*?[\)\]]", "", obj_str)
    obj_str = re.sub(r"\d+", "", obj_str)
    obj_str = obj_str.replace(":", "").replace("_", " ")
    return obj_str.strip().lower()

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
# 2. Scene Identification
# ---------------------------------------------------------
def clean_object_name(obj_str):
    """Cleans garbage like 'bowl (5)' into just 'bowl'."""
    obj_str = re.sub(r"[\(\[].*?[\)\]]", "", obj_str)
    obj_str = re.sub(r"\d+", "", obj_str)
    obj_str = obj_str.replace(":", "").replace("_", " ")
    return obj_str.strip().lower()

def build_reverse_index(priors_dict):
    """
    Creates a lookup table where EVERY item points back to its category.
    Example: 'spoon' -> ['kitchen', 'cutlery']
    """
    index = {}
    for category, items in priors_dict.items():
        # Map the category name itself
        if category not in index: index[category] = []
        index[category].append(category)
        
        # Map all items inside the category
        for item in items:
            # Clean item (e.g., "lego block" -> check both full string and parts)
            item_clean = item.lower().strip()
            if item_clean not in index:
                index[item_clean] = []
            if category not in index[item_clean]:
                index[item_clean].append(category)
    return index

def get_scene_objects(dataset_dir):
    ds_name = os.path.basename(dataset_dir).lower()
    
    # 1. Tokenize folder name (split by _ - or space)
    # "my_red_apple_scan" -> ["my", "red", "apple", "scan"]
    tokens = re.split(r'[_\-\s\.]+', ds_name)
    
    # 2. Build Reverse Index from our loaded JSON
    reverse_index = build_reverse_index(SCENE_PRIORS)
    
    detected_targets = []
    triggered_categories = set()
    
    print(f" [?] Analyzing Folder Name: '{ds_name}'")

    # 3. Match Tokens against Index
    for token in tokens:
        # Singularize simple plurals for better matching (apples -> apple)
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

    # 4. Check Log File (Supplemental)
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
        
        # If we found context via tokens, be strict (min 15). Else lenient (min 3).
        min_freq = 15 if detected_targets else 3
        
        for obj, count in object_counts.most_common():
            if count >= min_freq and obj not in detected_targets:
                print(f" [2] Log Discovery: Adding frequent object '{obj}' ({count} detections)")
                detected_targets.append(obj)

    # 5. Final Cleanup
    if detected_targets:
        # Deduplicate list
        unique_targets = list(set(detected_targets))
        print(f" [✓] Final Target List ({len(unique_targets)} items)")
        return unique_targets

    # --- FALLBACK ---
    print(f" [WARN] No match found in folder name or logs. Using generic fallback.")
    return ["chair", "table", "box", "door", "floor"]
# ---------------------------------------------------------
# 3. Mask Generation & Visualization
# ---------------------------------------------------------
def generate_scene_masks(dataset_dir, processed_images, target_objects):
    masks_dict = {}
    if not processed_images or not target_objects: return {}

    print(f"Loading models to isolate: {target_objects}")
    try:
        detector = YOLOWorld('yolov8l-world.pt') 
        segmenter = SAM('sam_b.pt') 
        detector.set_classes(target_objects)
    except Exception as e:
        print(f"Model load failed: {e}")
        return {}

    sorted_paths = sorted(processed_images.keys())
    
    # Setup Log and Cutout Dirs
    log_file_path = os.path.join(dataset_dir, "debug_frame_detections.txt")
    cutout_dir = os.path.join(dataset_dir, "cutouts")
    os.makedirs(cutout_dir, exist_ok=True)
    
    log_lines = []
    
    for filepath in tqdm(sorted_paths, desc="Extracting & Saving Cutouts"):
        img_array = processed_images[filepath]
        h, w = img_array.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        filename = os.path.basename(filepath)
        
        # Run detection
        det = detector.predict(img_array, verbose=False, conf=0.05)[0]
        
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

        # --- VISUALIZATION: SAVE CUTOUT ---
        if np.sum(combined_mask) > 0:
            # Apply mask (black out background)
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

# ---------------------------------------------------------
# 4. 3D Lifting (Standard Cleaning)
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
    # Standard cleaning 
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=75, std_ratio=1.0)
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
        if mask.shape[:2] != (h, w): mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if img_rgb.shape[:2] != (h, w): img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        valid = (mask > 0) & (depth > 0.1) & (depth < 10.0)
        if np.sum(valid) < 10: continue

        ys, xs = np.where(valid)
        z = depth[ys, xs]
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        points_cam = np.stack([x, y, z], axis=-1)
        
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

    output_data, dataset_map = process_dataset(MY_DATASET_PATH)
    if not output_data: return print("No images loaded")

    print("\n--- Processing Datasets ---")
    for ds_path, img_paths in dataset_map.items():
        ds_name = os.path.basename(ds_path)
        print(f"\nDataset: {ds_name}")
        
        # A. Get Targets (Using JSON SCENE_PRIORS)
        target_objs = get_scene_objects(ds_path)
        
        if not target_objs:
            print("No valid objects found to lift.")
            continue
        
        # B. Mask & Cutout Generation
        ds_imgs = {k: output_data[k] for k in img_paths}
        masks = generate_scene_masks(ds_path, ds_imgs, target_objs)
        
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