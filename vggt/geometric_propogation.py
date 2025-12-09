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
IGNORE_CLASSES = ["person", "man", "woman", "handbag", "backpack", "text", "drying rack", "rack", "clothes rack"] 
JSON_CONFIG_PATH = "scene_priors.json"
VOXEL_SIZE = 0.005      # 5mm grid
OUTLIER_NEIGHBORS = 50  # For final statistical cleaning
COLOR_TOLERANCE = 50    # Color threshold
PALETTE_SIZE = 5        # Number of colors to learn from Frame 0

# ---------------------------------------------------------
# 1. Helper Functions
# ---------------------------------------------------------
def load_scene_priors(json_path):
    if os.path.exists(json_path):
        print(f" [i] Loading scene priors from: {json_path}")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if "__comment__" in data: del data["__comment__"]
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

def clean_object_name(obj_str):
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
    dataset_map = {} 
    print(f"Scanning '{dataset_root_dir}'...")
    if not os.path.exists(dataset_root_dir):
        print(f" [!] ERROR: Dataset root does not exist: {dataset_root_dir}")
        return {}

    for root, dirs, files in os.walk(dataset_root_dir):
        if os.path.basename(root) == 'images':
            dataset_dir = os.path.dirname(root)
            dataset_map[dataset_dir] = []
            
            for file in sorted(files):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    full_path = os.path.join(root, file)
                    dataset_map[dataset_dir].append(full_path)
                        
    return dataset_map

def build_reverse_index(priors_dict):
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

# ---------------------------------------------------------
# 2. Mask Generation (Frame 1 ONLY)
# ---------------------------------------------------------
def get_frame1_anchor_mask(image_paths, target_objects):
    if not image_paths or not target_objects: return None, None

    print(f" [!] Frame 1 Detection Mode: Isolating {target_objects}")
    
    try:
        detector = YOLOWorld('yolov8l-world.pt') 
        segmenter = SAM('sam_b.pt') 
        detector.set_classes(target_objects)
    except Exception as e:
        print(f"Model load failed: {e}")
        return None, None

    first_path = sorted(image_paths)[0]
    print(f" [1] Processing Frame 1: {os.path.basename(first_path)}")
    
    img_array = preprocess_image_vggt_style(first_path)
    if img_array is None: return None, None
    
    h, w = img_array.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    det = detector.predict(img_array, verbose=False, conf=0.15)[0]
    
    detected_items = []
    if det.boxes:
        for i, cls_id in enumerate(det.boxes.cls):
            cls_name = detector.names[int(cls_id)]
            conf = float(det.boxes.conf[i])
            
            if cls_name.lower() in IGNORE_CLASSES:
                continue
                
            detected_items.append(f"{cls_name} ({conf:.2f})")

            bbox = det.boxes.xyxy[i:i+1] 
            sam = segmenter(img_array, bboxes=bbox, verbose=False)[0]
            
            if sam.masks:
                for m in sam.masks.data.cpu().numpy():
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
                    combined_mask = np.maximum(combined_mask, (m > 0.5).astype(np.uint8))
    else:
        print(" [X] No objects detected in Frame 1.")
        return None, None
    
    print(f"  -> Detected: {', '.join(detected_items)}")
    return combined_mask, first_path

def save_verification_mask(dataset_dir, img_path, mask):
    img_rgb = preprocess_image_vggt_style(img_path)
    if img_rgb is None: return
        
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    vis_mask = (mask * 255).astype(np.uint8)
    masked_view = cv2.bitwise_and(img_bgr, img_bgr, mask=vis_mask)
    
    debug_dir = os.path.join(dataset_dir, "debug_masks")
    os.makedirs(debug_dir, exist_ok=True)
    
    save_path = os.path.join(debug_dir, "anchor_check.jpg")
    cv2.imwrite(save_path, masked_view)
    print(f"\n [i] ANCHOR MASK SAVED: {save_path}")

# ---------------------------------------------------------
# 3. VGGT I/O & Geometric Fusion
# ---------------------------------------------------------
def load_vggt_outputs(dataset_dir):
    out_dir = os.path.join(dataset_dir, "outputs")
    depths = sorted(glob.glob(os.path.join(out_dir, "**", "*depth*.npy"), recursive=True))
    poses = sorted(glob.glob(os.path.join(out_dir, "**", "*extrinsic*.txt"), recursive=True))
    intrs = sorted(glob.glob(os.path.join(out_dir, "**", "*intrinsic*.txt"), recursive=True))
    return depths, poses, intrs

def read_matrix(path):
    return np.loadtxt(path)

def extract_dominant_palette(img_array, mask, k=5):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0: return None
    
    pixels = img_array[ys, xs] 
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    actual_k = min(k, len(pixels))
    
    try:
        ret, label, centers = cv2.kmeans(pixels, actual_k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return centers 
    except Exception as e:
        return np.mean(pixels, axis=0, keepdims=True)

def lift_anchor_to_3d(mask, depth_path, pose_path, intr_path, img_path_for_color=None):
    print(" [2] Lifting Anchor to 3D (Core Sampling)...")
    depth = np.load(depth_path)
    h, w = mask.shape[:2]
    
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.NEAREST)
        
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    
    if np.sum(eroded_mask) < 100:
        print("  [!] Warning: Object too small for erosion. Using raw mask.")
        eroded_mask = mask
        
    K = read_matrix(intr_path)
    T_w2c = read_matrix(pose_path)
    T_c2w = np.linalg.inv(T_w2c)
    
    ys, xs = np.where(eroded_mask > 0)
    z_val = depth[ys, xs]
    
    valid = (z_val > 0.1) & (z_val < 50.0)
    xs, ys, z_val = xs[valid], ys[valid], z_val[valid]
    
    colors = None
    palette = None
    
    if img_path_for_color:
        img_array = preprocess_image_vggt_style(img_path_for_color)
        if img_array is not None:
            colors = img_array[ys, xs] / 255.0
            colors = np.ascontiguousarray(colors)
            palette = extract_dominant_palette(img_array, eroded_mask, k=PALETTE_SIZE)

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x_cam = (xs - cx) * z_val / fx
    y_cam = (ys - cy) * z_val / fy
    
    pts_cam = np.vstack((x_cam, y_cam, z_val, np.ones_like(z_val)))
    pts_world = (T_c2w @ pts_cam).T[:, :3]
    
    return np.ascontiguousarray(pts_world), colors, palette

def propagate_and_harvest(anchor_pts, palette, img_paths, depths, poses, intrs):
    voxel_map = {} 
    anchor_h = np.hstack((anchor_pts, np.ones((len(anchor_pts), 1)))).T 
    
    print(f" [3] Propagating to {len(img_paths)} frames and harvesting...")
    
    # --- DYNAMIC VOTING THRESHOLD ---
    # Require object to be present in 30% of frames to eliminate ghost noise
    MIN_VOTES_DYNAMIC = max(3, int(len(img_paths) * 0.3))
    print(f" [i] Dynamic Consensus: Voxel must appear in {MIN_VOTES_DYNAMIC} frames to survive.")
    
    for i in tqdm(range(len(img_paths))):
        img_array = preprocess_image_vggt_style(img_paths[i])
        if img_array is None: continue
        
        h, w = img_array.shape[:2]
        curr_depth = np.load(depths[i])
        
        if curr_depth.shape[:2] != (h, w):
            curr_depth = cv2.resize(curr_depth, (w, h), interpolation=cv2.NEAREST)
            
        K = read_matrix(intrs[i])
        T_w2c = read_matrix(poses[i])
        
        pts_cam = T_w2c @ anchor_h
        X, Y, Z = pts_cam[0], pts_cam[1], pts_cam[2]
        
        valid_z = Z > 0.1
        
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        u = (X[valid_z] * fx / Z[valid_z]) + cx
        v = (Y[valid_z] * fy / Z[valid_z]) + cy
        
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        
        valid_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u_final, v_final = u[valid_uv], v[valid_uv]
        
        if len(u_final) < 20: continue 
        
        # --- CREATE MASK ---
        points_2d = np.column_stack((u_final, v_final))
        hull = cv2.convexHull(points_2d)
        geo_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(geo_mask, hull, 255)
        
        # --- CRITICAL FIX: REDUCED DILATION ---
        # 20x20 was causing the floor leakage. Reduced to 5x5.
        kernel = np.ones((5, 5), np.uint8) 
        geo_mask = cv2.dilate(geo_mask, kernel, iterations=1)
        
        ys, xs = np.where(geo_mask > 0)
        z_new = curr_depth[ys, xs]
        pixels_new = img_array[ys, xs] 
        c_new = pixels_new / 255.0
        
        ghost_mean_z = np.mean(Z[valid_z])
        tolerance = 0.5 
        
        valid_depth = (z_new > 0.1) & \
                      (z_new > ghost_mean_z - tolerance) & \
                      (z_new < ghost_mean_z + tolerance)
                      
        if palette is not None:
            diffs = pixels_new[:, np.newaxis, :] - palette[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2) 
            min_dists = np.min(dists, axis=1) 
            valid_color = min_dists < COLOR_TOLERANCE 
        else:
            valid_color = np.ones_like(valid_depth, dtype=bool)

        valid_h = valid_depth & valid_color
        
        if np.sum(valid_h) == 0: continue
        
        z_h = z_new[valid_h]
        x_h = xs[valid_h]
        y_h = ys[valid_h]
        c_h = c_new[valid_h]
        
        x_c = (x_h - cx) * z_h / fx
        y_c = (y_h - cy) * z_h / fy
        
        pts_c_harvest = np.vstack((x_c, y_c, z_h, np.ones_like(z_h)))
        T_c2w = np.linalg.inv(T_w2c)
        pts_w_harvest = (T_c2w @ pts_c_harvest).T[:, :3]
        
        voxel_indices = np.floor(pts_w_harvest / VOXEL_SIZE).astype(int)
        
        for idx, vox in enumerate(voxel_indices):
            v_key = tuple(vox)
            if v_key not in voxel_map:
                voxel_map[v_key] = [c_h[idx][0], c_h[idx][1], c_h[idx][2], 1] 
            else:
                entry = voxel_map[v_key]
                n = entry[3]
                entry[0] = (entry[0] * n + c_h[idx][0]) / (n + 1)
                entry[1] = (entry[1] * n + c_h[idx][1]) / (n + 1)
                entry[2] = (entry[2] * n + c_h[idx][2]) / (n + 1)
                entry[3] += 1

    final_pts = []
    final_colors = []
    
    print(f" [4] Voting Complete. Total Voxels: {len(voxel_map)}")
    
    for v_key, data in voxel_map.items():
        if data[3] >= MIN_VOTES_DYNAMIC: 
            pt = np.array(v_key) * VOXEL_SIZE + (VOXEL_SIZE/2)
            final_pts.append(pt)
            final_colors.append(data[:3])
            
    print(f"  -> Surviving Voxels (Votes >= {MIN_VOTES_DYNAMIC}): {len(final_pts)}")
    
    if len(final_pts) == 0:
        print(" [!] WARNING: Voting filtered out all points!")
        return o3d.geometry.PointCloud()
            
    pcd_frame = o3d.geometry.PointCloud()
    pcd_frame.points = o3d.utility.Vector3dVector(np.ascontiguousarray(final_pts))
    pcd_frame.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(final_colors))
        
    return pcd_frame

# ---------------------------------------------------------
# 4. NORMAL-BASED GEOMETRIC CLEANING (Replaces DBSCAN/RANSAC)
# ---------------------------------------------------------
def clean_and_save(pcd, save_path):
    print(f" [5] Post-Processing Cloud ({len(pcd.points)} points)...")
    
    if len(pcd.points) == 0:
        return

    # 1. ESTIMATE NORMALS
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)
    
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    # 2. IDENTIFY FLOOR DIRECTION (Lowest points are floor)
    # We find the axis with the lowest values and assume that's the bottom.
    min_z = np.min(points[:, 2])
    max_z = np.max(points[:, 2])
    
    # Define "Bottom Zone" as lowest 10% of the object
    bottom_threshold = min_z + (max_z - min_z) * 0.10
    bottom_indices = np.where(points[:, 2] < bottom_threshold)[0]
    
    # Safety: If Z isn't height, try Y
    if len(bottom_indices) < 10: 
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        bottom_threshold = min_y + (max_y - min_y) * 0.10
        bottom_indices = np.where(points[:, 1] < bottom_threshold)[0]

    if len(bottom_indices) > 0:
        # Calculate Average Floor Normal
        floor_normal = np.mean(normals[bottom_indices], axis=0)
        floor_normal /= np.linalg.norm(floor_normal)
        print(f"  -> Detected Floor Normal: {floor_normal}")

        # 3. SURGICAL FILTERING
        # Calculate angle difference (dot product) between every point and the floor
        dots = np.abs(np.dot(normals, floor_normal))
        
        # Check 1: Is the point physically at the bottom?
        # Re-detect axis to be safe
        if points[:, 2].std() < points[:, 1].std(): # Heuristic for "flat" axis
            is_low = points[:, 1] < bottom_threshold 
        else:
            is_low = points[:, 2] < bottom_threshold

        # Check 2: Is the point facing the same way as the floor?
        is_flat = dots > 0.85 # Point is parallel to floor
        
        # Condition: If it's LOW and FLAT, it's a puddle. If it's LOW and VERTICAL, it's a wall.
        noise_mask = np.logical_and(is_low, is_flat)
        
        valid_indices = np.where(~noise_mask)[0]
        pcd = pcd.select_by_index(valid_indices)
        print(f"  -> Removed {np.sum(noise_mask)} floor/skirt points based on normals.")
    else:
        print(" [!] Could not detect bottom of object. Skipping normal filter.")

    # 4. FINAL CLEANUP
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
    final_pcd = pcd.select_by_index(ind)
    
    o3d.io.write_point_cloud(save_path, final_pcd)
    print(f" [✓] Saved to: {save_path}")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # --- UPDATE THIS PATH ---
    DATASET_ROOT = "/projects/standard/csci5561/shared/G11/my_datasets" 
    
    if not os.path.exists(DATASET_ROOT) or DATASET_ROOT == "path/to/your/dataset":
        print(f"\n [!] CRITICAL ERROR: You must edit the 'DATASET_ROOT' variable.")
        exit()

    dataset_map = process_dataset(DATASET_ROOT)
    
    if not dataset_map:
        print(f" [!] No images found in '{DATASET_ROOT}'.")
        exit()
    
    for ds_path, img_paths in dataset_map.items():
        if not img_paths: continue
        
        print(f"\nProcessing: {ds_path}")
        
        targets = get_scene_objects(ds_path)
        v_depths, v_poses, v_intrs = load_vggt_outputs(ds_path)
        
        if not v_depths:
            print(" [!] No VGGT outputs found. Skipping.")
            continue
            
        min_len = min(len(img_paths), len(v_depths), len(v_poses), len(v_intrs))
        img_paths = sorted(img_paths)[:min_len]
        v_depths = v_depths[:min_len]
        v_poses = v_poses[:min_len]
        v_intrs = v_intrs[:min_len]
        
        anchor_mask, anchor_img_path = get_frame1_anchor_mask(img_paths, targets)
        
        if anchor_mask is not None:
            save_verification_mask(ds_path, anchor_img_path, anchor_mask)
            
            anchor_pts, anchor_colors, palette = lift_anchor_to_3d(anchor_mask, v_depths[0], v_poses[0], v_intrs[0], anchor_img_path)
            
            debug_pcd = o3d.geometry.PointCloud()
            debug_pcd.points = o3d.utility.Vector3dVector(anchor_pts)
            if anchor_colors is not None:
                debug_pcd.colors = o3d.utility.Vector3dVector(anchor_colors)
            o3d.io.write_point_cloud(os.path.join(ds_path, "debug_frame0_anchor.ply"), debug_pcd)

            full_cloud = propagate_and_harvest(anchor_pts, palette, img_paths, v_depths, v_poses, v_intrs)
            
            out_file = os.path.join(ds_path, "reconstructed_object.ply")
            clean_and_save(full_cloud, out_file)
            
    print("\nProcessing Complete.")