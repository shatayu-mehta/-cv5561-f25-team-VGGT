import os
import json
import re
import cv2
import numpy as np
import glob
import torch
from ultralytics import YOLOWorld, SAM 

# --- CONFIGURATION ---
JSON_CONFIG_PATH = "scene_priors.json"
OVERLAY_COLOR = (0, 255, 0) # Green
OVERLAY_ALPHA = 0.5

# --- PURE GEOMETRY SETTINGS ---
POSE_TYPE = "C2W" # Standard for VGGT/ScanNet. Change to "W2C" only if points disappear.

# =========================================================
# PART 1: HELPERS
# =========================================================
def load_scene_priors(json_path):
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if "__comment__" in data: del data["__comment__"]
                return data
        except: pass
    return {"office": ["chair", "desk", "box"], "apple": ["apple"]} 

SCENE_PRIORS = load_scene_priors(JSON_CONFIG_PATH)

def get_scene_objects(dataset_dir):
    ds_name = os.path.basename(dataset_dir).lower()
    targets = []
    for cat, items in SCENE_PRIORS.items():
        if cat in ds_name: targets.extend(items)
    return list(set(targets)) if targets else ["chair", "table", "box"]

def load_matrix(path):
    return np.loadtxt(path)

# =========================================================
# PART 2: DATA LOADING
# =========================================================
def process_dataset(base_path):
    output_data = {}
    dataset_map = {}
    dataset_dirs = [f.path for f in os.scandir(base_path) if f.is_dir()]
    
    for ds_path in dataset_dirs:
        images_dir = os.path.join(ds_path, "images")
        if not os.path.exists(images_dir): continue
        img_files = sorted(glob.glob(os.path.join(images_dir, "*")))
        img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if img_files:
            dataset_map[ds_path] = img_files
            for fpath in img_files:
                img = cv2.imread(fpath)
                if img is not None:
                    output_data[fpath] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return output_data, dataset_map

def get_vggt_files(dataset_dir):
    out_dir = os.path.join(dataset_dir, "outputs")
    depths = sorted(glob.glob(os.path.join(out_dir, "depths", "*.npy")))
    if not depths: depths = sorted(glob.glob(os.path.join(out_dir, "depths", "*.png")))
    poses = sorted(glob.glob(os.path.join(out_dir, "poses", "*extrinsic.txt")))
    intrinsics = sorted(glob.glob(os.path.join(out_dir, "poses", "*intrinsic.txt")))
    return depths, poses, intrinsics

# =========================================================
# PART 3: GENERATE MASK 0
# =========================================================
def generate_mask0(processed_images, target_objects, detector_model, sam_model):
    sorted_paths = sorted(processed_images.keys())
    first_path = sorted_paths[0]
    img = processed_images[first_path]
    h, w = img.shape[:2]
    
    detector_model.set_classes(target_objects)
    det = detector_model.predict(img, verbose=False, conf=0.05)[0]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    if det.boxes:
        sam_res = sam_model(img, bboxes=det.boxes.xyxy, verbose=False)[0]
        if sam_res.masks:
            for m in sam_res.masks.data.cpu().numpy():
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.maximum(combined_mask, (m > 0.5).astype(np.uint8))
    
    return combined_mask, first_path

# =========================================================
# PART 4: FORWARD WARPING (PURE DOTS)
# =========================================================
def forward_warp_mask(mask_src, depth_src, pose_src, pose_dst, K_src, K_dst, target_shape):
    H_tgt, W_tgt = target_shape 
    H_src, W_src = mask_src.shape

    # 1. Get ONLY the masked source pixels
    v_src, u_src = np.where(mask_src > 0)
    if len(v_src) == 0: return np.zeros((H_tgt, W_tgt), dtype=np.uint8)

    # Align Depth size
    if depth_src.shape[:2] != (H_src, W_src):
        depth_src = cv2.resize(depth_src, (W_src, H_src), interpolation=cv2.INTER_NEAREST)
    
    # Extract Raw Depth
    z_src = depth_src[v_src, u_src]
    
    # Filter valid depth
    valid_depth = z_src > 0
    u_src, v_src, z_src = u_src[valid_depth], v_src[valid_depth], z_src[valid_depth]
    
    # 2. Unproject to 3D (Camera 0 Frame)
    fx, fy = K_src[0, 0], K_src[1, 1]
    cx, cy = K_src[0, 2], K_src[1, 2]
    
    X_cam = (u_src - cx) * z_src / fx
    Y_cam = (v_src - cy) * z_src / fy
    Z_cam = z_src
    
    # Homogeneous Coordinates [x, y, z, 1]
    P_cam_src = np.stack([X_cam, Y_cam, Z_cam, np.ones_like(Z_cam)]) 

    # 3. Transform to Camera 1 Frame
    # Expand to 4x4 if needed
    if pose_src.shape == (3, 3): pose_src = np.vstack([pose_src, [0,0,0,1]])
    if pose_dst.shape == (3, 3): pose_dst = np.vstack([pose_dst, [0,0,0,1]])

    if POSE_TYPE == "C2W":
        # Standard: T = inv(Dst) @ Src
        T_src_to_dst = np.linalg.inv(pose_dst) @ pose_src
    else: 
        # Inverted: T = Dst @ inv(Src)
        T_src_to_dst = pose_dst @ np.linalg.inv(pose_src)
        
    P_cam_dst = T_src_to_dst @ P_cam_src 
    X_new, Y_new, Z_new = P_cam_dst[0, :], P_cam_dst[1, :], P_cam_dst[2, :]
    
    # 4. Project back to Pixels
    # Keep only points in front of camera
    valid_proj = Z_new > 0.01
    X_new, Y_new, Z_new = X_new[valid_proj], Y_new[valid_proj], Z_new[valid_proj]
    
    fx_new, fy_new = K_dst[0, 0], K_dst[1, 1]
    cx_new, cy_new = K_dst[0, 2], K_dst[1, 2]
    
    u_new = (X_new * fx_new / Z_new) + cx_new
    v_new = (Y_new * fy_new / Z_new) + cy_new
    
    # Round to nearest pixel integer (The purest form of rasterization)
    u_new = np.round(u_new).astype(int)
    v_new = np.round(v_new).astype(int)
    
    # 5. Filter Screen Bounds
    valid_bounds = (u_new >= 0) & (u_new < W_tgt) & (v_new >= 0) & (v_new < H_tgt)
    u_new, v_new = u_new[valid_bounds], v_new[valid_bounds]
    
    # 6. Create Mask (Just dots)
    mask_dst = np.zeros((H_tgt, W_tgt), dtype=np.uint8)
    mask_dst[v_new, u_new] = 255
    
    # NO MORPHOLOGY. NO HOLE FILLING.
    
    return mask_dst

# =========================================================
# MAIN
# =========================================================
def main():
    MY_DATASET_PATH = "/projects/standard/csci5561/shared/G11/my_datasets" 
    if not os.path.exists(MY_DATASET_PATH): return print("Path not found")
    output_data, dataset_map = process_dataset(MY_DATASET_PATH) 
    if not output_data: return print("No images.")

    print("\n--- Running Pure Geometric Propagation (DOTS ONLY) ---")
    
    for ds_path, img_paths in dataset_map.items():
        ds_name = os.path.basename(ds_path)
        print(f"\nDataset: {ds_name}")
        
        # Reload models
        try:
            detector = YOLOWorld('yolov8l-world.pt') 
            sam_model = SAM('sam_b.pt')
        except Exception as e:
            print(f"   [!] Failed to load models: {e}")
            continue

        depths, poses, intrinsics = get_vggt_files(ds_path)
        if not depths or len(depths) < 2: continue
            
        target_objs = get_scene_objects(ds_path)
        ds_imgs = {k: output_data[k] for k in img_paths if k in output_data}
        mask0, frame0_path = generate_mask0(ds_imgs, target_objs, detector, sam_model)
        
        if mask0 is None: continue

        print("   [i] Warping Mask0 to Frame 1...")
        frame1_path = sorted(ds_imgs.keys())[1]
        frame1_img_rgb = ds_imgs[frame1_path]
        frame1_shape = frame1_img_rgb.shape[:2]

        try:
            depth0 = np.load(depths[0])
            pose0 = load_matrix(poses[0])   
            pose1 = load_matrix(poses[1])   
            K0 = load_matrix(intrinsics[0]) 
            K1 = load_matrix(intrinsics[1]) 
        except: continue

        # Pure Geometric Warp
        warped_mask = forward_warp_mask(mask0, depth0, pose0, pose1, K0, K1, frame1_shape)
        
        prop_dir = os.path.join(ds_path, "propagated_masks")
        os.makedirs(prop_dir, exist_ok=True)
        if not os.access(prop_dir, os.W_OK): continue

        frame1_img_bgr = cv2.cvtColor(frame1_img_rgb, cv2.COLOR_RGB2BGR)
        
        # Overlay Logic
        colored_mask = np.zeros_like(frame1_img_bgr)
        colored_mask[warped_mask > 0] = OVERLAY_COLOR
        overlay_img = frame1_img_bgr.copy()
        
        roi = overlay_img[warped_mask > 0]
        colored_roi = colored_mask[warped_mask > 0]
        blended_roi = cv2.addWeighted(roi, 1 - OVERLAY_ALPHA, colored_roi, OVERLAY_ALPHA, 0)
        overlay_img[warped_mask > 0] = blended_roi

        save_path = os.path.join(prop_dir, f"pure_dots_overlay_{os.path.basename(frame1_path)}")
        cv2.imwrite(save_path, overlay_img)
        print(f"   [✓] Saved: {save_path}")

if __name__ == "__main__":
    main()