import os
import sys
import torch
import cv2
import numpy as np
import argparse
import traceback
import requests
import shutil  # Added for deleting folders
from tqdm import tqdm
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
USE_SKY_SEGMENTATION = False  # Set to True only for outdoor scenes
DEPTH_CUTOFF = 2.0           # TIGHTENED: Cut anything further than 2.0 units to remove walls

# -----------------------------------------------------------------------------
# IMPORTS & SETUP
# -----------------------------------------------------------------------------
# Assuming the script is running where 'vggt' package is accessible
sys.path.append("vggt/")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# Try importing onnxruntime for Sky Segmentation if enabled
sky_session = None
if USE_SKY_SEGMENTATION:
    try:
        import onnxruntime
        HAS_ONNX = True
    except ImportError:
        HAS_ONNX = False
        print("![WARNING] 'onnxruntime' not found. Sky filtering disabled.")

def download_file_from_url(url, filename):
    try:
        print(f"  > Downloading {filename}...")
        response = requests.get(url, allow_redirects=False)
        if response.status_code == 302:
            response = requests.get(response.headers["Location"], stream=True)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"    [ERROR] Download failed: {e}")

def get_sky_mask(image, session):
    if session is None: return np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
    
    # Preprocess
    img = cv2.resize(image, (320, 320))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    # The math here upcasts to float64 automatically
    img = (img / 255 - mean) / std
    
    # Transpose dimensions
    img = img.transpose(2, 0, 1)[None] # (1, 3, 320, 320)

    # --- FIX: FORCE FLOAT32 BEFORE INFERENCE ---
    img = img.astype(np.float32) 
    # -------------------------------------------

    # Inference
    iname = session.get_inputs()[0].name
    oname = session.get_outputs()[0].name
    
    # This will now accept the input
    mask = session.run([oname], {iname: img})[0].squeeze()

    # Post-process
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    norm_mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    
    # Binary: 1=Keep (Not Sky), 0=Remove (Sky)
    return (norm_mask > 0.12).astype(np.float32)

# -----------------------------------------------------------------------------
# GEOMETRY
# -----------------------------------------------------------------------------
def save_ply(points, colors, filename):
    if len(points) == 0: return
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def manual_unproject(depth, extrinsic, intrinsic):
    # Robust squeeze: verify dimensions before squeezing
    if depth.ndim > 2: depth = depth.squeeze()
    
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u, v = u.flatten(), v.flatten()
    z = depth.flatten()
    
    # Filter invalid and far points
    valid = (z > 0) & (z < DEPTH_CUTOFF) & np.isfinite(z)
    u, v, z = u[valid], v[valid], z[valid]
    
    # Intrinsics often come in as (1, 3, 3) or (3, 3)
    if intrinsic.ndim > 2: intrinsic = intrinsic.squeeze()
    
    fx, fy = float(intrinsic[0,0]), float(intrinsic[1,1])
    cx, cy = float(intrinsic[0,2]), float(intrinsic[1,2])
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    cam_pts = np.stack([x, y, z, np.ones_like(z)], axis=1)
    
    if extrinsic.ndim > 2: extrinsic = extrinsic.squeeze()

    try:
        # World = inv(Extrinsic) * Cam
        world_pts = (np.linalg.inv(extrinsic) @ cam_pts.T).T[:, :3]
        return world_pts, valid
    except:
        return np.zeros((0,3)), valid

# -----------------------------------------------------------------------------
# FILTERING (Official Logic + Spatial Crop)
# -----------------------------------------------------------------------------
def filter_spatial_outliers(points, colors, confs, sky_mask, sigma=1.0):
    """
    Automatically crops the scene to the central object.
    Calculates the centroid and keeps points within 'sigma' standard deviations
    (or IQR-based range) to remove background walls/floor.
    """
    if len(points) == 0: return points, colors, confs, sky_mask
    
    print("    - Running statistical spatial crop...")
    
    # 1. Find the 'center' of the scene (Median is robust to outliers)
    center = np.median(points, axis=0)
    
    # 2. Calculate distance of every point from the center
    dists = np.linalg.norm(points - center, axis=1)
    
    # 3. Find a cutoff distance using Interquartile Range (IQR)
    # This adapts to the size of the object.
    q25, q75 = np.percentile(dists, [25, 75])
    iqr = q75 - q25
    upper_bound = q75 + (sigma * iqr)
    
    print(f"      > Center: {center}")
    print(f"      > 75% of points are within {q75:.2f} units.")
    print(f"      > Cropping everything further than {upper_bound:.2f} units.")
    
    # 4. Create Mask
    mask = dists < upper_bound
    
    return points[mask], colors[mask], confs[mask], sky_mask[mask]

def apply_filters(points, colors, confs, sky_mask_flat, percentile=80):
    # 2. Color Filter (Black/White Backgrounds)
    # Black: Sum < 16
    not_black = colors.sum(axis=1) >= 16
    # White: All > 240
    not_white = ~((colors > 240).all(axis=1))
    
    color_mask = not_black & not_white
    
    # 3. Percentile Confidence
    # Filter points first by color to get clean stats
    valid_confs = confs[color_mask]
    if len(valid_confs) == 0: return np.array([]), np.array([])
    
    cutoff = np.percentile(valid_confs, percentile)
    final_mask = color_mask & (confs >= cutoff)
    
    return points[final_mask], colors[final_mask]

# -----------------------------------------------------------------------------
# PROCESS DATASET
# -----------------------------------------------------------------------------
def process_single_dataset(dataset_path, model, device):
    p = Path(dataset_path)
    dataset_name = p.name 
    image_dir = p / "images" if (p / "images").exists() else p
    out_dir = p / "outputs"
    
    # CLEAN UP OLD OUTPUTS
    if out_dir.exists():
        print(f"  > Cleaning up old outputs in {out_dir}...")
        shutil.rmtree(out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    images = sorted([str(x) for x in image_dir.glob("*") if x.suffix.lower() in ['.png','.jpg','.jpeg']])
    if not images: return

    num_frames = len(images)
    print(f"  > Processing {dataset_name} ({num_frames} frames)...")

    # A. Init Sky Model if needed
    global sky_session
    if USE_SKY_SEGMENTATION and HAS_ONNX and sky_session is None:
        if not os.path.exists("skyseg.onnx"):
            download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")
        try:
            sky_session = onnxruntime.InferenceSession("skyseg.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except:
            print("    (SkySeg failed to load, continuing without it)")

    # B. Run VGGT
    # Ensure batch size matches logic
    images_tensor = load_and_preprocess_images(images).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        preds = model(images_tensor)
    
    # C. Extract & Normalize Data (THE FIX)
    ext, intr = pose_encoding_to_extri_intri(preds["pose_enc"], images_tensor.shape[-2:])
    
    # 1. Depth: Force reshape to (Num_Frames, H, W)
    raw_depth = preds["depth"].float().cpu().numpy()
    # Remove last dim if channel is 1
    if raw_depth.shape[-1] == 1: raw_depth = raw_depth.squeeze(-1)
    
    H, W = raw_depth.shape[-2], raw_depth.shape[-1]
    # FORCE RESHAPE: if we have 41 frames, this MUST be (41, H, W)
    try:
        depth = raw_depth.reshape(num_frames, H, W)
    except ValueError:
        print(f"    [WARNING] Shape mismatch: Model output {raw_depth.shape} vs expected {num_frames} frames.")
        depth = raw_depth.reshape(-1, H, W) # Best effort
        num_frames = depth.shape[0] # Adjust loop

    # 2. Extrinsics: Force reshape to (Num_Frames, 4, 4)
    ext_flat = ext.float().cpu().numpy().flatten()
    # 4x4 = 16 elements. 3x4 = 12 elements.
    elements_per_matrix = ext_flat.shape[0] // num_frames
    
    if elements_per_matrix == 12: # 3x4 case
        ext_cpu = ext_flat.reshape(num_frames, 3, 4)
        # Add [0,0,0,1] row
        last_row = np.array([0, 0, 0, 1]).reshape(1, 1, 4).repeat(num_frames, axis=0)
        ext_cpu = np.concatenate([ext_cpu, last_row], axis=1)
    else: # 4x4 case
        ext_cpu = ext_flat.reshape(num_frames, 4, 4)

    # 3. Intrinsics: Force reshape to (Num_Frames, 3, 3)
    intr_cpu = intr.float().cpu().numpy().flatten().reshape(num_frames, 3, 3)
    
    # 4. Confidence
    conf_raw = preds.get("depth_conf", torch.zeros_like(preds["depth"])).float().cpu().numpy()
    # Force reshape to match depth shape
    if conf_raw.size == depth.size:
        conf_raw = conf_raw.reshape(num_frames, H, W)
    else:
        conf_raw = conf_raw.reshape(num_frames, -1) 

    # D. Accumulate Points
    all_pts, all_cols, all_conf, all_sky = [], [], [], []
    
    print(f"    (Reshaped Depth: {depth.shape}, Extrinsics: {ext_cpu.shape})")

    for i in tqdm(range(num_frames)):
        # 1. Unproject
        pts, mask = manual_unproject(depth[i], ext_cpu[i], intr_cpu[i])
        
        # 2. Color & Conf
        img = cv2.imread(images[i])
        # Ensure image matches depth map size
        img = cv2.resize(img, (W, H))
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)[mask]
        
        # Handle confidence resizing
        c_frame = conf_raw[i]
        if c_frame.shape != (H, W):
             if c_frame.size == H*W:
                 c_frame = c_frame.reshape(H,W)
             else:
                 c_frame = np.ones((H, W))
        
        cnf = cv2.resize(c_frame, (W, H), interpolation=cv2.INTER_NEAREST).flatten()[mask]
        
        # 3. Sky Mask
        sky_val = np.ones_like(cnf)
        if sky_session:
            smask = get_sky_mask(img, sky_session)
            smask = cv2.resize(smask, (W, H), interpolation=cv2.INTER_NEAREST).flatten()[mask]
            sky_val = smask

        all_pts.append(pts)
        all_cols.append(rgb)
        all_conf.append(cnf)
        all_sky.append(sky_val)

    # Concatenate
    pts_full = np.concatenate(all_pts)
    cols_full = np.concatenate(all_cols)
    conf_full = np.concatenate(all_conf)
    sky_full = np.concatenate(all_sky)

    # Apply Sky to Confidence (Zero out sky points)
    conf_full = conf_full * sky_full

    # E. Save Percentiles
    if conf_full.max() <= 1.0: conf_full *= 100.0
    
    # --- APPLY SPATIAL CROP BEFORE PERCENTILES ---
    # This removes the walls so the percentiles are calculated on the object, not the room
    pts_full, cols_full, conf_full, sky_full = filter_spatial_outliers(pts_full, cols_full, conf_full, sky_full, sigma=1.2)

    ply_folder = out_dir / "pointclouds"
    ply_folder.mkdir(exist_ok=True)

    save_ply(pts_full, cols_full, str(ply_folder / f"{dataset_name}_raw_cropped.ply"))

    # WIDER RANGE OF OUTPUTS [0, 10, ... 90, 95, 98, 99]
    # Percentile p means we keep top (100-p)% of points
    percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99]
    
    print("  > Saving percentile-filtered clouds...")
    for p in percentiles:
        f_pts, f_cols = apply_filters(pts_full, cols_full, conf_full, sky_full, percentile=p)
        fname = ply_folder / f"{dataset_name}_top{100-p}percent.ply"
        save_ply(f_pts, f_cols, str(fname))
        # print(f"    Saved {fname.name}") # Commented out to reduce clutter

    # F. Save Poses & Depth Maps
    (out_dir / "poses").mkdir(exist_ok=True)
    (out_dir / "depths").mkdir(exist_ok=True)
    
    print("  > Saving poses and depth maps (NPY & PNG)...")
    for i in range(num_frames):
        np.savetxt(out_dir / "poses" / f"{i:06d}_extrinsic.txt", ext_cpu[i])
        np.savetxt(out_dir / "poses" / f"{i:06d}_intrinsic.txt", intr_cpu[i])
        
        # Save Raw NPY
        d_map = depth[i]
        np.save(out_dir / "depths" / f"{i:06d}_depth.npy", d_map)
        
        # Save Visual PNG (Normalized 0-255)
        # Normalize valid range for better contrast
        d_valid = d_map[(d_map > 0) & (d_map < DEPTH_CUTOFF)]
        if len(d_valid) > 0:
            vmin, vmax = d_valid.min(), d_valid.max()
        else:
            vmin, vmax = d_map.min(), d_map.max()
            
        d_norm = (d_map - vmin) / (vmax - vmin + 1e-6)
        d_norm = np.clip(d_norm, 0, 1) * 255.0
        d_uint8 = d_norm.astype(np.uint8)
        
        # Apply Colormap (Inferno is good for depth)
        d_color = cv2.applyColorMap(d_uint8, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(out_dir / "depths" / f"{i:06d}_depth.png"), d_uint8)

    print(f"  > Done: {dataset_name}")

def main():
    # HARDCODED INPUT PATH
    import argparse
    args = argparse.Namespace()
    args.input = "/projects/standard/csci5561/shared/G11/my_datasets" 
    
    p = Path(args.input)
    if not p.exists(): return print("Path not found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading VGGT on {device}...")
    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location=device))
    model.eval().to(device)

    for sub in sorted([x for x in p.iterdir() if x.is_dir() and not x.name.startswith('.')]):
        try:
            process_single_dataset(sub, model, device)
        except Exception as e:
            print(f"Error on {sub.name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()