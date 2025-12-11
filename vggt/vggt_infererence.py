"""
VGGT Inference and Point Cloud Processing

This script runs the Visual Geometry Grounded Transformer (VGGT) model to:
1. Generate depth maps from multi-view RGB images
2. Estimate camera poses (intrinsics and extrinsics)
3. Create and filter 3D point clouds with confidence-based cleaning
4. Export results as PLY files and depth visualizations

Usage:
    python vggt_infererence.py
    
Input: Images organized in my_datasets/<scene_name>/images/
Output: Point clouds, depth maps, and camera poses in my_datasets/<scene_name>/outputs/
"""

import os
import sys
import torch
import cv2
import numpy as np
import argparse
import traceback
import requests
import shutil
from tqdm import tqdm
from pathlib import Path

# Configuration parameters for scene processing
USE_SKY_SEGMENTATION = False  # Enable for outdoor scenes to filter sky regions
DEPTH_CUTOFF = 2.0           # Maximum depth in meters to keep (filters distant background)

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
    """
    Download a file from URL with redirect handling.
    
    Args:
        url: Source URL to download from
        filename: Local path where file will be saved
    """
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
    """
    Generate binary mask to filter sky regions from outdoor scenes.
    
    Args:
        image: Input BGR image (H, W, 3)
        session: ONNX runtime session for sky segmentation model
        
    Returns:
        Binary mask where 1=keep (not sky), 0=remove (sky)
    """
    if session is None:
        return np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
    
    # Preprocess: resize and normalize using ImageNet statistics
    img = cv2.resize(image, (320, 320))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    img = (img / 255 - mean) / std
    
    # Reshape to NCHW format for ONNX model
    img = img.transpose(2, 0, 1)[None]
    img = img.astype(np.float32)

    # Run inference
    iname = session.get_inputs()[0].name
    oname = session.get_outputs()[0].name
    mask = session.run([oname], {iname: img})[0].squeeze()

    # Postprocess: resize to original dimensions and threshold
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    norm_mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    
    return (norm_mask > 0.12).astype(np.float32)

# -----------------------------------------------------------------------------
# GEOMETRY
# -----------------------------------------------------------------------------
def save_ply(points, colors, filename):
    """
    Export 3D point cloud to PLY format with RGB colors.
    
    Args:
        points: Nx3 array of 3D coordinates
        colors: Nx3 array of RGB values (0-255)
        filename: Output PLY file path
    """
    if len(points) == 0:
        return
        
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
    """
    Unproject 2D depth map to 3D world coordinates using camera parameters.
    
    This function:
    1. Converts pixel coordinates (u,v) and depth to camera space using intrinsics
    2. Transforms from camera space to world space using extrinsics
    3. Filters invalid/distant points based on DEPTH_CUTOFF
    
    Args:
        depth: (H, W) depth map in meters
        extrinsic: (4, 4) camera-to-world transformation matrix
        intrinsic: (3, 3) camera intrinsic matrix [fx, fy, cx, cy]
        
    Returns:
        world_pts: (N, 3) array of 3D points in world coordinates
        valid: Boolean mask indicating which pixels were unprojected
    """
    if depth.ndim > 2:
        depth = depth.squeeze()
    
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u, v = u.flatten(), v.flatten()
    z = depth.flatten()
    
    # Filter: remove invalid, distant, or non-finite depth values
    valid = (z > 0) & (z < DEPTH_CUTOFF) & np.isfinite(z)
    u, v, z = u[valid], v[valid], z[valid]
    
    if intrinsic.ndim > 2:
        intrinsic = intrinsic.squeeze()
    
    # Extract focal lengths and principal point from intrinsic matrix
    fx, fy = float(intrinsic[0,0]), float(intrinsic[1,1])
    cx, cy = float(intrinsic[0,2]), float(intrinsic[1,2])
    
    # Unproject to camera space
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    cam_pts = np.stack([x, y, z, np.ones_like(z)], axis=1)
    
    if extrinsic.ndim > 2:
        extrinsic = extrinsic.squeeze()

    try:
        # Transform to world coordinates: inv(extrinsic) maps camera to world
        world_pts = (np.linalg.inv(extrinsic) @ cam_pts.T).T[:, :3]
        return world_pts, valid
    except:
        return np.zeros((0,3)), valid

# -----------------------------------------------------------------------------
# FILTERING (Official Logic + Spatial Crop)
# -----------------------------------------------------------------------------
def filter_spatial_outliers(points, colors, confs, sky_mask, sigma=1.0):
    """
    Remove distant background points to focus on central object.
    
    Uses Interquartile Range (IQR) method to adaptively determine spatial bounds:
    - Computes scene centroid using median (robust to outliers)
    - Calculates distance of each point from center
    - Keeps points within q75 + sigma*IQR range
    
    This effectively removes walls, floors, and distant background while
    preserving the target object regardless of its size.
    
    Args:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors
        confs: (N,) array of confidence scores
        sky_mask: (N,) array of sky filter mask
        sigma: IQR multiplier for cutoff distance (higher = more inclusive)
        
    Returns:
        Filtered (points, colors, confs, sky_mask) tuples
    """
    if len(points) == 0:
        return points, colors, confs, sky_mask
    
    print("    - Running statistical spatial crop...")
    
    center = np.median(points, axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    
    # Adaptive cutoff using IQR method
    q25, q75 = np.percentile(dists, [25, 75])
    iqr = q75 - q25
    upper_bound = q75 + (sigma * iqr)
    
    print(f"      > Center: {center}")
    print(f"      > 75% of points are within {q75:.2f} units.")
    print(f"      > Cropping everything further than {upper_bound:.2f} units.")
    
    mask = dists < upper_bound
    return points[mask], colors[mask], confs[mask], sky_mask[mask]

def apply_filters(points, colors, confs, sky_mask_flat, percentile=80):
    """
    Apply confidence and color-based filtering to point cloud.
    
    Filtering steps:
    1. Remove pure black points (likely invalid/shadows): RGB sum < 16
    2. Remove pure white points (likely overexposed): all channels > 240
    3. Keep top (100 - percentile)% of points by confidence score
    
    Args:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors
        confs: (N,) array of confidence scores
        sky_mask_flat: (N,) sky mask (unused but kept for compatibility)
        percentile: Confidence threshold percentile (80 = keep top 20%)
        
    Returns:
        Filtered (points, colors) tuple
    """
    # Filter invalid colors
    not_black = colors.sum(axis=1) >= 16
    not_white = ~((colors > 240).all(axis=1))
    color_mask = not_black & not_white
    
    # Compute confidence threshold from valid points only
    valid_confs = confs[color_mask]
    if len(valid_confs) == 0:
        return np.array([]), np.array([])
    
    cutoff = np.percentile(valid_confs, percentile)
    final_mask = color_mask & (confs >= cutoff)
    
    return points[final_mask], colors[final_mask]

def process_single_dataset(dataset_path, model, device):
    """
    Run complete VGGT pipeline on a single dataset.
    
    Processing steps:
    1. Load and preprocess RGB images from dataset directory
    2. Run VGGT model to predict depth maps and camera poses
    3. Unproject depth to 3D points using camera parameters
    4. Apply spatial and confidence-based filtering
    5. Export multiple point cloud variants (raw and filtered at different percentiles)
    6. Save depth maps (NPY and PNG) and camera poses
    
    Args:
        dataset_path: Path to dataset directory containing images/ subfolder
        model: Loaded VGGT model instance
        device: Torch device ('cuda' or 'cpu')
        
    Output structure:
        <dataset>/outputs/
        ├── pointclouds/  # PLY files at various filtering levels
        ├── depths/       # Depth maps (.npy raw, .png visualization)
        └── poses/        # Camera intrinsics and extrinsics (.txt)
    """
    p = Path(dataset_path)
    dataset_name = p.name 
    image_dir = p / "images" if (p / "images").exists() else p
    out_dir = p / "outputs"
    
    # Clean up old outputs to ensure fresh results
    if out_dir.exists():
        print(f"  > Cleaning up old outputs in {out_dir}...")
        shutil.rmtree(out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    images = sorted([str(x) for x in image_dir.glob("*") if x.suffix.lower() in ['.png','.jpg','.jpeg']])
    if not images:
        return

    num_frames = len(images)
    print(f"  > Processing {dataset_name} ({num_frames} frames)...") # Initialize sky segmentation model for outdoor scenes (if enabled)
    global sky_session
    if USE_SKY_SEGMENTATION and HAS_ONNX and sky_session is None:
        if not os.path.exists("skyseg.onnx"):
            download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")
        try:
            sky_session = onnxruntime.InferenceSession("skyseg.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except:
            print("    (SkySeg failed to load, continuing without it)")

    # Run VGGT model inference with mixed precision for efficiency
    images_tensor = load_and_preprocess_images(images).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        preds = model(images_tensor)
    
    # Extract camera poses from model predictions
    ext, intr = pose_encoding_to_extri_intri(preds["pose_enc"], images_tensor.shape[-2:])
    
    # Reshape depth predictions to (num_frames, H, W) format
    raw_depth = preds["depth"].float().cpu().numpy()
    if raw_depth.shape[-1] == 1:
        raw_depth = raw_depth.squeeze(-1)
    
    H, W = raw_depth.shape[-2], raw_depth.shape[-1]
    
    try:
        depth = raw_depth.reshape(num_frames, H, W)
    except ValueError:
        print(f"    [WARNING] Shape mismatch: Model output {raw_depth.shape} vs expected {num_frames} frames.")
        depth = raw_depth.reshape(-1, H, W)
        num_frames = depth.shape[0]

    # Reshape extrinsics: handle both 3x4 and 4x4 matrix formats
    ext_flat = ext.float().cpu().numpy().flatten()
    elements_per_matrix = ext_flat.shape[0] // num_frames
    
    if elements_per_matrix == 12:  # 3x4 format needs bottom row added
        ext_cpu = ext_flat.reshape(num_frames, 3, 4)
        last_row = np.array([0, 0, 0, 1]).reshape(1, 1, 4).repeat(num_frames, axis=0)
        ext_cpu = np.concatenate([ext_cpu, last_row], axis=1)
    else:  # Already 4x4
        ext_cpu = ext_flat.reshape(num_frames, 4, 4)

    # Reshape intrinsics to (num_frames, 3, 3)
    intr_cpu = intr.float().cpu().numpy().flatten().reshape(num_frames, 3, 3)
    
    # Extract and reshape confidence scores
    conf_raw = preds.get("depth_conf", torch.zeros_like(preds["depth"])).float().cpu().numpy()
    if conf_raw.size == depth.size:
        conf_raw = conf_raw.reshape(num_frames, H, W)
    else:
        conf_raw = conf_raw.reshape(num_frames, -1) 

    # Accumulate 3D points, colors, and confidence from all frames
    all_pts, all_cols, all_conf, all_sky = [], [], [], []
    
    print(f"    (Reshaped Depth: {depth.shape}, Extrinsics: {ext_cpu.shape})")

    for i in tqdm(range(num_frames)):
        # Unproject 2D depth to 3D points in world coordinates
        pts, mask = manual_unproject(depth[i], ext_cpu[i], intr_cpu[i])
        
        # Load RGB image and extract colors for valid points
        img = cv2.imread(images[i])
        img = cv2.resize(img, (W, H))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)[mask]
        
        # Extract confidence values for valid points
        c_frame = conf_raw[i]
        if c_frame.shape != (H, W):
             if c_frame.size == H*W:
                 c_frame = c_frame.reshape(H,W)
             else:
                 c_frame = np.ones((H, W))
        
        cnf = cv2.resize(c_frame, (W, H), interpolation=cv2.INTER_NEAREST).flatten()[mask]
        
        # Apply sky segmentation mask if enabled
        sky_val = np.ones_like(cnf)
        if sky_session:
            smask = get_sky_mask(img, sky_session)
            smask = cv2.resize(smask, (W, H), interpolation=cv2.INTER_NEAREST).flatten()[mask]
            sky_val = smask

        all_pts.append(pts)
        all_cols.append(rgb)
        all_conf.append(cnf)
        all_sky.append(sky_val)

    # Merge points from all frames into single point cloud
    pts_full = np.concatenate(all_pts)
    cols_full = np.concatenate(all_cols)
    conf_full = np.concatenate(all_conf)
    sky_full = np.concatenate(all_sky)

    # Zero out confidence for sky regions (if sky segmentation enabled)
    conf_full = conf_full * sky_full

    # Normalize confidence scores to 0-100 range
    if conf_full.max() <= 1.0:
        conf_full *= 100.0
    
    # Apply spatial filtering to remove distant background before percentile calculation
    pts_full, cols_full, conf_full, sky_full = filter_spatial_outliers(
        pts_full, cols_full, conf_full, sky_full, sigma=1.2
    )

    # Export point clouds at multiple quality levels
    ply_folder = out_dir / "pointclouds"
    ply_folder.mkdir(exist_ok=True)

    # Save spatially filtered but unfiltered by confidence
    save_ply(pts_full, cols_full, str(ply_folder / f"{dataset_name}_raw_cropped.ply"))

    # Generate point clouds at various confidence thresholds
    # Lower percentile = stricter filtering (keeps fewer, higher quality points)
    percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99]
    
    print("  > Saving percentile-filtered clouds...")
    for p in percentiles:
        f_pts, f_cols = apply_filters(pts_full, cols_full, conf_full, sky_full, percentile=p)
        fname = ply_folder / f"{dataset_name}_top{100-p}percent.ply"
        save_ply(f_pts, f_cols, str(fname))

    # Save camera poses and depth maps for each frame
    (out_dir / "poses").mkdir(exist_ok=True)
    (out_dir / "depths").mkdir(exist_ok=True)
    
    print("  > Saving poses and depth maps (NPY & PNG)...")
    for i in range(num_frames):
        # Save camera extrinsics and intrinsics as text files
        np.savetxt(out_dir / "poses" / f"{i:06d}_extrinsic.txt", ext_cpu[i])
        np.savetxt(out_dir / "poses" / f"{i:06d}_intrinsic.txt", intr_cpu[i])
        
        # Save depth: raw NPY for numerical accuracy
        d_map = depth[i]
        np.save(out_dir / "depths" / f"{i:06d}_depth.npy", d_map)
        
        # Save depth: normalized PNG for visualization
        d_valid = d_map[(d_map > 0) & (d_map < DEPTH_CUTOFF)]
        if len(d_valid) > 0:
            vmin, vmax = d_valid.min(), d_valid.max()
        else:
            vmin, vmax = d_map.min(), d_map.max()
            
        d_norm = (d_map - vmin) / (vmax - vmin + 1e-6)
        d_norm = np.clip(d_norm, 0, 1) * 255.0
        d_uint8 = d_norm.astype(np.uint8)
        
        # Apply colormap for better depth visualization
        d_color = cv2.applyColorMap(d_uint8, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(out_dir / "depths" / f"{i:06d}_depth.png"), d_uint8)

    print(f"  > Done: {dataset_name}")

def main():
    """
    Main entry point for VGGT batch processing.
    
    Loads the VGGT-1B model from Hugging Face and processes all scene
    directories found in the my_datasets folder. Each scene is processed
    independently with error handling to prevent one failure from stopping
    the entire batch.
    
    Expected directory structure:
        my_datasets/
        ├── scene1/images/*.jpg
        ├── scene2/images/*.jpg
        └── ...
    """
    args = argparse.Namespace()
    args.input = "/projects/standard/csci5561/shared/G11/my_datasets" 
    
    p = Path(args.input)
    if not p.exists():
        return print("Path not found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading VGGT on {device}...")
    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location=device))
    model.eval().to(device)

    # Process each scene directory independently
    for sub in sorted([x for x in p.iterdir() if x.is_dir() and not x.name.startswith('.')]):
        try:
            process_single_dataset(sub, model, device)
        except Exception as e:
            print(f"Error on {sub.name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()