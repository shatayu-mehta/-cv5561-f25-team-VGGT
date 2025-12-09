import torch

import numpy as np

import cv2

import open3d as o3d

import os

import glob # <-- Added this import



# --- Import VGGT modules ---

# (This assumes you are in the 'vggt-env' virtual environment

# and have installed the requirements.txt)

try:

    from vggt.models.vggt import VGGT

    from vggt.utils.load_fn import load_and_preprocess_images

    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    from vggt.utils.geometry import unproject_depth_map_to_point_map

except ImportError as e:

    print(f"Error: {e}")

    print("Could not import VGGT modules. Did you activate the virtual environment?")

    print("Run: source vggt-env/bin/activate")

    print("And ensure you ran: pip install -r requirements.txt")

    exit(1)



# --- Helper function to save a point cloud ---

def save_point_cloud(points_np, colors_np, filename):

    """Saves a NumPy point cloud and colors to a .ply file."""

    # Ensure points and colors are flat arrays

    points_np = points_np.reshape(-1, 3)

    colors_np = colors_np.reshape(-1, 3)

    

    # Create Open3D point cloud object

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points_np)

    

    # Normalize colors to [0, 1] if they are in [0, 255]

    if colors_np.max() > 1.0:

        colors_np = colors_np / 255.0

        

    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    

    # Save to file

    o3d.io.write_point_cloud(filename, pcd)

    print(f"Saved point cloud to {filename}")



def main():

    print("Starting VGGT inference script...")



    # --- 1. Setup Device ---

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print(f"Using device: {device} with dtype: {dtype}")



    # --- 2. Initialize Model ---

    print("Loading pre-trained VGGT-1B model...")

    # This will automatically download the model weights (5-6 GB) the first time.

    try:

        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

    except Exception as e:

        print(f"Error loading model: {e}")

        print("This could be a network issue or a problem with 'huggingface_hub'.")

        print("Ensure you have an internet connection.")

        exit(1)

        

    print("Model loaded successfully.")



    # --- 3. Load and Preprocess Images (Paths Updated) ---

    

    IMAGE_PATH_DIR = "/projects/standard/csci5561/shared/G11/vggt/examples/kitchen/images"

    

    print(f"Searching for images in: {IMAGE_PATH_DIR}")

    

    # Find all .png images in the directory

    all_image_files = sorted(glob.glob(os.path.join(IMAGE_PATH_DIR, "*.png")))

    

    # Select the first 4 images for this test run (you can change this number)

    image_paths = all_image_files[0:4]

    

    # Check if we found any images

    if not image_paths:

        print("="*50)

        print(f"ERROR: No '.png' images found in directory:")

        print(f"{IMAGE_PATH_DIR}")

        print("Please check the path and ensure it contains .png files.")

        print("="*50)

        exit(1)

        

    # Check if files exist

    for p in image_paths:

        if not os.path.exists(p):

            print(f"ERROR: Image file not found at: {p}")

            print("Please check your image paths.")

            exit(1)



    print(f"Loading and preprocessing {len(image_paths)} images...")

    try:

        # The 'load_and_preprocess_images' function is your data pipeline.

        # It handles resizing, normalization, and tensor conversion.

        # It returns a tensor of shape [NumImages, 3, Height, Width]

        images_tensor = load_and_preprocess_images(image_paths).to(device)

        

        # The model expects a batch dimension. Add it.

        # Shape becomes: [1, NumImages, 3, Height, Width]

        images_tensor = images_tensor.unsqueeze(0)

    

    except Exception as e:

        print(f"Error during image preprocessing: {e}")

        print("This can happen if files are corrupt or not images.")

        exit(1)



    print(f"Image tensor created with shape: {images_tensor.shape}")



    # --- 4. Run Inference ---

    print("Running model inference...")

    with torch.no_grad():

        with torch.cuda.amp.autocast(dtype=dtype):

            # This follows the "Detailed Usage" in the VGGT README

            

            # 1. Aggregator: Encodes images into tokens

            aggregated_tokens_list, ps_idx = model.aggregator(images_tensor)

            

            # 2. Camera Head: Predicts camera poses

            pose_enc = model.camera_head(aggregated_tokens_list)[-1]

            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])

            

            # 3. Depth Head: Predicts depth maps

            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_tensor, ps_idx)

            

            # 4. Point Map Head (from depth): Unproject depth to 3D points

            # This is generally more accurate than the direct point_map output

            point_map_by_unprojection = unproject_depth_map_to_point_map(

                depth_map.squeeze(0), 

                extrinsic.squeeze(0), 

                intrinsic.squeeze(0)

            )



    print("Inference complete.")

    

    # --- 5. Save Outputs (Goal for Day 1) ---

    print("Saving outputs...")

    

    # --- Define Output Directories and ensure they exist ---

    POINT_CLOUD_OUTPUT_DIR = "/projects/standard/csci5561/shared/G11/Outputs/RGB-D_TUM/Pt_clds"

    DEPTH_MAP_OUTPUT_DIR = "/projects/standard/csci5561/shared/G11/Outputs/RGB-D_TUM/Depth_maps"

    

    os.makedirs(POINT_CLOUD_OUTPUT_DIR, exist_ok=True)

    os.makedirs(DEPTH_MAP_OUTPUT_DIR, exist_ok=True)



    # Detach tensors from GPU and convert to NumPy

    # Squeeze the batch dimension (dim 0)

    depth_map_np = depth_map.squeeze(0).squeeze(-1).cpu().numpy() # <-- MODIFIED: Added .squeeze(-1) to remove trailing channel dim

    point_map_np = point_map_by_unprojection # This is already a numpy array from the helper fn

    

    # We also need the original images as NumPy for colors

    # Re-load the first image for its color data (un-preprocessed)

    # This is simpler than reversing the preprocessing

    original_image_for_color = cv2.imread(image_paths[0])

    # Resize it to match the depth map's dimensions

    depth_h, depth_w = depth_map_np[0].shape

    original_image_for_color = cv2.resize(original_image_for_color, (depth_w, depth_h))

    original_image_for_color_rgb = cv2.cvtColor(original_image_for_color, cv2.COLOR_BGR2RGB)



    # --- Save Depth Map for the *first* image ---

    # depth_map_np has shape [NumImages, H, W]

    first_depth_map = depth_map_np[0]

    output_depth_file = os.path.join(DEPTH_MAP_OUTPUT_DIR, "depth_map_0.png")

    

    # Save as a 16-bit PNG. This preserves the float precision better than 8-bit.

    # You can't save floats directly to PNG, so normalize and scale to 16-bit integer range

    first_depth_map_normalized = cv2.normalize(first_depth_map, None, 0, 65535, cv2.NORM_MINMAX)

    first_depth_map_16bit = first_depth_map_normalized.astype(np.uint16)

    cv2.imwrite(output_depth_file, first_depth_map_16bit)

    print(f"Saved 16-bit depth map to {output_depth_file}")



    # --- Save Point Cloud for the *first* image ---

    # point_map_np has shape [NumImages, H, W, 3]

    first_point_map = point_map_np[0]

    output_ply_file = os.path.join(POINT_CLOUD_OUTPUT_DIR, "point_cloud_0.ply")

    

    # Use the helper function to save

    save_point_cloud(first_point_map, original_image_for_color_rgb, output_ply_file)



    print("\n--- Day 1 Success! ---")

    print("You have successfully generated:")

    print(f"1. A depth map: {output_depth_file}")

    print(f"2. A point cloud: {output_ply_file}")

    print("You are ready for Day 2 (Visualization).")





if __name__ == "__main__":

    # Removed 'inspect' as it's no longer needed for the error check

    

    # You may need to install these if you haven't already

    # pip install opencv-python-headless open3d

    main()
