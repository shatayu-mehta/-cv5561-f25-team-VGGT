# import os
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# from torchvision import transforms as TF
# from ultralytics import YOLO
# from tqdm import tqdm

# # ---------------------------------------------------------
# # 1. VGGT-Style Image Preprocessing
# # ---------------------------------------------------------
# def preprocess_image_vggt_style(image_path, target_size=518):
#     """
#     Resizes and crops an image exactly like VGGT's default inference mode.
#     Returns: numpy.ndarray (H, W, 3) in uint8 [0-255] (RGB format)
#     """
#     try:
#         img = Image.open(image_path)
#         if img.mode == "RGBA":
#             background = Image.new("RGBA", img.size, (255, 255, 255, 255))
#             img = Image.alpha_composite(background, img)
#         img = img.convert("RGB")

#         width, height = img.size
#         new_width = target_size
#         new_height = round(height * (new_width / width) / 14) * 14

#         img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

#         to_tensor = TF.ToTensor()
#         img_tensor = to_tensor(img) 

#         if new_height > target_size:
#             start_y = (new_height - target_size) // 2
#             img_tensor = img_tensor[:, start_y : start_y + target_size, :]

#         img_numpy = img_tensor.permute(1, 2, 0).numpy()
#         img_numpy = (img_numpy * 255).astype(np.uint8)

#         return img_numpy
#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")
#         return None

# # ---------------------------------------------------------
# # 2. Dataset Loading
# # ---------------------------------------------------------
# def process_dataset(dataset_root_dir):
#     processed_images = {}
#     print(f"Scanning '{dataset_root_dir}' for 'images' folders...")

#     for root, dirs, files in os.walk(dataset_root_dir):
#         if os.path.basename(root) == 'images':
#             print(f"Found images folder: {root}")
#             for file in files:
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#                     full_path = os.path.join(root, file)
#                     img_array = preprocess_image_vggt_style(full_path)
#                     if img_array is not None:
#                         processed_images[full_path] = img_array
#     return processed_images

# # ---------------------------------------------------------
# # 3. Object Cutout Generation (Background Removal)
# # ---------------------------------------------------------
# def generate_object_cutouts(processed_images, model_path='yolov8n-seg.pt'):
#     """
#     Runs YOLO segmentation. 
#     Keeps original pixels for detected objects, sets background to black.
#     """
#     cutout_images = {}
    
#     if not processed_images:
#         print("Error: No images found.")
#         return {}
        
#     sorted_filepaths = sorted(processed_images.keys())
    
#     print(f"Loading YOLO model: {model_path}...")
#     try:
#         model = YOLO(model_path)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return {}

#     print(f"Starting inference on {len(sorted_filepaths)} frames...")

#     for filepath in tqdm(sorted_filepaths, desc="Generating Cutouts"):
#         img_array = processed_images[filepath]
#         img_h, img_w = img_array.shape[:2]
        
#         # Start with a mask of all zeros (all background)
#         combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
#         results = model(img_array, verbose=False)
#         result = results[0]

#         if result.masks is not None:
#             raw_masks = result.masks.data.cpu().numpy()
            
#             # Combine all detected object masks into one
#             for raw_mask in raw_masks:
#                 resized_mask = cv2.resize(raw_mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
#                 binary_mask = (resized_mask > 0.5).astype(np.uint8)
#                 combined_mask = np.maximum(combined_mask, binary_mask)
        
#         # Apply the mask to the original image
#         # 1. Expand mask to 3 channels (H, W, 1) -> (H, W, 3) to match RGB image
#         mask_3ch = np.stack([combined_mask] * 3, axis=-1)
        
#         # 2. Multiply: Pixels with mask=1 stay, pixels with mask=0 become black
#         object_cutout = img_array * mask_3ch
        
#         cutout_images[filepath] = object_cutout

#     print("Inference complete.")
#     return cutout_images

# # ---------------------------------------------------------
# # 4. Main Execution
# # ---------------------------------------------------------
# def main():
#     MY_DATASET_PATH = "/projects/standard/csci5561/shared/G11/my_datasets" 
    
#     if not os.path.exists(MY_DATASET_PATH):
#         print(f"Error: Path '{MY_DATASET_PATH}' does not exist.")
#         return {}

#     output_data = process_dataset(MY_DATASET_PATH)
#     print(f"\nProcessing Complete. Total images: {len(output_data)}")
#     return output_data

# if __name__ == "__main__":
#     # 1. Load Data
#     output_data = main()
    
#     if output_data:
#         # 2. Generate Cutouts (Original Colors)
#         all_cutouts = generate_object_cutouts(output_data, model_path='yolov8n-seg.pt')
        
#         # 3. Debug Visualization
#         print("\n--- Saving Debug Images ---")
#         debug_dir = "debug_cutouts"
#         os.makedirs(debug_dir, exist_ok=True)

#         saved_counts = {} 

#         for filepath, cutout in all_cutouts.items():
#             # Extract dataset name
#             parent_dir = os.path.dirname(filepath)
#             dataset_name = os.path.basename(os.path.dirname(parent_dir))
#             filename = os.path.basename(filepath)
            
#             if dataset_name not in saved_counts:
#                 saved_counts[dataset_name] = 0
            
#             # Save first 2 examples per dataset
#             if saved_counts[dataset_name] < 2:
#                 save_name = f"{dataset_name}_{filename}"
#                 save_path = os.path.join(debug_dir, save_name)
                
#                 # IMPORTANT: Convert RGB (PIL/VGGT standard) to BGR (OpenCV standard) for saving
#                 cutout_bgr = cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR)
                
#                 cv2.imwrite(save_path, cutout_bgr)
#                 print(f"Saved: {save_name}")
#                 saved_counts[dataset_name] += 1

#         print(f"\nCheck the '{debug_dir}' folder. You should see your objects in original colors on black backgrounds.")


import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF
from ultralytics import YOLO
from tqdm import tqdm

# ---------------------------------------------------------
# 1. VGGT-Style Image Preprocessing
# ---------------------------------------------------------
def preprocess_image_vggt_style(image_path, target_size=518):
    """
    Resizes and crops an image exactly like VGGT's default inference mode.
    Returns: numpy.ndarray (H, W, 3) in uint8 [0-255] (RGB format)
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
        print(f"Error processing {image_path}: {e}")
        return None

# ---------------------------------------------------------
# 2. Dataset Loading
# ---------------------------------------------------------
def process_dataset(dataset_root_dir):
    """
    Parses directory structure to find 'images' folders and processes files.
    """
    processed_images = {}
    print(f"Scanning '{dataset_root_dir}' for 'images' folders...")

    for root, dirs, files in os.walk(dataset_root_dir):
        if os.path.basename(root) == 'images':
            print(f"Found images folder: {root}")
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    full_path = os.path.join(root, file)
                    img_array = preprocess_image_vggt_style(full_path)
                    if img_array is not None:
                        processed_images[full_path] = img_array
    return processed_images

# ---------------------------------------------------------
# 3. Main Processing Logic (Inference + Saving)
# ---------------------------------------------------------
def run_inference_and_save(processed_images, model_path='yolov8n-seg.pt'):
    """
    Runs YOLO segmentation, saves cutouts to local folders, and writes detection logs.
    """
    if not processed_images:
        print("Error: No images found.")
        return

    # Sort files
    sorted_filepaths = sorted(processed_images.keys())
    
    print(f"Loading YOLO model: {model_path}...")
    try:
        model = YOLO(model_path)
        class_names = model.names # Dictionary {0: 'person', 1: 'bicycle', ...}
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Starting inference on {len(sorted_filepaths)} frames...")

    # Dictionary to store detection logs per dataset
    # Structure: { '/path/to/dataset_dir': ["image01.png: chair", "image02.png: table"] }
    dataset_logs = {}

    for filepath in tqdm(sorted_filepaths, desc="Processing Images"):
        img_array = processed_images[filepath]
        img_h, img_w = img_array.shape[:2]
        
        # --- 1. Determine Output Paths ---
        # Input: .../my_dataset/kitchen/images/00.png
        # Goal:  .../my_dataset/kitchen/cutouts/00.png
        
        images_dir = os.path.dirname(filepath)       # .../kitchen/images
        dataset_dir = os.path.dirname(images_dir)    # .../kitchen
        filename = os.path.basename(filepath)        # 00.png
        
        cutouts_dir = os.path.join(dataset_dir, "cutouts")
        os.makedirs(cutouts_dir, exist_ok=True)
        save_path = os.path.join(cutouts_dir, filename)

        # Initialize log entry for this dataset if needed
        if dataset_dir not in dataset_logs:
            dataset_logs[dataset_dir] = []

        # --- 2. Run YOLO Inference ---
        results = model(img_array, verbose=False)
        result = results[0]

        # Initialize mask (background)
        combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        detected_objects = []

        # --- 3. Process Detections ---
        if result.masks is not None:
            raw_masks = result.masks.data.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for raw_mask, cls_id in zip(raw_masks, class_ids):
                # Resize mask
                resized_mask = cv2.resize(raw_mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                binary_mask = (resized_mask > 0.5).astype(np.uint8)
                
                # Add to combined mask
                combined_mask = np.maximum(combined_mask, binary_mask)
                
                # Record object name
                obj_name = class_names[cls_id]
                detected_objects.append(obj_name)
        
        # --- 4. Create Cutout (Background Removal) ---
        # Expand mask to 3 channels
        mask_3ch = np.stack([combined_mask] * 3, axis=-1)
        # Apply mask: Object pixels stay, background becomes black (0,0,0)
        object_cutout = img_array * mask_3ch
        
        # --- 5. Save Image to Disk ---
        # Convert RGB -> BGR for OpenCV
        cutout_bgr = cv2.cvtColor(object_cutout, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, cutout_bgr)

        # --- 6. Log Detections ---
        # Format: "00.png: chair, table" or "00.png: (no detection)"
        if detected_objects:
            # Remove duplicates if desired, or keep count
            obj_str = ", ".join(detected_objects)
        else:
            obj_str = "(no detection)"
            
        dataset_logs[dataset_dir].append(f"{filename}: {obj_str}")

    # --- 7. Save Text Files ---
    print("\nWriting detection logs...")
    for d_dir, logs in dataset_logs.items():
        log_path = os.path.join(d_dir, "detected_objects.txt")
        try:
            with open(log_path, "w") as f:
                f.write("\n".join(logs))
            print(f"Log saved: {log_path}")
        except Exception as e:
            print(f"Failed to write log for {d_dir}: {e}")

# ---------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------
def main():
    # --- Configuration ---
    MY_DATASET_PATH = "/projects/standard/csci5561/shared/G11/my_datasets" 
    # ---------------------

    if not os.path.exists(MY_DATASET_PATH):
        print(f"Error: Path '{MY_DATASET_PATH}' does not exist.")
        return

    # 1. Load Images
    output_data = process_dataset(MY_DATASET_PATH)
    print(f"\nImages loaded. Total: {len(output_data)}")
    
    if output_data:
        # 2. Run Inference, Save Images, Write Logs
        run_inference_and_save(output_data, model_path='yolov8n-seg.pt')
        print("\nAll done! Check your dataset folders for 'cutouts' directories and 'detected_objects.txt'.")

if __name__ == "__main__":
    main()