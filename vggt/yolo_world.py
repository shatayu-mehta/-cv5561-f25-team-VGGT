import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF
from ultralytics import YOLOWorld, SAM
from tqdm import tqdm

# ---------------------------------------------------------
# 1. VGGT-Style Image Preprocessing
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
        print(f"Error processing {image_path}: {e}")
        return None

# ---------------------------------------------------------
# 2. Dataset Loading
# ---------------------------------------------------------
def process_dataset(dataset_root_dir):
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
# 3. Grounded SAM Inference (YOLO-World + SAM)
# ---------------------------------------------------------
def run_grounded_sam_and_save(processed_images):
    if not processed_images:
        print("Error: No images found.")
        return

    sorted_filepaths = sorted(processed_images.keys())
    
    # 1. Load Models
    print("Loading models...")
    try:
        # Detector
        detector = YOLOWorld('yolov8l-world.pt') 
        # Segmenter
        segmenter = SAM('sam_b.pt') 
        
        # --- DEFINE CLASSES ---
        # 1. Your Custom Classes (for the drawers/cabinets)
        my_custom_objects = [
            "drawer unit", "filing cabinet", "office drawer", 
            "storage unit", "pedestal cabinet", "blue cabinet"
        ]
        
        # 2. Standard COCO Classes (for apples, people, etc.)
        # This list covers the 80 common objects standard models usually know.
        common_objects = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]
        
        # Combine them!
        all_classes = my_custom_objects + common_objects
        
        # Set the model to look for EVERYTHING in this list
        detector.set_classes(all_classes)
        print(f"Detector initialized. Looking for {len(all_classes)} distinct object types.")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    print(f"Starting inference on {len(sorted_filepaths)} frames...")
    dataset_logs = {}

    for filepath in tqdm(sorted_filepaths, desc="Processing (Grounded SAM)"):
        img_array = processed_images[filepath]
        img_h, img_w = img_array.shape[:2]
        
        # Setup paths
        images_dir = os.path.dirname(filepath)
        dataset_dir = os.path.dirname(images_dir)
        filename = os.path.basename(filepath)
        
        cutouts_dir = os.path.join(dataset_dir, "cutouts")
        os.makedirs(cutouts_dir, exist_ok=True)
        save_path = os.path.join(cutouts_dir, filename)

        if dataset_dir not in dataset_logs:
            dataset_logs[dataset_dir] = []

        # --- STEP A: Detect with YOLO-World ---
        # conf=0.05 is very low to ensure we catch objects
        det_results = detector.predict(img_array, verbose=False, conf=0.05)
        det_result = det_results[0]

        combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        detected_objects = []

        # If YOLO found boxes, pass them to SAM
        if det_result.boxes is not None and len(det_result.boxes) > 0:
            bboxes = det_result.boxes.xyxy
            class_ids = det_result.boxes.cls.cpu().numpy().astype(int)
            names = det_result.names
            
            for cls_id in class_ids:
                # Add check to ensure ID exists in names (safety)
                if cls_id in names:
                    detected_objects.append(names[cls_id])

            # --- STEP B: Segment with SAM ---
            sam_results = segmenter(img_array, bboxes=bboxes, verbose=False)
            
            if sam_results[0].masks is not None:
                raw_masks = sam_results[0].masks.data.cpu().numpy()
                
                for raw_mask in raw_masks:
                    # Convert boolean mask to uint8 (0 or 1) BEFORE resizing
                    mask_uint8 = raw_mask.astype(np.uint8)
                    resized_mask = cv2.resize(mask_uint8, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Threshold back to binary 0/1
                    binary_mask = (resized_mask > 0.5).astype(np.uint8)
                    combined_mask = np.maximum(combined_mask, binary_mask)

        # --- STEP C: Create Cutout ---
        mask_3ch = np.stack([combined_mask] * 3, axis=-1)
        object_cutout = img_array * mask_3ch
        
        # Save (RGB -> BGR)
        cutout_bgr = cv2.cvtColor(object_cutout, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, cutout_bgr)

        # Log
        if detected_objects:
            # Remove duplicates for cleaner logs
            unique_objs = list(set(detected_objects))
            obj_str = ", ".join(unique_objs)
        else:
            obj_str = "(no detection)"
        dataset_logs[dataset_dir].append(f"{filename}: {obj_str}")

    # Save Logs
    print("\nWriting detection logs...")
    for d_dir, logs in dataset_logs.items():
        log_path = os.path.join(d_dir, "detected_objects.txt")
        with open(log_path, "w") as f:
            f.write("\n".join(logs))
        print(f"Log saved: {log_path}")

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

    output_data = process_dataset(MY_DATASET_PATH)
    print(f"\nImages loaded. Total: {len(output_data)}")
    
    if output_data:
        run_grounded_sam_and_save(output_data)
        print("\nAll done!")

if __name__ == "__main__":
    main()