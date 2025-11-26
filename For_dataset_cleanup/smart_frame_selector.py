import cv2
import numpy as np
import os
import glob
import json
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================

# MODE 1: CALIBRATION (Run this first)
# Path to your manually selected 40 "Good" frames
REF_GOOD_DIR = "/path/to/your/40_good_frames"

# MODE 2: SELECTION (Run this second)
# Path to the new dataset you want to sort
TARGET_DATASET_DIR = "/path/to/new/dataset/rgb"
OUTPUT_JSON = "smart_selected_frames.json"

# DIVERSITY SETTINGS (Crucial for "Multiple Views")
# How different must a frame be from the previous one to be kept?
# 0 = Keep duplicates. 255 = Must be totally opposite.
# Start with 15.0. If you get too many similar frames, increase to 25.0.
DIVERSITY_THRESHOLD = 15.0 

# ==========================================

def calculate_metrics(image_path):
    """Calculates the 'DNA' of a frame: Blur and Content Detail."""
    img = cv2.imread(image_path)
    if img is None: return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Blur Score (Laplacian Variance)
    # High = Sharp, Low = Blurry
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Entropy (Detail/Texture)
    # High = Complex scene, Low = Flat wall/floor
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / (hist.sum() + 1e-7)
    # Filter zero entries to avoid log(0)
    hist_norm = hist_norm[hist_norm > 0]
    entropy = -np.sum(hist_norm * np.log2(hist_norm))
    
    return blur_var, entropy, gray

def calibrate(reference_dir):
    """
    Analyzes your 'Good' frames to find the baseline quality.
    """
    print(f"--- CALIBRATION MODE ---")
    print(f"Scanning reference frames in: {reference_dir}")
    
    files = sorted(glob.glob(os.path.join(reference_dir, "*.png")))
    if not files:
        print("Error: No images found in reference directory.")
        return None, None

    blurs = []
    entropies = []
    
    for f in tqdm(files):
        b, e, _ = calculate_metrics(f)
        blurs.append(b)
        entropies.append(e)
        
    blurs = np.array(blurs)
    entropies = np.array(entropies)
    
    # We set the threshold to include ~90% of your good frames.
    # (Mean - 1 Standard Deviation) is a robust statistical floor.
    suggested_blur_thresh = max(10.0, blurs.mean() - blurs.std())
    suggested_entropy_thresh = max(1.0, entropies.mean() - entropies.std())
    
    print("\n=== CALIBRATION RESULTS ===")
    print(f"Avg Blur: {blurs.mean():.2f} (Std: {blurs.std():.2f})")
    print(f"Avg Entropy: {entropies.mean():.2f} (Std: {entropies.std():.2f})")
    print("-" * 30)
    print(f"RECOMMENDED THRESHOLDS:")
    print(f"MIN_BLUR = {suggested_blur_thresh:.2f}")
    print(f"MIN_ENTROPY = {suggested_entropy_thresh:.2f}")
    print("Use these numbers in the select() function below.")
    print("=" * 30)
    
    return suggested_blur_thresh, suggested_entropy_thresh

def select_frames(dataset_dir, min_blur, min_entropy):
    """
    Selects frames that meet quality standards AND are visually unique.
    """
    print(f"\n--- SELECTION MODE ---")
    print(f"Filtering {dataset_dir}...")
    print(f"Criteria: Blur > {min_blur:.2f}, Entropy > {min_entropy:.2f}, Diff > {DIVERSITY_THRESHOLD}")
    
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.png")))
    selected_frames = []
    
    # For Diversity Check
    last_kept_img = None
    last_kept_idx = -1
    
    dropped_quality = 0
    dropped_redundant = 0
    
    for i, f in enumerate(tqdm(files)):
        # 1. Quality Check
        curr_blur, curr_entropy, curr_img_gray = calculate_metrics(f)
        
        if curr_blur < min_blur or curr_entropy < min_entropy:
            dropped_quality += 1
            continue
            
        # 2. Diversity/Novelty Check (The "Multiple Views" logic)
        is_novel = True
        if last_kept_img is not None:
            # Calculate absolute difference between current and last kept frame
            # We resize to small 64x64 to speed up comparison and ignore pixel noise
            s1 = cv2.resize(last_kept_img, (64, 64))
            s2 = cv2.resize(curr_img_gray, (64, 64))
            
            diff_score = np.mean(cv2.absdiff(s1, s2))
            
            if diff_score < DIVERSITY_THRESHOLD:
                is_novel = False
                dropped_redundant += 1
        
        # 3. Final Decision
        # We ALWAYS keep the first frame that passes quality
        if is_novel or last_kept_img is None:
            # Parse timestamp for TUM format
            try:
                ts = float(os.path.basename(f).replace(".png", ""))
            except:
                ts = float(i)
                
            selected_frames.append({
                "path": f,
                "timestamp": ts,
                "original_index": i,
                "blur_score": curr_blur,
                "diff_score": diff_score if last_kept_img is not None else 100.0
            })
            
            last_kept_img = curr_img_gray
            last_kept_idx = i

    print("\n=== SELECTION SUMMARY ===")
    print(f"Total Frames: {len(files)}")
    print(f"Selected: {len(selected_frames)}")
    print(f"Dropped (Low Quality): {dropped_quality}")
    print(f"Dropped (Redundant/Similar): {dropped_redundant}")
    
    with open(OUTPUT_JSON, 'w') as f_out:
        json.dump(selected_frames, f_out, indent=4)
    print(f"Saved list to {OUTPUT_JSON}")

if __name__ == "__main__":
    # STEP 1: Uncomment this line to calculate thresholds from your 40 good frames
    # threshold_blur, threshold_entropy = calibrate(REF_GOOD_DIR)
    
    # STEP 2: Once you have the numbers from Step 1, hardcode them here and run this:
    # (Example values, replace with yours)
    MY_CALIBRATED_BLUR = 120.0  
    MY_CALIBRATED_ENTROPY = 4.5
    
    select_frames(TARGET_DATASET_DIR, MY_CALIBRATED_BLUR, MY_CALIBRATED_ENTROPY)