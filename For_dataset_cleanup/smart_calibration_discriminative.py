import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURATION ---
# 1. Folder containing ONLY your 40 selected 'Good' images
GOOD_DIR = "/path/to/your/40_good_frames"

# 2. Folder containing the ENTIRE dataset (Good + Bad mixed is fine)
FULL_DATASET_DIR = "/path/to/full/dataset/rgb"

def get_metrics(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur (Variance of Laplacian)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Entropy (Shannon)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / (hist.sum() + 1e-7)
    hist_norm = hist_norm[hist_norm > 0]
    entropy = -np.sum(hist_norm * np.log2(hist_norm))
    
    return blur, entropy

def find_optimal_threshold(good_vals, bad_vals, metric_name):
    """
    Finds the cutoff that best separates the two distributions.
    Uses a simple Linear Discriminant Analysis (LDA) approximation.
    """
    good_mean = np.mean(good_vals)
    bad_mean = np.mean(bad_vals)
    
    good_std = np.std(good_vals)
    bad_std = np.std(bad_vals)
    
    # Calculate the intersection point of the two Gaussian distributions
    # Simple weighted average based on standard deviation (inverse variance weighting)
    # If bad data is very messy (high std), we trust the good data more.
    
    if good_mean > bad_mean:
        # We expect Good > Bad (like Blur)
        # Optimal cut is somewhere between the means
        cutoff = (good_mean * bad_std + bad_mean * good_std) / (good_std + bad_std)
        
        # Safety: Ensure cutoff isn't higher than the worst 'Good' frame
        # (We don't want to throw away frames you explicitly said were good)
        min_good = np.min(good_vals)
        if cutoff > min_good:
            print(f"  [Adjusting] Calculated optimal {metric_name} ({cutoff:.2f}) was too strict.")
            print(f"  Lowering to minimum 'Good' value: {min_good:.2f}")
            cutoff = min_good * 0.95 # 5% safety margin
            
    else:
        # We expect Good < Bad (rare for these metrics, but possible for others)
        cutoff = (good_mean * bad_std + bad_mean * good_std) / (good_std + bad_std)

    return cutoff

def main():
    print("--- 1. Scanning 'Good' Frames ---")
    good_files = set(os.path.basename(f) for f in glob.glob(os.path.join(GOOD_DIR, "*.png")))
    good_paths = glob.glob(os.path.join(GOOD_DIR, "*.png"))
    
    good_blurs = []
    good_entropies = []
    
    for f in tqdm(good_paths):
        b, e = get_metrics(f)
        good_blurs.append(b)
        good_entropies.append(e)

    print(f"\n--- 2. Scanning 'Bad' Frames (From Full Dataset) ---")
    # We identify 'Bad' frames as anything in Full Dataset NOT in the Good folder
    all_paths = glob.glob(os.path.join(FULL_DATASET_DIR, "*.png"))
    
    bad_blurs = []
    bad_entropies = []
    
    count = 0
    # Limit processing to ~500 bad frames to save time if dataset is huge
    MAX_BAD_SAMPLES = 500 
    
    for f in tqdm(all_paths):
        if os.path.basename(f) in good_files:
            continue # Skip the good ones
        
        # This is a "Bad" (or unselected) frame
        b, e = get_metrics(f)
        bad_blurs.append(b)
        bad_entropies.append(e)
        
        count += 1
        if count >= MAX_BAD_SAMPLES: break
    
    # Convert to arrays
    good_blurs = np.array(good_blurs)
    bad_blurs = np.array(bad_blurs)
    good_entropies = np.array(good_entropies)
    bad_entropies = np.array(bad_entropies)

    print("\n" + "="*40)
    print("       OPTIMAL THRESHOLD CALCULATION")
    print("="*40)

    # --- BLUR ANALYSIS ---
    print(f"\n[BLUR METRIC]")
    print(f"  Good Mean: {good_blurs.mean():.2f} (Std: {good_blurs.std():.2f})")
    print(f"  Bad Mean:  {bad_blurs.mean():.2f}  (Std: {bad_blurs.std():.2f})")
    
    blur_cutoff = find_optimal_threshold(good_blurs, bad_blurs, "Blur")
    print(f"  >> RECOMMENDED BLUR THRESHOLD: {blur_cutoff:.2f}")

    # --- ENTROPY ANALYSIS ---
    print(f"\n[ENTROPY METRIC]")
    print(f"  Good Mean: {good_entropies.mean():.2f} (Std: {good_entropies.std():.2f})")
    print(f"  Bad Mean:  {bad_entropies.mean():.2f}  (Std: {bad_entropies.std():.2f})")
    
    entropy_cutoff = find_optimal_threshold(good_entropies, bad_entropies, "Entropy")
    print(f"  >> RECOMMENDED ENTROPY THRESHOLD: {entropy_cutoff:.2f}")
    
    # --- VISUALIZATION ---
    print("\nGenerating distribution plots...")
    plt.figure(figsize=(12, 5))
    
    # Plot Blur
    plt.subplot(1, 2, 1)
    plt.hist(good_blurs, bins=20, alpha=0.6, label='Good', color='green', density=True)
    plt.hist(bad_blurs, bins=20, alpha=0.6, label='Bad/Ignored', color='red', density=True)
    plt.axvline(blur_cutoff, color='black', linestyle='--', linewidth=2, label=f'Cutoff: {blur_cutoff:.1f}')
    plt.title('Blur Distribution')
    plt.legend()
    
    # Plot Entropy
    plt.subplot(1, 2, 2)
    plt.hist(good_entropies, bins=20, alpha=0.6, label='Good', color='green', density=True)
    plt.hist(bad_entropies, bins=20, alpha=0.6, label='Bad/Ignored', color='red', density=True)
    plt.axvline(entropy_cutoff, color='black', linestyle='--', linewidth=2, label=f'Cutoff: {entropy_cutoff:.2f}')
    plt.title('Entropy Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("calibration_result.png")
    print("Saved plot to 'calibration_result.png'. Check it to verify the separation.")
    print("="*40)

if __name__ == "__main__":
    main()