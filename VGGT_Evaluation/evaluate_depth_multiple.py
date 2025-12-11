#!/usr/bin/env python3
"""
Depth evaluation between TUM RGB-D ground truth and VGGT prediction.

Features:
- Single-pair mode
- Folder mode: evaluate all GT/pred pairs in two folders, names matched by basename.
- Handles different resolutions.
- Masks out invalid depth (including black borders where depth == 0).
- Performs global scale alignment (median or L2).
- Computes standard depth metrics.
- Aggregates results over all frames and plots per-metric curves with cumulative mean.

Requires:
    numpy, opencv-python, matplotlib
"""

import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ---------- I/O helpers ----------

def load_depth(path, is_tum=False, depth_scale=5000.0):
    """
    Load depth from .png or .npy.
    If is_tum=True, assume TUM RGB-D format (uint16, value/5000 = meters).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        depth = np.load(path).astype(np.float32)
    else:
        # PNG or other image
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if img.ndim == 3:
            # drop color channels if present
            img = img[..., 0]
        depth = img.astype(np.float32)

        if is_tum:
            # TUM convention: uint16, depth[m] = raw / 5000.0, 0 == invalid
            depth = depth / float(depth_scale)

    return depth


def resize_to_match(pred, gt_shape):
    """Resize prediction depth map to match ground-truth resolution."""
    H_gt, W_gt = gt_shape
    H_p, W_p = pred.shape
    if (H_gt, W_gt) == (H_p, W_p):
        return pred
    pred_resized = cv2.resize(pred, (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)
    return pred_resized


# ---------- Scale alignment ----------

def compute_scale_factor(gt, pred, mask, method="median"):
    """
    Compute global scale factor s such that pred_aligned = s * pred.

    method:
      - "median": s = median(gt) / median(pred)
      - "l2": least-squares s = sum(gt*pred) / sum(pred^2)
    """
    gt_valid = gt[mask]
    pred_valid = pred[mask]

    if pred_valid.size == 0:
        raise ValueError("No valid pixels after masking; check masks.")

    if method == "median":
        med_gt = np.median(gt_valid)
        med_pred = np.median(pred_valid)
        if med_pred == 0:
            raise ValueError("Median of predicted depth is zero; cannot median-scale.")
        s = med_gt / med_pred
    elif method == "l2":
        denom = np.sum(pred_valid ** 2)
        if denom == 0:
            raise ValueError("Sum of squared prediction is zero; cannot L2-scale.")
        s = np.sum(gt_valid * pred_valid) / denom
    else:
        raise ValueError(f"Unknown scale method: {method}")

    return float(s)


# ---------- Metrics ----------

def depth_metrics(gt, pred):
    """
    Compute standard depth metrics given already aligned & masked 1D arrays.

    Inputs:
      gt   : (N,) ground truth depths (meters)
      pred : (N,) predicted depths (meters, already scale-aligned)

    Returns dict of metrics.
    """
    assert gt.shape == pred.shape
    # avoid log(0) / log(negative)
    eps = 1e-6
    gt_safe = np.clip(gt, eps, None)
    pred_safe = np.clip(pred, eps, None)

    diff = pred_safe - gt_safe
    abs_diff = np.abs(diff)

    # Common metrics
    abs_rel = np.mean(abs_diff / gt_safe)
    sq_rel = np.mean((diff ** 2) / gt_safe)
    rmse = np.sqrt(np.mean(diff ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt_safe) - np.log(pred_safe)) ** 2))

    # Scale-invariant log RMSE (silog)
    log_diff = np.log(pred_safe) - np.log(gt_safe)
    silog = np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2)) * 100.0

    # Threshold accuracies
    ratio = np.maximum(gt_safe / pred_safe, pred_safe / gt_safe)
    delta1 = np.mean(ratio < 1.25)
    delta2 = np.mean(ratio < 1.25 ** 2)
    delta3 = np.mean(ratio < 1.25 ** 3)

    # MAE
    mae = np.mean(abs_diff)

    return {
        "num_pixels": int(gt_safe.size),
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "rmse_log": float(rmse_log),
        "silog": float(silog),
        "mae": float(mae),
        "delta1": float(delta1),
        "delta2": float(delta2),
        "delta3": float(delta3),
    }


# ---------- Per-pair evaluation ----------

def evaluate_pair(
    gt_path,
    pred_path,
    gt_is_tum=True,
    gt_depth_scale=5000.0,
    pred_is_tum=False,
    pred_depth_scale=1.0,
    max_depth=None,
    scale_method="median",
    verbose=True,
):
    """
    Evaluate one GT/pred pair.

    Returns a dict:
    {
        "metrics": <metrics dict>,
        "scale": float,
        "valid_pixels": int,
        "total_pixels": int,
        "gt_shape": (H, W),
        "pred_shape": (H, W)
    }
    """
    # Load
    gt = load_depth(gt_path, is_tum=gt_is_tum, depth_scale=gt_depth_scale)
    pred = load_depth(pred_path, is_tum=pred_is_tum, depth_scale=pred_depth_scale)

    H_gt, W_gt = gt.shape
    H_p, W_p = pred.shape

    if verbose:
        print(f"GT path:   {gt_path}")
        print(f"Pred path: {pred_path}")
        print(f"GT shape:   {gt.shape}, dtype={gt.dtype}")
        print(f"Pred shape: {pred.shape}, dtype={pred.dtype}")

    # Resize prediction to GT resolution
    pred_resized = resize_to_match(pred, gt.shape)

    # Build mask:
    # GT > 0 to remove black border and invalid pixels
    # Pred finite and > 0
    mask = np.isfinite(gt) & np.isfinite(pred_resized) & (gt > 0) & (pred_resized > 0)

    if max_depth is not None:
        mask &= (gt <= max_depth)

    valid_pixels = int(mask.sum())
    total_pixels = int(gt.size)

    if valid_pixels == 0:
        raise ValueError("Mask removed all pixels; check your data and thresholds.")

    if verbose:
        print(f"Valid pixels after masking: {valid_pixels} / {total_pixels}")

    # Scale alignment
    s = compute_scale_factor(gt, pred_resized, mask, method=scale_method)
    if verbose:
        print(f"Scale factor (pred -> gt) using '{scale_method}': {s:.6f}")

    pred_aligned = pred_resized * s

    # Compute metrics on valid pixels only
    gt_valid = gt[mask]
    pred_valid = pred_aligned[mask]

    metrics = depth_metrics(gt_valid, pred_valid)

    if verbose:
        print("\n=== Depth Evaluation Results ===")
        for k, v in metrics.items():
            if k == "num_pixels":
                print(f"{k:10s}: {v}")
            else:
                print(f"{k:10s}: {v:.6f}")
        print("-" * 60)

    return {
        "metrics": metrics,
        "scale": s,
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "gt_shape": (H_gt, W_gt),
        "pred_shape": (H_p, W_p),
    }


# ---------- Folder evaluation + aggregation ----------

def list_basename_pairs(gt_dir, pred_dir, gt_ext=".png", pred_ext=".npy"):
    """
    Find matching basenames between gt_dir and pred_dir.

    Assumes:
      - GT files live in gt_dir, predictions in pred_dir.
      - Basenames (without extension) are the same.
      - GT extension and pred extension can differ.

    Returns: list of (gt_path, pred_path, basename), sorted by basename.
    """
    gt_files = []
    for f in os.listdir(gt_dir):
        if gt_ext is not None:
            if not f.lower().endswith(gt_ext.lower()):
                continue
        full = os.path.join(gt_dir, f)
        if os.path.isfile(full):
            gt_files.append(full)

    if not gt_files:
        raise RuntimeError(f"No GT files found in {gt_dir} with ext={gt_ext}")

    gt_files = sorted(gt_files)
    pairs = []

    for gt_path in gt_files:
        base = os.path.splitext(os.path.basename(gt_path))[0]
        pred_path = os.path.join(pred_dir, base + pred_ext)
        if not os.path.isfile(pred_path):
            print(f"[WARN] No prediction found for basename '{base}' at {pred_path}, skipping.")
            continue
        pairs.append((gt_path, pred_path, base))

    if not pairs:
        raise RuntimeError("No GT/pred pairs found. Check naming and extensions.")

    return pairs


def evaluate_folder(
    gt_dir,
    pred_dir,
    gt_is_tum=True,
    gt_depth_scale=5000.0,
    pred_is_tum=False,
    pred_depth_scale=1.0,
    max_depth=None,
    scale_method="median",
    gt_ext=".png",
    pred_ext=".npy",
    make_plots=True,
    out_dir=".",
):
    """
    Evaluate all GT/pred pairs from two folders and aggregate metrics.

    Assumes same basenames in both folders (only extension differs).
    """
    pairs = list_basename_pairs(gt_dir, pred_dir, gt_ext=gt_ext, pred_ext=pred_ext)
    print(f"Found {len(pairs)} GT/pred pairs.")

    metrics_per_frame = []
    scales = []
    basenames = []

    for idx, (gt_path, pred_path, base) in enumerate(pairs):
        print(f"\n=== Pair {idx+1}/{len(pairs)}: {base} ===")
        result = evaluate_pair(
            gt_path=gt_path,
            pred_path=pred_path,
            gt_is_tum=gt_is_tum,
            gt_depth_scale=gt_depth_scale,
            pred_is_tum=pred_is_tum,
            pred_depth_scale=pred_depth_scale,
            max_depth=max_depth,
            scale_method=scale_method,
            verbose=True,
        )
        metrics_per_frame.append(result["metrics"])
        scales.append(result["scale"])
        basenames.append(base)

    # Aggregate metrics
    metric_keys = ["abs_rel", "sq_rel", "rmse", "rmse_log",
                   "silog", "mae", "delta1", "delta2", "delta3"]
    print("\n\n===== AGGREGATED RESULTS OVER ALL PAIRS =====")
    for key in metric_keys:
        vals = np.array([m[key] for m in metrics_per_frame], dtype=np.float64)
        print(f"{key:10s}: mean={vals.mean():.6f}, std={vals.std():.6f}, "
              f"min={vals.min():.6f}, max={vals.max():.6f}")

    scales = np.array(scales, dtype=np.float64)
    print(f"\nScale factor stats (per-frame, {scale_method}): "
          f"mean={scales.mean():.6f}, std={scales.std():.6f}, "
          f"min={scales.min():.6f}, max={scales.max():.6f}")

    # Plots: per-frame + cumulative mean for each metric
    if make_plots:
        os.makedirs(out_dir, exist_ok=True)
        x = np.arange(len(basenames))
        for key in metric_keys:
            vals = np.array([m[key] for m in metrics_per_frame], dtype=np.float64)
            cum_mean = np.cumsum(vals) / (np.arange(len(vals)) + 1)

            plt.figure(figsize=(8, 4))
            plt.plot(x, vals, marker="o", linestyle="-", label="per-frame")
            plt.plot(x, cum_mean, linestyle="--", label="cumulative mean")
            plt.xlabel("frame index")
            plt.ylabel(key)
            plt.title(f"{key} per frame ({scale_method} scale) with cumulative mean")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()

            out_path = os.path.join(out_dir, f"{key}_curve.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Saved plot: {out_path}")

        # Also plot scale factors over frames
        plt.figure(figsize=(8, 4))
        cum_scale_mean = np.cumsum(scales) / (np.arange(len(scales)) + 1)
        plt.plot(x, scales, marker="o", linestyle="-", label="per-frame scale")
        plt.plot(x, cum_scale_mean, linestyle="--", label="cumulative mean scale")
        plt.xlabel("frame index")
        plt.ylabel("scale factor")
        plt.title(f"Scale factors per frame ({scale_method})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, "scale_factors_curve.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved plot: {out_path}")


# ---------- CLI ----

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate depth prediction vs TUM RGB-D ground truth "
                    "(single pair or entire folders)."
    )

    # Single-pair mode
    parser.add_argument("--gt", type=str, default=None,
                        help="Path to single TUM ground-truth depth (PNG or NPY).")
    parser.add_argument("--pred", type=str, default=None,
                        help="Path to single predicted depth (PNG or NPY).")

    # Folder mode
    parser.add_argument("--gt-dir", type=str, default=None,
                        help="Directory containing GT depth maps (e.g., TUM PNGs).")
    parser.add_argument("--pred-dir", type=str, default=None,
                        help="Directory containing predicted depth maps (.npy or images).")
    parser.add_argument("--gt-ext", type=str, default=".png",
                        help="GT file extension to filter in gt-dir (default: .png).")
    parser.add_argument("--pred-ext", type=str, default=".npy",
                        help="Prediction file extension in pred-dir (default: .npy).")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Directory to save plots (folder mode).")

    # Common options
    parser.add_argument("--gt-is-tum", action="store_true", default=True,
                        help="Assume GT is TUM format (depth[m] = raw/5000, 0=invalid). Default: True.")
    parser.add_argument("--no-gt-is-tum", dest="gt_is_tum", action="store_false")

    parser.add_argument("--gt-depth-scale", type=float, default=5000.0,
                        help="Scale for TUM depths: depth[m] = raw/scale. Default: 5000.")

    parser.add_argument("--pred-is-tum", action="store_true", default=False,
                        help="If prediction is also in TUM-style PNG.")
    parser.add_argument("--pred-depth-scale", type=float, default=1.0,
                        help="Scale for prediction if it's raw depth (usually leave at 1.0).")

    parser.add_argument("--max-depth", type=float, default=None,
                        help="Optional max GT depth (in meters) to include, e.g. 10.0. Default: no limit.")

    parser.add_argument("--scale-method", type=str, default="median",
                        choices=["median", "l2"],
                        help="Scale alignment method. 'median' or 'l2' (least squares).")

    parser.add_argument("--no-plots", action="store_true",
                        help="Disable plotting in folder mode.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    single_mode = args.gt is not None and args.pred is not None
    folder_mode = args.gt_dir is not None and args.pred_dir is not None

    if single_mode and folder_mode:
        raise SystemExit("Error: Use EITHER (--gt & --pred) OR (--gt-dir & --pred-dir), not both.")

    if not single_mode and not folder_mode:
        raise SystemExit("Error: You must specify either single-pair mode (--gt & --pred) "
                         "or folder mode (--gt-dir & --pred-dir).")

    if single_mode:
        # single pair
        evaluate_pair(
            gt_path=args.gt,
            pred_path=args.pred,
            gt_is_tum=args.gt_is_tum,
            gt_depth_scale=args.gt_depth_scale,
            pred_is_tum=args.pred_is_tum,
            pred_depth_scale=args.pred_depth_scale,
            max_depth=args.max_depth,
            scale_method=args.scale_method,
            verbose=True,
        )
    else:
        # Folder mode
        evaluate_folder(
            gt_dir=args.gt_dir,
            pred_dir=args.pred_dir,
            gt_is_tum=args.gt_is_tum,
            gt_depth_scale=args.gt_depth_scale,
            pred_is_tum=args.pred_is_tum,
            pred_depth_scale=args.pred_depth_scale,
            max_depth=args.max_depth,
            scale_method=args.scale_method,
            gt_ext=args.gt_ext,
            pred_ext=args.pred_ext,
            make_plots=not args.no_plots,
            out_dir=args.out_dir,
        )
