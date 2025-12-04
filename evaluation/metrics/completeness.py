import numpy as np
from scipy.spatial import cKDTree

def completeness_score(reconstructed_pts, gt_pts, threshold=0.05):
    if len(reconstructed_pts) == 0 or len(gt_pts) == 0:
        return 0.0

    tree = cKDTree(reconstructed_pts)
    dist, _ = tree.query(gt_pts)

    good = (dist < threshold).sum()
    total = len(gt_pts)

    return good / total
