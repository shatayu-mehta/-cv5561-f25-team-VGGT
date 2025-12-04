import numpy as np
from scipy.spatial import cKDTree

def chamfer_distance(points_a, points_b):
    if len(points_a) == 0 or len(points_b) == 0:
        return np.inf

    kdtree_a = cKDTree(points_a)
    kdtree_b = cKDTree(points_b)

    dist_ab, _ = kdtree_a.query(points_b)
    dist_ba, _ = kdtree_b.query(points_a)

    return dist_ab.mean() + dist_ba.mean()
