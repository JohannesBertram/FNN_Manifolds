"""Graph connectivity utilities.

Covers:
  - compute_graph_statistics   — degree and feature-map connection stats
  - handle_disconnected_points — update graph after IAN finds isolated points
  - connected_comp_helper      — bridge disconnected graph components
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

DISCONNECTED_BRIDGING_FACTOR = 10.0


def compute_graph_statistics(wG, neurons_used, nonoutliers):
    """Compute graph connectivity statistics including degrees and feature-map connections.

    Args:
        wG: sparse weighted graph
        neurons_used: array of neuron metadata (column 1 = feature-map label)
        nonoutliers: indices of non-outlier neurons

    Returns:
        dict with keys: degree_{mean,min,max,std}, weighted_degree_*, same_fmap_*, other_fmap_*,
            degrees, weighted_degrees, same_fmaps, other_fmaps
    """
    dense_wG = wG.toarray()
    c = neurons_used[nonoutliers, 1]
    ulbls, labels = np.unique(c, return_inverse=True)

    same_fmaps, other_fmaps, weighted_degrees, degrees = [], [], [], []

    for i in range(len(dense_wG)):
        weighted_degrees.append(np.sum(dense_wG[i]) - 1)
        degrees.append(np.sum(dense_wG[i] > 0) - 1)
        num_same, num_other = 0, 0
        for j in range(len(dense_wG)):
            if dense_wG[i, j] > 0 and i != j:
                if labels[i] == labels[j]:
                    num_same += 1
                else:
                    num_other += 1
        same_fmaps.append(num_same)
        other_fmaps.append(num_other)

    return {
        'degree_mean': np.mean(degrees),
        'degree_min': np.min(degrees),
        'degree_max': np.max(degrees),
        'degree_std': np.std(degrees),
        'weighted_degree_mean': np.mean(weighted_degrees),
        'weighted_degree_min': np.min(weighted_degrees),
        'weighted_degree_max': np.max(weighted_degrees),
        'weighted_degree_std': np.std(weighted_degrees),
        'same_fmap_mean': np.mean(same_fmaps),
        'same_fmap_min': np.min(same_fmaps),
        'same_fmap_max': np.max(same_fmaps),
        'same_fmap_std': np.std(same_fmaps),
        'other_fmap_mean': np.mean(other_fmaps),
        'other_fmap_min': np.min(other_fmaps),
        'other_fmap_max': np.max(other_fmaps),
        'other_fmap_std': np.std(other_fmaps),
        'degrees': degrees,
        'weighted_degrees': weighted_degrees,
        'same_fmaps': same_fmaps,
        'other_fmaps': other_fmaps,
    }


def handle_disconnected_points(disc_pts, optScales, G, D2, outliers_list, neurons_used, X):
    """Handle disconnected points from IAN by updating graphs and outlier lists.

    Args:
        disc_pts: list of disconnected points from IAN
        optScales: optimal scales from IAN
        G: graph matrix
        D2: distance matrix
        outliers_list: current list of outlier indices
        neurons_used: neuron metadata array
        X: data matrix

    Returns:
        tuple: (wG, G, outliers_list, nonoutliers, nonout_mask)
    """
    from ian.ian import getSparseMultiScaleK

    new_outliers = [disc_pts[di][0] for di in range(len(disc_pts))]
    nonout_mask = np.ones(optScales.size, dtype=bool)
    nonout_mask[new_outliers] = False

    if new_outliers:
        wG = getSparseMultiScaleK(D2[nonout_mask][:, nonout_mask], optScales[nonout_mask])
        G = G[nonout_mask][:, nonout_mask]

        index_map = np.ones(len(neurons_used), dtype=bool)
        index_map[outliers_list] = False
        original_indices = np.where(index_map)[0]
        mapped_second_outliers = original_indices[new_outliers]

        outliers_list = np.append(outliers_list, mapped_second_outliers)
        nonoutliers = np.array([i for i in range(X.shape[0]) if i not in outliers_list])

        return wG, G, outliers_list, nonoutliers, nonout_mask
    else:
        wG = getSparseMultiScaleK(D2, optScales)
        nonoutliers = np.array([i for i in range(X.shape[0]) if i not in outliers_list])
        return wG, G, outliers_list, nonoutliers, nonout_mask


def connected_comp_helper(
    A: Optional[np.ndarray],
    X: np.ndarray,
    connect: bool = True,
) -> Optional[np.ndarray]:
    """Ensure graph connectivity by bridging disconnected components.

    Args:
        A: square adjacency matrix (or None)
        X: data matrix used to find nearest cross-component pairs
        connect: if True, bridge disconnected components; if False, log and return unchanged

    Returns:
        Updated adjacency matrix (or None if input was None)
    """
    if A is None:
        logger.warning("Adjacency is None => skipping connectivity.")
        return A

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square. shape={A.shape}")

    n_components, comp_labels = connected_components(A, directed=False, return_labels=True)
    if n_components > 1:
        if connect:
            logger.info(f"Graph has {n_components} disconnected components => bridging them.")
            finite_mask = np.isfinite(A) & (A > 0)
            bridging_val = 1e6 if not np.any(finite_mask) else np.max(A[finite_mask]) * DISCONNECTED_BRIDGING_FACTOR

            for c in range(n_components - 1):
                comp_i = np.where(comp_labels == c)[0]
                comp_j = np.where(comp_labels == c + 1)[0]
                dist_ij = cdist(X[comp_i], X[comp_j], metric='euclidean')
                min_idx = np.unravel_index(np.argmin(dist_ij), dist_ij.shape)
                vi = comp_i[min_idx[0]]
                vj = comp_j[min_idx[1]]
                A[vi, vj] = bridging_val
                A[vj, vi] = bridging_val
        else:
            logger.info(f"Graph has {n_components} disconnected parts, not bridging.")

    return A
