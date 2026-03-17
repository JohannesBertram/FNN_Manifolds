"""
subpop_utils.py
---------------
Utilities for subpopulation analysis of neural encoding and decoding manifolds.

Organized into four groups:
  A. Neuron Selection   — dynamic filtering or manifold-region selection
  B. Decoding Analysis  — PCA-based decoding manifold and trajectories
  C. Encoding Analysis  — IAN + diffusion maps + MDS for a subpopulation
  D. Visualization      — publication-style 2D/3D plots
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

from .manifold_utils import compute_mds_embedding, run_hdbscan_clustering


# ---------------------------------------------------------------------------
# Group A: Neuron Selection
# ---------------------------------------------------------------------------

def compute_dynamic_metrics(tensor4d, target_stim_idx=None):
    """Per-neuron dynamic metrics from raw tensor.

    Reproduces the inline metric computation from partial_decoding.ipynb cell 5.

    Parameters
    ----------
    tensor4d : ndarray, shape (N, S, D, T)
    target_stim_idx : int or None
        If an int, compute per-stimulus metrics for that stimulus index.
        If None (default), compute global averages across all stimuli.

    Returns
    -------
    dict with keys:
        'speed'            : (N,)  early transient magnitude
        'stability'        : (N,)  late transient magnitude (lower = more stable)
        'curvature'        : (N,)  nonlinearity of early response
        'classifiability'  : (N,)  stimulus discriminability
        'pc_contrib'       : (N,)  loading on first global PC
    """
    N_neurons, N_stim, N_ori, N_time = tensor4d.shape

    # Window length for early-transient and curvature metrics.
    # Biological data has T=135 so the full 40-step window applies;
    # FNN data has T=37, so we clamp to avoid a reshape error.
    T_early = min(40, N_time)

    # ---- A. Transients / Speed (early vs late) ----------------------------
    d_tensor = np.diff(tensor4d, axis=3)                              # (N, S, D, T-1)
    d_tensor_mean_ori = np.mean(np.abs(d_tensor), axis=2)            # (N, S, T-1)
    early_transients_per_stim = np.mean(d_tensor_mean_ori[:, :, :T_early], axis=2)   # (N, S)
    late_transients_per_stim  = np.mean(d_tensor_mean_ori[:, :, -T_early:], axis=2)  # (N, S)

    # ---- B. Curvature / Non-linearity (early T_early steps) --------------
    early_traces_per_stim = np.mean(tensor4d[:, :, :, :T_early], axis=2)  # (N, S, T_early)
    smoothed_early = gaussian_filter1d(early_traces_per_stim, sigma=2, axis=2)

    flat_traces = smoothed_early.reshape(-1, T_early).T               # (T_early, N*S)
    X_time = np.vstack([np.arange(T_early), np.ones(T_early)]).T     # (T_early, 2)
    slopes_intercepts, _, _, _ = np.linalg.lstsq(X_time, flat_traces, rcond=None)
    linear_fit = X_time @ slopes_intercepts                           # (T_early, N*S)
    residuals = flat_traces - linear_fit
    curvature_flat = np.sqrt(np.mean(residuals**2, axis=0))           # (N*S,)
    curvature_per_stim = curvature_flat.reshape(N_neurons, N_stim)   # (N, S)

    # ---- C. Stimulus classifiability / selectivity -----------------------
    steady_state_means = np.mean(
        np.mean(tensor4d[:, :, :, -10:], axis=3), axis=2)     # (N, S)
    global_means = np.mean(steady_state_means, axis=1, keepdims=True)  # (N, 1)
    distinctiveness_per_stim = np.abs(steady_state_means - global_means)  # (N, S)
    global_selectivity = np.var(steady_state_means, axis=1)   # (N,)

    # ---- D. Select metrics based on configuration -----------------------
    if target_stim_idx is not None:
        si = target_stim_idx
        metric_speed           = early_transients_per_stim[:, si]
        metric_stability       = late_transients_per_stim[:, si]
        metric_curvature       = curvature_per_stim[:, si]
        metric_classifiability = distinctiveness_per_stim[:, si]
    else:
        metric_speed           = np.mean(early_transients_per_stim, axis=1)
        metric_stability       = np.mean(late_transients_per_stim,  axis=1)
        metric_curvature       = np.mean(curvature_per_stim,        axis=1)
        metric_classifiability = global_selectivity

    # ---- E. PC contribution (global, always) ----------------------------
    tensor_flat = tensor4d.transpose(1, 2, 3, 0).reshape(-1, N_neurons)  # (S*D*T, N)
    pca_global = PCA(n_components=1)
    pca_global.fit(tensor_flat)
    metric_pc_contrib = np.sqrt(np.sum(pca_global.components_[:1, :]**2, axis=0))  # (N,)

    return {
        'speed':           metric_speed,
        'stability':       metric_stability,
        'curvature':       metric_curvature,
        'classifiability': metric_classifiability,
        'pc_contrib':      metric_pc_contrib,
    }


def filter_neurons_by_metric(metrics, metric_name,
                              percentile_gt=None, percentile_lt=None):
    """Boolean mask (N,) for neurons passing a percentile threshold.

    Multiple calls can be ANDed for compound filters::

        mask = (filter_neurons_by_metric(m, 'speed', percentile_gt=99.75) &
                filter_neurons_by_metric(m, 'stability', percentile_lt=50))

    Parameters
    ----------
    metrics : dict  (output of compute_dynamic_metrics)
    metric_name : str
    percentile_gt : float or None  — keep neurons ABOVE this percentile
    percentile_lt : float or None  — keep neurons BELOW this percentile

    Returns
    -------
    mask : ndarray of bool, shape (N,)
    """
    vals = metrics[metric_name]
    mask = np.ones(len(vals), dtype=bool)
    if percentile_gt is not None:
        mask &= vals > np.percentile(vals, percentile_gt)
    if percentile_lt is not None:
        mask &= vals < np.percentile(vals, percentile_lt)
    return mask


# -- Method 2: Manifold Region Selection ------------------------------------

def select_neurons_by_cluster(cluster_labels, target_clusters):
    """Boolean mask for neurons in the given HDBSCAN cluster label(s).

    Parameters
    ----------
    cluster_labels : ndarray, shape (N_nonout,)
    target_clusters : int or list of int
        Pass -1 to include noise points.

    Returns
    -------
    mask : ndarray of bool, shape (N_nonout,)
    """
    if np.isscalar(target_clusters):
        target_clusters = [target_clusters]
    mask = np.zeros(len(cluster_labels), dtype=bool)
    for lbl in target_clusters:
        mask |= (cluster_labels == lbl)
    return mask


def select_neurons_by_bbox(embedding, bounds):
    """Boolean mask for neurons inside a bounding box in embedding space.

    Parameters
    ----------
    embedding : ndarray, shape (N_nonout, D)
    bounds : dict mapping dim_index -> (min_val, max_val)
        Example: {0: (-0.5, 0.5), 1: (0.0, 1.0)}

    Returns
    -------
    mask : ndarray of bool, shape (N_nonout,)
    """
    mask = np.ones(len(embedding), dtype=bool)
    for dim, (lo, hi) in bounds.items():
        mask &= (embedding[:, dim] >= lo) & (embedding[:, dim] <= hi)
    return mask


def select_neurons_by_radius(embedding, seed_idx, radius):
    """Boolean mask for neurons within Euclidean radius of a seed point.

    Parameters
    ----------
    embedding : ndarray, shape (N_nonout, D)
    seed_idx : int
    radius : float

    Returns
    -------
    mask : ndarray of bool, shape (N_nonout,)
    """
    dists = np.linalg.norm(embedding - embedding[seed_idx], axis=1)
    return dists <= radius


# ---------------------------------------------------------------------------
# Group B: Decoding Analysis
# ---------------------------------------------------------------------------

def compute_decoding_manifold(tensor4d_sub, n_components=3):
    """PCA on stimulus-averaged activity → decoding manifold.

    Mirrors partial_decoding.ipynb cell 6 ("For Decoding Manifold").

    Steps:
      1. transpose (N, S, D, T) → (T, S, D, N)
      2. mean over T → (S, D, N)
      3. reshape → (S*D, N)
      4. PCA.fit_transform → (S*D, n_components)

    Parameters
    ----------
    tensor4d_sub : ndarray, shape (N, S, D, T)
    n_components : int

    Returns
    -------
    coords : ndarray, shape (S*D, n_components)
    pca    : fitted PCA object
    """
    N, N_stim, N_dir, T = tensor4d_sub.shape
    transposed = np.transpose(tensor4d_sub, (3, 1, 2, 0))  # (T, S, D, N)
    meaned = np.mean(transposed, axis=0)                   # (S, D, N)
    reshaped = meaned.reshape(-1, N)                       # (S*D, N)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(reshaped)                   # (S*D, n_components)
    return coords, pca


def compute_decoding_trajectories(tensor4d_sub, n_components=3):
    """PCA on time-resolved activity → decoding trajectories.

    Mirrors partial_decoding.ipynb cell 6 ("For Trajectories").

    Parameters
    ----------
    tensor4d_sub : ndarray, shape (N, S, D, T)
    n_components : int

    Returns
    -------
    trajectories : ndarray, shape (S*D, T, n_components)
    pca          : fitted PCA object
    """
    N, N_stim, N_dir, T = tensor4d_sub.shape
    transposed = np.transpose(tensor4d_sub, (3, 1, 2, 0))  # (T, S, D, N)
    reshaped = transposed.reshape(-1, N)                   # (T*S*D, N)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(reshaped)               # (T*S*D, n_components)
    # Reshape to (T, S*D, n_components) then transpose to (S*D, T, n_components)
    reshaped_result = pca_result.reshape(T, N_stim * N_dir, n_components)
    trajectories = np.transpose(reshaped_result, (1, 0, 2))
    return trajectories, pca


# ---------------------------------------------------------------------------
# Group C: Encoding Analysis (rebuild manifold for subpopulation)
# ---------------------------------------------------------------------------

def _handle_disc_pts_simple(disc_pts, optScales, G, D2):
    """Remove disconnected points from IAN result and recompute sparse kernel.

    Simplified version of handle_disconnected_points() that works within
    subpopulation space without requiring neurons_used metadata.

    Parameters
    ----------
    disc_pts  : list of disconnected-point records from IAN
    optScales : ndarray, shape (N,)
    G         : binary adjacency, shape (N, N)
    D2        : squared-distance matrix, shape (N, N)

    Returns
    -------
    wG          : sparse weighted adjacency, shape (N_clean, N_clean)
    G           : binary adjacency, shape (N_clean, N_clean)
    nonoutliers : ndarray of int, shape (N_clean,) — indices into original N
    """
    from ian.ian import getSparseMultiScaleK

    new_outliers = [disc_pts[di][0] for di in range(len(disc_pts))]
    N = optScales.size
    nonout_mask = np.ones(N, dtype=bool)
    nonout_mask[new_outliers] = False

    if new_outliers:
        wG = getSparseMultiScaleK(
            D2[nonout_mask][:, nonout_mask], optScales[nonout_mask])
        G = G[nonout_mask][:, nonout_mask]
        nonoutliers = np.where(nonout_mask)[0]
    else:
        wG = getSparseMultiScaleK(D2, optScales)
        nonoutliers = np.arange(N)

    return wG, G, nonoutliers


def build_encoding_manifold_for_subpop(
        X_full, subpop_indices, nPCs,
        solver=None, n_diffmap_components=20, n_mds_components=10):
    """Rebuild IAN graph + diffusion maps + MDS for a subpopulation.

    Mirrors encoding_manifolds.ipynb cells 24–36, but operates on a subset
    of the already PCA-reduced, nonoutlier-filtered neural matrix.

    Parameters
    ----------
    X_full         : ndarray, shape (N_nonout, nPCs)
                     PCA-reduced, nonoutlier-filtered neural matrix (``myX``
                     in encoding_manifolds.ipynb).
    subpop_indices : array-like of int
                     Indices into X_full rows (nonoutlier-space).
    nPCs           : int
                     Number of diffusion coordinates to use for MDS embedding.
    solver         : str or None
                     cvxpy solver for IAN (e.g. 'GUROBI'). None = default.
    n_diffmap_components : int  (default 20)
    n_mds_components     : int  (default 10)

    Returns
    -------
    dict with keys:
        'X_sub'         : (N_sub_clean, nPCs)
        'nonoutliers'   : (N_sub_clean,) indices into subpop_indices
        'wG'            : sparse (N_sub_clean, N_sub_clean)
        'G'             : dense binary adjacency (N_sub_clean, N_sub_clean)
        'diffmap_y'     : (N_sub_clean, n_diffmap_components-1)
        'diffmap_evals' : (n_diffmap_components-1,)
        'embedding_'    : (N_sub_clean, n_mds_components)
    """
    from ian.ian import IAN
    from ian.utils import pwdists
    from ian.embed_utils import diffusionMapSparseK

    subpop_indices = np.asarray(subpop_indices)

    # 1. Subset
    X_sub = X_full[subpop_indices]                         # (K, nPCs)

    # 2. Squared pairwise distances
    D2 = pwdists(X_sub, sqdists=True)                      # (K, K)

    # 3. IAN
    G, _wG_init, optScales, disc_pts = IAN(
        'exact-precomputed-sq', D2, solver=solver)

    # 4. Handle disconnected points
    wG, G, nonoutliers = _handle_disc_pts_simple(disc_pts, optScales, G, D2)

    # 5. Diffusion maps
    diffmap_y, diffmap_evals = diffusionMapSparseK(
        csr_matrix(wG), n_diffmap_components, 1, t=1)

    # 6. MDS embedding
    embedding_ = compute_mds_embedding(diffmap_y, nPCs,
                                       n_components=n_mds_components)

    return {
        'X_sub':         X_sub[nonoutliers],
        'nonoutliers':   nonoutliers,
        'wG':            wG,
        'G':             G,
        'diffmap_y':     diffmap_y,
        'diffmap_evals': diffmap_evals,
        'embedding_':    embedding_,
    }


# ---------------------------------------------------------------------------
# Group D: Visualization
# ---------------------------------------------------------------------------

def plot_metric_distributions(metrics, selected_mask=None):
    """Histogram panel for all 5 dynamic metrics.

    Parameters
    ----------
    metrics       : dict (output of compute_dynamic_metrics)
    selected_mask : ndarray of bool, shape (N,) or None
        If provided, overlays the selected subpopulation in orange.

    Returns
    -------
    fig, axes
    """
    keys = ['speed', 'stability', 'curvature', 'classifiability', 'pc_contrib']
    titles = ['Early Speed', 'Late Stability', 'Curvature', 'Classifiability', 'PC Contrib']

    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.5))
    for ax, key, title in zip(axes, keys, titles):
        vals = metrics[key]
        ax.hist(vals, bins=40, color='steelblue', alpha=0.7, label='all')
        if selected_mask is not None:
            ax.hist(vals[selected_mask], bins=40, color='darkorange',
                    alpha=0.8, label='selected')
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(key, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig, axes


def plot_neuron_locations_in_manifold(embedding, selected_mask, dcs=(0, 1, 2),
                                      s_bg=10, s_sel=30,
                                      color_bg='lightgray', color_sel='darkorange'):
    """3D scatter: full population in gray, subpopulation highlighted.

    Parameters
    ----------
    embedding     : ndarray, shape (N_nonout, D)
    selected_mask : ndarray of bool, shape (N_nonout,)
    dcs           : tuple of 3 ints — which embedding dims to plot

    Returns
    -------
    fig, ax
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    Y = embedding[:, list(dcs)]
    ax.scatter(*Y[~selected_mask].T, c=color_bg, s=s_bg,
               alpha=0.4, edgecolors='none', depthshade=False, label='rest')
    ax.scatter(*Y[selected_mask].T, c=color_sel, s=s_sel,
               alpha=0.9, edgecolors='none', depthshade=False, label='selected')

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    ax.set_xlabel(f'DC{dcs[0]}', fontsize=8)
    ax.set_ylabel(f'DC{dcs[1]}', fontsize=8)
    ax.set_zlabel(f'DC{dcs[2]}', fontsize=8)
    ax.legend(fontsize=8)
    ax.view_init(elev=30, azim=45)
    fig.tight_layout()
    return fig, ax


def plot_decoding_manifold(coords, n_stim, n_dir, stimulus_colors, labels,
                           dcs=(0, 1, 2), s=100, title='Decoding Manifold'):
    """3D scatter of decoding manifold.

    Mirrors decoding_analysis.ipynb cell 8.

    Parameters
    ----------
    coords          : ndarray, shape (n_stim*n_dir, 3)
    n_stim          : int
    n_dir           : int
    stimulus_colors : list of str, length n_stim
    labels          : list of str, length n_stim
    dcs             : tuple of 3 ints (which PCA dims to plot)

    Returns
    -------
    fig, ax
    """
    clrs = np.array([c for c in stimulus_colors for _ in range(n_dir)])
    legend_labels = np.array([lb for lb in labels for _ in range(n_dir)])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    for i, label in enumerate(labels):
        mask = legend_labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, dcs[0]], coords[mask, dcs[1]], coords[mask, dcs[2]],
            color=stimulus_colors[i], s=s, alpha=1.0,
            edgecolors='none', label=label, depthshade=False)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    ax.view_init(elev=30, azim=45)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_decoding_trajectories(trajectories, n_stim, n_dir, stimulus_colors,
                               dcs=(0, 1, 2), linewidth=1.5, alpha=0.35,
                               z_scale=0.5, title='Decoding Trajectories'):
    """3D trajectory lines.

    Mirrors plot_trajectories_matplotlib() in partial_decoding.ipynb cell 10.

    Parameters
    ----------
    trajectories    : ndarray, shape (n_stim*n_dir, T, 3)
    n_stim          : int
    n_dir           : int
    stimulus_colors : list of str, length n_stim
    dcs             : tuple of 3 ints

    Returns
    -------
    fig, ax
    """
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, traj in enumerate(trajectories):
        stim_idx = i // n_dir
        color = stimulus_colors[stim_idx % len(stimulus_colors)]
        x = traj[:, dcs[0]]
        y = traj[:, dcs[1]]
        z = traj[:, dcs[2]] * z_scale
        ax.plot(x, y, z, color=color, linewidth=linewidth, alpha=alpha)
        ax.scatter(x[0], y[0], z[0], color='black', s=15, depthshade=False,
                   edgecolors='none')
        ax.scatter(x[-1], y[-1], z[-1], color=color, s=60, depthshade=False,
                   edgecolors='none')

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    ax.xaxis.line.set_color('black')
    ax.yaxis.line.set_color('black')
    ax.zaxis.line.set_color('black')
    ax.view_init(elev=30, azim=45)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_encoding_manifold_3d(embedding, color_labels, palette, dcs=(0, 1, 2),
                               s=40, alpha=0.8, title='Encoding Manifold (3D)'):
    """3D scatter of MDS embedding colored by cluster/feature-map labels.

    Mirrors encoding_manifolds.ipynb cell 38.

    Parameters
    ----------
    embedding     : ndarray, shape (N, D)
    color_labels  : ndarray, shape (N,) — integer or string labels
    palette       : array-like of color strings
    dcs           : tuple of 3 ints

    Returns
    -------
    fig, ax
    """
    ulbls = np.unique(color_labels)
    palette = np.asarray(palette)
    # Extend palette if needed
    while len(palette) < len(ulbls):
        palette = np.r_[palette, palette]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for fi, lbl in enumerate(ulbls):
        mask = color_labels == lbl
        ax.scatter(
            embedding[mask, dcs[0]],
            embedding[mask, dcs[1]],
            embedding[mask, dcs[2]],
            c=palette[fi % len(palette)],
            s=s, alpha=alpha, edgecolors='none', depthshade=False)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=45)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_encoding_manifold_2d(diffmap_y, color_labels, palette, dcs=(0, 1),
                               s=6, title='Encoding Manifold (2D)'):
    """2D diffusion coordinate scatter.

    Mirrors encoding_manifolds.ipynb cell 34.

    Parameters
    ----------
    diffmap_y    : ndarray, shape (N, K)
    color_labels : ndarray, shape (N,)
    palette      : array-like of color strings
    dcs          : tuple of 2 ints

    Returns
    -------
    fig, ax
    """
    ulbls = np.unique(color_labels)
    palette = np.asarray(palette)
    while len(palette) < len(ulbls):
        palette = np.r_[palette, palette]

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    for fi, lbl in enumerate(ulbls):
        mask = color_labels == lbl
        ax.scatter(
            diffmap_y[mask, dcs[0]],
            diffmap_y[mask, dcs[1]],
            c=palette[fi % len(palette)],
            s=s, alpha=0.8, edgecolors='none')

    ax.set_xlabel(f'DC{dcs[0]}', fontsize=9)
    ax.set_ylabel(f'DC{dcs[1]}', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig, ax
