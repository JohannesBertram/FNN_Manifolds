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


def select_top_k_by_metric(metrics, metric_name, k, high=True):
    """Return integer indices of top-k neurons by metric_name.

    Parameters
    ----------
    metrics     : dict (output of compute_dynamic_metrics)
    metric_name : str
    k           : int — number of neurons to select
    high        : bool
        True  → highest-k values (e.g. fast neurons)
        False → lowest-k values  (e.g. slow neurons)

    Returns
    -------
    indices : ndarray of int, shape (k,)
    """
    vals = metrics[metric_name]
    order = np.argsort(vals)
    return order[-k:] if high else order[:k]


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
    meaned = np.nanmean(transposed, axis=0)                # (S, D, N) — ignores NaN-padded frames
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

    # Identify padded frames: all neurons NaN at that (t, s, d) position
    padded_mask = np.all(np.isnan(transposed), axis=-1)    # (T, S, D)
    flat = transposed.reshape(-1, N)                       # (T*S*D, N)
    valid_rows = ~padded_mask.reshape(-1)                  # (T*S*D,)

    # Fit PCA only on valid (non-padded) frames — padded rows never touch the PCA
    valid_data = flat[valid_rows]                          # (num_valid, N)
    valid_data = np.where(np.isfinite(valid_data), valid_data, 0.0)
    pca = PCA(n_components=n_components)
    valid_pca = pca.fit_transform(valid_data)              # (num_valid, n_components)

    # Place results back into a full array, NaN for padded positions
    full_pca = np.full((T * N_stim * N_dir, n_components), np.nan)
    full_pca[valid_rows] = valid_pca
    reshaped_result = full_pca.reshape(T, N_stim * N_dir, n_components)
    trajectories = np.transpose(reshaped_result, (1, 0, 2))  # (S*D, T, n_components)
    return trajectories, pca


def knn_decoding_accuracy(coords, stim_labels, n_neighbors=5):
    """Leave-one-out k-NN accuracy on (S*D, n_components) decoding manifold.

    Returns np.nan when there are too few points for LOO k-NN
    (requires len(coords) > n_neighbors).

    Parameters
    ----------
    coords      : ndarray, shape (S*D, n_components)
    stim_labels : ndarray, shape (S*D,) int — stimulus index for each point
    n_neighbors : int

    Returns
    -------
    accuracy : float in [0, 1], or np.nan
    """
    from sklearn.neighbors import KNeighborsClassifier

    n = len(stim_labels)
    if n <= n_neighbors:
        return np.nan
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    correct = sum(
        knn.fit(np.delete(coords, i, 0), np.delete(stim_labels, i))
           .predict(coords[[i]])[0] == stim_labels[i]
        for i in range(n)
    )
    return correct / n


def procrustes_r2(coords_ref, coords_sub):
    """Procrustes R² between reference and subpop decoding manifold.

    Uses scipy.spatial.procrustes which normalises by the reference
    Frobenius norm; disparity = 1 - R².

    Returns np.nan when the arrays have different shapes (e.g. when the
    subpop manifold has fewer PCA components than the reference).

    Parameters
    ----------
    coords_ref : ndarray, shape (S*D, n_components)
    coords_sub : ndarray, shape (S*D, n_components)

    Returns
    -------
    r2 : float in [0, 1], or np.nan
    """
    from scipy.spatial import procrustes

    if coords_ref.shape != coords_sub.shape:
        return np.nan
    try:
        _, _, disparity = procrustes(coords_ref, coords_sub)
    except ValueError:
        return np.nan
    return 1.0 - disparity


def rdm_correlation(tensor4d_ref, tensor4d_sub):
    """Spearman rank correlation between full-pop and subpop RDMs.

    Works in native neural space (no PCA). Time-averages the tensor internally.
    Directly measures whether the subpop preserves the same stimulus-similarity
    structure as the full population, without any dimensionality reduction.

    Parameters
    ----------
    tensor4d_ref : ndarray, shape (N_ref, S, D, T)
    tensor4d_sub : ndarray, shape (N_sub, S, D, T)

    Returns
    -------
    rho : float in [-1, 1]
    """
    from scipy.stats import spearmanr

    def _act_matrix(t):
        N, S, D, T = t.shape
        return t.mean(axis=3).transpose(1, 2, 0).reshape(-1, N)  # (S*D, N)

    A_ref = _act_matrix(tensor4d_ref)
    A_sub = _act_matrix(tensor4d_sub)
    rdm_ref = squareform(pdist(A_ref, metric='euclidean'))
    rdm_sub = squareform(pdist(A_sub, metric='euclidean'))
    idx = np.triu_indices(len(rdm_ref), k=1)
    rho, _ = spearmanr(rdm_ref[idx], rdm_sub[idx])
    return float(rho)


def linear_cka(tensor4d_ref, tensor4d_sub):
    """Linear CKA between full-pop and subpop stimulus representations.

    Invariant to rotation and isotropic scaling. No PCA or point correspondence
    needed beyond row-ordering (each row = one stimulus-direction pair).

    Parameters
    ----------
    tensor4d_ref : ndarray, shape (N_ref, S, D, T)
    tensor4d_sub : ndarray, shape (N_sub, S, D, T)

    Returns
    -------
    cka : float in [0, 1]
    """
    def _act_matrix(t):
        N, S, D, T = t.shape
        return t.mean(axis=3).transpose(1, 2, 0).reshape(-1, N)  # (S*D, N)

    def _center_kernel(K):
        n = len(K)
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    X = _act_matrix(tensor4d_ref)
    Y = _act_matrix(tensor4d_sub)
    Kx = _center_kernel(X @ X.T)
    Ky = _center_kernel(Y @ Y.T)
    num = np.sum(Kx * Ky)
    den = np.linalg.norm(Kx, 'fro') * np.linalg.norm(Ky, 'fro')
    return float(num / den) if den > 0 else np.nan


def variance_reproduced(coords_ref, coords_sub):
    """Fraction of full-pop decoding manifold variance reproduced by sub-pop.

    Computes the ratio of total variance (sum of per-component variances) of
    coords_sub to coords_ref, using the leading min(n_ref, n_sub) components.
    Values > 1 are possible when the sub-pop is geometrically more spread than
    the full population (common for very small, noisy subsets).

    Parameters
    ----------
    coords_ref : ndarray, shape (S*D, n_components)
    coords_sub : ndarray, shape (S*D, n_components_sub)

    Returns
    -------
    ratio : float, or np.nan if coords_ref has zero variance
    """
    n = min(coords_ref.shape[1], coords_sub.shape[1])
    var_ref = np.sum(np.var(coords_ref[:, :n], axis=0))
    if var_ref == 0:
        return np.nan
    return np.sum(np.var(coords_sub[:, :n], axis=0)) / var_ref


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
# Group C2: Synthetic Population Construction
# ---------------------------------------------------------------------------

def create_synthetic_clustered_population(
        tensor4d_nonout, embedding, n_seeds=20, n_neighbors=8, rng_seed=42):
    """Create a synthetic population with clustered encoding manifold topology.

    Selects ``n_seeds`` seed neurons evenly spread across the embedding via
    K-means, finds ``n_neighbors`` nearest manifold neighbours for each seed,
    then constructs ``N // n_seeds`` new neurons per cluster by shuffling the
    per-stimulus responses of those neighbours.  The result has ≈N neurons
    organised in ``n_seeds`` tight clusters in neural-tuning space, while each
    neuron's per-stimulus response still originates from a real, locally-similar
    neighbour — preserving decodability but breaking the smooth manifold topology.

    Parameters
    ----------
    tensor4d_nonout : ndarray, shape (N, S, D, T)
        Neural responses of the nonoutlier population in (neurons, stimuli,
        directions, time) order.
    embedding : ndarray, shape (N, n_dims)
        MDS embedding coordinates of the same population.
    n_seeds : int
        Number of cluster centres.
    n_neighbors : int
        Neighbourhood size for per-stimulus response shuffling.
    rng_seed : int

    Returns
    -------
    tensor4d_synthetic : ndarray, shape (n_seeds * n_per_seed, S, D, T)
        Synthetic population; n_per_seed = max(1, N // n_seeds).
    """
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors

    N, S, D, T = tensor4d_nonout.shape
    n_per_seed = max(1, N // n_seeds)
    rng = np.random.default_rng(rng_seed)

    # 1. K-means on embedding to find evenly-spread seed locations
    km = KMeans(n_clusters=n_seeds, random_state=rng_seed, n_init=10)
    km.fit(embedding)

    # Snap each cluster centre to the nearest real neuron
    nbrs_seed = NearestNeighbors(n_neighbors=1).fit(embedding)
    _, seed_nn = nbrs_seed.kneighbors(km.cluster_centers_)
    seed_indices = seed_nn.flatten()          # (n_seeds,) in nonout-space

    # 2. For each seed, find n_neighbors nearest neighbours in embedding
    # (ask for n_neighbors+1 and drop the first column to exclude the seed itself)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embedding)
    _, neighbor_indices = nbrs.kneighbors(embedding[seed_indices])
    neighbor_indices = neighbor_indices[:, 1:]   # (n_seeds, n_neighbors)

    # 3. Create synthetic neurons: for each stimulus, copy a random neighbour's
    #    full directional × temporal response (shape D × T).
    synthetic_list = []
    for seed_i in range(n_seeds):
        nbr_idx = neighbor_indices[seed_i]                 # (n_neighbors,)
        nbr_responses = tensor4d_nonout[nbr_idx]           # (n_neighbors, S, D, T)
        chosen = rng.integers(0, n_neighbors, size=(n_per_seed, S))  # (n_per_seed, S)
        for syn_i in range(n_per_seed):
            new_neuron = np.empty((S, D, T), dtype=tensor4d_nonout.dtype)
            for s in range(S):
                new_neuron[s] = nbr_responses[chosen[syn_i, s], s]
            synthetic_list.append(new_neuron)

    return np.stack(synthetic_list, axis=0)       # (n_seeds * n_per_seed, S, D, T)


def compute_encoding_manifold_from_tensor(
        tensor4d, n_diffmap_components=20, n_mds_components=10,
        min_expl_var=0.8, n_far=2, n_close=5, solver=None):
    """Build an encoding manifold directly from a tensor4d (no CP decomposition).

    Provides an alternative to the CP-based pipeline in ``load_for_explorer`` /
    ``build_encoding_manifold_for_subpop``.  Uses mean-over-time PCA as the
    neural representation, then runs the same IAN → diffusion maps → MDS
    pipeline.  Intended for synthetic populations where no CP factors exist,
    but can also be applied to real data for a fair same-method comparison.

    Parameters
    ----------
    tensor4d : ndarray, shape (N, S, D, T)
        Neural tensor in (neurons, stimuli, directions, time) order.
        NaN values (padded frames) are replaced with 0 before PCA.
    n_diffmap_components : int
    n_mds_components : int
    min_expl_var : float
        Cumulative variance threshold used to select the number of PCA components.
    n_far : int
        Number of most-isolated neurons removed as outliers.
    n_close : int
        Number of most-similar neurons removed as outliers.
    solver : str or None
        IAN solver (None = default).

    Returns
    -------
    dict with keys:
        'embedding_'  : ndarray, shape (N_clean, n_mds_components)
        'nonoutliers' : ndarray, shape (N_clean,) — indices into original N
        'wG'          : sparse (N_clean, N_clean) weighted adjacency
        'diffmap_y'   : ndarray, shape (N_clean, n_diffmap_components - 1)
        'nPCs'        : int
    """
    from ian.ian import IAN
    from ian.utils import pwdists
    from ian.embed_utils import diffusionMapSparseK

    N, S, D, T = tensor4d.shape

    # 1. Mean over time, flatten stimulus × direction → feature vector
    X_flat = np.nan_to_num(tensor4d.mean(axis=3), nan=0.0).reshape(N, S * D)

    # 2. PCA — choose nPCs to explain ≥ min_expl_var of variance
    n_comps = min(N - 1, S * D)
    pca = PCA(n_components=n_comps)
    pcaX = pca.fit_transform(X_flat)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    matches = np.flatnonzero(cumvar > min_expl_var)
    nPCs = int(matches[0]) + 1 if len(matches) else n_comps
    myX = pcaX[:, :nPCs]

    # 3. Outlier removal (mirrors cache_utils.load_for_explorer steps 7–8)
    from ian.utils import pwdists as _pwdists
    D2_full = _pwdists(myX, sqdists=True)
    D1 = np.sqrt(D2_full)
    mindists = np.min(D1 + np.eye(N) * D1.max(), axis=0)
    outls_far   = np.argsort(mindists)[::-1][:n_far]
    outls_close = np.argsort(mindists)[:n_close]
    outliers_set = set(np.append(outls_far, outls_close).tolist())
    keep = np.array([i for i in range(N) if i not in outliers_set])
    myX_clean = myX[keep]                       # (N_clean1, nPCs)

    # 4. Distances in cleaned space
    D2 = _pwdists(myX_clean, sqdists=True)

    # 5. IAN
    G, _wG_init, optScales, disc_pts = IAN('exact-precomputed-sq', D2, solver=solver)

    # 6. Handle disconnected points; disc_nonout indexes into myX_clean
    wG, G, disc_nonout = _handle_disc_pts_simple(disc_pts, optScales, G, D2)
    nonoutliers = keep[disc_nonout]             # indices into original N

    # 7. Diffusion maps
    diffmap_y, _ = diffusionMapSparseK(
        csr_matrix(wG), n_diffmap_components, 1, t=1)

    # 8. MDS embedding
    embedding_ = compute_mds_embedding(diffmap_y, nPCs, n_components=n_mds_components)

    return {
        'embedding_':  embedding_,
        'nonoutliers': nonoutliers,
        'wG':          wG,
        'diffmap_y':   diffmap_y,
        'nPCs':        nPCs,
    }


# ---------------------------------------------------------------------------
# Group C3: Alternative Synthetic Population Constructors
# ---------------------------------------------------------------------------

def create_clustered_by_amplification(
        tensor4d_nonout, embedding, n_seeds=20, n_neighbors=8,
        alpha_range=(0.6, 1.0), rng_seed=42):
    """Create a clustered synthetic population via seed-response amplification.

    Unlike ``create_synthetic_clustered_population`` (which shuffles per-stimulus
    responses across neighbours independently), this method generates each
    synthetic neuron by *blending* the seed's response with a randomly chosen
    neighbour's response.  The blend weight ``alpha`` is drawn uniformly from
    ``alpha_range`` per (neuron, stimulus) pair.  At ``alpha=1`` the synthetic
    neuron is a pure copy of the seed; at ``alpha=0.6`` it is 60% seed + 40%
    neighbour.  Because all synthetic neurons within a cluster remain close to
    the *same* seed response, the resulting manifold is more tightly clustered
    than the shuffle-based method while still exhibiting realistic per-stimulus
    variability.

    Parameters
    ----------
    tensor4d_nonout : ndarray, shape (N, S, D, T)
    embedding : ndarray, shape (N, n_dims)
    n_seeds : int
    n_neighbors : int
    alpha_range : tuple (min_alpha, max_alpha)
        Blend-weight range for seed vs neighbour mixing.
    rng_seed : int

    Returns
    -------
    ndarray, shape (n_seeds * n_per_seed, S, D, T)
    """
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors

    N, S, D, T = tensor4d_nonout.shape
    n_per_seed = max(1, N // n_seeds)
    rng = np.random.default_rng(rng_seed)

    km = KMeans(n_clusters=n_seeds, random_state=rng_seed, n_init=10)
    km.fit(embedding)
    nbrs_seed = NearestNeighbors(n_neighbors=1).fit(embedding)
    _, seed_nn = nbrs_seed.kneighbors(km.cluster_centers_)
    seed_indices = seed_nn.flatten()

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embedding)
    _, neighbor_indices = nbrs.kneighbors(embedding[seed_indices])
    neighbor_indices = neighbor_indices[:, 1:]   # (n_seeds, n_neighbors)

    synthetic_list = []
    a_lo, a_hi = alpha_range
    for seed_i in range(n_seeds):
        seed_resp = tensor4d_nonout[seed_indices[seed_i]]   # (S, D, T)
        nbr_idx = neighbor_indices[seed_i]                  # (n_neighbors,)
        nbr_responses = tensor4d_nonout[nbr_idx]            # (n_neighbors, S, D, T)
        for _ in range(n_per_seed):
            alphas = rng.uniform(a_lo, a_hi, size=S)        # per-stimulus blend
            nbr_choice = rng.integers(0, n_neighbors, size=S)
            new_neuron = np.empty((S, D, T), dtype=tensor4d_nonout.dtype)
            for s in range(S):
                alpha = alphas[s]
                new_neuron[s] = (alpha * seed_resp[s]
                                 + (1.0 - alpha) * nbr_responses[nbr_choice[s], s])
            synthetic_list.append(new_neuron)

    return np.stack(synthetic_list, axis=0)


def create_clustered_by_centroid(
        tensor4d_nonout, embedding, n_seeds=20, noise_scale=0.05, rng_seed=42):
    """Create a clustered synthetic population from K-means centroid responses.

    K-means clusters the embedding into ``n_seeds`` groups.  For each cluster,
    the *mean* response across all assigned neurons defines a centroid template.
    Synthetic neurons are then generated by adding scaled Gaussian noise (matched
    to the per-neuron response standard deviation) to the centroid template.
    The result is a population of ≈N neurons arranged in ``n_seeds`` very tight
    clusters; the centroid templates span the original embedding, so decoding
    information is preserved.

    Parameters
    ----------
    tensor4d_nonout : ndarray, shape (N, S, D, T)
    embedding : ndarray, shape (N, n_dims)
    n_seeds : int
    noise_scale : float
        Noise amplitude as a fraction of the per-neuron response std.
    rng_seed : int

    Returns
    -------
    ndarray, shape (n_seeds * n_per_seed, S, D, T)
    """
    from sklearn.cluster import KMeans

    N, S, D, T = tensor4d_nonout.shape
    n_per_seed = max(1, N // n_seeds)
    rng = np.random.default_rng(rng_seed)

    km = KMeans(n_clusters=n_seeds, random_state=rng_seed, n_init=10)
    cluster_labels = km.fit_predict(embedding)

    # Global response std for noise scaling
    global_std = float(np.nanstd(tensor4d_nonout))

    synthetic_list = []
    for c in range(n_seeds):
        members = np.where(cluster_labels == c)[0]
        if len(members) == 0:
            continue
        centroid = np.nanmean(tensor4d_nonout[members], axis=0)  # (S, D, T)
        for _ in range(n_per_seed):
            noise = rng.normal(0.0, noise_scale * global_std, size=(S, D, T))
            synthetic_list.append(centroid + noise)

    return np.stack(synthetic_list, axis=0)


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
