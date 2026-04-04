"""Neural response metrics and distance measures.

Covers:
  - compute_osi_and_pref_stim      — orientation selectivity index
  - compute_temporal_variance      — log mean temporal variance per neuron
  - compute_mean_activation        — mean activation per neuron
  - compute_gromov_wasserstein     — GW cost via POT library (exact solver)
  - compute_entropic_gw            — GW cost via entropic/regularized solver (fast, large N)
  - compute_fgw                    — Fused GW cost (structural + feature terms)
  - compute_gromov_hausdorff_approx — approx GH distance from point clouds
  - compute_single_linkage_ultrametric — custom single-linkage ultrametric
  - approximate_gh_on_ultrametrics — GH distance via GW on ultrametrics
  - normalize_distance_matrix      — scale matrix to max=1
  - build_index_maps               — bidirectional neuron index maps (nat-movie ↔ manifold)
  - natmovie_to_manifold_indices   — map natural-movie indices → manifold indices
  - manifold_to_natmovie_indices   — map manifold indices → natural-movie indices
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------------------
# Neural response metrics
# ---------------------------------------------------------------------------

def compute_osi_and_pref_stim(tensorX, n_stim=6, n_dir=8):
    """Compute Orientation Selectivity Index (OSI) and preferred stimulus.

    Args:
        tensorX: processed tensor data, shape (N, NSTIMS, NDIRS * TRIAL_LEN)
        n_stim: number of stimuli (default 6)
        n_dir: number of directions (default 8)

    Returns:
        tuple: (OSI_final, pref_stim)  — arrays of length N
    """
    tensor_reshaped = tensorX.reshape((len(tensorX), n_stim, n_dir, -1))
    data_avg = tensor_reshaped.mean(axis=3)  # (N, n_stim, n_dir)

    OSI_per_stimulus = np.zeros((len(tensor_reshaped), n_stim))
    orth_offset = max(1, n_dir // 4)

    for stim_idx in range(n_stim):
        tuning_curves = data_avg[:, stim_idx, :]
        pref_ori = tuning_curves.argmax(axis=1)
        orth_ori_pos = (pref_ori + orth_offset) % n_dir
        orth_ori_neg = (pref_ori - orth_offset) % n_dir

        pref_responses = tuning_curves[np.arange(len(tensor_reshaped)), pref_ori]
        orth_responses = (
            tuning_curves[np.arange(len(tensor_reshaped)), orth_ori_pos]
            + tuning_curves[np.arange(len(tensor_reshaped)), orth_ori_neg]
        ) / 2

        OSI_per_stimulus[:, stim_idx] = (pref_responses - orth_responses) / (
            pref_responses + orth_responses + 1e-12
        )

    OSI_final = OSI_per_stimulus.max(axis=1)
    stim_responses = data_avg.mean(axis=2)
    pref_stim = stim_responses.argmax(axis=1)

    return OSI_final, pref_stim


def compute_temporal_variance(tensorX):
    """Per-neuron log mean temporal variance across stimuli. Shape: (N,)."""
    v = np.var(tensorX, axis=2)
    v = np.nanmean(v, axis=1)
    return np.log(v + 1e-12)


def compute_mean_activation(tensorX):
    """Per-neuron mean activation across all stimuli and time. Shape: (N,)."""
    return np.mean(tensorX, axis=(1, 2))


def compute_dsi(tensorX, n_stim, n_dir=8):
    """Direction Selectivity Index per neuron.

    DSI = (R_pref - R_opp) / (R_pref + R_opp)
    where R_opp is the response at the direction 180° opposite to the preferred.
    Computed per stimulus then max taken, consistent with compute_osi_and_pref_stim.

    Args:
        tensorX: processed tensor, shape (N, NSTIMS, NDIRS * TRIAL_LEN)
        n_stim: number of stimuli
        n_dir: number of directions (default 8)

    Returns:
        ndarray of shape (N,) with DSI values in [0, 1]
    """
    tensor_reshaped = tensorX.reshape((len(tensorX), n_stim, n_dir, -1))
    data_avg = tensor_reshaped.mean(axis=3)   # (N, n_stim, n_dir)
    opp_offset = n_dir // 2

    DSI_per_stimulus = np.zeros((len(tensorX), n_stim))
    for stim_idx in range(n_stim):
        tc = data_avg[:, stim_idx, :]
        pref_dir = tc.argmax(axis=1)
        opp_dir  = (pref_dir + opp_offset) % n_dir
        R_pref   = tc[np.arange(len(tensorX)), pref_dir]
        R_opp    = tc[np.arange(len(tensorX)), opp_dir]
        DSI_per_stimulus[:, stim_idx] = (R_pref - R_opp) / (R_pref + R_opp + 1e-12)

    return DSI_per_stimulus.max(axis=1)


# ---------------------------------------------------------------------------
# Distance matrix helpers
# ---------------------------------------------------------------------------

def normalize_distance_matrix(D):
    """Scale matrix so that max=1. If max=0, return D unchanged."""
    dmax = np.max(D)
    if dmax > 0:
        return D / dmax
    return D


# ---------------------------------------------------------------------------
# Gromov-Wasserstein / Gromov-Hausdorff
# ---------------------------------------------------------------------------

def compute_gromov_wasserstein(C1, C2, p, q, loss_fun='square_loss', max_iter=10000, tol=1e-4):
    """GW cost between cost/distance matrices C1, C2 via POT library.

    Args:
        C1, C2: distance matrices for the two spaces
        p, q: probability distributions over the two spaces
        loss_fun: POT loss function (default 'square_loss')

    Returns:
        float: GW cost (not the sqrt)
    """
    import ot
    return ot.gromov.gromov_wasserstein2(C1, C2, p, q, loss_fun=loss_fun, max_iter=max_iter, tol=tol)


def compute_gromov_hausdorff_approx(X, Y, metric='euclidean'):
    """Approximate GH distance ~ sqrt(GW) from point clouds X and Y.

    Returns the raw GW cost; take sqrt() to get the GH approximation.
    """
    import ot
    Cx = squareform(pdist(X, metric=metric))
    Cy = squareform(pdist(Y, metric=metric))
    N, M = len(X), len(Y)
    p = np.ones(N) / N
    q = np.ones(M) / M
    return ot.gromov.gromov_wasserstein2(Cx, Cy, p, q, loss_fun='square_loss', max_iter=10000, tol=1e-4)


def compute_single_linkage_ultrametric(points, metric='euclidean'):
    """NxD points → NxN ultrametric matrix via custom single-linkage algorithm.

    Implements the algorithm with alpha = sqrt(2) and k = d * log(n).
    """
    from scipy.spatial import cKDTree
    from math import log, ceil

    n, d = points.shape
    if n < 2:
        return np.zeros((n, n), dtype=float)

    alpha = np.sqrt(2)
    k = max(2, ceil(d * log(n)))

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k + 1, p=2)
    rk = distances[:, -1]

    max_rk = np.max(rk)
    pairs = tree.query_pairs(r=alpha * max_rk, p=2)

    if pairs:
        pair_list = np.array(list(pairs))
        diffs = points[pair_list[:, 0]] - points[pair_list[:, 1]]
        pair_distances = np.linalg.norm(diffs, axis=1)
        sorted_indices = np.argsort(pair_distances)
        sorted_pairs = pair_list[sorted_indices]
        sorted_distances = pair_distances[sorted_indices]
    else:
        sorted_pairs = np.empty((0, 2), dtype=int)
        sorted_distances = np.array([])

    parent = np.arange(n)
    rank_union = np.zeros(n, dtype=int)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu == pv:
            return False
        if rank_union[pu] < rank_union[pv]:
            parent[pu] = pv
        else:
            parent[pv] = pu
            if rank_union[pu] == rank_union[pv]:
                rank_union[pu] += 1
        return True

    def compress_paths():
        for u in range(len(parent)):
            find(u)

    U = np.full((n, n), np.inf)
    np.fill_diagonal(U, 0)

    for (i, j), dist in zip(sorted_pairs, sorted_distances):
        current_r = max(rk[i], rk[j])
        if dist > alpha * current_r:
            continue
        if union(i, j):
            compress_paths()
            root = find(i)
            members = np.where(parent == root)[0]
            for m1 in members:
                for m2 in members:
                    if m1 < m2:
                        U[m1, m2] = min(U[m1, m2], current_r)
                        U[m2, m1] = U[m1, m2]

    U[U == np.inf] = max_rk
    return U


def compute_entropic_gw(C1, C2, p, q, epsilon=0.01, solver='PPA', max_iter=200, tol=1e-5):
    """Entropic GW cost via POT library (faster approximation for large N).

    Args:
        C1, C2: normalized distance matrices for the two spaces
        p, q: probability distributions over the two spaces
        epsilon: regularization strength (larger = faster, less accurate)
        solver: 'PPA' (proximal point, recommended) or 'sinkhorn'

    Returns:
        float: GW cost (take sqrt for GH approximation)
    """
    import ot
    return ot.gromov.entropic_gromov_wasserstein2(
        C1, C2, p, q, epsilon=epsilon, solver=solver, max_iter=max_iter, tol=tol)


def compute_fgw(M, C1, C2, p, q, alpha=0.5, max_iter=200, tol=1e-4):
    """Fused GW cost: structural (GW) + feature (Wasserstein) terms.

    Args:
        M: feature cost matrix, shape (N1, N2), should be normalized to max=1
        C1, C2: structural distance matrices
        p, q: probability distributions
        alpha: weight for structural GW term (1-alpha weights the feature term)

    Returns:
        float: FGW cost (take sqrt for distance interpretation)
    """
    import ot
    return ot.gromov.fused_gromov_wasserstein2(
        M, C1, C2, p, q, alpha=alpha, max_iter=max_iter, tol=tol)


def approximate_gh_on_ultrametrics(U1, U2, loss_fun='square_loss', max_iter=10000, tol=1e-4):
    """Approximate GH distance via GW on normalised ultrametric matrices.

    Args:
        U1, U2: NxN and MxM ultrametric distance matrices

    Returns:
        float: sqrt(GW cost) on normalised matrices
    """
    import ot
    N, M = U1.shape[0], U2.shape[0]
    p = np.ones(N) / N
    q = np.ones(M) / M
    U1n = normalize_distance_matrix(U1)
    U2n = normalize_distance_matrix(U2)
    cost = ot.gromov.gromov_wasserstein2(U1n, U2n, p, q, loss_fun=loss_fun, max_iter=max_iter, tol=tol)
    return np.sqrt(abs(cost))


# ---------------------------------------------------------------------------
# Natural-movie ↔ manifold index maps
# ---------------------------------------------------------------------------

def build_index_maps(
    session_uids_used: list[tuple[int, int]],
    cell_ids: list[tuple[int, int]],
) -> tuple[dict[int, int], dict[int, int]]:
    """Build bidirectional index maps between natural-movie and manifold neuron spaces.

    Args:
        session_uids_used: (session_id, unit_id) pairs in natural-movie order.
        cell_ids: (session_id, unit_id) pairs in manifold order.

    Returns:
        nm2m: dict mapping natural-movie index → manifold index.
        m2nm: dict mapping manifold index → natural-movie index.
    """
    manifold_lookup = {uid_pair: j for j, uid_pair in enumerate(cell_ids)}
    nm2m: dict[int, int] = {}
    m2nm: dict[int, int] = {}
    for i, uid_pair in enumerate(session_uids_used):
        if uid_pair in manifold_lookup:
            j = manifold_lookup[uid_pair]
            nm2m[i] = j
            m2nm[j] = i
    print(
        f"Shared neurons: {len(nm2m)} of {len(session_uids_used)} nat-movie, "
        f"{len(m2nm)} of {len(cell_ids)} manifold"
    )
    return nm2m, m2nm


def natmovie_to_manifold_indices(natmovie_ixs: list[int], nm2m: dict[int, int]) -> list[int]:
    """Map natural-movie neuron indices to manifold indices (dropping unmatched)."""
    return [nm2m[i] for i in natmovie_ixs if i in nm2m]


def manifold_to_natmovie_indices(manifold_ixs: list[int], m2nm: dict[int, int]) -> list[int]:
    """Map manifold neuron indices to natural-movie indices (dropping unmatched)."""
    return [m2nm[j] for j in manifold_ixs if j in m2nm]
