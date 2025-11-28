#!/usr/bin/env python3
"""
Tubularity experiments: clustering, centerlines, and two scores (S_tight, S_cross).
Grid of scenarios:
  - Single-tube radius sweep
  - Two-parallel-tubes: separation × radius
  - Two-crossing-tubes: angle × radius

Plots (matplotlib only; one chart per figure; no explicit colors):
  1) Overview: curves + fitted centerlines per basic scenario.
  2) Per-cluster bar charts for S_tight and S_cross.
  3) Radius profiles R_q(u).
  4) Grid summaries with means ± sqrt(variance).

If clustering returns no valid clusters, we fall back to a single cluster to avoid NaNs.

Run:
  python tubularity_experiments.py
  python tubularity_experiments.py --lite   # skip heavy overview plots, show grid summaries only
"""
import argparse, math
import numpy as np
import matplotlib.pyplot as plt

# Optional deps
try:
    import hdbscan  # type: ignore
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    from scipy.interpolate import UnivariateSpline, splrep, splev
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ----------------------------
# Utilities
# ----------------------------
def arclength_resample(curve_xy, M):
    X = np.asarray(curve_xy, dtype=float)
    diffs = np.diff(X, axis=0)
    seglens = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglens)])
    total = s[-1] if s[-1] > 0 else 1.0
    u_new = np.linspace(0.0, 1.0, M)
    s_new = u_new * total
    res = np.zeros((M, X.shape[1]))
    j = 0
    for k, sn in enumerate(s_new):
        while j < len(s) - 2 and s[j+1] < sn:
            j += 1
        t = 0.0 if s[j+1] - s[j] == 0 else (sn - s[j]) / (s[j+1] - s[j])
        res[k] = X[j] * (1 - t) + X[j+1] * t
    if M > 1:
        w = np.ones(M) * (total / (M - 1))
        w[0] *= 0.5; w[-1] *= 0.5
    else:
        w = np.array([total])
    return res, w, total

def finite_difference_derivative(curve_u):
    X = np.asarray(curve_u, dtype=float)
    M = X.shape[0]
    du = 1.0 / (M - 1) if M > 1 else 1.0
    dX = np.zeros_like(X)
    if M == 1:
        return dX
    dX[0] = (X[1] - X[0]) / du
    dX[-1] = (X[-1] - X[-2]) / du
    if M > 2:
        dX[1:-1] = (X[2:] - X[:-2]) / (2 * du)
    return dX


# ----------------------------
# Distances (L2 and H1)
# ----------------------------
def curve_L2_distance_matrix(curves):
    m = len(curves)
    Dmat = np.zeros((m, m))
    for i in range(m):
        Xi = curves[i]['uX']
        for j in range(i+1, m):
            Xj = curves[j]['uX']
            diff = Xi - Xj
            val2 = np.mean(np.sum(diff**2, axis=1))
            d = math.sqrt(max(val2, 0.0))
            Dmat[i, j] = Dmat[j, i] = d
    return Dmat

def curve_H1_distance_matrix(curves, alpha=0.5):
    m = len(curves)
    Dmat = np.zeros((m, m))
    for i in range(m):
        Xi = curves[i]['uX']; Di = curves[i]['dX']
        for j in range(i+1, m):
            Xj = curves[j]['uX']; Dj = curves[j]['dX']
            diff0 = Xi - Xj
            diff1 = Di - Dj
            val2 = np.mean(np.sum(diff0**2, axis=1)) + alpha * np.mean(np.sum(diff1**2, axis=1))
            d = math.sqrt(max(val2, 0.0))
            Dmat[i, j] = Dmat[j, i] = d
    return Dmat


# ----------------------------
# Clustering
# ----------------------------
def _labels_or_fallback(labels, min_valid_size=2):
    """If no valid clusters (size >= min_valid_size), fallback to single cluster label 0."""
    if labels is None or len(labels) == 0:
        return np.zeros(0, dtype=int)
    labs, counts = np.unique(labels, return_counts=True)
    # valid = non -1 and size >= min_valid_size
    ok = [(lab, c) for lab, c in zip(labs, counts) if (lab != -1 and c >= min_valid_size)]
    if len(ok) == 0:
        return np.zeros_like(labels, dtype=int)  # everyone in one cluster
    return labels

def cluster_curves_precomputed(D, method="hdbscan", min_cluster_size=6, min_samples=None, eps=None):
    """Return labels in {-1,0,1,...}. Robust: falls back to single cluster if everything is noise."""
    m = D.shape[0]
    labels = None
    if method == "hdbscan" and HAS_HDBSCAN:
        ms = min_samples if (min_samples is not None) else max(3, min_cluster_size//2)
        try:
            clusterer = hdbscan.HDBSCAN(metric='precomputed',
                                        min_cluster_size=max(3, min_cluster_size),
                                        min_samples=ms,
                                        cluster_selection_method='eom')
            labels = clusterer.fit_predict(D)
        except Exception:
            labels = None
    if labels is None and HAS_SKLEARN:
        # Adaptive eps: aim for a connected graph; relax if too sparse
        tri = D[np.triu_indices(m, 1)]
        med = np.median(tri) if tri.size > 0 else 1.0
        base_eps = med * 1.2
        tried = []
        for scale in [1.2, 1.5, 2.0, 3.0]:
            e = med * scale if np.isfinite(med) and med > 0 else 1.0 * scale
            tried.append(e)
            try:
                clusterer = DBSCAN(eps=e, min_samples=max(3, min_cluster_size//2), metric='precomputed')
                labels = clusterer.fit_predict(D)
                labs, cnts = np.unique(labels, return_counts=True)
                if np.any((labs != -1) & (cnts >= 2)):
                    break
            except Exception:
                labels = None
        # if still None, will fallback below
    if labels is None:
        # simple threshold fallback
        tri = D[np.triu_indices(m, 1)]
        thr = np.median(tri) * 1.2 if tri.size > 0 else 1.0
        adj = (D <= thr).astype(int)
        np.fill_diagonal(adj, 0)
        visited = np.zeros(m, dtype=bool)
        labels = -np.ones(m, dtype=int)
        cur = 0
        for i in range(m):
            if not visited[i]:
                q = [i]; comp = []; visited[i] = True
                while q:
                    u = q.pop(0); comp.append(u)
                    neigh = np.where(adj[u] == 1)[0]
                    for v in neigh:
                        if not visited[v]:
                            visited[v] = True; q.append(v)
                if len(comp) >= 2:
                    for v in comp: labels[v] = cur
                    cur += 1
        # if no cluster found, everyone together
        if np.all(labels == -1):
            labels = np.zeros(m, dtype=int)
    return _labels_or_fallback(labels, min_valid_size=2)


# ----------------------------
# Centerline fitting
# ----------------------------
def _make_strictly_increasing(x, eps=1e-9):
    x = np.asarray(x, float).copy()
    for i in range(1, x.size):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + eps
    return np.clip(x, 0.0, 1.0)

def _compress_duplicates(uh, Xh):
    uh = np.asarray(uh, float)
    Xh = np.asarray(Xh, float)
    uu, inv, cnt = np.unique(uh, return_inverse=True, return_counts=True)
    sums = np.zeros((uu.shape[0], Xh.shape[1]))
    for i, c in enumerate(inv):
        sums[c] += Xh[i]
    Xavg = sums / cnt[:, None]
    return uu, Xavg

def fit_centerline_mean(curves_uX, smoothing=2.0, max_iters=10, grid_eval=400, tol=1e-3):
    """
    Centerline from the pointwise mean of curves with optional smoothing.

    Args:
        curves_uX: List of curve arrays, each shape (M, D) where M is samples, D is dims.
        smoothing: Smoothing strength. If HAS_SCIPY, passed as "s" to UnivariateSpline
                   per dimension (s=0 gives interpolation; larger s = smoother). If
                   SciPy is unavailable, applies a moving-average with a window that
                   grows with "smoothing". Set <= 0 to disable smoothing.
        max_iters: Unused (kept for compatibility).
        grid_eval: Unused here (kept for compatibility with other fitters).
        tol:       Unused (kept for compatibility).

    Returns:
        c_of_u: Callable u -> R^D that evaluates the smoothed mean centerline for u in [0,1].
    """
    # Compute pointwise mean curve on the native uniform grid
    smoothing = 0.01
    M, D = curves_uX[0].shape
    u_grid = np.linspace(0.0, 1.0, M)
    mean_curve = np.mean(np.stack(curves_uX, axis=0), axis=0)

    # If smoothing disabled, fall back to simple linear interpolation of the mean
    if smoothing is None or smoothing <= 0:
        print("Here")
        def c_of_u(u):
            u = np.clip(np.asarray(u), 0.0, 1.0)
            idx = np.searchsorted(u_grid, u, side='left')
            idx = np.clip(idx, 1, M - 1)
            t = (u - u_grid[idx - 1]) / (u_grid[idx] - u_grid[idx - 1] + 1e-12)
            return (1 - t)[:, None] * mean_curve[idx - 1] + t[:, None] * mean_curve[idx]
        return c_of_u

    # Prefer spline smoothing when SciPy is available
    if HAS_SCIPY:
        spls = [UnivariateSpline(u_grid, mean_curve[:, d], s=float(smoothing)) for d in range(D)]
        def c_of_u(u):
            u = np.clip(np.asarray(u), 0.0, 1.0)
            return np.stack([spl(u) for spl in spls], axis=1)
        return c_of_u

    # Fallback: moving-average smoothing of the mean curve, then linear interp
    def _moving_avg(y, win):
        pad = win // 2
        yp = np.pad(y, (pad, pad), mode='edge')
        ker = np.ones(win) / win
        return np.convolve(yp, ker, mode='valid')

    # Window scales gently with smoothing and sample count; ensure odd
    base = max(3, M // 20)
    win = int(max(5, 2 * int(base * float(smoothing)) + 1))
    if win >= M:
        win = M - 1 if (M % 2 == 0) else M - 2
        if win < 3:
            win = 3
    if win % 2 == 0:
        win += 1

    smooth_curve = np.zeros_like(mean_curve)
    for d in range(D):
        smooth_curve[:, d] = _moving_avg(mean_curve[:, d], win)

    def c_of_u(u):
        u = np.clip(np.asarray(u), 0.0, 1.0)
        idx = np.searchsorted(u_grid, u, side='left')
        idx = np.clip(idx, 1, M - 1)
        t = (u - u_grid[idx - 1]) / (u_grid[idx] - u_grid[idx - 1] + 1e-12)
        return (1 - t)[:, None] * smooth_curve[idx - 1] + t[:, None] * smooth_curve[idx]
    return c_of_u

def fit_centerline_alternating(curves_uX, smoothing=2.0, max_iters=10, grid_eval=400, tol=1e-3):
    """Alternate: project to current curve (polyline) → refit cubic smoothing splines per dim."""
    M, D = curves_uX[0].shape
    u_grid = np.linspace(0.0, 1.0, M)
    mean_curve = np.mean(np.stack(curves_uX, axis=0), axis=0)

    if not HAS_SCIPY:
        # Fallback: moving-average smoothing + linear interpolation
        win = max(5, (M // 20) * 2 + 1)
        def moving_avg(y, w):
            pad = w // 2
            yp = np.pad(y, (pad, pad), mode='edge')
            ker = np.ones(w) / w
            return np.convolve(yp, ker, mode='valid')
        smooth = np.zeros_like(mean_curve)
        for d in range(D):
            smooth[:, d] = moving_avg(mean_curve[:, d], win)
        def c_of_u(u):
            u = np.clip(np.asarray(u), 0.0, 1.0)
            idx = np.searchsorted(u_grid, u, side='left')
            idx = np.clip(idx, 1, M - 1)
            t = (u - u_grid[idx - 1]) / (u_grid[idx] - u_grid[idx - 1] + 1e-12)
            return (1 - t)[:, None] * smooth[idx - 1] + t[:, None] * smooth[idx]
        return c_of_u

    pooled_X = np.concatenate(curves_uX, axis=0)
    u_eval = np.linspace(0.0, 1.0, grid_eval)

    mean_eval = np.zeros((grid_eval, D))
    for d in range(D):
        tck = splrep(u_grid, mean_curve[:, d], s=0)
        mean_eval[:, d] = splev(u_eval, tck)

    def project_to_polyline(points, poly_u, poly_pts):
        P = points
        U = np.zeros(P.shape[0]); Y = np.zeros_like(P)
        segs = poly_pts[1:] - poly_pts[:-1]
        seglen2 = np.sum(segs**2, axis=1) + 1e-12
        for i in range(P.shape[0]):
            p = P[i]
            v = p - poly_pts[:-1]
            t = np.sum(v * segs, axis=1) / seglen2
            t = np.clip(t, 0.0, 1.0)
            proj = poly_pts[:-1] + (t[:, None] * segs)
            d2 = np.sum((proj - p)**2, axis=1)
            k = int(np.argmin(d2))
            u_proj = poly_u[k] * (1 - t[k]) + poly_u[k+1] * t[k]
            U[i] = u_proj; Y[i] = proj[k]
        return U, Y

    u_hat, _ = project_to_polyline(pooled_X, u_eval, mean_eval)
    prev = u_hat.copy()

    for _ in range(max_iters):
        order = np.argsort(u_hat)
        uh = u_hat[order]; Xh = pooled_X[order]
        uh, Xh = _compress_duplicates(uh, Xh)
        uh = _make_strictly_increasing(uh, eps=1e-9)
        spls = [UnivariateSpline(uh, Xh[:, d], s=smoothing) for d in range(D)]
        c_eval = np.stack([spls[d](u_eval) for d in range(D)], axis=1)
        u_hat, _ = project_to_polyline(pooled_X, u_eval, c_eval)
        if np.max(np.abs(u_hat - prev)) < tol:
            break
        prev = u_hat.copy()

    order = np.argsort(u_hat)
    uh = u_hat[order]; Xh = pooled_X[order]
    uh, Xh = _compress_duplicates(uh, Xh)
    uh = _make_strictly_increasing(uh, eps=1e-9)
    spls = [UnivariateSpline(uh, Xh[:, d], s=smoothing) for d in range(D)]
    def c_of_u(u):
        u = np.clip(np.asarray(u), 0.0, 1.0)
        return np.stack([spls[d](u) for d in range(D)], axis=1)
    return c_of_u


# ----------------------------
# Epsilon (inter-curve scale)
# ----------------------------
def cluster_epsilon(curves_in_cluster):
    pts, owners = [], []
    for idx, c in enumerate(curves_in_cluster):
        X = c['uX']
        pts.append(X); owners.append(np.full(X.shape[0], idx, dtype=int))
    P = np.vstack(pts); O = np.concatenate(owners)
    if HAS_SCIPY:
        tree = cKDTree(P)
        dists = []
        kq = 10
        dd, ii = tree.query(P, k=kq)
        if kq == 1:
            dd = dd[:, None]; ii = ii[:, None]
        for i in range(P.shape[0]):
            ow = O[i]; found = None
            for kk in range(1, kq):
                j = ii[i, kk]
                if j == -1:
                    continue
                if O[j] != ow:
                    found = dd[i, kk]; break
            if found is None:
                found = float('inf')
            dists.append(found)
        d = np.array(dists); d = d[np.isfinite(d)]
        eps = float(np.median(d)) if d.size > 0 else 1.0
        return max(eps, 1e-12)
    # slow fallback
    dmin = []
    for i in range(P.shape[0]):
        ow = O[i]
        diffs = P - P[i]
        dist = np.linalg.norm(diffs, axis=1)
        dist[O == ow] = np.inf
        mv = np.min(dist)
        if np.isfinite(mv):
            dmin.append(mv)
    eps = float(np.median(dmin)) if dmin else 1.0
    return max(eps, 1e-12)


# ----------------------------
# Scores
# ----------------------------
def centerline_mean_distance(centerline_a, centerline_b, n_points=100):
    u = np.linspace(0, 1, n_points)
    pts_a = centerline_a(u)
    pts_b = centerline_b(u)
    return np.mean(np.linalg.norm(pts_a - pts_b, axis=1))

def compute_S_tight(curves_in_cluster, c_of_u, q=0.9, B=30, contextual_normalization=True, all_centerlines=None, tube_idx=None, n_points=100):
    """
    Compute tightness score for a tube bundle.
    If contextual_normalization is True, normalize by mean distance to other centerlines.
    all_centerlines: list of centerline functions for all tubes (required if contextual_normalization=True)
    tube_idx: index of current tube in all_centerlines (required if contextual_normalization=True)
    """
    eps = cluster_epsilon(curves_in_cluster)
    print("  Eps:", eps)
    M = curves_in_cluster[0]['uX'].shape[0]
    u_bins = np.linspace(0.0, 1.0, B+1)
    centers = 0.5 * (u_bins[:-1] + u_bins[1:])
    Rq = np.zeros(B)
    for b in range(B):
        u_c = centers[b]
        Cb = c_of_u(np.array([u_c]))[0]
        res = []
        for c in curves_in_cluster:
            X = c['uX']
            u_grid = np.linspace(0.0, 1.0, M)
            mask = ((u_grid >= u_bins[b]) & (u_grid < u_bins[b+1])) if b < B-1 else ((u_grid >= u_bins[b]) & (u_grid <= u_bins[b+1]))
            if not np.any(mask):
                continue
            res.extend(np.linalg.norm(X[mask] - Cb[None, :], axis=1).tolist())
        Rq[b] = 0.0 if len(res) == 0 else float(np.quantile(np.array(res), q))

    S_tight_raw = float(np.mean(Rq))

    if contextual_normalization and all_centerlines is not None and tube_idx is not None:
        other_centerlines = [c for i, c in enumerate(all_centerlines) if i != tube_idx]
        if other_centerlines:
            dists = [centerline_mean_distance(c_of_u, c_other, n_points=n_points) for c_other in other_centerlines]
            mean_dist = np.min(dists)
        else:
            mean_dist = 1.0
        S_tight = S_tight_raw / mean_dist if mean_dist > 0 else S_tight_raw
        return S_tight, mean_dist, (centers, Rq, mean_dist)
    else:
        # Normalize by centerline length instead of eps
        # Approximate centerline length by sampling c_of_u on a dense grid
        uu = np.linspace(0.0, 1.0, max(200, B*4))
        C = c_of_u(uu)
        segs = np.diff(C, axis=0)
        L = float(np.sum(np.linalg.norm(segs, axis=1)))
        S_tight = S_tight_raw / (L + 1e-12)
        return S_tight, eps, (centers, Rq, eps)

def compute_S_cross(curves_in_cluster, eps, neighbor_radius_mult=3.0, centerline_length=None):
    pts, owners, tans, weights = [], [], [], []
    for idx, c in enumerate(curves_in_cluster):
        X = c['uX']; dX = c['dX']
        M = X.shape[0]
        T = dX.copy()
        nrm = np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        T = T / nrm
        w = np.ones(M) / M
        pts.append(X); owners.append(np.full(M, idx, dtype=int)); tans.append(T); weights.append(w)
    P = np.vstack(pts); O = np.concatenate(owners); T = np.vstack(tans); W = np.concatenate(weights)

    rad = neighbor_radius_mult * eps
    if HAS_SCIPY:
        tree = cKDTree(P)
        idxs = tree.query_ball_point(P, r=rad)
    else:
        diffs = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(diffs, axis=2)
        idxs = [list(np.where(dist[i] <= rad)[0]) for i in range(P.shape[0])]

    m_k = len(curves_in_cluster)
    total = 0.0
    for i in range(P.shape[0]):
        for j in idxs[i]:
            if j <= i:
                continue
            if O[i] == O[j]:
                continue
            rho2 = float(np.sum((P[i] - P[j])**2))
            kernel = math.exp(-rho2 / (2.0 * eps * eps))
            phi = 1.0 - float(np.dot(T[i], T[j])**2)
            total += kernel * phi * W[i] * W[j]

    X_eps = (2.0 / (m_k * (m_k - 1))) * total if m_k >= 2 else 0.0
    if centerline_length is None:
        # fallback to legacy eps-based normalization if length not provided
        S_cross = X_eps / (eps * eps + 1e-12)
    else:
        S_cross = X_eps / (float(centerline_length) + 1e-12)
    return float(S_cross)


# ----------------------------
# Simulation
# ----------------------------
def simulate_single_tube(n_curves=25, radius=0.02, curvature=0.0, seed=0):
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_curves):
        n = rng.integers(90, 130)
        t = np.linspace(0.0, 1.0, n)
        base = np.stack([t, curvature*(t-0.5)*(1-t)], axis=1)
        y = rng.normal(0, radius, size=n)
        X = base + np.stack([np.zeros(n), y], axis=1)
        curves.append(X)
    return curves

def simulate_two_parallel_tubes(n_curves_per=20, sep=0.3, radius=0.04, seed=1):
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_curves_per):
        n = rng.integers(90, 130)
        t = np.linspace(0.0, 1.0, n)
        base = np.stack([t, np.zeros_like(t) + sep/2], axis=1)
        X = base + rng.normal(0, radius, size=(n, 2)) * np.array([0.2, 1.0])
        curves.append(X)
    for _ in range(n_curves_per):
        n = rng.integers(90, 130)
        t = np.linspace(0.0, 1.0, n)
        base = np.stack([t, np.zeros_like(t) - sep/2], axis=1)
        X = base + rng.normal(0, radius, size=(n, 2)) * np.array([0.2, 1.0])
        curves.append(X)
    return curves

def simulate_two_crossing_tubes(n_curves_per=18, angle_deg=90, radius=0.02, seed=2):
    rng = np.random.default_rng(seed)
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    curves = []
    for _ in range(n_curves_per):
        n = rng.integers(90, 130)
        t = np.linspace(-0.6, 0.6, n)
        base = np.stack([t, np.zeros_like(t)], axis=1)
        X = base + rng.normal(0, radius, size=(n, 2)) * np.array([0.2, 1.0])
        X = X + np.array([0.5, 0.0])
        curves.append(X)
    for _ in range(n_curves_per):
        n = rng.integers(90, 130)
        t = np.linspace(-0.6, 0.6, n)
        base = np.stack([t, np.zeros_like(t)], axis=1) @ R.T
        X = base + (rng.normal(0, radius, size=(n, 2)) @ R.T) * np.array([0.2, 1.0])
        X = X + np.array([0.5, 0.0])
        curves.append(X)
    return curves


# ----------------------------
# Pipeline
# ----------------------------
def preprocess_curves(curves_raw, M=160):
    curves = []
    for X in curves_raw:
        uX, w, L = arclength_resample(X, M)
        dX = finite_difference_derivative(uX)
        curves.append({'uX': uX, 'dX': dX, 'length': L})
    return curves

def run_clustering(curves, metric="H1", alpha=0.5, min_cluster_size=6):
    if metric.lower() == "l2":
        D = curve_L2_distance_matrix(curves)
    else:
        D = curve_H1_distance_matrix(curves, alpha=alpha)
    tri = D[np.triu_indices(len(curves), 1)]
    scale = np.median(tri) if tri.size > 0 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    Dn = D / scale
    labels = cluster_curves_precomputed(Dn, method="hdbscan", min_cluster_size=min_cluster_size)
    return labels, Dn

def compute_cluster_metrics(curves, labels, smoothing=2.0, q=0.9, B=30):
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(curves[i])
    results = []
    centerlines = []
    cluster_keys = list(clusters.keys())
    # First, fit all centerlines
    for lab in cluster_keys:
        cur_list = clusters[lab]
        if len(cur_list) < 2:
            centerlines.append(None)
            continue
        c_of_u = fit_centerline_mean([c['uX'] for c in cur_list], smoothing=smoothing, max_iters=10, grid_eval=400, tol=1e-3)
        centerlines.append(c_of_u)
    # Now, compute metrics with contextual normalization
    for idx, lab in enumerate(cluster_keys):
        cur_list = clusters[lab]
        if len(cur_list) < 2:
            continue
        c_of_u = centerlines[idx]
        S_tight, eps, (u_centers, Rq, eps_used) = compute_S_tight(
            cur_list, c_of_u, q=q, B=B,
            contextual_normalization=False,
            all_centerlines=[c for c in centerlines if c is not None],
            tube_idx=sum([1 for i in range(idx) if centerlines[i] is not None])
        )
        # Compute centerline length for normalization of S_cross
        uu_len = np.linspace(0.0, 1.0, 400)
        C_len = c_of_u(uu_len)
        segs_len = np.diff(C_len, axis=0)
        L_center = float(np.sum(np.linalg.norm(segs_len, axis=1)))
        S_cross = compute_S_cross(cur_list, eps_used, centerline_length=L_center)
        results.append({
            'label': lab,
            'n_curves': len(cur_list),
            'epsilon': eps_used,
            'S_tight': S_tight,
            'S_cross': S_cross,
            'centerline': c_of_u,
            'members': cur_list,
            'Rq_profile': (u_centers, Rq)
        })
    # Final guard: if still empty (e.g., all singletons), force one cluster
    if not results and len(curves) >= 2:
        c_of_u = fit_centerline_mean([c['uX'] for c in curves],
                                            smoothing=smoothing, max_iters=10, grid_eval=400, tol=1e-3)
        S_tight, eps, (u_centers, Rq, eps_used) = compute_S_tight(curves, c_of_u, q=q, B=B, contextual_normalization=True)
        # Compute centerline length for normalization of S_cross
        uu_len = np.linspace(0.0, 1.0, 400)
        C_len = c_of_u(uu_len)
        segs_len = np.diff(C_len, axis=0)
        L_center = float(np.sum(np.linalg.norm(segs_len, axis=1)))
        S_cross = compute_S_cross(curves, eps_used, centerline_length=L_center)
        results.append({
            'label': 0, 'n_curves': len(curves), 'epsilon': eps_used,
            'S_tight': S_tight, 'S_cross': S_cross,
            'centerline': c_of_u, 'members': curves, 'Rq_profile': (u_centers, Rq)
        })
    return results

def aggregate_metrics(results):
    if not results:
        return np.nan, np.nan, np.nan, np.nan
    total = sum(r['n_curves'] for r in results)
    wts = np.array([r['n_curves'] / total for r in results])
    tight = np.array([r['S_tight'] for r in results])
    cross = np.array([r['S_cross'] for r in results])
    tight_avg = float(np.sum(wts * tight))
    cross_avg = float(np.sum(wts * cross))
    tight_var = float(np.sum(wts * (tight - tight_avg)**2))
    cross_var = float(np.sum(wts * (cross - cross_avg)**2))
    return tight_avg, cross_avg, tight_var, cross_var


# ----------------------------
# Plots
# ----------------------------
def plot_scenario_overview(curves_raw, results, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for X in curves_raw:
        ax.plot(X[:,0], X[:,1], linewidth=0.8, alpha=0.7)
    for r in results:
        uu = np.linspace(0.0, 1.0, 400)
        C = r['centerline'](uu)
        print(len(C))
        ax.plot(C[:,0], C[:,1], linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.show()

def plot_scenario_overview_proper(curves_raw, curves_processed, results, labels, title, layer=None):
    """
    Improved version that uses actual cluster labels for proper coloring.
    
    Args:
        curves_raw: List of original curve arrays (before preprocessing)
        curves_processed: List of preprocessed curves (with 'uX', 'dX', etc.)
        results: List of cluster results from compute_cluster_metrics
        labels: Cluster labels for each curve (from clustering step)
        title: Plot title
    """
    from tueplots import bundles, axes
    from tueplots.constants.color import rgb

    if True:
    #with plt.rc_context({**bundles.neurips2024(), **axes.lines()}):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Get unique cluster labels and assign colors
        unique_labels = np.unique(labels)
        colors = stimulus_colors = [
            '#0070C0', '#00B0F0', '#00B050', '#92D050', '#FF0000', '#FFC000', 
            '#00B0F0', '#00B050', '#92D050', '#FF0000', '#FFC000'
        ]
        label_to_color = dict(zip(unique_labels, colors))
        
        # Plot individual curves colored by their actual cluster assignment
        for i, X in enumerate(curves_raw):
            cluster_label = labels[i]
            color = label_to_color[cluster_label]
            alpha = 0.7 if cluster_label != -1 else 0.3  # Dim noise points
            ax.plot(X[:,0], X[:,1], linewidth=0.8, alpha=alpha, color=color)
        
        # Plot centerlines with matching colors
        for r in results:
            uu = np.linspace(0.0, 1.0, 400)
            C = r['centerline'](uu)
            color = label_to_color[r['label']]
            ax.plot(C[:,0], C[:,1], linewidth=3.0, color=color, 
                    label=f"Bundle {r['label']}")# ({r['n_curves']} curves)")
        
        #ax.set_title(title)
        ax.set_xlabel("PC1", fontsize=20)
        ax.set_ylabel("PC2", fontsize=20)
        ax.legend(fontsize=20)
        plt.xticks([])
        plt.yticks([])
        if layer:
            plt.savefig(f"Tubes_{layer}.png")
        else:
            #ax.grid(True, alpha=0.3)
            plt.show()

def plot_per_cluster_bars(results, scenario_name, layer=None):
    labels = [f"C{r['label']}" for r in results]
    tight_vals = [r['S_tight'] for r in results]
    cross_vals = [r['S_cross'] for r in results]
    fig1 = plt.figure(); ax1 = fig1.add_subplot(111)
    ax1.bar(np.arange(len(results)), tight_vals)
    ax1.set_xticks(np.arange(len(results))); ax1.set_xticklabels(labels)
    ax1.set_ylabel("S_tight")
    ax1.set_title(f"{scenario_name}: per-cluster S_tight")
    plt.savefig(f"fig/bars/Tightness_{layer}.png")
    fig2 = plt.figure(); ax2 = fig2.add_subplot(111)
    ax2.bar(np.arange(len(results)), cross_vals)
    ax2.set_xticks(np.arange(len(results))); ax2.set_xticklabels(labels)
    ax2.set_ylabel("S_cross")
    ax2.set_title(f"{scenario_name}: per-cluster S_cross")
    plt.savefig(f"fig/bars/Crossings_{layer}.png")

def plot_radius_profiles(results, scenario_name):
    fig = plt.figure(); ax = fig.add_subplot(111)
    for r in results:
        u_centers, Rq = r['Rq_profile']
        ax.plot(u_centers, Rq, linewidth=1.5, alpha=0.9, label=f"C{r['label']}")
    ax.set_title(f"{scenario_name}: R_q(u) profiles")
    ax.set_xlabel("u (arc-length)"); ax.set_ylabel("R_q(u)")
    ax.legend()
    #plt.show()

def plot_metric_vs_param(x_vals, mean_vals, var_vals, xlabel, ylabel, title, xtick_labels=None):
    mean_vals = np.asarray(mean_vals, float)
    var_vals = np.asarray(var_vals, float)
    err = np.sqrt(np.maximum(0.0, var_vals))
    fig = plt.figure(); ax = fig.add_subplot(111)
    ax.errorbar(x_vals, mean_vals, yerr=err, marker='o')
    if xtick_labels is not None:
        ax.set_xticks(x_vals); ax.set_xticklabels(xtick_labels)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    #plt.show()

def plot_lines_over_param(x_vals, lines, xlabel, ylabel, title, legend_labels):
    fig = plt.figure(); ax = fig.add_subplot(111)
    for y in lines:
        ax.plot(x_vals, y, marker='o')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(legend_labels)
    #plt.show()


# ----------------------------
# Experiments
# ----------------------------
def run_basic_scenarios(show_overviews=True):
    basics = []
    curves_A = simulate_single_tube(n_curves=30, radius=0.015, curvature=0.05, seed=10)
    basics.append(("A: single tight tube", curves_A))
    curves_B = simulate_two_parallel_tubes(n_curves_per=20, sep=0.4, radius=0.05, seed=20)
    basics.append(("B: two parallel tubes (wide)", curves_B))
    curves_C = simulate_two_crossing_tubes(n_curves_per=18, angle_deg=90, radius=0.02, seed=30)
    basics.append(("C: two crossing tubes (≈90°)", curves_C))

    summaries = []
    for name, curves_raw in basics:
        curves = preprocess_curves(curves_raw, M=160)
        labels, _ = run_clustering(curves, metric="H1", alpha=0.5, min_cluster_size=6)
        results = compute_cluster_metrics(curves, labels, smoothing=2.0, q=0.9, B=30)
        t_avg, c_avg, t_var, c_var = aggregate_metrics(results)
        print(f"{name}: mean S_tight={t_avg:.3f} (var={t_var:.3f}), mean S_cross={c_avg:.3f} (var={c_var:.3f})")
        if show_overviews:
            plot_scenario_overview(curves_raw, results, title=f"{name}: curves + fitted centerlines")
            if results:
                plot_per_cluster_bars(results, scenario_name=name)
                plot_radius_profiles(results, scenario_name=name)
        summaries.append((name, t_avg, c_avg, t_var, c_var))

    names = [s[0] for s in summaries]
    xs = np.arange(len(names))
    plot_metric_vs_param(xs, [s[1] for s in summaries], [s[3] for s in summaries],
                         xlabel="Scenario", ylabel="Avg S_tight", title="Basic scenarios: S_tight",
                         xtick_labels=names)
    plot_metric_vs_param(xs, [s[2] for s in summaries], [s[4] for s in summaries],
                         xlabel="Scenario", ylabel="Avg S_cross", title="Basic scenarios: S_cross",
                         xtick_labels=names)

def radius_sweep_single_tube(radii, n_curves=30, seed=41):
    mt, vt, mc, vc = [], [], [], []
    for r in radii:
        curves = simulate_single_tube(n_curves=n_curves, radius=r, curvature=0.05, seed=seed)
        cs = preprocess_curves(curves, M=160)
        labels, _ = run_clustering(cs, metric="H1", alpha=0.5, min_cluster_size=6)
        results = compute_cluster_metrics(cs, labels, smoothing=2.0, q=0.9, B=30)
        t_avg, c_avg, t_var, c_var = aggregate_metrics(results)
        mt.append(t_avg); vt.append(t_var); mc.append(c_avg); vc.append(c_var)
        print(f"Single-tube radius={r:.3f}: S_tight={t_avg:.3f} (var={t_var:.3f}), S_cross={c_avg:.3f} (var={c_var:.3f})")
    return np.array(mt), np.array(vt), np.array(mc), np.array(vc)

def parallel_grid_sweep(seps, radii, n_per=18, seed=52):
    grid = {sep: {'tight': [], 'tight_var': [], 'cross': [], 'cross_var': []} for sep in seps}
    for sep in seps:
        for r in radii:
            curves = simulate_two_parallel_tubes(n_curves_per=n_per, sep=sep, radius=r, seed=seed + int(sep*100) + int(r*1000))
            cs = preprocess_curves(curves, M=160)
            labels, _ = run_clustering(cs, metric="H1", alpha=0.5, min_cluster_size=6)
            results = compute_cluster_metrics(cs, labels, smoothing=2.0, q=0.9, B=30)
            t_avg, c_avg, t_var, c_var = aggregate_metrics(results)
            grid[sep]['tight'].append(t_avg); grid[sep]['tight_var'].append(t_var)
            grid[sep]['cross'].append(c_avg); grid[sep]['cross_var'].append(c_var)
            print(f"Parallel sep={sep:.2f}, radius={r:.3f}: S_tight={t_avg:.3f}, S_cross={c_avg:.3f}")
    return grid

def crossing_grid_sweep(angles_deg, radii, n_per=16, seed=63):
    # returns dict: radius -> (tight_means, cross_means) arrays over angles
    result = {}
    for r in radii:
        tt, cc = [], []
        for ang in angles_deg:
            curves = simulate_two_crossing_tubes(n_curves_per=n_per, angle_deg=ang, radius=r, seed=seed + int(ang*10) + int(r*1000))
            cs = preprocess_curves(curves, M=160)
            labels, _ = run_clustering(cs, metric="H1", alpha=0.5, min_cluster_size=6)
            results = compute_cluster_metrics(cs, labels, smoothing=2.0, q=0.9, B=30)
            t_avg, c_avg, _, _ = aggregate_metrics(results)
            tt.append(t_avg); cc.append(c_avg)
            print(f"Crossing angle={ang:>3}°, radius={r:.3f}: S_tight={t_avg:.3f}, S_cross={c_avg:.3f}")
        result[r] = (np.array(tt), np.array(cc))
    return result

def run_grid_experiments(lite=False):
    # 1) Single tube radius sweep
    radii_single = [0.01, 0.02, 0.04, 0.08]
    mt, vt, mc, vc = radius_sweep_single_tube(radii_single, n_curves=30, seed=41)
    plot_metric_vs_param(radii_single, mt, vt, xlabel="Radius", ylabel="Avg S_tight",
                         title="Single tube: S_tight vs radius")
    plot_metric_vs_param(radii_single, mc, vc, xlabel="Radius", ylabel="Avg S_cross",
                         title="Single tube: S_cross vs radius")

    # 2) Two parallel tubes: separation × radius
    seps = [0.2, 0.4, 0.6]
    radii_par = [0.02, 0.05]
    grid = parallel_grid_sweep(seps, radii_par, n_per=18, seed=52)
    # lines vs separation for each fixed radius index j
    tight_lines = []
    cross_lines = []
    for j, r in enumerate(radii_par):
        tight_lines.append([grid[sep]['tight'][j] for sep in seps])
        cross_lines.append([grid[sep]['cross'][j] for sep in seps])
    plot_lines_over_param(seps, tight_lines, xlabel="Separation", ylabel="Avg S_tight",
                          title="Parallel tubes: S_tight vs separation",
                          legend_labels=[f"radius={r}" for r in radii_par])
    plot_lines_over_param(seps, cross_lines, xlabel="Separation", ylabel="Avg S_cross",
                          title="Parallel tubes: S_cross vs separation",
                          legend_labels=[f"radius={r}" for r in radii_par])

    # 3) Two crossing tubes: angle × radius
    angles = [15, 30, 45, 60, 90, 120, 150]
    radii_cross = [0.01, 0.03, 0.06]
    cross_grid = crossing_grid_sweep(angles, radii_cross, n_per=16, seed=63)
    cross_lines2 = [cross_grid[r][1] for r in radii_cross]
    plot_lines_over_param(angles, cross_lines2, xlabel="Crossing angle (deg)", ylabel="Avg S_cross",
                          title="Crossing tubes: S_cross vs angle",
                          legend_labels=[f"radius={r}" for r in radii_cross])
    tight_lines2 = [cross_grid[r][0] for r in radii_cross]
    plot_lines_over_param(angles, tight_lines2, xlabel="Crossing angle (deg)", ylabel="Avg S_tight",
                          title="Crossing tubes: S_tight vs angle",
                          legend_labels=[f"radius={r}" for r in radii_cross])


# ----------------------------
# Main
# ---------------------------- 
def main():
    parser = argparse.ArgumentParser(description="Tubularity experiments with a grid of scenarios")
    parser.add_argument("--lite", action="store_true", help="Skip overview plots; show grid summaries only")
    args = parser.parse_args()

    # Basic scenarios help validate intuitions and verify visuals
    if not args.lite:
        run_basic_scenarios(show_overviews=True)
    else:
        run_basic_scenarios(show_overviews=False)

    # Grid experiments
    run_grid_experiments(lite=args.lite)

if __name__ == "__main__":
    main()
