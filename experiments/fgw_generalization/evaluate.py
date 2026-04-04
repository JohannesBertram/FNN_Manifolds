"""Evaluation utilities: compute encoding manifold metrics and GW distances."""

import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist


def compute_encoding_manifold(model, X_stimuli, y_stimuli, n_ori):
    """Compute structural cost matrix and tuning features for a stimulus set.

    Parameters
    ----------
    model      : MLP
    X_stimuli  : (S, input_dim) float32 tensor
    y_stimuli  : (S,) int64 tensor
    n_ori      : int

    Returns
    -------
    C        : (N, N) float32 numpy — cosine distance on response profiles
    tuning   : (N, n_ori) float32 numpy — normalized mean response per class
    osi      : (N,) float32 numpy — circular resultant length (OSI proxy)
    pref_ori : (N,) int numpy — preferred orientation class
    """
    model.eval()
    with torch.no_grad():
        h1 = model.get_h1(X_stimuli)    # (S, N)

    N = h1.shape[1]
    h1_np  = h1.numpy()
    y_np   = y_stimuli.numpy()

    # Structural cost matrix: cosine distance on neuron response profiles
    A      = h1.T                        # (N, S)
    A_norm = F.normalize(A, dim=1)
    C = (1 - A_norm @ A_norm.T).clamp(min=0).numpy().astype(np.float32)
    np.fill_diagonal(C, 0.0)

    # Tuning vectors: mean response per orientation class
    tuning = np.zeros((N, n_ori), dtype=np.float32)
    for k in range(n_ori):
        mask = (y_np == k)
        if mask.sum() > 0:
            tuning[:, k] = h1_np[mask].mean(axis=0)
    tuning = np.clip(tuning, 0, None)
    nrm    = np.linalg.norm(tuning, axis=1, keepdims=True) + 1e-8
    tuning = tuning / nrm

    # OSI proxy: circular resultant length
    thetas    = np.linspace(0, 2 * np.pi, n_ori, endpoint=False)
    R_norm    = tuning / (tuning.sum(axis=1, keepdims=True) + 1e-8)
    resultant = np.abs((R_norm * np.exp(1j * thetas)).sum(axis=1))
    osi       = resultant.astype(np.float32)
    pref_ori  = tuning.argmax(axis=1).astype(np.int32)

    return C, tuning, osi, pref_ori


def accuracy(model, X, y):
    """Classification accuracy."""
    model.eval()
    with torch.no_grad():
        logits, _ = model(X)
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean().item())


def gw_distance(C1, C2, epsilon=0.05, max_iter=200):
    """sqrt(entropic GW cost) between two cosine distance matrices.

    Normalizes both matrices to max=1 before computing GW.

    Parameters
    ----------
    C1, C2  : (N1, N1) and (N2, N2) float32/64 numpy arrays
    epsilon : entropic regularization
    max_iter: solver iterations

    Returns
    -------
    float — sqrt(GW cost)
    """
    import ot

    N1, N2 = C1.shape[0], C2.shape[0]
    p = np.ones(N1, dtype=np.float64) / N1
    q = np.ones(N2, dtype=np.float64) / N2

    mx = max(float(C1.max()), float(C2.max()))
    if mx > 0:
        C1n = (C1 / mx).astype(np.float64)
        C2n = (C2 / mx).astype(np.float64)
    else:
        C1n, C2n = C1.astype(np.float64), C2.astype(np.float64)

    cost = ot.gromov.entropic_gromov_wasserstein2(
        C1n, C2n, p, q,
        epsilon=epsilon, solver='PPA',
        max_iter=max_iter, tol=1e-5,
    )
    return float(np.sqrt(abs(cost)))


def compute_all_metrics(model, X_train, y_train, X_test, y_test,
                         C_teacher_train, C_teacher_test, n_ori,
                         gw_epsilon=0.05):
    """Compute accuracy, GW distance, and OSI on both train and test splits.

    Parameters
    ----------
    model            : trained MLP
    X_train, y_train : train stimuli tensors
    X_test,  y_test  : test  stimuli tensors
    C_teacher_train  : (N_t, N_t) teacher structural cost on train stimuli
    C_teacher_test   : (N_t, N_t) teacher structural cost on test  stimuli
    n_ori            : int

    Returns
    -------
    dict with keys:
        acc_train, acc_test,
        gw_train,  gw_test,
        osi_mean_train, osi_mean_test,
        osi_median_train, osi_median_test,
        C_train (ndarray), C_test (ndarray)
    """
    # Encoding manifolds on both splits
    C_tr, _, osi_tr, _ = compute_encoding_manifold(model, X_train, y_train, n_ori)
    C_te, _, osi_te, _ = compute_encoding_manifold(model, X_test,  y_test,  n_ori)

    return dict(
        acc_train        = accuracy(model, X_train, y_train),
        acc_test         = accuracy(model, X_test,  y_test),
        gw_train         = gw_distance(C_tr, C_teacher_train, epsilon=gw_epsilon),
        gw_test          = gw_distance(C_te, C_teacher_test,  epsilon=gw_epsilon),
        osi_mean_train   = float(osi_tr.mean()),
        osi_mean_test    = float(osi_te.mean()),
        osi_median_train = float(np.median(osi_tr)),
        osi_median_test  = float(np.median(osi_te)),
        C_train          = C_tr,
        C_test           = C_te,
    )


def pca_embedding(C, n_components=2):
    """Simple PCA-based 2D embedding of an encoding manifold (for visualization).

    Uses kernel PCA with a precomputed centered distance kernel.
    """
    from sklearn.decomposition import KernelPCA
    G = -0.5 * (C ** 2)
    kpca = KernelPCA(n_components=n_components, kernel='precomputed')
    return kpca.fit_transform(G)


def save_metrics(metrics_dict, path):
    """Save scalar metrics to JSON (arrays are excluded)."""
    scalar_dict = {k: v for k, v in metrics_dict.items()
                   if not isinstance(v, np.ndarray)}
    with open(path, 'w') as f:
        json.dump(scalar_dict, f, indent=2)
