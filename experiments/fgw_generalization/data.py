"""Grating generation and train/test split utilities."""

import numpy as np


def make_gratings(n_ori=8, n_freq=4, n_reps=8, grid=8, noise=0.15,
                  freqs=None, seed=0):
    """Generate synthetic sinusoidal grating stimuli.

    Generation order: rep (outer) → orientation → spatial frequency (inner).
    This makes noise-based splitting trivial: rep r occupies rows
    [r*n_ori*n_freq : (r+1)*n_ori*n_freq].

    Parameters
    ----------
    n_ori  : int — number of orientation classes
    n_freq : int — number of spatial frequencies
    n_reps : int — noise repetitions per (ori, SF) pair
    grid   : int — image side length (pixels); total input dim = grid²
    noise  : float — Gaussian noise std
    freqs  : list of float or None — spatial frequencies (default: linspace 1..4)
    seed   : int — global numpy seed for reproducibility

    Returns
    -------
    X : ndarray (n_ori*n_freq*n_reps, grid²) float32, z-scored
    y : ndarray (n_ori*n_freq*n_reps,) int64 — orientation class labels
    rep_ids : ndarray (n_ori*n_freq*n_reps,) int — repetition index (0..n_reps-1)
    sf_ids  : ndarray (n_ori*n_freq*n_reps,) int — SF index (0..n_freq-1)
    """
    rng = np.random.default_rng(seed)
    if freqs is None:
        freqs = np.linspace(1.0, 4.0, n_freq)
    freqs = np.asarray(freqs)

    thetas_rad = np.deg2rad(np.linspace(0, 180, n_ori, endpoint=False))
    x = np.linspace(0, 1, grid)
    xg, yg = np.meshgrid(x, x)

    samples, labels, rep_ids, sf_ids = [], [], [], []
    for rep in range(n_reps):
        for ori_idx, theta in enumerate(thetas_rad):
            for sf_idx, freq in enumerate(freqs):
                s = np.cos(2 * np.pi * freq * (xg * np.cos(theta) + yg * np.sin(theta)))
                s = s + noise * rng.standard_normal(s.shape)
                samples.append(s.flatten().astype(np.float32))
                labels.append(ori_idx)
                rep_ids.append(rep)
                sf_ids.append(sf_idx)

    X = np.stack(samples)            # (N, 64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y = np.array(labels, dtype=np.int64)
    rep_ids = np.array(rep_ids, dtype=np.int64)
    sf_ids  = np.array(sf_ids,  dtype=np.int64)
    return X, y, rep_ids, sf_ids


def noise_split(X, y, rep_ids, sf_ids, n_reps_train):
    """Split by repetition index.

    Train: reps 0..n_reps_train-1
    Test:  reps n_reps_train..end

    Returns X_train, y_train, X_test, y_test (same dtype as inputs).
    """
    tr = rep_ids < n_reps_train
    te = ~tr
    return X[tr], y[tr], X[te], y[te]


def sf_split(X, y, rep_ids, sf_ids, n_sf_train):
    """Split by spatial frequency index.

    Train: SF indices 0..n_sf_train-1
    Test:  SF indices n_sf_train..end

    Returns X_train, y_train, X_test, y_test.
    """
    tr = sf_ids < n_sf_train
    te = ~tr
    return X[tr], y[tr], X[te], y[te]


def sf_interp_split(X, y, rep_ids, sf_ids):
    """Interpolation SF split: train on even-indexed SFs, test on odd-indexed SFs.

    Assumes the freq list was constructed as interleaved:
        even indices (0,2,4,...) → training SFs
        odd  indices (1,3,5,...) → test SFs (each sitting between two train SFs)

    Example with 7 SFs [1.0,1.5,2.0,2.5,3.0,3.5,4.0]:
        train: idx 0,2,4,6 → {1.0, 2.0, 3.0, 4.0}
        test:  idx 1,3,5   → {1.5, 2.5, 3.5}

    Returns X_train, y_train, X_test, y_test.
    """
    tr = (sf_ids % 2) == 0
    te = ~tr
    return X[tr], y[tr], X[te], y[te]
