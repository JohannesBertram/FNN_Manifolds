"""Natural movie subpopulation analysis utilities.

Importable module — no top-level execution. All paths are passed as arguments.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_natural_movie_data(
    data_dir: str,
    session_ids: list[int],
    area: str = "VISp",
    stimulus: str = "natural_movie_one",
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Load per-session trial firing rate data and concatenate across sessions.

    Args:
        data_dir: directory containing ``{stimulus}/{area}/`` session files.
        session_ids: list of session IDs to load.
        area: brain area string (default ``"VISp"``).
        stimulus: stimulus name (default ``"natural_movie_one"``).

    Returns:
        trial_avgs: ``(N, 900)`` trial-averaged firing rates.
        all_neurons: ``(N, trials_max, 900)`` per-trial firing rates (NaN-padded).
        session_uids_used: list of ``(session_id, unit_id)`` pairs, one per neuron row.
    """
    trial_avgs_list = []
    all_neurons_list = []
    session_uids_used: list[tuple[int, int]] = []

    movie_dir = os.path.join(data_dir, area)

    for session_id in session_ids:
        data_path = os.path.join(movie_dir, f"s{session_id}_{area}_{stimulus}_trialFRs_trial_data.npy")
        info_path = os.path.join(movie_dir, f"s{session_id}_{area}_{stimulus}_trialFRs_trial_info.pkl")

        trialX = np.load(data_path)  # (neurons, total_trials_flat)
        with open(info_path, "rb") as f:
            trial_info = pickle.load(f)

        n_frames = len(trial_info["stims"])
        max_trials = max(trial_info["stim_ntrials"])

        # Build (neurons, frames, trials_max) tensor with NaN padding
        trialT = np.full((trialX.shape[0], n_frames, max_trials), np.nan)
        for ni in range(trialX.shape[0]):
            i = 0
            for si in range(n_frames):
                j = i + trial_info["stim_ntrials"][si]
                trialT[ni, si, : j - i] = trialX[ni, i:j]
                i = j

        trial_avgs = np.nanmean(trialT, axis=2)  # (neurons, frames)
        all_neurons_session = np.transpose(trialT, (0, 2, 1))  # (neurons, trials_max, frames)

        trial_avgs_list.append(trial_avgs)
        all_neurons_list.append(all_neurons_session)

        unit_ids = trial_info["uis"]
        for uid in unit_ids:
            session_uids_used.append((session_id, uid))

        print(f"  {session_id} — {trialX.shape[0]} units")

    trial_avgs = np.concatenate(trial_avgs_list, axis=0)
    all_neurons = np.concatenate(all_neurons_list, axis=0)

    print(f"Loaded {trial_avgs.shape[0]} neurons, {trial_avgs.shape[1]} frames")
    return trial_avgs, all_neurons, session_uids_used


def load_encoding_manifold(enc_mfd_dir: str) -> tuple[np.ndarray, list]:
    """Load precomputed encoding manifold and cell IDs.

    Args:
        enc_mfd_dir: directory containing ``VISp-manifold.npy`` and
            ``cell_ids_to_use_VISp_dg.pkl``.

    Returns:
        embedding: ``(M, 6)`` manifold coordinates.
        cell_ids: list of ``(session_id, unit_id)`` pairs of length M.
    """
    embedding = np.load(os.path.join(enc_mfd_dir, "VISp-manifold.npy"))
    with open(os.path.join(enc_mfd_dir, "cell_ids_to_use_VISp_dg.pkl"), "rb") as f:
        cell_ids = pickle.load(f)
    print(f"Encoding manifold: {embedding.shape[0]} neurons, {embedding.shape[1]} dims")
    return embedding, cell_ids


def build_index_maps(
    session_uids_used: list[tuple[int, int]],
    cell_ids: list[tuple[int, int]],
) -> tuple[dict[int, int], dict[int, int]]:
    """Build bidirectional index maps between natural-movie and manifold neuron spaces.

    Args:
        session_uids_used: ``(session_id, unit_id)`` pairs in natural-movie order.
        cell_ids: ``(session_id, unit_id)`` pairs in manifold order.

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
    print(f"Shared neurons: {len(nm2m)} of {len(session_uids_used)} nat-movie, "
          f"{len(m2nm)} of {len(cell_ids)} manifold")
    return nm2m, m2nm


def natmovie_to_manifold_indices(natmovie_ixs: list[int], nm2m: dict[int, int]) -> list[int]:
    """Map natural-movie neuron indices to manifold indices (dropping unmatched)."""
    return [nm2m[i] for i in natmovie_ixs if i in nm2m]


def manifold_to_natmovie_indices(manifold_ixs: list[int], m2nm: dict[int, int]) -> list[int]:
    """Map manifold neuron indices to natural-movie indices (dropping unmatched)."""
    return [m2nm[j] for j in manifold_ixs if j in m2nm]


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------

def compute_population_trajectory(
    X_neurons_frames: np.ndarray,
    n_components: int = 3,
    pca: Optional[PCA] = None,
    neuron_indices: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, PCA]:
    """Embed frame-wise population activity into PCA space.

    Args:
        X_neurons_frames: ``(N, T)`` activity matrix.
        n_components: number of PCA components (default 3).
        pca: if provided, project onto this pre-fitted PCA; otherwise fit new PCA.
        neuron_indices: when projecting onto a pre-fitted PCA that was trained on a
            larger population, pass the integer indices of the neurons in
            ``X_neurons_frames`` relative to the full population. The projection then
            uses only those columns of ``pca.components_`` and ``pca.mean_``, allowing
            a subpopulation to be embedded in the same axes as the full population.
            Ignored when ``pca`` is None.

    Returns:
        embedding: ``(T, n_components)`` trajectory.
        pca: fitted PCA object.
    """
    X_frames_neurons = X_neurons_frames.T  # (T, N)
    if pca is None:
        pca = PCA(n_components=n_components, random_state=0)
        embedding = pca.fit_transform(X_frames_neurons)
    elif neuron_indices is not None:
        # Manual projection using only the subpop columns of the full-pop PCA.
        # pca.components_: (n_components, N_full); pca.mean_: (N_full,)
        idx = np.asarray(neuron_indices)
        components_sub = pca.components_[:, idx]   # (n_components, n_sub)
        mean_sub = pca.mean_[idx]                  # (n_sub,)
        embedding = (X_frames_neurons - mean_sub) @ components_sub.T
    else:
        embedding = pca.transform(X_frames_neurons)
    return embedding, pca


def compute_trial_bundle(
    all_neurons_subpop: np.ndarray,
    pca: PCA,
    neuron_indices: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project each trial's activity matrix through PCA.

    Args:
        all_neurons_subpop: ``(N, trials_max, 900)`` per-trial data (NaN-padded).
        pca: fitted PCA object to project onto.
        neuron_indices: when ``all_neurons_subpop`` is a subpopulation drawn from
            the larger population on which ``pca`` was fitted, pass the integer
            indices of those neurons so that only the corresponding PCA columns are
            used. Same semantics as in :func:`compute_population_trajectory`.

    Returns:
        bundle: ``(n_valid_trials, 900, n_components)`` projected trial trajectories.
        trial_mask: boolean array of length trials_max indicating valid (non-NaN) trials.
    """
    n_neurons, trials_max, n_frames = all_neurons_subpop.shape
    n_components = pca.n_components_

    # Precompute subpop PCA matrices once
    if neuron_indices is not None:
        idx = np.asarray(neuron_indices)
        components_sub = pca.components_[:, idx]   # (n_components, n_sub)
        mean_sub = pca.mean_[idx]                  # (n_sub,)
    else:
        components_sub = None
        mean_sub = None

    bundle_list = []
    valid_mask = []

    for tr in range(trials_max):
        X = all_neurons_subpop[:, tr, :].T  # (frames, neurons)
        # Skip trials that are entirely NaN
        if np.all(np.isnan(X)):
            valid_mask.append(False)
            continue

        # Fill NaNs: use per-neuron mean across frames; fallback to 0
        m = np.nanmean(X, axis=0)
        m = np.where(np.isfinite(m), m, 0.0)
        nan_idx = np.where(~np.isfinite(X))
        if nan_idx[0].size:
            X = X.copy()
            X[nan_idx] = m[nan_idx[1]]

        if components_sub is not None:
            proj = (X - mean_sub) @ components_sub.T
        else:
            proj = pca.transform(X)
        bundle_list.append(proj)
        valid_mask.append(True)

    bundle = np.stack(bundle_list, axis=0) if bundle_list else np.empty((0, n_frames, n_components))
    return bundle, np.array(valid_mask)


# ---------------------------------------------------------------------------
# Neuron metrics
# ---------------------------------------------------------------------------

def compute_neuron_metrics(
    all_neurons: np.ndarray,
    trial_avgs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute per-neuron summary metrics from natural movie responses.

    Args:
        all_neurons: ``(N, trials_max, 900)`` per-trial data (NaN-padded).
        trial_avgs: ``(N, 900)`` trial-averaged firing rates.

    Returns:
        dict with arrays of length N:
            ``'reliability'``: mean pairwise Pearson r across trials (NaN → 0).
            ``'speed'``: mean absolute frame-to-frame change in trial_avg.
            ``'selectivity'``: std of trial_avg over time.
    """
    N, trials_max, n_frames = all_neurons.shape

    reliability = np.zeros(N)
    for ni in range(N):
        X = all_neurons[ni]  # (trials_max, n_frames)
        # Only use trials that have at least some valid data
        valid = [tr for tr in range(trials_max) if not np.all(np.isnan(X[tr]))]
        if len(valid) < 2:
            reliability[ni] = np.nan
            continue
        # Fill NaNs with row mean for correlation computation
        traces = []
        for tr in valid:
            t = X[tr].copy()
            m = np.nanmean(t)
            t = np.where(np.isnan(t), m if np.isfinite(m) else 0.0, t)
            traces.append(t)
        traces = np.array(traces)  # (n_valid, n_frames)
        # Pairwise Pearson r
        rs = []
        for a in range(len(traces)):
            for b in range(a + 1, len(traces)):
                ta, tb = traces[a], traces[b]
                if np.std(ta) == 0 or np.std(tb) == 0:
                    continue
                r = np.corrcoef(ta, tb)[0, 1]
                if np.isfinite(r):
                    rs.append(r)
        reliability[ni] = np.mean(rs) if rs else np.nan

    speed = np.mean(np.abs(np.diff(trial_avgs, axis=1)), axis=1)
    selectivity = np.std(trial_avgs, axis=1)

    return {
        "reliability": reliability,
        "speed": speed,
        "selectivity": selectivity,
    }


def select_by_metric(
    metrics: dict[str, np.ndarray],
    metric_name: str,
    percentile_lo: float = 0,
    percentile_hi: float = 100,
) -> np.ndarray:
    """Select neuron indices by metric percentile range.

    Args:
        metrics: dict from :func:`compute_neuron_metrics`.
        metric_name: key in metrics dict.
        percentile_lo: lower bound percentile (inclusive).
        percentile_hi: upper bound percentile (inclusive).

    Returns:
        Indices of neurons within the specified percentile range.
    """
    values = metrics[metric_name]
    valid = np.isfinite(values)
    lo = np.nanpercentile(values[valid], percentile_lo)
    hi = np.nanpercentile(values[valid], percentile_hi)
    return np.where(valid & (values >= lo) & (values <= hi))[0]


# ---------------------------------------------------------------------------
# Scene-based clip segmentation
# ---------------------------------------------------------------------------

def segment_movie_into_clips(
    trial_avgs: np.ndarray,
    frame_diff: np.ndarray,
    K: int = 8,
    threshold_k: float = 2.5,
    min_scene_len: int = 30,
) -> tuple[np.ndarray, list[int], int]:
    """Segment the movie into K clips based on scene detection.

    Detects scene cuts by thresholding ``frame_diff`` at
    ``mean + threshold_k * std``, enforces a minimum scene length, and
    recursively splits scenes longer than ``4 * min_scene_len`` at their
    internal pixel-change peak. Selects the K longest scenes and returns
    per-clip mean neural activity.

    Args:
        trial_avgs: ``(N, T)`` trial-averaged firing rates.
        frame_diff: ``(T-1,)`` mean absolute frame-to-frame pixel change.
        K: number of clips to select (default 8).
        threshold_k: scene cut threshold = mean + threshold_k * std (default 2.5).
        min_scene_len: minimum scene length in frames (default 30).

    Returns:
        clip_avgs: ``(N, K)`` per-clip mean activity.
        clip_starts: list of K scene start frame indices.
        clip_len: length of each clip in frames (= shortest selected scene).
    """
    N, T = trial_avgs.shape

    # Scene cut detection
    threshold = np.mean(frame_diff) + threshold_k * np.std(frame_diff)
    cut_frames = (np.where(frame_diff > threshold)[0] + 1).tolist()

    all_boundaries = [0] + cut_frames + [T]
    filtered = [0]
    for b in all_boundaries[1:]:
        if b - filtered[-1] >= min_scene_len:
            filtered.append(b)
    if filtered[-1] != T:
        filtered.append(T)

    scenes = [(filtered[i], filtered[i + 1]) for i in range(len(filtered) - 1)]

    # Recursively split scenes longer than 4 * min_scene_len
    max_scene_len = min_scene_len * 4

    def _split(start: int, end: int) -> list:
        if end - start <= max_scene_len:
            return [(start, end)]
        lo = start + min_scene_len
        hi = end - min_scene_len
        if lo > hi:
            return [(start, end)]
        fd_slice = frame_diff[lo - 1 : hi]
        cut = lo + int(np.argmax(fd_slice))
        return _split(start, cut) + _split(cut, end)

    scenes = [seg for s, e in scenes for seg in _split(s, e)]

    # Select top-K scenes by duration
    durations = sorted([(e - s, s, e) for s, e in scenes], reverse=True)
    top_k = durations[:K]

    clip_len = min(dur for dur, _, _ in top_k)
    clip_starts = [s for _, s, _ in top_k]

    clip_avgs = np.stack(
        [trial_avgs[:, s : s + clip_len].mean(axis=1) for s in clip_starts],
        axis=1,
    )  # (N, K)

    return clip_avgs, clip_starts, clip_len


# ---------------------------------------------------------------------------
# Frame decodability
# ---------------------------------------------------------------------------

def compute_frame_decodability(
    trial_avgs_subpop: np.ndarray,
    test_frac: float = 0.2,
    n_neighbors: int = 5,
    seed: int = 0,
) -> dict:
    """Decode frame index from population activity using K-NN regression.

    Features are the ``(N_subpop,)`` activity vector at each frame.
    Target is the frame index (0 … T-1).

    Args:
        trial_avgs_subpop: ``(N, T)`` trial-averaged activity.
        test_frac: fraction of frames held out for testing.
        n_neighbors: K for KNN regressor.
        seed: random seed for train/test split.

    Returns:
        dict with keys ``r2``, ``mae``, ``predictions``, ``true_frames``.
    """
    X = trial_avgs_subpop.T  # (T, N)
    y = np.arange(X.shape[0], dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed
    )

    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    return {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "predictions": y_pred,
        "true_frames": y_test,
    }


# ---------------------------------------------------------------------------
# Neuron clustering
# ---------------------------------------------------------------------------

def cluster_neurons_by_response(
    trial_avgs: np.ndarray,
    n_pcs: int = 20,
    min_cluster_size: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster neurons by their 900-frame response pattern.

    Runs PCA on ``(N, 900)`` to reduce to ``n_pcs`` dimensions, then HDBSCAN.
    Returns a 2D embedding for visualisation (first 2 PCs of response space).

    Args:
        trial_avgs: ``(N, 900)`` trial-averaged activity.
        n_pcs: number of PCA components before clustering.
        min_cluster_size: HDBSCAN ``min_cluster_size``.

    Returns:
        cluster_labels: ``(N,)`` integer labels (``-1`` = noise).
        embedding_2d: ``(N, 2)`` PCA projection for scatter visualisation.
    """
    import hdbscan

    pca = PCA(n_components=min(n_pcs, trial_avgs.shape[0], trial_avgs.shape[1]), random_state=0)
    X_pca = pca.fit_transform(trial_avgs)  # (N, n_pcs)
    embedding_2d = X_pca[:, :2]

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(X_pca)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = np.sum(cluster_labels == -1)
    print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise neurons")

    return cluster_labels, embedding_2d


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _time_colored_3d(ax, traj, cmap_name="viridis", alpha=1.0, linewidth=1.5):
    """Draw a time-colored 3D trajectory on an existing Axes3D."""
    T = traj.shape[0]
    t = np.arange(T)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(t / (T - 1 if T > 1 else 1))
    for i in range(T - 1):
        ax.plot(
            traj[i : i + 2, 0],
            traj[i : i + 2, 1],
            traj[i : i + 2, 2],
            color=colors[i],
            alpha=alpha,
            linewidth=linewidth,
        )
    return colors


def plot_single_trajectory(
    traj: np.ndarray,
    title: str = "",
    out_path: Optional[str] = None,
    alpha: float = 1.0,
    linewidth: float = 1.5,
) -> plt.Figure:
    """Save a single 3D trajectory, colored by time.

    Args:
        traj: ``(T, 3)`` trajectory array.
        title: plot title.
        out_path: if provided, save figure here.

    Returns:
        Matplotlib Figure.
    """
    T = traj.shape[0]
    t = np.arange(T)
    cmap = plt.get_cmap("viridis")
    colors = cmap(t / (T - 1 if T > 1 else 1))

    fig = plt.figure(figsize=(7, 6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    for i in range(T - 1):
        ax.plot(
            traj[i : i + 2, 0], traj[i : i + 2, 1], traj[i : i + 2, 2],
            color=colors[i], alpha=alpha, linewidth=linewidth,
        )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=T - 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.1, fraction=0.035, label="frame")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_trial_bundle(
    bundle: np.ndarray,
    title: str = "",
    out_path: Optional[str] = None,
    alpha: float = 0.35,
    linewidth: float = 0.9,
) -> plt.Figure:
    """Plot multiple trial trajectories in one 3D plot, colored by time.

    Args:
        bundle: ``(n_trials, T, 3)`` trial bundle.
        title: plot title.
        out_path: if provided, save figure here.

    Returns:
        Matplotlib Figure.
    """
    n_trials, T, _ = bundle.shape
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.arange(T) / (T - 1 if T > 1 else 1))

    fig = plt.figure(figsize=(7, 6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    for tr in range(n_trials):
        X = bundle[tr]
        if not np.all(np.isfinite(X)):
            continue
        for i in range(T - 1):
            ax.plot(
                X[i : i + 2, 0], X[i : i + 2, 1], X[i : i + 2, 2],
                color=colors[i], alpha=alpha, linewidth=linewidth,
            )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=T - 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.1, fraction=0.035, label="frame")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_trajectory_comparison(
    avg_traj_full: np.ndarray,
    avg_traj_sub: np.ndarray,
    title_full: str = "Full population",
    title_sub: str = "Subpopulation",
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side 3D trajectories colored by time.

    Args:
        avg_traj_full: ``(T, 3)`` full-population trajectory.
        avg_traj_sub: ``(T, 3)`` subpopulation trajectory.
        title_full: title for left panel.
        title_sub: title for right panel.
        out_path: if provided, save figure here.

    Returns:
        Matplotlib Figure.
    """
    fig = plt.figure(figsize=(12, 5), dpi=150)
    T = avg_traj_full.shape[0]
    cmap = plt.get_cmap("viridis")

    for col, (traj, title) in enumerate([(avg_traj_full, title_full), (avg_traj_sub, title_sub)]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        _time_colored_3d(ax, traj)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=T - 1))
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, label="frame", fraction=0.02, pad=0.04)

    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_trial_bundle_comparison(
    bundle_full: np.ndarray,
    bundle_sub: np.ndarray,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side trial bundles, time-colored with alpha.

    Args:
        bundle_full: ``(n_trials, T, 3)`` full-population trial bundle.
        bundle_sub: ``(n_trials, T, 3)`` subpopulation trial bundle.
        out_path: if provided, save figure here.

    Returns:
        Matplotlib Figure.
    """
    fig = plt.figure(figsize=(12, 5), dpi=150)
    T = bundle_full.shape[1]
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.arange(T) / (T - 1 if T > 1 else 1))

    for col, (bundle, label) in enumerate([
        (bundle_full, "Full population"),
        (bundle_sub, "Subpopulation"),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        for tr in range(bundle.shape[0]):
            X = bundle[tr]
            if not np.all(np.isfinite(X)):
                continue
            for i in range(T - 1):
                ax.plot(
                    X[i : i + 2, 0],
                    X[i : i + 2, 1],
                    X[i : i + 2, 2],
                    color=colors[i],
                    alpha=0.3,
                    linewidth=0.8,
                )
        ax.set_title(label)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=T - 1))
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, label="frame", fraction=0.02, pad=0.04)
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_manifold_with_highlighted_subpop(
    enc_embedding: np.ndarray,
    all_idxs: list[int],
    highlight_idxs: list[int],
    color_values: Optional[np.ndarray] = None,
    cmap: str = "plasma",
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter of encoding manifold with a highlighted subpopulation.

    Args:
        enc_embedding: ``(M, dims)`` manifold embedding (first 2 dims used).
        all_idxs: indices of neurons shared between nat-movie and manifold.
        highlight_idxs: manifold indices of the subpopulation.
        color_values: optional ``(len(highlight_idxs),)`` values to color highlights.
        cmap: colormap for highlight colors.
        out_path: if provided, save figure here.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    ax.scatter(
        enc_embedding[:, 0], enc_embedding[:, 1],
        s=5, color="lightgrey", alpha=0.5, linewidths=0, label="all neurons", zorder=1,
    )

    if color_values is not None:
        sc = ax.scatter(
            enc_embedding[highlight_idxs, 0], enc_embedding[highlight_idxs, 1],
            s=25, c=color_values, cmap=cmap, alpha=0.9, linewidths=0, zorder=3,
        )
        plt.colorbar(sc, ax=ax)
    else:
        ax.scatter(
            enc_embedding[highlight_idxs, 0], enc_embedding[highlight_idxs, 1],
            s=25, color="crimson", alpha=0.9, linewidths=0, label="subpop", zorder=3,
        )

    ax.set_xlabel("DC1")
    ax.set_ylabel("DC2")
    ax.set_title("Encoding manifold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_metric_on_manifold(
    enc_embedding: np.ndarray,
    nm2m: dict[int, int],
    metric_values: np.ndarray,
    metric_name: str,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Color the encoding manifold scatter by a neuron metric.

    Only neurons shared between datasets are colored; the rest are shown in grey.

    Args:
        enc_embedding: ``(M, dims)`` manifold embedding.
        nm2m: natural-movie-to-manifold index map.
        metric_values: ``(N,)`` metric values in natural-movie order.
        metric_name: label string for colorbar.
        out_path: if provided, save figure here.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    ax.scatter(
        enc_embedding[:, 0], enc_embedding[:, 1],
        s=5, color="lightgrey", alpha=0.4, linewidths=0, zorder=1,
    )

    nm_idxs = sorted(nm2m.keys())
    m_idxs = [nm2m[i] for i in nm_idxs]
    vals = metric_values[nm_idxs]
    valid = np.isfinite(vals)

    sc = ax.scatter(
        enc_embedding[np.array(m_idxs)[valid], 0],
        enc_embedding[np.array(m_idxs)[valid], 1],
        s=15, c=vals[valid], cmap="plasma", alpha=0.85, linewidths=0, zorder=2,
    )
    plt.colorbar(sc, ax=ax, label=metric_name)

    ax.set_xlabel("DC1")
    ax.set_ylabel("DC2")
    ax.set_title(f"Encoding manifold colored by {metric_name}")
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_neuron_traces(
    trial_avgs_subpop: np.ndarray,
    neuron_idxs: Optional[list[int]] = None,
    movie_frames: Optional[np.ndarray] = None,
    scene_starts: Optional[list[int]] = None,
    out_path: Optional[str] = None,
    max_individual: int = 10,
) -> plt.Figure:
    """Plot firing rate traces over time for a subpopulation.

    For <= max_individual neurons: individual traces per neuron.
    For larger subpops: population-averaged trace.

    When ``scene_starts`` is provided, vertical dashed lines mark each scene
    cut. When ``movie_frames`` is also provided, a small thumbnail of the
    frame at each scene cut is shown above the top trace panel.

    Args:
        trial_avgs_subpop: ``(N, T)`` trial-averaged activity.
        neuron_idxs: optional list of original neuron indices (for labelling).
        movie_frames: optional ``(T, H, W)`` frames; thumbnails shown at scene cuts.
        scene_starts: optional list of frame indices at which scenes start.
        out_path: if provided, save figure here.
        max_individual: threshold above which to show mean ± std instead.

    Returns:
        Matplotlib Figure.
    """
    N, T = trial_avgs_subpop.shape
    t = np.arange(T)

    if N <= max_individual:
        fig, axes = plt.subplots(N, 1, figsize=(10, 1.8 * N), dpi=120, squeeze=False)
        for row, ni in enumerate(range(N)):
            ax = axes[row, 0]
            ax.plot(t, trial_avgs_subpop[ni], lw=1.2, color="steelblue")
            label = f"neuron {neuron_idxs[ni]}" if neuron_idxs is not None else f"neuron {ni}"
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            if row < N - 1:
                ax.set_xticks([])
            if scene_starts is not None:
                for s in scene_starts:
                    ax.axvline(s, color="tomato", lw=0.8, linestyle="--", alpha=0.7)
        axes[-1, 0].set_xlabel("frame")
        top_ax = axes[0, 0]
    else:
        mean_trace = np.nanmean(trial_avgs_subpop, axis=0)
        std_trace = np.nanstd(trial_avgs_subpop, axis=0)
        fig, ax = plt.subplots(figsize=(10, 3), dpi=120)
        ax.plot(t, mean_trace, lw=1.5, color="steelblue", label="mean")
        ax.fill_between(t, mean_trace - std_trace, mean_trace + std_trace,
                        alpha=0.25, color="steelblue")
        ax.set_xlabel("frame")
        ax.set_ylabel("firing rate")
        ax.set_title(f"Population-averaged trace (N={N})")
        ax.legend(fontsize=8)
        if scene_starts is not None:
            for s in scene_starts:
                ax.axvline(s, color="tomato", lw=0.8, linestyle="--", alpha=0.7)
        top_ax = ax

    # Frame thumbnails above the top panel at each scene start
    if scene_starts is not None and movie_frames is not None:
        img_w = 0.06   # axes-fraction width per thumbnail
        img_h = 0.25   # axes-fraction height per thumbnail
        y0 = 1.04      # bottom of thumbnail (just above the axis top)
        for s in scene_starts:
            x_center = s / T
            x0 = max(0.0, min(1.0 - img_w, x_center - img_w / 2))
            inset = top_ax.inset_axes([x0, y0, img_w, img_h])
            inset.imshow(movie_frames[s], cmap="gray", aspect="auto")
            inset.axis("off")

    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_frame_decodability_comparison(
    results: dict[str, dict],
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing frame decodability R² across conditions.

    Args:
        results: dict mapping condition name → output of :func:`compute_frame_decodability`.
        out_path: if provided, save figure here.

    Returns:
        Matplotlib Figure.
    """
    labels = list(results.keys())
    r2_vals = [results[k]["r2"] for k in labels]
    mae_vals = [results[k]["mae"] for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=120)

    axes[0].bar(labels, r2_vals, color="steelblue", alpha=0.8)
    axes[0].set_ylabel("R²")
    axes[0].set_title("Frame decodability (R²)")
    axes[0].set_ylim(0, 1)

    axes[1].bar(labels, mae_vals, color="salmon", alpha=0.8)
    axes[1].set_ylabel("MAE (frames)")
    axes[1].set_title("Frame decodability (MAE)")

    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_cluster_embedding(
    embedding_2d: np.ndarray,
    cluster_labels: np.ndarray,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of neuron clusters in 2D response space.

    Args:
        embedding_2d: ``(N, 2)`` 2D embedding (first 2 PCs of response space).
        cluster_labels: ``(N,)`` cluster labels (``-1`` = noise).
        out_path: if provided, save figure here.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    unique_labels = sorted(set(cluster_labels))
    cmap = plt.get_cmap("tab10")

    for c in unique_labels:
        mask = cluster_labels == c
        color = "lightgrey" if c == -1 else cmap(c % 10)
        label = "noise" if c == -1 else f"cluster {c} (n={mask.sum()})"
        ax.scatter(
            embedding_2d[mask, 0], embedding_2d[mask, 1],
            s=8, color=color, alpha=0.7, linewidths=0, label=label,
        )

    ax.set_xlabel("PC1 (response space)")
    ax.set_ylabel("PC2 (response space)")
    ax.set_title("Neuron clusters by NM response pattern")
    ax.legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig
