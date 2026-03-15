"""Natural movie analysis utilities.

This script:
- loads trial-wise firing rate traces per neuron for the same natural movie (900 frames)
- constructs
    - all_neurons: (neurons, trials, frames)
    - trial_avgs_all_neurons: (neurons, frames)
- runs a PCA over neurons to embed frame-wise population activity into 3D
- saves trajectory plots as PNG
"""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import ConnectionPatch
from matplotlib.animation import PillowWriter



movie_frames = np.load("movie1.npy") # shape (t, height, width)
print("movie_frames:", movie_frames.shape)



my_session_ids = [719161530, 737581020, 746083955, 798911424, 743475441, 744228101, 756029989, 758798717, 
                  715093703, 762602078, 797828357, 754312389, 760345702, 739448407, 763673393, 799864342, 
                  721123822, 761418226, 732592105, 742951821, 762120172, 750749662, 791319847, 760693773, 
                  755434585, 759883607, 750332458, 757216464, 754829445, 757970808, 773418906, 751348571]

stimulus = 'natural_movie_one'
AREA = 'VISp'
moviedir = "natural_movie_one/VISp"

#### compute mean n. of spikes per frame for all units from all exps

trial_avgs_all_neurons = []
all_neurons_list = []

for session_id in my_session_ids:
    
    
    trialX = np.load(f'{stimulus}/{AREA}/s{session_id}_{AREA}_{stimulus}_trialFRs_trial_data.npy')
#     print(trialX.shape, "neurons x (trials x frames)")

    with open(f'{moviedir}/s{session_id}_{AREA}_{stimulus}_trialFRs_trial_info.pkl', 'rb') as f:
        trial_info = pickle.load(f)
#     print(trial_info.keys())
    
    # simply reshaping as below doesnt work b/c different frames might have a different # of trials    
    # trialT = trialX.reshape(trialX.shape[0],len(trial_info['stims']),-1) BAD
    
    # this works: start with a 3-tensor full of NaNs and fill in each frame individually
    # trialT: (neurons, frames, trials_max)
    trialT = np.empty((trialX.shape[0], len(trial_info['stims']), max(trial_info['stim_ntrials'])))
    trialT.fill(np.nan)
#     print(trialT.shape, "neurons x frames x trials")
    
    for ni in range(trialX.shape[0]):
        i = 0
        for si in range(len(trial_info['stims'])):
            j = i + trial_info['stim_ntrials'][si]
            assert j - i <= max(trial_info['stim_ntrials'])
            trialT[ni,si,:j-i] = trialX[ni,i:j]
            i = j
                      
    # then, to compute the avg # spikes per trial use `np.nanmean`:
    trial_avgs = np.nanmean(trialT, axis=2)  # (neurons, frames)
    
    
    trial_avgs_all_neurons.append(trial_avgs)
    # keep non-averaged data around (neurons, trials, frames)
    all_neurons_list.append(np.transpose(trialT, (0, 2, 1)))
    
    print(f'{session_id} - {trial_avgs.shape[0]} units')

trial_avgs_all_neurons = np.concatenate(trial_avgs_all_neurons) # SHAPE (tot. neurons, frames)
all_neurons = np.concatenate(all_neurons_list)  # SHAPE (tot. neurons, trials_max, frames)

print("trial_avgs_all_neurons:", trial_avgs_all_neurons.shape)
print("all_neurons:", all_neurons.shape)


# ----------------------------
# Encoding manifold index mapping
# ----------------------------

with open("enc_mfds/cell_ids_to_use_VISp_dg.pkl", "rb") as f:
    _cell_ids_to_use = pickle.load(f)

_manifold_lookup = {uid_pair: j for j, uid_pair in enumerate(_cell_ids_to_use)}
_natmovie_to_manifold: dict[int, int] = {}
_manifold_to_natmovie: dict[int, int] = {}
for _i, _uid_pair in enumerate(session_uids_used):
    if _uid_pair in _manifold_lookup:
        _j = _manifold_lookup[_uid_pair]
        _natmovie_to_manifold[_i] = _j
        _manifold_to_natmovie[_j] = _i

print(f"Neurons shared between natural movie and encoding manifold: {len(_natmovie_to_manifold)}")


def natmovie_to_manifold_indices(natmovie_ixs: list[int]) -> list[int]:
    """Map natural movie neuron indices to encoding manifold indices.

    Args:
        natmovie_ixs: indices into natural movie data (i.e. into session_uids_used /
                      trial_avgs_all_neurons / all_neurons first axis)

    Returns:
        Manifold indices for neurons present in both datasets, in the same order.
        Neurons not shared with the manifold are silently dropped.
    """
    return [_natmovie_to_manifold[i] for i in natmovie_ixs if i in _natmovie_to_manifold]


def manifold_to_natmovie_indices(manifold_ixs: list[int]) -> list[int]:
    """Map encoding manifold neuron indices to natural movie indices.

    Args:
        manifold_ixs: indices into the encoding manifold (i.e. into cell_ids_to_use_VISp_dg)

    Returns:
        Natural movie indices for neurons present in both datasets, in the same order.
        Neurons not shared with the natural movie data are silently dropped.
    """
    return [_manifold_to_natmovie[j] for j in manifold_ixs if j in _manifold_to_natmovie]


def _pca_3d_over_neurons(X_neurons_frames: np.ndarray) -> np.ndarray:
    """PCA over neurons.

    Args:
        X_neurons_frames: array shape (neurons, frames)

    Returns:
        X_frames_3d: array shape (frames, 3)
    """
    if X_neurons_frames.ndim != 2:
        raise ValueError(f"Expected 2D (neurons, frames), got {X_neurons_frames.shape}")

    # PCA expects samples x features. We want: samples=frames, features=neurons.
    X_frames_neurons = X_neurons_frames.T
    if X_frames_neurons.shape[0] < 3:
        raise ValueError("Need at least 3 frames for a 3D PCA embedding")

    pca = PCA(n_components=3, random_state=0)
    return pca.fit_transform(X_frames_neurons)


def _save_3d_time_colored_trace(
    X_frames_3d: np.ndarray,
    out_path: str,
    title: str,
    *,
    alpha: float = 1.0,
    linewidth: float = 1.5,
):
    """Save a single 3D trace, colored by time."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    t = np.arange(X_frames_3d.shape[0])
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(7, 6), dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    # draw small time-colored line segments
    colors = cmap(t / (t.max() if t.max() > 0 else 1))
    for i in range(X_frames_3d.shape[0] - 1):
        ax.plot(
            X_frames_3d[i : i + 2, 0],
            X_frames_3d[i : i + 2, 1],
            X_frames_3d[i : i + 2, 2],
            color=colors[i],
            alpha=alpha,
            linewidth=linewidth,
        )

    # add a colorbar for time
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=t.min(), vmax=t.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, fraction=0.035)
    cbar.set_label("frame", rotation=270, labelpad=12)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_3d_time_colored_traces(
    X_trials_frames_3d: np.ndarray,
    out_path: str,
    title: str,
    *,
    alpha: float = 0.35,
    linewidth: float = 0.9,
):
    """Save multiple 3D traces in one plot, each colored by time."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if X_trials_frames_3d.ndim != 3 or X_trials_frames_3d.shape[-1] != 3:
        raise ValueError(f"Expected (trials, frames, 3), got {X_trials_frames_3d.shape}")

    frames = X_trials_frames_3d.shape[1]
    t = np.arange(frames)
    cmap = plt.get_cmap("viridis")
    colors = cmap(t / (t.max() if t.max() > 0 else 1))

    fig = plt.figure(figsize=(7, 6), dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    for tr in range(X_trials_frames_3d.shape[0]):
        X = X_trials_frames_3d[tr]
        if np.any(~np.isfinite(X)):
            # should not happen after nan-handling, but skip if it does
            continue
        for i in range(frames - 1):
            ax.plot(
                X[i : i + 2, 0],
                X[i : i + 2, 1],
                X[i : i + 2, 2],
                color=colors[i],
                alpha=alpha,
                linewidth=linewidth,
            )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=t.min(), vmax=t.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, fraction=0.035)
    cbar.set_label("frame", rotation=270, labelpad=12)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _blend_with_white(rgb: tuple[float, float, float], strength: float) -> tuple[float, float, float]:
    """Blend a base RGB color with white.

    strength=0 -> white
    strength=1 -> base color
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    r, g, b = rgb
    return (1.0 - strength) + strength * r, (1.0 - strength) + strength * g, (1.0 - strength) + strength * b


def _save_3d_trial_and_time_colored_traces(
    X_trials_frames_3d: np.ndarray,
    out_path: str,
    title: str,
    *,
    alpha: float = 0.55,
    linewidth: float = 1.0,
):
    """Save multiple 3D traces in one plot.

    Each trace gets a distinct base color; time is encoded by blending from white (early)
    to the base color (late).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if X_trials_frames_3d.ndim != 3 or X_trials_frames_3d.shape[-1] != 3:
        raise ValueError(f"Expected (trials, frames, 3), got {X_trials_frames_3d.shape}")

    n_trials, frames, _ = X_trials_frames_3d.shape
    t = np.arange(frames)
    t_norm = (t - t.min()) / (t.max() - t.min() if t.max() > t.min() else 1)

    base_cmap = plt.get_cmap("tab20")

    fig = plt.figure(figsize=(7, 6), dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    for tr in range(n_trials):
        X = X_trials_frames_3d[tr]
        if np.any(~np.isfinite(X)):
            continue
        base_rgb = base_cmap(tr % base_cmap.N)[:3]
        for i in range(frames - 1):
            c = _blend_with_white(base_rgb, t_norm[i])
            ax.plot(
                X[i : i + 2, 0],
                X[i : i + 2, 1],
                X[i : i + 2, 2],
                color=c,
                alpha=alpha,
                linewidth=linewidth,
            )

    # Add a legend-like hint for how time is encoded
    # (a single colorbar doesn't make sense because each trial has its own base hue)
    ax.text2D(0.02, 0.02, "time: white → strong color", transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_avg_trace_with_movie_frames(
    Xavg_frames_3d: np.ndarray,
    movie_frames: np.ndarray,
    out_path: str,
    title: str,
    *,
    n_frames_to_show: int = 10,
):
    """3D averaged trace plus selected movie frames, each connected by an arrow.

    movie_frames is expected shape (frames, H, W) or (frames, H, W, C).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    frames = Xavg_frames_3d.shape[0]
    if movie_frames.shape[0] != frames:
        raise ValueError(
            f"movie_frames has {movie_frames.shape[0]} frames but neural trace has {frames}. "
            "Make sure you're loading the same 900-frame movie array."
        )

    idxs = np.linspace(0, frames - 1, n_frames_to_show, dtype=int)
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.arange(frames) / (frames - 1 if frames > 1 else 1))

    fig = plt.figure(figsize=(8.5, 6.5), dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    # averaged trace (time-colored)
    for i in range(frames - 1):
        ax.plot(
            Xavg_frames_3d[i : i + 2, 0],
            Xavg_frames_3d[i : i + 2, 1],
            Xavg_frames_3d[i : i + 2, 2],
            color=colors[i],
            linewidth=1.4,
            alpha=0.95,
        )

    # We'll place frame images on a "panel" to the right of the 3D trajectory.
    # This is done in 2D axes coordinates, so it stays readable.
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    # Add inset images + arrows pointing to corresponding 3D points
    # Note: annotate3D isn't standard; we use 2D annotation anchored to the projected positions.
    # The arrow start is the inset center in figure coords; end is the projected 3D point.
    fig.canvas.draw()
    proj = ax.get_proj()

    # Inset layout (figure coordinates)
    left = 0.75
    top = 0.92
    h = 0.12
    w = 0.20
    pad = 0.01

    for k, fi in enumerate(idxs):
        # inset axis for the frame
        y0 = top - (k + 1) * h - k * pad
        inset = fig.add_axes([left, y0, w, h])
        frame = movie_frames[fi]
        if frame.ndim == 2:
            inset.imshow(frame, cmap="gray", interpolation="nearest")
        else:
            inset.imshow(frame, interpolation="nearest")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title(f"f={fi}", fontsize=8)

        # Project the 3D point to 2D figure coordinates (for arrow target).
        # Matplotlib's 3D projection returns 4 values (homogeneous coords), so don't
        # unpack into 3 variables.
        x3, y3, z3 = Xavg_frames_3d[fi]

        # Robust: use the 3D axis' projection to get screen x/y
        x2, y2, _z2 = proj3d.proj_transform(x3, y3, z3, ax.get_proj())
        disp = ax.transData.transform((x2, y2))
        fig_coords = fig.transFigure.inverted().transform(disp)

        # Arrow from inset center to the projected 3D point.
        # Figure doesn't have annotate() in many matplotlib versions.
        # Use a ConnectionPatch between the inset axes (axes-fraction) and the 3D axes (data coords).
        con = ConnectionPatch(
            xyA=(0.5, 0.5),
            coordsA=inset.transAxes,
            xyB=(x2, y2),
            coordsB=ax.transData,
            arrowstyle="->",
            lw=0.8,
            color="black",
            alpha=0.7,
            shrinkA=2,
            shrinkB=2,
        )
        fig.add_artist(con)

        # mark the actual 3D point
        ax.scatter([x3], [y3], [z3], s=12, color="black", alpha=0.8)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_clusters_with_movie_frames(
    Xavg_frames_3d: np.ndarray,
    movie_frames: np.ndarray,
    out_path: str,
    title: str,
    *,
    n_clusters: int = 6,
):
    """Cluster frames in 3D activity space and show representative movie frames per cluster."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    frames = Xavg_frames_3d.shape[0]
    if movie_frames.shape[0] != frames:
        raise ValueError(
            f"movie_frames has {movie_frames.shape[0]} frames but neural trace has {frames}. "
            "Make sure you're loading the same 900-frame movie array."
        )

    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
    labels = km.fit_predict(Xavg_frames_3d)

    cmap = plt.get_cmap("tab10")

    # Create a figure with 3D scatter + a strip of representative frames
    fig = plt.figure(figsize=(10.5, 6.5), dpi=200)
    ax = fig.add_subplot(121, projection="3d")

    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(
            Xavg_frames_3d[mask, 0],
            Xavg_frames_3d[mask, 1],
            Xavg_frames_3d[mask, 2],
            s=8,
            alpha=0.65,
            color=cmap(c % cmap.N),
            label=f"C{c} (n={mask.sum()})",
        )
        ax.scatter(
            [km.cluster_centers_[c, 0]],
            [km.cluster_centers_[c, 1]],
            [km.cluster_centers_[c, 2]],
            s=60,
            marker="x",
            color="black",
            linewidths=1.5,
        )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(loc="upper left", fontsize=7)

    # Representative frames: choose the frame closest to each centroid
    ax2 = fig.add_subplot(122)
    ax2.axis("off")
    for c in range(n_clusters):
        d = np.linalg.norm(Xavg_frames_3d - km.cluster_centers_[c], axis=1)
        fi = int(np.argmin(d))

        # place each representative frame as an inset
        # Coordinates in axes fraction
        y0 = 1.0 - (c + 1) / n_clusters
        inset = ax2.inset_axes([0.05, y0 + 0.02, 0.9, 1.0 / n_clusters - 0.04])
        frame = movie_frames[fi]
        if frame.ndim == 2:
            inset.imshow(frame, cmap="gray", interpolation="nearest")
        else:
            inset.imshow(frame, interpolation="nearest")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title(f"Cluster {c}: rep frame {fi}", fontsize=8, color=cmap(c % cmap.N))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_movie_with_trajectory(
    movie_frames: np.ndarray,
    Xavg_frames_3d: np.ndarray,
    out_path: str,
    *,
    fps: int = 30,
    dpi: int = 150,
):
    """Create a GIF: left=movie frame, right=3D average trajectory with current point highlighted."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    frames = Xavg_frames_3d.shape[0]
    if movie_frames.shape[0] != frames:
        raise ValueError(
            f"movie_frames has {movie_frames.shape[0]} frames but neural trace has {frames}. "
            "Make sure you're loading the same movie array."
        )

    # Precompute time colors for the trajectory
    cmap = plt.get_cmap("viridis")
    t = np.arange(frames)
    colors = cmap(t / (frames - 1 if frames > 1 else 1))

    fig = plt.figure(figsize=(11, 5.5), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_traj = fig.add_subplot(gs[0, 1], projection="3d")

    # Left: initial frame
    frame0 = movie_frames[0]
    if frame0.ndim == 2:
        im = ax_img.imshow(frame0, cmap="gray", interpolation="nearest")
    else:
        im = ax_img.imshow(frame0, interpolation="nearest")
    ax_img.set_title("movie frame")
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # Right: static trajectory
    for i in range(frames - 1):
        ax_traj.plot(
            Xavg_frames_3d[i : i + 2, 0],
            Xavg_frames_3d[i : i + 2, 1],
            Xavg_frames_3d[i : i + 2, 2],
            color=colors[i],
            linewidth=1.2,
            alpha=0.9,
        )
    ax_traj.set_title("avg trajectory (PCA over neurons)")
    ax_traj.set_xlabel("PC1")
    ax_traj.set_ylabel("PC2")
    ax_traj.set_zlabel("PC3")

    # Highlight: current point
    hl = ax_traj.scatter(
        [Xavg_frames_3d[0, 0]],
        [Xavg_frames_3d[0, 1]],
        [Xavg_frames_3d[0, 2]],
        s=60,
        color="red",
        depthshade=False,
        zorder=10,
    )

    # Add a small time label
    time_txt = fig.text(0.5, 0.02, "frame 0", ha="center", va="bottom")

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, out_path, dpi=dpi):
        for fi in range(frames):
            # update image
            im.set_data(movie_frames[fi])
            ax_img.set_title(f"movie frame {fi}")

            # update highlight point
            hl._offsets3d = (
                np.array([Xavg_frames_3d[fi, 0]]),
                np.array([Xavg_frames_3d[fi, 1]]),
                np.array([Xavg_frames_3d[fi, 2]]),
            )
            time_txt.set_text(f"frame {fi}")

            writer.grab_frame()

    plt.close(fig)


# ----------------------------
# Scene detection helpers
# ----------------------------


def _compute_frame_diff(movie_frames: np.ndarray) -> np.ndarray:
    """Compute mean absolute frame-to-frame pixel change.

    Returns shape (T-1,).
    """
    return np.mean(
        np.abs(movie_frames[1:].astype(np.float32) - movie_frames[:-1].astype(np.float32)),
        axis=(1, 2),
    )


def _detect_scene_cuts(
    frame_diff: np.ndarray,
    n_frames: int,
    *,
    threshold_k: float = 2.5,
    min_scene_len: int = 10,
    max_scene_len: int = 120,
) -> list:
    """Detect scene cuts based on large frame-to-frame visual changes.

    threshold = mean + threshold_k * std(frame_diff)
    Short segments (< min_scene_len) are merged into the following segment.
    Long segments (> max_scene_len) are recursively split at their largest
    internal frame_diff until every segment is <= max_scene_len.

    Returns list of (start, end) tuples where end is exclusive.
    """
    threshold = np.mean(frame_diff) + threshold_k * np.std(frame_diff)
    cut_frames = (np.where(frame_diff > threshold)[0] + 1).tolist()

    # Build scene boundaries, filtering cuts that would create short segments
    all_boundaries = [0] + cut_frames + [n_frames]
    filtered = [0]
    for b in all_boundaries[1:]:
        if b - filtered[-1] >= min_scene_len:
            filtered.append(b)
    if filtered[-1] != n_frames:
        filtered.append(n_frames)

    scenes = [(filtered[i], filtered[i + 1]) for i in range(len(filtered) - 1)]

    # Recursively split any scene longer than max_scene_len at its internal peak,
    # always keeping both sub-segments >= min_scene_len so no tiny scenes are created.
    def _split(start: int, end: int) -> list:
        if end - start <= max_scene_len:
            return [(start, end)]
        # Constrain cut to [lo, hi] so both halves have >= min_scene_len frames.
        lo = start + min_scene_len      # smallest valid cut
        hi = end - min_scene_len        # largest valid cut (inclusive)
        if lo > hi:
            return [(start, end)]       # too short to split while respecting min_scene_len
        # frame_diff[c-1] is the visual jump arriving at frame c (cut point c
        # means first scene = [start, c), second = [c, end)).
        # For cuts c in [lo, hi] we need frame_diff[lo-1 : hi].
        fd_slice = frame_diff[lo - 1 : hi]   # length == hi - lo + 1
        cut = lo + int(np.argmax(fd_slice))
        return _split(start, cut) + _split(cut, end)

    scenes = [seg for s, e in scenes for seg in _split(s, e)]

    n_scenes = len(scenes)
    if n_scenes < 2:
        print(
            f"WARNING: Only {n_scenes} scene(s) detected. "
            f"Consider lowering threshold_k (currently {threshold_k})."
        )
    elif n_scenes > 30:
        print(
            f"WARNING: {n_scenes} scenes detected. "
            f"Consider raising threshold_k (currently {threshold_k})."
        )
    return scenes


# ----------------------------
# Analysis A: Scene-relative trajectory recentering
# ----------------------------


def _save_scene_relative_trajectories(
    X_trials_3d: np.ndarray,
    scenes: list,
    out_path: str,
    title: str,
    *,
    max_scenes_shown: int = 9,
    alpha: float = 0.45,
    linewidth: float = 0.9,
) -> None:
    """Per-scene subplot grid: trial trajectories recentered to scene-start.

    X_trials_3d: (trials, 900, 3)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Select up to max_scenes_shown longest scenes, re-sort by start frame
    scene_lens = sorted(enumerate(scenes), key=lambda x: x[1][1] - x[1][0], reverse=True)
    shown_idxs = sorted([i for i, _ in scene_lens[:max_scenes_shown]])
    shown_scenes = [scenes[i] for i in shown_idxs]
    n_shown = len(shown_scenes)

    n_cols = min(3, n_shown)
    n_rows = int(np.ceil(n_shown / n_cols))

    fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows), dpi=150)
    fig.suptitle(title, y=1.01, fontsize=10)

    cmap = plt.get_cmap("viridis")
    n_trials = X_trials_3d.shape[0]

    for plot_idx, (start, end) in enumerate(shown_scenes):
        ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1, projection="3d")
        seg_len = end - start
        t_norm = np.linspace(0, 1, seg_len)
        colors = cmap(t_norm)

        for tr in range(n_trials):
            seg = X_trials_3d[tr, start:end, :]  # (seg_len, 3)
            if not np.all(np.isfinite(seg[0])):   # skip ghost trials
                continue
            seg_rel = seg - seg[0:1, :]            # recenter to origin

            for i in range(seg_len - 1):
                ax.plot(
                    seg_rel[i : i + 2, 0],
                    seg_rel[i : i + 2, 1],
                    seg_rel[i : i + 2, 2],
                    color=colors[i],
                    alpha=alpha,
                    linewidth=linewidth,
                )

        ax.scatter([0], [0], [0], color="red", s=20, zorder=10)
        ax.set_title(f"f{start}\u2013{end} ({seg_len} frames)", fontsize=8)
        ax.set_xlabel("PC1", fontsize=6)
        ax.set_ylabel("PC2", fontsize=6)
        ax.set_zlabel("PC3", fontsize=6)
        ax.tick_params(labelsize=5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _smooth_trajectory(seg: np.ndarray, window: int) -> np.ndarray:
    """Apply a box-car moving average along the time axis of a (T, 3) trajectory."""
    if window <= 1:
        return seg
    kernel = np.ones(window) / window
    out = np.empty_like(seg)
    for d in range(seg.shape[1]):
        out[:, d] = np.convolve(seg[:, d], kernel, mode="same")
    return out


def _save_scene_relative_trajectories_smoothed(
    X_trials_3d: np.ndarray,
    scenes: list,
    out_path: str,
    title: str,
    *,
    max_scenes_shown: int = 9,
    smooth_window: int = 7,
    alpha: float = 0.55,
    linewidth: float = 1.1,
) -> None:
    """Like _save_scene_relative_trajectories but each trial segment is smoothed
    with a box-car filter before plotting, making common trajectory shapes visible.

    X_trials_3d: (trials, 900, 3)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    scene_lens = sorted(enumerate(scenes), key=lambda x: x[1][1] - x[1][0], reverse=True)
    shown_idxs = sorted([i for i, _ in scene_lens[:max_scenes_shown]])
    shown_scenes = [scenes[i] for i in shown_idxs]
    n_shown = len(shown_scenes)

    n_cols = min(3, n_shown)
    n_rows = int(np.ceil(n_shown / n_cols))

    fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows), dpi=150)
    fig.suptitle(title, y=1.01, fontsize=10)

    cmap = plt.get_cmap("viridis")
    n_trials = X_trials_3d.shape[0]

    for plot_idx, (start, end) in enumerate(shown_scenes):
        ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1, projection="3d")
        seg_len = end - start
        t_norm = np.linspace(0, 1, seg_len)
        colors = cmap(t_norm)

        for tr in range(n_trials):
            seg = X_trials_3d[tr, start:end, :]  # (seg_len, 3)
            if not np.all(np.isfinite(seg[0])):
                continue
            seg_rel = seg - seg[0:1, :]           # recenter
            seg_smooth = _smooth_trajectory(seg_rel, smooth_window)

            for i in range(seg_len - 1):
                ax.plot(
                    seg_smooth[i : i + 2, 0],
                    seg_smooth[i : i + 2, 1],
                    seg_smooth[i : i + 2, 2],
                    color=colors[i],
                    alpha=alpha,
                    linewidth=linewidth,
                )

        ax.scatter([0], [0], [0], color="red", s=20, zorder=10)
        ax.set_title(f"f{start}\u2013{end} ({seg_len} fr, w={smooth_window})", fontsize=8)
        ax.set_xlabel("PC1", fontsize=6)
        ax.set_ylabel("PC2", fontsize=6)
        ax.set_zlabel("PC3", fontsize=6)
        ax.tick_params(labelsize=5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_scene_overlay_trajectories(
    Xavg_3d: np.ndarray,
    scenes: list,
    out_path: str,
    title: str,
    *,
    max_scenes_shown: int = 12,
    alpha: float = 0.7,
    linewidth: float = 1.2,
) -> None:
    """Overlay scene trajectories from the avg trace, all recentered to origin.

    Xavg_3d: (900, 3)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    scene_lens = sorted(enumerate(scenes), key=lambda x: x[1][1] - x[1][0], reverse=True)
    shown_idxs = sorted([i for i, _ in scene_lens[:max_scenes_shown]])
    shown_scenes = [scenes[i] for i in shown_idxs]

    tab20 = plt.get_cmap("tab20")
    fig = plt.figure(figsize=(8, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=9)

    for plot_idx, (start, end) in enumerate(shown_scenes):
        color = tab20(plot_idx % 20)
        seg = Xavg_3d[start:end]
        avg_seg = seg - seg[0:1, :]  # recenter

        orig_idx = shown_idxs[plot_idx]
        ax.plot(
            avg_seg[:, 0], avg_seg[:, 1], avg_seg[:, 2],
            color=color, alpha=alpha, linewidth=linewidth,
            label=f"s{orig_idx} f{start}\u2013{end}",
        )
        ax.scatter(
            [avg_seg[0, 0]], [avg_seg[0, 1]], [avg_seg[0, 2]],
            color=color, marker="o", s=20, alpha=0.9, zorder=10,
        )
        ax.scatter(
            [avg_seg[-1, 0]], [avg_seg[-1, 1]], [avg_seg[-1, 2]],
            color=color, marker="s", s=20, alpha=0.9, zorder=10,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(loc="upper left", fontsize=6, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Analysis B: Neural speed vs visual change
# ----------------------------


def _save_neural_speed_vs_visual_change(
    Xavg_3d: np.ndarray,
    frame_diff: np.ndarray,
    scenes: list,
    out_path: str,
    title: str,
) -> None:
    """Dual time-series and scatter of neural speed vs. pixel change.

    Xavg_3d: (900, 3); frame_diff: (899,)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    neural_speed = np.linalg.norm(np.diff(Xavg_3d, axis=0), axis=1)  # (899,)

    def _zscore(x: np.ndarray) -> np.ndarray:
        s = np.std(x)
        return (x - np.mean(x)) / (s if s > 0 else 1.0)

    ns_z = _zscore(neural_speed)
    fd_z = _zscore(frame_diff)
    r = float(np.corrcoef(neural_speed, frame_diff)[0, 1])

    t = np.arange(len(neural_speed))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), dpi=150)

    ax1.plot(t, ns_z, label="neural speed (z)", color="steelblue", lw=0.8, alpha=0.85)
    ax1.plot(t, fd_z, label="visual change (z)", color="darkorange", lw=0.8, alpha=0.85)
    for s, e in scenes:
        if s > 0:
            ax1.axvline(s, color="gray", lw=0.6, linestyle="--", alpha=0.6)
    ax1.set_xlabel("frame")
    ax1.set_ylabel("z-score")
    ax1.set_title(title)
    ax1.legend(fontsize=9)

    ax2.scatter(frame_diff, neural_speed, s=3, alpha=0.4, color="purple")
    ax2.set_xlabel("visual change (pixel diff)")
    ax2.set_ylabel("neural speed (PC distance/frame)")
    ax2.set_title(f"Pearson r = {r:.3f}")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Analysis C: Cross-trial variability per frame
# ----------------------------


def _save_cross_trial_variability(
    X_trials_3d: np.ndarray,
    frame_diff: np.ndarray,
    scenes: list,
    out_path: str,
    title: str,
) -> None:
    """Per-frame cross-trial variability (avg std across PC dims).

    X_trials_3d: (trials, 900, 3); frame_diff: (899,)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n_frames = X_trials_3d.shape[1]
    variability = np.zeros(n_frames)
    for f in range(n_frames):
        pts = X_trials_3d[:, f, :]
        valid = pts[np.all(np.isfinite(pts), axis=1)]
        if len(valid) > 1:
            variability[f] = float(np.std(valid, axis=0).mean())

    r = float(np.corrcoef(variability[1:], frame_diff)[0, 1])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), dpi=150)

    t = np.arange(n_frames)
    ax1.plot(t, variability, color="steelblue", lw=0.8)
    for s, e in scenes:
        if s > 0:
            ax1.axvline(s, color="gray", lw=0.6, linestyle="--", alpha=0.6)
    ax1.set_ylabel("cross-trial std (avg PC)")
    ax1.set_title(title)

    t2 = np.arange(len(frame_diff))
    ax2.plot(t2, frame_diff, color="darkorange", lw=0.8)
    for s, e in scenes:
        if s > 0:
            ax2.axvline(s, color="gray", lw=0.6, linestyle="--", alpha=0.6)
    ax2.set_ylabel("pixel frame diff")
    ax2.set_xlabel("frame")

    ax3.scatter(frame_diff, variability[1:], s=3, alpha=0.4, color="purple")
    ax3.set_xlabel("visual change (pixel diff)")
    ax3.set_ylabel("cross-trial variability")
    ax3.set_title(f"Pearson r = {r:.3f}")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Analysis D: Neural and visual similarity matrices
# ----------------------------


def _save_similarity_matrices(
    Xavg_3d: np.ndarray,
    movie_frames: np.ndarray,
    scenes: list,
    out_path: str,
    title: str,
    *,
    frame_subsample: int = 3,
) -> None:
    """Side-by-side neural distance and pixel similarity matrices.

    Xavg_3d: (900, 3); movie_frames: (900, H, W) uint8
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    idx = np.arange(0, len(Xavg_3d), frame_subsample)
    Xsub = Xavg_3d[idx]  # (N, 3)

    # Neural pairwise distance
    diff = Xsub[:, None, :] - Xsub[None, :, :]   # (N, N, 3)
    neural_dist = np.linalg.norm(diff, axis=-1)    # (N, N)

    # Pixel similarity: z-score frames then dot product / n_pixels
    F = movie_frames[idx].astype(np.float32)       # (N, H, W)
    n_pixels = F.shape[1] * F.shape[2]
    F = F.reshape(len(idx), -1)                    # (N, P)
    mu = F.mean(axis=1, keepdims=True)
    sigma = F.std(axis=1, keepdims=True)
    sigma = np.where(sigma > 0, sigma, 1.0)
    F_z = (F - mu) / sigma
    pixel_sim = (F_z @ F_z.T) / n_pixels          # (N, N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=150)

    ext = [int(idx[0]), int(idx[-1]), int(idx[-1]), int(idx[0])]
    im1 = ax1.imshow(neural_dist, aspect="auto", cmap="viridis_r", extent=ext)
    ax1.set_title("Neural distance (avg trajectory)")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("frame")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(pixel_sim, aspect="auto", cmap="plasma", extent=ext)
    ax2.set_title("Pixel similarity (z-scored dot product / pixel)")
    ax2.set_xlabel("frame")
    ax2.set_ylabel("frame")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Overlay scene boundaries
    for s, e in scenes:
        for ax in (ax1, ax2):
            ax.axhline(s, color="white", lw=0.6, alpha=0.7)
            ax.axvline(s, color="white", lw=0.6, alpha=0.7)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Analysis E: Scene tightness
# ----------------------------


def _save_scene_tightness(
    X_trials_3d: np.ndarray,
    scenes: list,
    Xavg_3d: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    """Horizontal bar chart of per-scene avg cross-trial variability (tightness).

    X_trials_3d: (trials, 900, 3); Xavg_3d: (900, 3)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n_frames = X_trials_3d.shape[1]
    variability = np.zeros(n_frames)
    for f in range(n_frames):
        pts = X_trials_3d[:, f, :]
        valid = pts[np.all(np.isfinite(pts), axis=1)]
        if len(valid) > 1:
            variability[f] = float(np.std(valid, axis=0).mean())

    tightness = np.array([variability[s:e].mean() for s, e in scenes])
    durations = np.array([e - s for s, e in scenes])
    labels = [f"s{i} f{s}\u2013{e}" for i, (s, e) in enumerate(scenes)]

    order = np.argsort(tightness)
    tight_sorted = tightness[order]
    labels_sorted = [labels[i] for i in order]
    dur_sorted = durations[order]

    dur_range = float(dur_sorted.max() - dur_sorted.min())
    dur_norm = (dur_sorted - dur_sorted.min()) / (dur_range if dur_range > 0 else 1.0)
    cmap = plt.get_cmap("viridis")
    bar_colors = [cmap(float(v)) for v in dur_norm]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(scenes))), dpi=150)
    ax.barh(range(len(scenes)), tight_sorted, color=bar_colors)
    ax.set_yticks(range(len(scenes)))
    ax.set_yticklabels(labels_sorted, fontsize=8)
    ax.set_xlabel("avg cross-trial variability (tighter = lower)")
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=int(dur_sorted.min()), vmax=int(dur_sorted.max())),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.01)
    cbar.set_label("scene duration (frames)", rotation=270, labelpad=14)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# PCA + plotting
# ----------------------------

out_dir = os.path.join("fig", "natural_movies")

# PCA on averaged trace
Xavg_3d = _pca_3d_over_neurons(trial_avgs_all_neurons)
_save_3d_time_colored_trace(
    Xavg_3d,
    os.path.join(out_dir, "pca3d_avg_trace.png"),
    "Natural movie: averaged trace (PCA over neurons)",
)

# PCA for each trial using a PCA fit on the averaged trace (so all trials share axes)
pca = PCA(n_components=3, random_state=0)
pca.fit(trial_avgs_all_neurons.T)  # samples=frames, features=neurons

# all_neurons: (neurons, trials, frames) -> (trials, frames, neurons)
X_trials_frames_neurons = np.transpose(all_neurons, (1, 2, 0))

# Handle NaNs (some frames have fewer trials in a session, padded with NaN).
# We fill NaNs per trial+neuron with that neuron's mean across frames for that trial,
# and if a neuron is all-NaN for a trial, we fill with 0.
X_trials_filled = X_trials_frames_neurons.copy()
for tr in range(X_trials_filled.shape[0]):
    X = X_trials_filled[tr]  # (frames, neurons)
    # mean across frames for each neuron
    m = np.nanmean(X, axis=0)
    m = np.where(np.isfinite(m), m, 0.0)
    inds = np.where(~np.isfinite(X))
    if inds[0].size:
        X[inds] = m[inds[1]]
    X_trials_filled[tr] = X

X_trials_3d = np.stack([pca.transform(X_trials_filled[tr]) for tr in range(X_trials_filled.shape[0])], axis=0)

_save_3d_time_colored_traces(
    X_trials_3d,
    os.path.join(out_dir, "pca3d_all_trials.png"),
    "Natural movie: all trials (PCA over neurons, shared axes)",
)

# Requested: also encode both trace identity and time (fade from white → trial color)
_save_3d_trial_and_time_colored_traces(
    X_trials_3d,
    os.path.join(out_dir, "pca3d_all_trials_trial_and_time_color.png"),
    "Natural movie: all trials (trace color + time fade)",
)

# Movie-frame ↔ activity linkage plot
_save_avg_trace_with_movie_frames(
    Xavg_3d,
    movie_frames,
    os.path.join(out_dir, "pca3d_avg_with_movie_frames.png"),
    "Averaged activity trajectory with example movie frames",
    n_frames_to_show=8,
)

# Clustering plot: cluster frames in activity space + show representative frames
_save_clusters_with_movie_frames(
    Xavg_3d,
    movie_frames,
    os.path.join(out_dir, "pca3d_avg_clusters_with_frames.png"),
    "Clusters of frames in activity space (KMeans on 3D PCA)",
    n_clusters=6,
)

# Movie: left=natural movie, right=avg trajectory with current point highlighted
_save_movie_with_trajectory(
    movie_frames,
    Xavg_3d,
    os.path.join(out_dir, "natural_movie_with_avg_trajectory.gif"),
    fps=30,
)

# ----------------------------
# Scene-based analyses
# ----------------------------

frame_diff = _compute_frame_diff(movie_frames)  # (899,)
scenes = _detect_scene_cuts(frame_diff, len(movie_frames))
print(f"Detected {len(scenes)} scenes: {scenes}")

_save_neural_speed_vs_visual_change(
    Xavg_3d,
    frame_diff,
    scenes,
    os.path.join(out_dir, "neural_speed_vs_visual_change.png"),
    "Neural speed vs. visual change",
)
print("Saved neural_speed_vs_visual_change.png")

_save_similarity_matrices(
    Xavg_3d,
    movie_frames,
    scenes,
    os.path.join(out_dir, "similarity_matrices.png"),
    "Neural distance and pixel similarity matrices",
)
print("Saved similarity_matrices.png")

_save_scene_relative_trajectories(
    X_trials_3d,
    scenes,
    os.path.join(out_dir, "scene_trajectories_per_scene.png"),
    "Scene-relative trial trajectories (recentered to scene start)",
)
print("Saved scene_trajectories_per_scene.png")

_save_scene_relative_trajectories_smoothed(
    X_trials_3d,
    scenes,
    os.path.join(out_dir, "scene_trajectories_per_scene_smoothed.png"),
    "Scene-relative trial trajectories (smoothed, w=7 frames)",
    smooth_window=7,
)
print("Saved scene_trajectories_per_scene_smoothed.png")

_save_scene_overlay_trajectories(
    Xavg_3d,
    scenes,
    os.path.join(out_dir, "scene_trajectories_overlay.png"),
    "Scene trajectories overlaid (avg trace, recentered)",
)
print("Saved scene_trajectories_overlay.png")

_save_cross_trial_variability(
    X_trials_3d,
    frame_diff,
    scenes,
    os.path.join(out_dir, "cross_trial_variability.png"),
    "Cross-trial variability vs. visual change",
)
print("Saved cross_trial_variability.png")

_save_scene_tightness(
    X_trials_3d,
    scenes,
    Xavg_3d,
    os.path.join(out_dir, "scene_tightness.png"),
    "Scene tightness (avg cross-trial variability per scene)",
)
print("Saved scene_tightness.png")


# ----------------------------
# Analysis F: Top rate-of-change neurons highlighted on encoding manifold
# ----------------------------


def _save_top_roc_on_manifold(
    trial_avgs: np.ndarray,
    session_uids: list,
    manifold: np.ndarray,
    out_path: str,
    *,
    n_top: int = 50,
) -> None:
    """Scatter the encoding manifold (DC1 vs DC2) and highlight the neurons
    with highest average frame-to-frame rate of change in the natural movie.

    Args:
        trial_avgs: (n_neurons, n_frames) mean firing rate per frame
        session_uids: list of (session_id, unit_id) in natural-movie neuron order
        manifold: (n_manifold_neurons, >=2) encoding manifold embedding
        out_path: where to save the figure
        n_top: number of high-ROC neurons to highlight
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Per-neuron average absolute frame-to-frame change
    roc = np.mean(np.abs(np.diff(trial_avgs, axis=1)), axis=1)  # (n_neurons,)

    # Top-n natural-movie indices by ROC
    top_natmovie = list(np.argsort(roc)[::-1][:n_top])

    # Map to manifold indices (drop any not present in manifold)
    top_manifold = natmovie_to_manifold_indices(top_natmovie)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)

    ax.scatter(
        manifold[:, 0], manifold[:, 1],
        s=6, color="steelblue", alpha=0.5, linewidths=0,
        label=f"all neurons (n={len(manifold)})",
        zorder=1,
    )
    ax.scatter(
        manifold[top_manifold, 0], manifold[top_manifold, 1],
        s=40, color="crimson", alpha=0.85, linewidths=0.4, edgecolors="white",
        label=f"top-{n_top} ROC neurons (n={len(top_manifold)} on manifold)",
        zorder=2,
    )

    ax.set_xlabel("DC1")
    ax.set_ylabel("DC2")
    ax.set_title(f"Encoding manifold: top-{n_top} neurons by avg frame-to-frame ΔFR")
    ax.legend(fontsize=9, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  matched {len(top_manifold)}/{n_top} top-ROC neurons onto manifold")


enc_manifold = np.load("enc_mfds/VISp-manifold.npy")
_save_top_roc_on_manifold(
    trial_avgs_all_neurons,
    session_uids_used,
    enc_manifold,
    os.path.join(out_dir, "top_roc_neurons_on_manifold.png"),
    n_top=50,
)
print("Saved top_roc_neurons_on_manifold.png")
