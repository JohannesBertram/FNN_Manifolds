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


