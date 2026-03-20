"""
ManifoldExplorer — Interactive 3D manifold explorer with subpopulation selection,
PSTH plotting, and a live decoding panel.

Usage (in encoding_manifolds.ipynb, after all analysis cells):

    from src.explorer_utils import ManifoldExplorer
    import numpy as np

    # Reshape tensorX (N_all, N_STIM, NDIRS*T) → (N_all, N_STIM, T, NDIRS),
    # then select nonoutliers so indexing is direct.
    tensorX_4d = np.reshape(tensorX, (len(tensorX), NSTIMS, TRIAL_LEN, NDIRS))
    tensorX_4d_nonout = tensorX_4d[nonoutliers]   # (N_nonout, N_STIM, T, N_DIR)

    explorer = ManifoldExplorer(
        embedding_     = embedding_,
        tensor4d       = tensorX_4d_nonout,
        nonoutliers    = nonoutliers,
        neurons_used   = neurons_used,
        categories     = my_stims,
        NDIRS          = NDIRS,
        cluster_labels = None,   # or cluster_labels if HDBSCAN was run
    )
    explorer.show()
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, Javascript
import ipywidgets as widgets
import plotly.graph_objects as go

from .subpop_utils import compute_dynamic_metrics, select_top_k_by_metric


def _make_bbox_lines(x0, x1, y0, y1, z0, z1):
    """Return (xs, ys, zs) for 12 box edges (None-separated for Plotly lines)."""
    edges = [
        # 4 bottom edges
        ((x0,y0,z0),(x1,y0,z0)), ((x1,y0,z0),(x1,y1,z0)),
        ((x1,y1,z0),(x0,y1,z0)), ((x0,y1,z0),(x0,y0,z0)),
        # 4 top edges
        ((x0,y0,z1),(x1,y0,z1)), ((x1,y0,z1),(x1,y1,z1)),
        ((x1,y1,z1),(x0,y1,z1)), ((x0,y1,z1),(x0,y0,z1)),
        # 4 verticals
        ((x0,y0,z0),(x0,y0,z1)), ((x1,y0,z0),(x1,y0,z1)),
        ((x1,y1,z0),(x1,y1,z1)), ((x0,y1,z0),(x0,y1,z1)),
    ]
    xs, ys, zs = [], [], []
    for (p0, p1) in edges:
        xs += [p0[0], p1[0], None]
        ys += [p0[1], p1[1], None]
        zs += [p0[2], p1[2], None]
    return xs, ys, zs


# ─────────────────────────────────────────────────────────────────────────────
# PSTH helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_individual_psths(sub, idxs, categories):
    """Per-neuron PSTH strip: (k × N_STIM) heatmap grid.

    Parameters
    ----------
    sub        : (k, N_STIM, T, N_DIR) float array
    idxs       : list of int — nonoutlier-space indices (for axis labels)
    categories : list of str — stimulus labels
    """
    k, N_STIM, T, N_DIR = sub.shape
    vmin, vmax = sub.min(), sub.max()

    fig, axs = plt.subplots(k, N_STIM, figsize=(2.5 * N_STIM, 2 * k),
                            squeeze=False)
    for ni, (neu, idx) in enumerate(zip(sub, idxs)):
        for stim in range(N_STIM):
            ax = axs[ni, stim]
            pst = neu[stim].T          # (N_DIR, T)
            opt_dir = pst.mean(1).argmax()
            pst = np.roll(pst, (2 - opt_dir) % N_DIR, axis=0)
            pst = np.concatenate([np.zeros((N_DIR, 5)), pst], axis=1)
            ax.imshow(pst, aspect='auto', interpolation='quadric',
                      cmap='hot', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if ni == 0:
                lbl = categories[stim] if stim < len(categories) else str(stim)
                ax.set_title(lbl, fontsize=8)
            if stim == 0:
                ax.set_ylabel(f'n={idx}', fontsize=7)
    fig.suptitle(f'Individual PSTHs ({k} neurons)', fontsize=10)
    fig.tight_layout()


def _plot_averaged_psths(avg_sub, categories):
    """Population-averaged PSTH: 3×2 heatmap grid.

    Parameters
    ----------
    avg_sub    : (N_STIM, T, N_DIR) float array — already averaged over neurons
    categories : list of str — stimulus labels
    """
    N_STIM, T, N_DIR = avg_sub.shape

    panels = []
    for stim in range(N_STIM):
        pst = avg_sub[stim].T          # (N_DIR, T)
        opt_dir = pst.mean(1).argmax()
        pst = np.roll(pst, (2 - opt_dir) % N_DIR, axis=0)
        pst = np.concatenate([np.zeros((N_DIR, 5)), pst], axis=1)
        panels.append(pst)

    vmin = min(p.min() for p in panels)
    vmax = max(p.max() for p in panels)

    ncols = min(N_STIM, 3)
    nrows = math.ceil(N_STIM / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows), squeeze=False)
    for stim, pst in enumerate(panels):
        ax = axs[stim // ncols, stim % ncols]
        ax.imshow(pst, aspect='auto', interpolation='quadric',
                  cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        lbl = categories[stim] if stim < len(categories) else str(stim)
        ax.set_title(lbl, fontsize=8)
    fig.suptitle('Population-averaged PSTHs', fontsize=10)
    fig.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# ManifoldExplorer
# ─────────────────────────────────────────────────────────────────────────────

# Palette used in decoding_analysis.ipynb (applied to decoding figures)
_DECODING_PALETTE = [
    '#0070C0', '#00B0F0', '#00B050', '#92D050', '#FF0000', '#FFC000',
    '#7030A0', '#FF6600', '#339933', '#CC0066', '#009999', '#996633',
]

_DEFAULT_PALETTE = np.array([
    '#03579b', '#0488d1', '#03a9f4', '#4fc3f7', '#b3e5fc',
    '#19237e', '#303f9f', '#3f51b5', '#7986cb', '#c5cae9',
    '#4a198c', '#7b21a2', '#9c27b0', '#ba68c8', '#e1bee7',
    '#88144f', '#c21f5b', '#e92663', '#f06292', '#f8bbd0',
    '#bf360c', '#e64a18', '#ff5722', '#ff8a65', '#ffccbc',
    '#f67f17', '#fbc02c', '#ffec3a', '#fff177', '#fdf9c3',
    '#33691d', '#689f38', '#8bc34a', '#aed581', '#ddedc8',
    '#253137', '#455a64', '#607d8b', '#90a4ae', '#cfd8dc',
])


class ManifoldExplorer:
    """Interactive 3D manifold explorer with subpopulation selection, PSTH plotting,
    and a live decoding panel.

    Selection modes
    ---------------
    Click       — click individual points on the scatter to toggle them
    Cluster     — select all neurons in a given HDBSCAN cluster
    Bounding box — select neurons inside an axis-aligned box in embedding space
    Radius      — select neurons within a Euclidean radius of a seed point

    PSTH display
    ------------
    ≤5 neurons selected  → individual per-neuron heatmap strip
    ≥6 neurons selected  → population-averaged 3×2 heatmap grid

    Decoding panel
    --------------
    Shows PCA of neural activity: time-averaged (manifold scatter) and
    time-resolved (trajectory lines) for the current (sub)population.
    Updates live when "Plot PSTHs ▶" is clicked; resets on "Clear selection".
    """

    def __init__(self, embedding_, tensor4d, nonoutliers, neurons_used,
                 categories, NDIRS, cluster_labels=None, palette=None,
                 extra_colorings=None):
        """
        Parameters
        ----------
        embedding_    : (N_nonout, n_components) float array — MDS embedding
        tensor4d      : (N_nonout, N_STIM, T, N_DIR) float array — already in
                        nonoutlier space (index directly with nonoutlier indices)
        nonoutliers   : (N_nonout,) int array — indices into full population
                        (stored for export / display only)
        neurons_used  : (N_all, k) int/float array — neuron metadata;
                        column 1 is used for feature-map / cell-type colouring
        categories    : list[str] — stimulus category labels (length = N_STIM)
        NDIRS         : int — number of directions (N_DIR axis of tensor4d)
        cluster_labels: (N_nonout,) int array or None — HDBSCAN cluster labels
        palette       : array-like of hex strings or None (uses default 40-colour palette)
        """
        self.emb           = np.asarray(embedding_)
        self.tensor4d      = np.asarray(tensor4d)
        self.nonoutliers   = np.asarray(nonoutliers)
        self.neurons_used  = neurons_used
        self.categories    = list(categories)
        self.NDIRS         = NDIRS
        self.N_STIM        = self.tensor4d.shape[1]
        self.cluster_labels = (np.asarray(cluster_labels)
                               if cluster_labels is not None else None)
        self.palette = (np.asarray(palette) if palette is not None
                        else _DEFAULT_PALETTE.copy())

        self.extra_colorings = dict(extra_colorings) if extra_colorings else {}
        # Auto-detect type: integer dtype or ≤20 unique values → categorical (palette)
        # otherwise → continuous (Viridis colorscale)
        self._coloring_type = {}
        for name, arr in self.extra_colorings.items():
            arr = np.asarray(arr)
            self.extra_colorings[name] = arr   # ensure ndarray
            finite_vals = arr[np.isfinite(arr.astype(float))]
            if np.issubdtype(arr.dtype, np.integer) or len(np.unique(finite_vals)) <= 20:
                self._coloring_type[name] = 'categorical'
            else:
                self._coloring_type[name] = 'continuous'

        self._selected_idxs   = set()   # indices in nonoutlier space
        self._last_clicked_idx = None   # for setting radius seed
        self._seed_point       = None   # embedding coords of radius seed
        self._cached_metrics   = None   # computed on first Metric tab Run

        # Decoding panel state (two separate figures: manifold + trajectories)
        self._decoding_full_coords = None   # (S*D, 3) precomputed for full population
        self._decoding_full_trajs  = None   # (S*D, T, 3) precomputed for full population

        self._build_ui()

    # ── colour helpers ────────────────────────────────────────────────────────

    def _colors_for_mode(self, mode):
        """Return (color_list, colorscale_or_None) for the base scatter trace."""
        N = len(self.emb)
        pal = self.palette
        if mode in self.extra_colorings:
            arr = np.asarray(self.extra_colorings[mode])[self.nonoutliers]
            if self._coloring_type[mode] == 'categorical':
                ulbls = np.unique(arr)
                pal = self.palette.copy()
                while len(pal) < len(ulbls):
                    pal = np.r_[pal, self.palette]
                cmap = {lbl: pal[i] for i, lbl in enumerate(ulbls)}
                return [cmap.get(v, '#aaaaaa') for v in arr], None
            else:
                arr_f = arr.astype(float)
                if np.any(np.isnan(arr_f)):
                    import matplotlib.cm as _mcm
                    import matplotlib.colors as _mc
                    _cmap = _mcm.get_cmap('viridis')
                    finite = arr_f[np.isfinite(arr_f)]
                    _norm = _mc.Normalize(vmin=finite.min(), vmax=finite.max())
                    colors = []
                    for v in arr_f:
                        if np.isnan(v):
                            colors.append('rgba(0,0,0,0)')
                        else:
                            r, g, b, _ = _cmap(_norm(v))
                            colors.append(f'rgba({int(r*255)},{int(g*255)},{int(b*255)},1)')
                    return colors, None
                return list(arr_f), 'Viridis'
        elif mode == 'feature_map' and self.neurons_used is not None:
            raw = self.neurons_used[self.nonoutliers, 1]
        elif mode == 'cluster' and self.cluster_labels is not None:
            raw = self.cluster_labels
        else:
            return list(range(N)), 'Viridis'

        ulbls = np.unique(raw)
        while len(pal) < len(ulbls):
            pal = np.r_[pal, self.palette]
        cmap = {lbl: pal[i] for i, lbl in enumerate(ulbls)}
        return [cmap.get(v, '#aaaaaa') for v in raw], None

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        N       = len(self.emb)
        n_dims  = self.emb.shape[1]

        # ── DC pair dropdowns ────────────────────────────────────────────────
        dim_opts = list(range(n_dims))
        self._dc_x = widgets.Dropdown(options=dim_opts, value=0,
                                      description='x:', style={'description_width': '20px'},
                                      layout=widgets.Layout(width='80px'))
        self._dc_y = widgets.Dropdown(options=dim_opts, value=1,
                                      description='y:', style={'description_width': '20px'},
                                      layout=widgets.Layout(width='80px'))
        self._dc_z = widgets.Dropdown(options=dim_opts, value=2,
                                      description='z:', style={'description_width': '20px'},
                                      layout=widgets.Layout(width='80px'))
        for w in (self._dc_x, self._dc_y, self._dc_z):
            w.observe(self._on_dc_change, names='value')

        # ── Color-by dropdown ────────────────────────────────────────────────
        base_opts = ['feature_map', 'cluster', 'index']
        all_opts  = base_opts + list(self.extra_colorings.keys())
        self._color_dd = widgets.Dropdown(
            options=all_opts, value='feature_map',
            description='Color by:', layout=widgets.Layout(width='200px'))
        self._color_dd.observe(self._on_color_change, names='value')

        # ── 3-D Plotly FigureWidget (encoding) ───────────────────────────────
        colors, cscale = self._colors_for_mode('feature_map')
        hover = [f'idx={i}  (orig={self.nonoutliers[i]})' for i in range(N)]

        self._trace_base = go.Scatter3d(
            x=self.emb[:, 0], y=self.emb[:, 1], z=self.emb[:, 2],
            mode='markers',
            marker=dict(size=3, color=colors, colorscale=cscale, opacity=0.6),
            text=hover, hoverinfo='text', name='all',
        )
        self._trace_sel = go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(size=6, color='orange', opacity=1.0),
            text=[], hoverinfo='text', name='selected',
        )
        self._trace_bbox = go.Scatter3d(
            x=[], y=[], z=[], mode='lines',
            line=dict(color='lime', width=3),
            name='bbox', hoverinfo='skip', visible=False,
        )

        fig_layout = go.Layout(
            scene=dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(showticklabels=False, title=''),
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            width=420, height=480, showlegend=True,
            legend=dict(x=0, y=1),
            title=dict(text='Encoding Manifold', x=0.5),
        )
        self._fig = go.FigureWidget(
            data=[self._trace_base, self._trace_sel, self._trace_bbox],
            layout=fig_layout)
        # Reassign to live FigureWidget trace references (FigureWidget copies on init)
        self._trace_base = self._fig.data[0]
        self._trace_sel  = self._fig.data[1]
        self._trace_bbox = self._fig.data[2]
        self._fig.data[0].on_click(self._on_click)

        self._fig_container = widgets.Box(
            [self._fig],
            layout=widgets.Layout(width='420px'),
        )

        # ── Decoding FigureWidget ─────────────────────────────────────────────
        self._build_decoding_fig()

        # ── Selection tabs ────────────────────────────────────────────────────

        # Tab 0: Click (no extra controls)
        click_tab = widgets.HTML(
            '<i>Click points on the 3-D plot to toggle selection.</i>')

        # Tab 1: Cluster
        if self.cluster_labels is not None:
            n_cl  = int(self.cluster_labels.max()) + 1
            opts  = [('-1 (noise)', -1)] + [(f'cluster {i}', i) for i in range(n_cl)]
            self._cluster_dd = widgets.Dropdown(
                options=opts, description='Cluster:',
                layout=widgets.Layout(width='220px'))
            cl_btn = widgets.Button(description='Select cluster',
                                    button_style='info')
            cl_btn.on_click(self._on_cluster_select)
            cluster_tab = widgets.VBox([self._cluster_dd, cl_btn])
        else:
            cluster_tab = widgets.HTML('<i>No cluster labels provided.</i>')
            self._cluster_dd = None

        # Tab 2: Bounding box
        emin, emax = self.emb.min(axis=0), self.emb.max(axis=0)
        self._bbox_sliders = []
        for di in range(min(3, n_dims)):
            lo, hi = float(emin[di]), float(emax[di])
            step = max((hi - lo) / 200, 1e-6)
            sl = widgets.FloatRangeSlider(
                value=[lo, hi], min=lo, max=hi, step=step,
                description=f'dim {di}:',
                layout=widgets.Layout(width='420px'),
                continuous_update=True, readout=True, readout_format='.2f',
            )
            sl.observe(self._on_bbox_change, names='value')
            self._bbox_sliders.append(sl)
        bbox_tab = widgets.VBox(self._bbox_sliders)

        # Tab 3: Radius
        self._seed_lbl = widgets.Label('Seed: (click a point first)')
        self._seed_btn = widgets.Button(
            description='Use last click as seed', button_style='warning')
        self._seed_btn.on_click(self._on_set_seed)
        self._radius_sl = widgets.FloatSlider(
            value=0.05, min=1e-4, max=1.0, step=1e-3,
            description='Radius:', layout=widgets.Layout(width='420px'),
            continuous_update=True, readout=True, readout_format='.3f',
        )
        self._radius_sl.observe(self._on_radius_change, names='value')
        radius_tab = widgets.VBox([
            widgets.HTML('<i>Click a point, then press the button to set as seed:</i>'),
            self._seed_lbl,
            self._seed_btn,
            self._radius_sl,
        ])

        # Tab 4: Metric
        self._metric_dd = widgets.Dropdown(
            options=['speed', 'stability', 'curvature', 'classifiability', 'pc_contrib'],
            description='Property:', layout=widgets.Layout(width='230px'))
        self._metric_dir = widgets.ToggleButtons(
            options=['High', 'Low'], value='High',
            description='Keep:', layout=widgets.Layout(width='230px'))
        self._metric_pct = widgets.IntSlider(
            value=20, min=1, max=100,
            description='Top %:', continuous_update=False,
            layout=widgets.Layout(width='280px'))
        self._metric_run = widgets.Button(
            description='Run', button_style='primary',
            layout=widgets.Layout(width='80px'))
        self._metric_run.on_click(self._on_metric_run)
        self._sweep_out = widgets.Output()
        metric_controls = widgets.VBox([
            self._metric_dd, self._metric_dir, self._metric_pct, self._metric_run])
        metric_tab = widgets.HBox([metric_controls, self._sweep_out])

        self._tabs = widgets.Tab(
            children=[click_tab, cluster_tab, bbox_tab, radius_tab, metric_tab])
        for i, title in enumerate(['Click', 'Cluster', 'Bounding Box', 'Radius', 'Metric']):
            self._tabs.set_title(i, title)
        self._tabs.observe(self._on_tab_change, names='selected_index')

        # ── Info + action row ─────────────────────────────────────────────────
        self._info_lbl = widgets.HTML('Selected: <b>0</b> neurons  [indices: ]')

        clear_btn = widgets.Button(description='Clear selection',
                                   button_style='danger', icon='trash')
        clear_btn.on_click(self._on_clear)

        copy_btn = widgets.Button(description='Copy indices',
                                  button_style='', icon='clipboard')
        copy_btn.on_click(self._on_copy)

        psth_btn = widgets.Button(description='Plot PSTHs ▶',
                                  button_style='success')
        psth_btn.on_click(lambda _b: self._plot_psths())

        action_row = widgets.HBox([clear_btn, copy_btn, psth_btn])

        # ── PSTH output area ──────────────────────────────────────────────────
        self._psth_out = widgets.Output()

        # ── Top control bar ───────────────────────────────────────────────────
        ctrl_row = widgets.HBox([
            self._color_dd,
            widgets.Label('  DC pair:'),
            self._dc_x, self._dc_y, self._dc_z,
        ])

        # ── Full layout ───────────────────────────────────────────────────────
        figs_row = widgets.HBox(
            [self._fig_container,
             self._dec_manifold_container,
             self._dec_traj_container],
            layout=widgets.Layout(flex_flow='row nowrap', overflow='visible'),
        )
        self._root = widgets.VBox([
            ctrl_row,
            figs_row,
            self._tabs,
            self._info_lbl,
            action_row,
            self._psth_out,
        ])

    # ── Decoding figure construction ──────────────────────────────────────────

    def _dec_color(self, s):
        """Stimulus color from the decoding palette."""
        return _DECODING_PALETTE[s % len(_DECODING_PALETTE)]

    def _build_decoding_fig(self):
        """Build two decoding FigureWidgets:
          - _dec_manifold_fig : time-averaged points (small markers, per-stimulus color)
          - _dec_traj_fig     : trajectory lines + black start + colored end points
        """
        N_STIM = self.N_STIM
        NDIRS  = self.NDIRS

        _scene = dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
        )
        _margin = dict(l=0, r=0, t=30, b=0)

        # ── Manifold figure (time-averaged points) ────────────────────────────
        mf_traces = []
        self._dec_mf_traces = []
        for s in range(N_STIM):
            color = self._dec_color(s)
            lbl   = self.categories[s] if s < len(self.categories) else f'stim {s}'
            t = go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers',
                marker=dict(size=4, color=color, opacity=1.0),
                name=lbl, hoverinfo='skip',
            )
            mf_traces.append(t)
            self._dec_mf_traces.append(t)

        self._dec_manifold_fig = go.FigureWidget(
            data=mf_traces,
            layout=go.Layout(
                scene=_scene, margin=_margin, width=260, height=380,
                showlegend=False,
                title=dict(text='Decoding manifold (full pop.)', x=0.5),
            ),
        )
        self._dec_mf_traces = list(self._dec_manifold_fig.data)

        self._dec_manifold_container = widgets.Box(
            [self._dec_manifold_fig],
            layout=widgets.Layout(width='260px'),
        )

        # ── Trajectory figure (lines + start/end points) ──────────────────────
        # Per stimulus: 1 line trace + 1 start scatter (black) + 1 end scatter (colored)
        # Trace order in figure: [line_0, start_0, end_0, line_1, start_1, end_1, ...]
        tj_traces = []
        self._dec_tj_lines  = []
        self._dec_tj_starts = []
        self._dec_tj_ends   = []
        for s in range(N_STIM):
            color = self._dec_color(s)
            lbl   = self.categories[s] if s < len(self.categories) else f'stim {s}'
            t_line = go.Scatter3d(
                x=[], y=[], z=[],
                mode='lines',
                line=dict(color=color, width=2),
                opacity=0.35,
                name=lbl, hoverinfo='skip',
            )
            t_start = go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers',
                marker=dict(size=3, color='black', opacity=1.0),
                name=lbl + ' start', hoverinfo='skip', showlegend=False,
            )
            t_end = go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers',
                marker=dict(size=7, color=color, opacity=1.0),
                name=lbl + ' end', hoverinfo='skip', showlegend=False,
            )
            tj_traces += [t_line, t_start, t_end]
            self._dec_tj_lines.append(t_line)
            self._dec_tj_starts.append(t_start)
            self._dec_tj_ends.append(t_end)

        self._dec_traj_fig = go.FigureWidget(
            data=tj_traces,
            layout=go.Layout(
                scene=_scene, margin=_margin, width=420, height=480,
                showlegend=False,
                title=dict(text='Decoding trajectories (full pop.)', x=0.5),
            ),
        )
        # Reassign to live trace references
        self._dec_tj_lines  = [self._dec_traj_fig.data[3 * s]     for s in range(N_STIM)]
        self._dec_tj_starts = [self._dec_traj_fig.data[3 * s + 1] for s in range(N_STIM)]
        self._dec_tj_ends   = [self._dec_traj_fig.data[3 * s + 2] for s in range(N_STIM)]

        self._dec_traj_container = widgets.Box(
            [self._dec_traj_fig],
            layout=widgets.Layout(width='420px'),
        )

        # Precompute full-population decoding data
        try:
            self._decoding_full_coords, self._decoding_full_trajs = (
                self._compute_decoding_data(self.tensor4d)
            )
            self._update_decoding_fig(
                self._decoding_full_coords,
                self._decoding_full_trajs,
                suffix='full pop.',
            )
        except Exception as e:
            print(f'Warning: decoding panel initialisation failed: {e}')

    def _compute_decoding_data(self, tensor4d_sub):
        """Compute decoding manifold and trajectories via PCA.

        Parameters
        ----------
        tensor4d_sub : (k, N_STIM, T, N_DIR) — explorer tensor format

        Returns
        -------
        coords : (S*D, 3)
        trajs  : (S*D, T, 3)
        """
        from .subpop_utils import compute_decoding_manifold, compute_decoding_trajectories
        # subpop_utils expects (N, S, D, T); explorer stores (N, S, T, D)
        sub = tensor4d_sub.transpose(0, 1, 3, 2)
        coords, _ = compute_decoding_manifold(sub, n_components=3)
        trajs,  _ = compute_decoding_trajectories(sub, n_components=3)
        return coords, trajs

    def _update_decoding_fig(self, coords, trajs, suffix):
        """Update both decoding figures in-place via batch_update.

        Parameters
        ----------
        coords : (S*D, 3)
        trajs  : (S*D, T, 3)
        suffix : str  — appended to each figure title, e.g. 'full pop.' or '42 neurons'
        """
        NDIRS  = self.NDIRS
        N_STIM = self.N_STIM

        # ── Manifold figure ───────────────────────────────────────────────────
        with self._dec_manifold_fig.batch_update():
            for s in range(N_STIM):
                s_coords = coords[s * NDIRS:(s + 1) * NDIRS]   # (NDIRS, 3)
                self._dec_mf_traces[s].x = s_coords[:, 0].tolist()
                self._dec_mf_traces[s].y = s_coords[:, 1].tolist()
                self._dec_mf_traces[s].z = s_coords[:, 2].tolist()
            self._dec_manifold_fig.layout.title.text = f'Decoding manifold ({suffix})'

        # ── Trajectory figure ─────────────────────────────────────────────────
        with self._dec_traj_fig.batch_update():
            for s in range(N_STIM):
                # Lines: all NDIRS trajectories concatenated with None separators
                xs, ys, zs = [], [], []
                sx, sy, sz = [], [], []   # start points
                ex, ey, ez = [], [], []   # end points
                for d in range(NDIRS):
                    traj = trajs[s * NDIRS + d]   # (T, 3)
                    xs += list(traj[:, 0]) + [None]
                    ys += list(traj[:, 1]) + [None]
                    zs += list(traj[:, 2]) + [None]
                    sx.append(float(traj[0, 0]))
                    sy.append(float(traj[0, 1]))
                    sz.append(float(traj[0, 2]))
                    ex.append(float(traj[-1, 0]))
                    ey.append(float(traj[-1, 1]))
                    ez.append(float(traj[-1, 2]))
                self._dec_tj_lines[s].x  = xs
                self._dec_tj_lines[s].y  = ys
                self._dec_tj_lines[s].z  = zs
                self._dec_tj_starts[s].x = sx
                self._dec_tj_starts[s].y = sy
                self._dec_tj_starts[s].z = sz
                self._dec_tj_ends[s].x   = ex
                self._dec_tj_ends[s].y   = ey
                self._dec_tj_ends[s].z   = ez
            self._dec_traj_fig.layout.title.text = f'Decoding trajectories ({suffix})'

    # ── Plotly click callback ─────────────────────────────────────────────────

    def _on_click(self, trace, points, selector):
        """Toggle clicked point (click mode only); always update last-clicked."""
        for idx in points.point_inds:
            self._last_clicked_idx = idx
            if self._tabs.selected_index == 0:   # click mode
                if idx in self._selected_idxs:
                    self._selected_idxs.discard(idx)
                else:
                    self._selected_idxs.add(idx)
        if self._tabs.selected_index == 0:
            self._update_selection()

    # ── Selection mode callbacks ──────────────────────────────────────────────

    def _on_cluster_select(self, _btn):
        if self._cluster_dd is None or self.cluster_labels is None:
            return
        k = self._cluster_dd.value
        self._selected_idxs = set(
            int(i) for i in np.where(self.cluster_labels == k)[0])
        self._update_selection()

    def _on_bbox_change(self, _change):
        if self._tabs.selected_index != 2:
            return
        self._apply_bbox()

    def _apply_bbox(self):
        dcs = [self._dc_x.value, self._dc_y.value, self._dc_z.value]
        new_sel = set()
        for i in range(len(self.emb)):
            if all(sl.value[0] <= self.emb[i, dcs[di]] <= sl.value[1]
                   for di, sl in enumerate(self._bbox_sliders)):
                new_sel.add(i)
        self._selected_idxs = new_sel
        self._update_selection()
        self._update_bbox_trace()

    def _update_bbox_trace(self):
        dcs = [self._dc_x.value, self._dc_y.value, self._dc_z.value]
        x0, x1 = self._bbox_sliders[0].value
        y0, y1 = self._bbox_sliders[1].value
        z0, z1 = self._bbox_sliders[2].value
        xs, ys, zs = _make_bbox_lines(x0, x1, y0, y1, z0, z1)
        with self._fig.batch_update():
            self._trace_bbox.x = xs
            self._trace_bbox.y = ys
            self._trace_bbox.z = zs
            self._trace_bbox.visible = True

    def _on_tab_change(self, change):
        if change['new'] == 2:        # switched to bounding box tab
            self._apply_bbox()
        else:
            self._trace_bbox.visible = False

    def _on_set_seed(self, _btn):
        if self._last_clicked_idx is None:
            return
        self._seed_point = self.emb[self._last_clicked_idx].copy()
        orig = self.nonoutliers[self._last_clicked_idx]
        self._seed_lbl.value = (
            f'Seed: nonout_idx={self._last_clicked_idx}  (full_idx={orig})')
        self._apply_radius()

    def _run_fraction_sweep(self, metric_name):
        """Lightweight sweep for one metric: hi/lo + random. Returns (fracs, lines).
        lines[strategy] = {'acc': [(mean, std), ...], 'r2': [(mean, std), ...]}
        Only runs 7 fractions × 3 strategies × 2 metrics (acc, r2). Fast (~2–5 s).
        """
        from .subpop_utils import (
            compute_decoding_manifold, knn_decoding_accuracy,
            procrustes_r2, select_top_k_by_metric,
        )
        rng = np.random.default_rng(0)
        N_SEEDS = 5
        FRACS = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0])

        # Explorer stores (N, S, T, D); subpop_utils expects (N, S, D, T)
        tensor_sdt = self.tensor4d.transpose(0, 1, 3, 2)
        N = tensor_sdt.shape[0]
        NSTIMS, NDIRS = tensor_sdt.shape[1], tensor_sdt.shape[2]
        stim_labels = np.repeat(np.arange(NSTIMS), NDIRS)

        coords_full, _ = compute_decoding_manifold(tensor_sdt, n_components=3)

        def _pad(cs, ref):
            if cs.shape[1] < ref.shape[1]:
                return np.hstack([cs, np.zeros((cs.shape[0], ref.shape[1] - cs.shape[1]))])
            return cs

        strategies = [metric_name, f'{metric_name} [lo]', 'random']
        lines = {s: {'acc': [], 'r2': []} for s in strategies}

        for f in FRACS:
            k = max(1, int(round(f * N)))
            n_comp = min(3, k)

            # random baseline
            _accs, _r2s = [], []
            for _ in range(N_SEEDS):
                idx = rng.choice(N, k, replace=False)
                t_sub = tensor_sdt[idx]
                cs, _ = compute_decoding_manifold(t_sub, n_components=n_comp)
                _accs.append(knn_decoding_accuracy(cs, stim_labels))
                _r2s.append(procrustes_r2(coords_full, _pad(cs, coords_full)))
            lines['random']['acc'].append((np.nanmean(_accs), np.nanstd(_accs)))
            lines['random']['r2'].append((np.nanmean(_r2s),  np.nanstd(_r2s)))

            # hi and lo
            for high, sname in [(True, metric_name), (False, f'{metric_name} [lo]')]:
                idx = select_top_k_by_metric(self._cached_metrics, metric_name, k=k, high=high)
                if len(idx) < 1:
                    lines[sname]['acc'].append((np.nan, 0.0))
                    lines[sname]['r2'].append((np.nan, 0.0))
                    continue
                t_sub = tensor_sdt[idx]
                cs, _ = compute_decoding_manifold(t_sub, n_components=min(3, len(idx)))
                lines[sname]['acc'].append((knn_decoding_accuracy(cs, stim_labels), 0.0))
                lines[sname]['r2'].append((procrustes_r2(coords_full, _pad(cs, coords_full)), 0.0))

        return FRACS, lines

    def _on_metric_run(self, _b):
        if self._cached_metrics is None:
            sub = self.tensor4d.transpose(0, 1, 3, 2)   # (N_nonout, S, D, T)
            self._cached_metrics = compute_dynamic_metrics(sub)
        k = max(1, round(self._metric_pct.value / 100 * len(self.nonoutliers)))
        high = (self._metric_dir.value == 'High')
        idxs = select_top_k_by_metric(self._cached_metrics, self._metric_dd.value, k, high=high)
        self._selected_idxs = set(idxs.tolist())
        self._update_selection()
        # Update decoding panel for selected subpopulation
        try:
            sub = self.tensor4d[sorted(self._selected_idxs)]
            coords, trajs = self._compute_decoding_data(sub)
            self._update_decoding_fig(coords, trajs, suffix=f'{len(self._selected_idxs)} neurons')
        except Exception as e:
            print(f'Warning: decoding update failed: {e}')

        # Run fraction sweep and render inline
        metric_name = self._metric_dd.value
        with self._sweep_out:
            clear_output(wait=True)
            print(f'Running sweep for {metric_name}...')

        try:
            fracs, lines = self._run_fraction_sweep(metric_name)
        except Exception as e:
            with self._sweep_out:
                clear_output(wait=True)
                print(f'Sweep failed: {e}')
            return

        _SWEEP_COLORS = {
            metric_name:           '#1565c0',
            f'{metric_name} [lo]': '#90caf9',
            'random':              '#607d8b',
        }
        with self._sweep_out:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))
            for ax, key, title in [
                (axes[0], 'acc', 'Decoding accuracy'),
                (axes[1], 'r2',  'Geometric fidelity (R²)'),
            ]:
                for sname, data in lines.items():
                    vals = np.array([v[0] for v in data[key]])
                    std  = np.array([v[1] for v in data[key]])
                    c    = _SWEEP_COLORS.get(sname, 'gray')
                    ls   = '--' if sname == 'random' else '-'
                    ax.plot(fracs, vals, color=c, lw=1.5, ls=ls, label=sname)
                    if sname == 'random':
                        ax.fill_between(fracs, vals - std, vals + std,
                                        color=c, alpha=0.2)
                ax.set_xlim(1.0, fracs[0])   # inverted: full pop on left
                ax.set_xlabel('Fraction', fontsize=8)
                ax.set_title(title, fontsize=8)
                ax.legend(fontsize=7, loc='lower right')
                ax.tick_params(labelsize=7)
            plt.tight_layout()
            plt.show()

    def _on_radius_change(self, _change):
        if self._tabs.selected_index != 3 or self._seed_point is None:
            return
        self._apply_radius()

    def _apply_radius(self):
        r = float(self._radius_sl.value)
        dists = np.linalg.norm(self.emb - self._seed_point[np.newaxis, :], axis=1)
        self._selected_idxs = set(int(i) for i in np.where(dists < r)[0])
        self._update_selection()

    # ── Axes / colour callbacks ───────────────────────────────────────────────

    def _on_dc_change(self, _change):
        dcs = [self._dc_x.value, self._dc_y.value, self._dc_z.value]
        with self._fig.batch_update():
            self._trace_base.x = self.emb[:, dcs[0]]
            self._trace_base.y = self.emb[:, dcs[1]]
            self._trace_base.z = self.emb[:, dcs[2]]
        self._update_selection_trace()
        if self._tabs.selected_index == 2:
            self._update_bbox_trace()

    def _on_color_change(self, change):
        colors, cscale = self._colors_for_mode(change['new'])
        with self._fig.batch_update():
            self._trace_base.marker.color = colors
            self._trace_base.marker.colorscale = cscale

    # ── Clear / copy callbacks ────────────────────────────────────────────────

    def _on_clear(self, _btn):
        self._selected_idxs.clear()
        self._update_selection()
        with self._psth_out:
            clear_output()
        # Reset decoding to full population
        if self._decoding_full_coords is not None:
            self._update_decoding_fig(
                self._decoding_full_coords,
                self._decoding_full_trajs,
                suffix='full pop.',
            )

    def _on_copy(self, _btn):
        full_idxs = sorted(int(self.nonoutliers[i]) for i in self._selected_idxs)
        js = (
            f"navigator.clipboard.writeText('{full_idxs}')"
            ".then(()=>console.log('Copied to clipboard'));"
        )
        display(Javascript(js))

    # ── Selection update ──────────────────────────────────────────────────────

    def _update_selection(self):
        self._update_selection_trace()
        idxs = sorted(self._selected_idxs)
        full = [int(self.nonoutliers[i]) for i in idxs]
        preview = str(full[:10]) + ('...' if len(full) > 10 else '')
        self._info_lbl.value = (
            f'Selected: <b>{len(idxs)}</b> neurons  '
            f'[tensor4d idxs: {preview}]'
        )

    def _update_selection_trace(self):
        dcs  = [self._dc_x.value, self._dc_y.value, self._dc_z.value]
        idxs = sorted(self._selected_idxs)
        if not idxs:
            with self._fig.batch_update():
                self._trace_sel.x = []
                self._trace_sel.y = []
                self._trace_sel.z = []
                self._trace_sel.text = []
        else:
            sel = self.emb[idxs]
            hover = [f'nonout={i}  orig={self.nonoutliers[i]}' for i in idxs]
            with self._fig.batch_update():
                self._trace_sel.x = sel[:, dcs[0]]
                self._trace_sel.y = sel[:, dcs[1]]
                self._trace_sel.z = sel[:, dcs[2]]
                self._trace_sel.text = hover

    # ── PSTH plotting ─────────────────────────────────────────────────────────

    def _plot_psths(self):
        idxs = sorted(self._selected_idxs)
        with self._psth_out:
            clear_output(wait=True)
            if not idxs:
                print('No neurons selected.')
                # Reset decoding to full population
                if self._decoding_full_coords is not None:
                    self._update_decoding_fig(
                        self._decoding_full_coords,
                        self._decoding_full_trajs,
                        suffix='full pop.',
                    )
                return
            sub = self.tensor4d[idxs]     # (k, N_STIM, T, N_DIR)
            if len(idxs) <= 5:
                _plot_individual_psths(sub, idxs, self.categories)
            else:
                _plot_averaged_psths(sub.mean(axis=0), self.categories)
            plt.show()
        # Update decoding panel for the selected subpopulation
        try:
            coords, trajs = self._compute_decoding_data(sub)
            self._update_decoding_fig(
                coords, trajs,
                suffix=f'{len(idxs)} neurons',
            )
        except Exception as e:
            print(f'Warning: decoding update failed: {e}')

    # ── Public API ────────────────────────────────────────────────────────────

    def show(self):
        """Display the interactive explorer widget."""
        display(self._root)

    @property
    def selected_indices(self):
        """Currently selected indices in nonoutlier space (sorted list)."""
        return sorted(self._selected_idxs)

    @property
    def selected_full_indices(self):
        """Currently selected indices in full-population space (sorted list)."""
        return sorted(int(self.nonoutliers[i]) for i in self._selected_idxs)
