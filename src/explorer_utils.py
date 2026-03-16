"""
ManifoldExplorer — Interactive 3D manifold explorer with subpopulation selection
and PSTH plotting.

Usage (in encoding_manifolds.ipynb, after all analysis cells):

    from explorer_utils import ManifoldExplorer
    import numpy as np

    # Reshape tensorX (N_all, N_STIM, NDIRS*T) → (N_all, N_STIM, T, NDIRS),
    # then select nonoutliers so indexing is direct.
    tensorX_4d = np.reshape(tensorX, (len(tensorX), NSTIMS, -1, NDIRS))
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
    """Interactive 3D manifold explorer with subpopulation selection and PSTH plotting.

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
                return list(arr.astype(float)), 'Viridis'
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

        # ── 3-D Plotly FigureWidget ──────────────────────────────────────────
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
            height=500, showlegend=True,
            legend=dict(x=0, y=1),
            title=dict(text='Manifold Explorer', x=0.5),
        )
        self._fig = go.FigureWidget(
            data=[self._trace_base, self._trace_sel, self._trace_bbox],
            layout=fig_layout)
        # Reassign to live FigureWidget trace references (FigureWidget copies on init)
        self._trace_base = self._fig.data[0]
        self._trace_sel  = self._fig.data[1]
        self._trace_bbox = self._fig.data[2]
        self._fig.data[0].on_click(self._on_click)

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

        self._tabs = widgets.Tab(
            children=[click_tab, cluster_tab, bbox_tab, radius_tab])
        for i, title in enumerate(['Click', 'Cluster', 'Bounding Box', 'Radius']):
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
        self._root = widgets.VBox([
            ctrl_row,
            self._fig,
            self._tabs,
            self._info_lbl,
            action_row,
            self._psth_out,
        ])

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
                return
            sub = self.tensor4d[idxs]     # (k, N_STIM, T, N_DIR)
            if len(idxs) <= 5:
                _plot_individual_psths(sub, idxs, self.categories)
            else:
                _plot_averaged_psths(sub.mean(axis=0), self.categories)
            plt.show()

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
