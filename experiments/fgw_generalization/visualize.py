"""Publication-quality sorted C-matrix heatmaps + preferred-orientation scatter.

Run with:
    cd experiments/fgw_generalization
    ../../.venv/bin/python visualize.py

Reads:  results/teacher_Cs.npz, results/models/*.pt, results/metrics.json
Writes: results/figures/c_matrices_noise.png
        results/figures/c_matrices_sf.png
        results/figures/pref_ori_scatter.png
"""

import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config   import CONFIG
from model    import MLP
from data     import make_gratings, noise_split, sf_interp_split
from evaluate import compute_encoding_manifold, pca_embedding

C       = CONFIG
RESULTS = os.path.join(_HERE, 'results')
MODELS  = os.path.join(RESULTS, 'models')
FIGS    = os.path.join(RESULTS, 'figures')
os.makedirs(FIGS, exist_ok=True)

_PALETTE = ['#0070C0', '#00B0F0', '#00B050', '#92D050',
            '#FF0000', '#FFC000', '#7030A0', '#FF6600']


# ── Re-generate data (deterministic) ────────────────────────────────────────
print('Regenerating gratings ...')
X_all, y_all, rep_ids, sf_ids = make_gratings(
    n_ori=C['n_ori'], n_freq=C['n_freq'], n_reps=C['n_reps_total'],
    grid=C['grid'], noise=C['noise'], freqs=C['freqs'], seed=0,
)
X_n_tr, y_n_tr, X_n_te, y_n_te = noise_split(X_all, y_all, rep_ids, sf_ids, C['n_reps_train'])
X_s_tr, y_s_tr, X_s_te, y_s_te = sf_interp_split(X_all, y_all, rep_ids, sf_ids)

def to_tensor(X, y):
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

Xn_tr_t, yn_tr_t = to_tensor(X_n_tr, y_n_tr)
Xn_te_t, yn_te_t = to_tensor(X_n_te, y_n_te)
Xs_tr_t, ys_tr_t = to_tensor(X_s_tr, y_s_tr)
Xs_te_t, ys_te_t = to_tensor(X_s_te, y_s_te)


# ── Load teacher manifolds ───────────────────────────────────────────────────
tCs      = np.load(os.path.join(RESULTS, 'teacher_Cs.npz'))
C_t_n_tr = tCs['C_noise_train'];  tun_t_n_tr = tCs['tuning_noise_train']
C_t_n_te = tCs['C_noise_test'];   tun_t_n_te = tCs['tuning_noise_test']
C_t_s_tr = tCs['C_sf_train'];     tun_t_s_tr = tCs['tuning_sf_train']
C_t_s_te = tCs['C_sf_test'];      tun_t_s_te = tCs['tuning_sf_test']


# ── Pick the visualization lambda ────────────────────────────────────────────
lam_vis = 0.1      # mid sweep value


def load_model(path):
    m = MLP(C['input_dim'], C['h1_dim'], C['h2_dim'], C['output_dim'])
    m.load_state_dict(torch.load(path, weights_only=True))
    m.eval()
    return m


def sorted_C(C_mat, tun):
    order = np.argsort(tun.argmax(axis=1))
    return C_mat[np.ix_(order, order)]


def pref_colors(tun):
    pref = tun.argmax(axis=1)
    return [_PALETTE[p % len(_PALETTE)] for p in pref]


# ── Figure 1: Sorted C-matrix heatmaps ──────────────────────────────────────
#   Rows: Teacher | Baseline | FGW(lam_vis)
#   Cols per split: train | test
#   Produces one figure per split

for split_name, Xtr, ytr, Xte, yte, C_tr, tun_tr, C_te, tun_te in [
    ('noise', Xn_tr_t, yn_tr_t, Xn_te_t, yn_te_t, C_t_n_tr, tun_t_n_tr, C_t_n_te, tun_t_n_te),
    ('sf',    Xs_tr_t, ys_tr_t, Xs_te_t, ys_te_t, C_t_s_tr, tun_t_s_tr, C_t_s_te, tun_t_s_te),
]:
    print(f'\nLoading models for {split_name} split ...')

    # Load baseline and fgw (seed 0)
    base_path = os.path.join(MODELS, f'{split_name}_baseline_lam{lam_vis}_s0.pt')
    fgw_path  = os.path.join(MODELS, f'{split_name}_fgw_lam{lam_vis}_s0.pt')

    try:
        base = load_model(base_path)
        fgw  = load_model(fgw_path)
    except FileNotFoundError as e:
        print(f'  SKIP (model not found): {e}')
        continue

    # Compute encoding manifolds for baseline + FGW on train and test
    C_b_tr, tun_b_tr, _, _ = compute_encoding_manifold(base, Xtr, ytr, C['n_ori'])
    C_b_te, tun_b_te, _, _ = compute_encoding_manifold(base, Xte, yte, C['n_ori'])
    C_f_tr, tun_f_tr, _, _ = compute_encoding_manifold(fgw,  Xtr, ytr, C['n_ori'])
    C_f_te, tun_f_te, _, _ = compute_encoding_manifold(fgw,  Xte, yte, C['n_ori'])

    rows_data = [
        ('Teacher',                  C_tr,   tun_tr,   C_te,   tun_te),
        ('Baseline (CE only)',        C_b_tr, tun_b_tr, C_b_te, tun_b_te),
        (f'FGW student (λ={lam_vis})', C_f_tr, tun_f_tr, C_f_te, tun_f_te),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(9, 12))
    fig.suptitle(f'Sorted C matrices — {split_name} split\n'
                 f'(cols sorted by preferred orientation)', fontsize=11)

    for ri, (label, Ctr_m, tun_tr_m, Cte_m, tun_te_m) in enumerate(rows_data):
        vmax = max(Ctr_m.max(), Cte_m.max())
        for ci, (C_m, tun_m, side) in enumerate([
            (Ctr_m, tun_tr_m, 'train'), (Cte_m, tun_te_m, 'test')
        ]):
            ax = axes[ri, ci]
            Cs = sorted_C(C_m, tun_m)
            im = ax.imshow(Cs, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
            ax.set_title(f'{label} — {side}', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

            # Orientation strip on top axis
            N = Cs.shape[0]
            order = np.argsort(tun_m.argmax(axis=1))
            strip_colors = [_PALETTE[tun_m[i].argmax() % len(_PALETTE)] for i in order]
            strip = np.array([[plt.matplotlib.colors.to_rgb(c) for c in strip_colors]])
            # Place strip above image
            ax_strip = ax.inset_axes([0, 1.01, 1, 0.04])
            ax_strip.imshow(strip, aspect='auto')
            ax_strip.set_xticks([]); ax_strip.set_yticks([])
            ax_strip.set_xlim(0, N)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIGS, f'c_matrices_{split_name}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


# ── Figure 2: 2D preferred-orientation scatter ───────────────────────────────
#   PCA embedding of C matrix, colored by preferred orientation
#   Rows: Teacher | Baseline | FGW   ×   Cols: noise train/test | sf train/test

print('\nBuilding pref-ori scatter figure ...')

all_panels = []
for split_name, Xtr, ytr, Xte, yte, C_tr, tun_tr, C_te, tun_te in [
    ('noise', Xn_tr_t, yn_tr_t, Xn_te_t, yn_te_t, C_t_n_tr, tun_t_n_tr, C_t_n_te, tun_t_n_te),
    ('sf',    Xs_tr_t, ys_tr_t, Xs_te_t, ys_te_t, C_t_s_tr, tun_t_s_tr, C_t_s_te, tun_t_s_te),
]:
    base_path = os.path.join(MODELS, f'{split_name}_baseline_lam{lam_vis}_s0.pt')
    fgw_path  = os.path.join(MODELS, f'{split_name}_fgw_lam{lam_vis}_s0.pt')
    try:
        base = load_model(base_path)
        fgw  = load_model(fgw_path)
    except FileNotFoundError:
        continue

    C_b_tr, tun_b_tr, _, _ = compute_encoding_manifold(base, Xtr, ytr, C['n_ori'])
    C_b_te, tun_b_te, _, _ = compute_encoding_manifold(base, Xte, yte, C['n_ori'])
    C_f_tr, tun_f_tr, _, _ = compute_encoding_manifold(fgw,  Xtr, ytr, C['n_ori'])
    C_f_te, tun_f_te, _, _ = compute_encoding_manifold(fgw,  Xte, yte, C['n_ori'])

    for label, C_m, tun_m, side in [
        ('Teacher — train',    C_tr,   tun_tr,   f'{split_name} train'),
        ('Teacher — test',     C_te,   tun_te,   f'{split_name} test'),
        ('Baseline — train',   C_b_tr, tun_b_tr, f'{split_name} train'),
        ('Baseline — test',    C_b_te, tun_b_te, f'{split_name} test'),
        (f'FGW — train',       C_f_tr, tun_f_tr, f'{split_name} train'),
        (f'FGW — test',        C_f_te, tun_f_te, f'{split_name} test'),
    ]:
        emb = pca_embedding(C_m)
        colors = pref_colors(tun_m)
        all_panels.append((f'{split_name}\n{label}', emb, colors))

n_panels = len(all_panels)
ncols = 6
nrows = (n_panels + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
axes = np.array(axes).flatten()

# Legend patches
legend_patches = [plt.matplotlib.patches.Patch(color=_PALETTE[i],
                  label=f'Ori {i}') for i in range(C['n_ori'])]

for ai, (title, emb, colors) in enumerate(all_panels):
    ax = axes[ai]
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=25, alpha=0.85, edgecolors='none')
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])

for ai in range(len(all_panels), len(axes)):
    axes[ai].set_visible(False)

fig.legend(handles=legend_patches, loc='lower right', fontsize=8, ncol=4,
           title='Preferred orientation')
fig.suptitle(f'Preferred-orientation scatter — λ={lam_vis}', fontsize=12)
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
out = os.path.join(FIGS, 'pref_ori_scatter.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {out}')

print('\nDone.')
