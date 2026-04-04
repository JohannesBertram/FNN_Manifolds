"""Load saved results and produce summary table + figures.

Run with:
    cd experiments/fgw_generalization
    ../../.venv/bin/python analyze.py

Reads:  results/metrics.json, results/embeddings.npz, results/teacher_Cs.npz
Writes: results/figures/*.png
"""

import os
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, 'results')
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================================
# Load data
# ============================================================================
with open(os.path.join(RESULTS_DIR, 'metrics.json')) as f:
    rows = json.load(f)

emb  = np.load(os.path.join(RESULTS_DIR, 'embeddings.npz'), allow_pickle=False)
tCs  = np.load(os.path.join(RESULTS_DIR, 'teacher_Cs.npz'), allow_pickle=False)


# ============================================================================
# Helper: aggregate over seeds
# ============================================================================

def _filter(rows, split, model, lam):
    return [r for r in rows if r['split'] == split
            and r['model'] == model and r['lam'] == lam]


def _mean_std(rows, key):
    vals = [r[key] for r in rows if key in r]
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))


# ============================================================================
# 1. Summary table
# ============================================================================

print('=' * 90)
print(f'{"split":>6}  {"λ":>5}  {"model":>8}  '
      f'{"acc_tr":>7}  {"acc_te":>7}  '
      f'{"GW_tr":>8}  {"GW_te":>8}  '
      f'{"OSI_tr":>7}  {"OSI_te":>7}')
print('-' * 90)

for split in ['noise', 'sf']:
    lambdas_all = sorted(set(r['lam'] for r in rows if r['split'] == split))
    for lam in lambdas_all:
        for model in ['baseline', 'fgw']:
            rs = _filter(rows, split, model, lam)
            if not rs:
                continue
            acc_tr  = _mean_std(rs, 'acc_train')
            acc_te  = _mean_std(rs, 'acc_test')
            gw_tr   = _mean_std(rs, 'gw_train')
            gw_te   = _mean_std(rs, 'gw_test')
            osi_tr  = _mean_std(rs, 'osi_mean_train')
            osi_te  = _mean_std(rs, 'osi_mean_test')
            print(f'{split:>6}  {lam:>5.2f}  {model:>8}  '
                  f'{acc_tr[0]:>6.3f}±{acc_tr[1]:.2f}  '
                  f'{acc_te[0]:>6.3f}±{acc_te[1]:.2f}  '
                  f'{gw_tr[0]:>7.4f}±{gw_tr[1]:.4f}  '
                  f'{gw_te[0]:>7.4f}±{gw_te[1]:.4f}  '
                  f'{osi_tr[0]:>6.3f}±{osi_tr[1]:.3f}  '
                  f'{osi_te[0]:>6.3f}±{osi_te[1]:.3f}')
    print()

print('=' * 90)


# ============================================================================
# 2. Lambda sweep figure: GW_train and GW_test vs lambda
# ============================================================================

def plot_lambda_sweep(rows, split, fig_path):
    lambdas = sorted(set(r['lam'] for r in rows
                         if r['split'] == split and r['model'] in ('baseline', 'fgw')))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f'Lambda sweep — {split} split', fontsize=12)

    metrics_info = [
        ('gw_train',  'gw_test',  'GW distance to teacher',    'GW dist (sqrt)'),
        ('acc_train', 'acc_test', 'Task accuracy',              'Accuracy'),
        ('osi_mean_train', 'osi_mean_test', 'Mean OSI proxy', 'OSI'),
    ]

    for ax, (key_tr, key_te, title, ylabel) in zip(axes, metrics_info):
        for model, color, ls in [('baseline', '#444444', '--'),
                                  ('fgw',      '#2196F3', '-')]:
            tr_means, tr_stds, te_means, te_stds = [], [], [], []
            lam_used = []
            for lam in lambdas:
                rs = _filter(rows, split, model, lam)
                if not rs:
                    continue
                lam_used.append(lam)
                m, s = _mean_std(rs, key_tr)
                tr_means.append(m); tr_stds.append(s)
                m, s = _mean_std(rs, key_te)
                te_means.append(m); te_stds.append(s)

            if not lam_used:
                continue

            lam_used = np.array(lam_used)
            tr_means = np.array(tr_means)
            te_means = np.array(te_means)
            tr_stds  = np.array(tr_stds)
            te_stds  = np.array(te_stds)

            # Train curve (filled markers)
            ax.plot(lam_used, tr_means, color=color, ls=ls,
                    marker='o', ms=5, label=f'{model} train')
            ax.fill_between(lam_used, tr_means - tr_stds,
                            tr_means + tr_stds, alpha=0.15, color=color)
            # Test curve (open markers)
            ax.plot(lam_used, te_means, color=color, ls=ls,
                    marker='s', ms=5, mfc='none', label=f'{model} test')
            ax.fill_between(lam_used, te_means - te_stds,
                            te_means + te_stds, alpha=0.08, color=color)

        ax.set_xlabel('λ_FGW'); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f'Saved {fig_path}')


plot_lambda_sweep(rows, 'noise',
                  os.path.join(FIG_DIR, 'lambda_sweep_noise.png'))
plot_lambda_sweep(rows, 'sf',
                  os.path.join(FIG_DIR, 'lambda_sweep_sf.png'))


# ============================================================================
# 3. Encoding manifold comparison: 3×2 grid for one lambda value
# ============================================================================

def plot_encoding_manifolds(emb, rows, split, lam, fig_path):
    """3-row (teacher / baseline / FGW) × 2-col (train / test) PCA scatter."""
    key_sfx = lambda model, split2, lam2, side: (
        f'{split2}_{model}_lam{lam2}_{side}'.replace('.', 'p')
        if model != 'teacher'
        else f'{split2}_teacher_{side}'
    )

    fig, axes = plt.subplots(3, 2, figsize=(8, 10),
                             subplot_kw={'aspect': 'equal'})
    fig.suptitle(f'Encoding manifolds — {split} split  λ={lam}', fontsize=12)
    row_labels = ['Teacher', 'Baseline (CE only)', f'FGW student (λ={lam})']
    model_keys = ['teacher', 'baseline', 'fgw']

    for ri, (model, label) in enumerate(zip(model_keys, row_labels)):
        for ci, side in enumerate(['train', 'test']):
            ax = axes[ri, ci]
            if model == 'teacher':
                k = f'{split}_teacher_{side}'
            else:
                k = f'{split}_{model}_lam{lam}_{side}'.replace('.', 'p')

            if k not in emb:
                ax.set_visible(False)
                continue

            E = emb[k]          # (N, 2)
            # Color by row index as a proxy for manifold position
            col = np.arange(len(E))
            ax.scatter(E[:, 0], E[:, 1], c=col, cmap='viridis',
                       s=8, alpha=0.7, edgecolors='none')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{label}\n({side})', fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f'Saved {fig_path}')


# Pick the middle lambda for the manifold comparison
for split in ['noise', 'sf']:
    lambdas_fgw = sorted(set(
        r['lam'] for r in rows
        if r['split'] == split and r['model'] == 'fgw'))
    if lambdas_fgw:
        mid_lam = lambdas_fgw[len(lambdas_fgw) // 2]
        plot_encoding_manifolds(
            emb, rows, split, mid_lam,
            os.path.join(FIG_DIR, f'encoding_manifolds_{split}.png'))


# ============================================================================
# 4. Generalization gap bar chart (train vs test GW at each lambda)
# ============================================================================

def plot_gen_gap(rows, split, fig_path):
    lambdas_fgw = sorted(set(
        r['lam'] for r in rows
        if r['split'] == split and r['model'] == 'fgw'))
    if not lambdas_fgw:
        return

    n = len(lambdas_fgw)
    x  = np.arange(n)
    w  = 0.35

    gw_tr_fgw, gw_te_fgw = [], []
    gw_tr_base, gw_te_base = [], []
    for lam in lambdas_fgw:
        rf = _filter(rows, split, 'fgw', lam)
        rb = _filter(rows, split, 'baseline', lam)
        gw_tr_fgw.append(_mean_std(rf, 'gw_train')[0])
        gw_te_fgw.append(_mean_std(rf, 'gw_test')[0])
        gw_tr_base.append(_mean_std(rb, 'gw_train')[0])
        gw_te_base.append(_mean_std(rb, 'gw_test')[0])

    gw_tr_fgw  = np.array(gw_tr_fgw)
    gw_te_fgw  = np.array(gw_te_fgw)
    gw_tr_base = np.array(gw_tr_base)
    gw_te_base = np.array(gw_te_base)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, gw_tr_fgw,  w, label='FGW — train',    color='#2196F3', alpha=0.85)
    ax.bar(x + w/2, gw_te_fgw,  w, label='FGW — test',     color='#2196F3', alpha=0.45)
    # Overlay baseline as horizontal dashed lines
    for i, (gtr, gte) in enumerate(zip(gw_tr_base, gw_te_base)):
        ax.hlines(gtr, i - w - 0.1, i + w + 0.1,
                  colors='#444444', ls='--', lw=1.2)
        ax.hlines(gte, i - w - 0.1, i + w + 0.1,
                  colors='#444444', ls=':', lw=1.2)

    legend_extra = [
        Line2D([0], [0], color='#444444', ls='--', lw=1.2, label='Baseline — train'),
        Line2D([0], [0], color='#444444', ls=':',  lw=1.2, label='Baseline — test'),
    ]
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles + legend_extra, labels_leg + ['Baseline — train', 'Baseline — test'],
              fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'λ={l}' for l in lambdas_fgw])
    ax.set_ylabel('GW distance to teacher'); ax.set_xlabel('FGW weight λ')
    ax.set_title(f'Generalization gap — {split} split', fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f'Saved {fig_path}')


plot_gen_gap(rows, 'noise', os.path.join(FIG_DIR, 'gen_gap_noise.png'))
plot_gen_gap(rows, 'sf',    os.path.join(FIG_DIR, 'gen_gap_sf.png'))


# ============================================================================
# 5. OSI distribution histograms (teacher vs baseline vs FGW, train vs test)
# ============================================================================

# This figure uses the per-seed raw values (just print means; histograms would need
# re-running evaluation — aggregate scalars suffice here)

print('\nAll figures saved to', FIG_DIR)
print('Analysis complete.')
