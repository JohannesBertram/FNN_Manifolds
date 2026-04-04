"""FGW generalization toward a biological V1 teacher (grating stimuli only).

Uses the grating subset of V1 recordings so that both teacher and student
manifolds are computed from the same stimulus domain (sinusoidal gratings).

V1 tensor4d: (637, 6, 8, 135) — neurons × stimulus_category × direction × time
Stimulus ordering (from notebooks/11_biologically_constrained_model.ipynb):
  index 0: grat_W1  (LF drifting grating, 8 directions)
  index 1: grat_W2  (HF drifting grating, 8 directions)
  indices 2-5: dot-flow stimuli (excluded)

We extract indices [0, 1], time-average → (637, 2, 8), flatten → (637, 16).
This gives a grating-domain response profile per neuron directly comparable to
the student's grating activations. The 8 directions of the drifting grating map
to the same orientation space as the student's 8 orientation classes.

Orientation tuning: mean across the 2 grating SF conditions → (637, 8), L2-norm.
This is semantically identical to the student tuning returned by
compute_encoding_manifold (mean response per class, L2-normalized).

Run with:
    cd experiments/fgw_generalization
    ../../.venv/bin/python run_bio.py

Outputs saved to results/bio_metrics.json, results/bio_teacher_C.npz
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config   import CONFIG
from data     import make_gratings, noise_split, sf_interp_split
from model    import MLP
from train    import train_baseline, train_fgw_student
from evaluate import compute_encoding_manifold, compute_all_metrics, gw_distance, pca_embedding, accuracy

RESULTS  = os.path.join(_HERE, 'results')
MODELS   = os.path.join(RESULTS, 'models')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(MODELS,  exist_ok=True)

C = CONFIG
DATA_DIR = os.path.join(_HERE, '..', '..', 'data', 'sampled')


# ── 1. Load & preprocess V1 data (grating stimuli only) ──────────────────────
print('Loading V1 data ...')
tensor4d = np.load(os.path.join(DATA_DIR, 'tensor4d_V1.npy'))  # (637, 6, 8, 135)
n_neurons, n_stims, n_dirs, n_time = tensor4d.shape
print(f'  V1 tensor shape: {tensor4d.shape}')

# Use only grating stimuli: index 0 = grat_W1 (LF), index 1 = grat_W2 (HF)
# dot-flow stimuli (indices 2-5) are excluded — different stimulus domain
GRATING_IDX = [0, 1]
R_grat = tensor4d[:, GRATING_IDX, :, :].mean(axis=3)  # (637, 2, 8), time-avg
R_flat = R_grat.reshape(n_neurons, -1).astype(np.float32)  # (637, 16)
print(f'  Using grating stimuli {GRATING_IDX} → response matrix: {R_flat.shape}')

# Cosine distance matrix C_V1_full (637×637) on grating responses
R_t   = torch.tensor(R_flat)
R_norm = F.normalize(R_t, dim=1)
C_V1_full = (1 - R_norm @ R_norm.T).clamp(min=0).numpy().astype(np.float32)
np.fill_diagonal(C_V1_full, 0.0)

# Orientation tuning: average across the 2 grating SF conditions → (637, 8)
# Each of the 8 columns corresponds to a drifting-grating direction, which maps
# to an orientation class — directly comparable to the student's n_ori=8 tuning.
tun_V1_raw = R_grat.mean(axis=1)          # (637, 8) — mean across LF/HF
tun_V1_raw = np.clip(tun_V1_raw, 0, None)
tun_V1_full = (tun_V1_raw /
               (np.linalg.norm(tun_V1_raw, axis=1, keepdims=True) + 1e-8)
               ).astype(np.float32)       # L2-normalized, same as compute_encoding_manifold

# OSI from grating responses (for reporting)
thetas  = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
R_norm_sum = tun_V1_full / (tun_V1_full.sum(axis=1, keepdims=True) + 1e-8)
OSI_v1 = np.abs((R_norm_sum * np.exp(1j * thetas)).sum(axis=1)).astype(np.float32)
print(f'  V1 OSI (grating): mean={OSI_v1.mean():.3f}  median={np.median(OSI_v1):.3f}')

# Subsample V1 to N_BIO neurons via k-means on the grating-domain C matrix
N_BIO = 150
print(f'\nSubsampling V1 to {N_BIO} neurons via k-means ...')
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA

G    = -0.5 * (C_V1_full ** 2)
kpca = KernelPCA(n_components=10, kernel='precomputed')
emb_v1 = kpca.fit_transform(G)  # (637, 10)

km     = KMeans(n_clusters=N_BIO, random_state=42, n_init=5)
labels = km.fit_predict(emb_v1)

sub_indices = []
for k in range(N_BIO):
    cluster_mask = (labels == k)
    cluster_idx  = np.where(cluster_mask)[0]
    if len(cluster_idx) == 0:
        continue
    center = emb_v1[cluster_mask].mean(axis=0)
    dists  = np.linalg.norm(emb_v1[cluster_idx] - center, axis=1)
    sub_indices.append(cluster_idx[dists.argmin()])

sub_indices = np.array(sorted(sub_indices))
print(f'  Selected {len(sub_indices)} representative neurons')

# Subsampled C and tuning — both in the grating domain, matching the student
C_V1     = C_V1_full[np.ix_(sub_indices, sub_indices)].astype(np.float32)
tun_V1   = tun_V1_full[sub_indices]                    # (N_BIO, 8) — real tuning
OSI_v1_sub = OSI_v1[sub_indices]

print(f'  C_V1 subsampled: {C_V1.shape}  max={C_V1.max():.3f}')
print(f'  tun_V1 shape: {tun_V1.shape}  (8-d orientation tuning, L2-normalized)')

np.savez(os.path.join(RESULTS, 'bio_teacher_C.npz'),
         C_V1_full=C_V1_full, C_V1_sub=C_V1,
         tun_V1_full=tun_V1_full, tun_V1_sub=tun_V1,
         OSI_v1_full=OSI_v1, OSI_v1_sub=OSI_v1_sub,
         sub_indices=sub_indices)
print('  Saved bio_teacher_C.npz')


# ── 2. Generate grating data (noise split only) ───────────────────────────────
print('\nGenerating gratings (noise split) ...')
X_all, y_all, rep_ids, sf_ids = make_gratings(
    n_ori=C['n_ori'], n_freq=C['n_freq'], n_reps=C['n_reps_total'],
    grid=C['grid'], noise=C['noise'], freqs=C['freqs'], seed=0,
)
X_tr, y_tr, X_te, y_te = noise_split(X_all, y_all, rep_ids, sf_ids, C['n_reps_train'])

def to_tensor(X, y):
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

Xtr_t, ytr_t = to_tensor(X_tr, y_tr)
Xte_t, yte_t = to_tensor(X_te, y_te)

print(f'  train: {X_tr.shape[0]}  test: {X_te.shape[0]}')


# ── 3. Lambda sweep ───────────────────────────────────────────────────────────
all_metrics = []
all_curves  = []

# Also compare against the GW distance to the *full* V1 (using only train stims)
# for interpretability: does alignment to sub-V1 generalize to test stims?

for lam in C['lambda_fgw_sweep']:
    for seed in range(C['n_seeds']):
        t0 = time.time()

        # Baseline
        torch.manual_seed(seed)
        base = MLP(C['input_dim'], C['h1_dim'], C['h2_dim'], C['output_dim'])
        log_b = train_baseline(base, Xtr_t, ytr_t, n_epochs=C['n_epochs'], lr=C['lr'])

        C_b_tr, _, osi_b_tr, _ = compute_encoding_manifold(base, Xtr_t, ytr_t, C['n_ori'])
        C_b_te, _, osi_b_te, _ = compute_encoding_manifold(base, Xte_t, yte_t, C['n_ori'])

        row_b = dict(
            split='bio_noise', model='baseline', lam=lam, seed=seed,
            acc_train  = accuracy(base, Xtr_t, ytr_t),
            acc_test   = accuracy(base, Xte_t, yte_t),
            gw_train   = gw_distance(C_b_tr, C_V1, epsilon=C['gw_epsilon']),
            gw_test    = gw_distance(C_b_te, C_V1, epsilon=C['gw_epsilon']),
            osi_mean_train  = float(osi_b_tr.mean()),
            osi_mean_test   = float(osi_b_te.mean()),
        )
        all_metrics.append(row_b)
        all_curves.append(dict(split='bio_noise', model='baseline', lam=lam, seed=seed, **log_b))

        # FGW student (only if lam > 0)
        if lam > 0:
            torch.manual_seed(seed)
            fgw = MLP(C['input_dim'], C['h1_dim'], C['h2_dim'], C['output_dim'])
            log_f = train_fgw_student(
                fgw, Xtr_t, ytr_t,
                C_V1, tun_V1,
                n_ori=C['n_ori'],
                n_epochs=C['n_epochs'], lr=C['lr'],
                lam_fgw=lam, alpha=C['alpha_fgw'],
                T_update_every=C['T_update_every'],
            )
            C_f_tr, _, osi_f_tr, _ = compute_encoding_manifold(fgw, Xtr_t, ytr_t, C['n_ori'])
            C_f_te, _, osi_f_te, _ = compute_encoding_manifold(fgw, Xte_t, yte_t, C['n_ori'])

            row_f = dict(
                split='bio_noise', model='fgw', lam=lam, seed=seed,
                acc_train  = accuracy(fgw, Xtr_t, ytr_t),
                acc_test   = accuracy(fgw, Xte_t, yte_t),
                gw_train   = gw_distance(C_f_tr, C_V1, epsilon=C['gw_epsilon']),
                gw_test    = gw_distance(C_f_te, C_V1, epsilon=C['gw_epsilon']),
                osi_mean_train  = float(osi_f_tr.mean()),
                osi_mean_test   = float(osi_f_te.mean()),
            )
            all_metrics.append(row_f)
            all_curves.append(dict(split='bio_noise', model='fgw', lam=lam, seed=seed, **log_f))

        elapsed = time.time() - t0
        gw_tr_b = row_b['gw_train'];  gw_te_b = row_b['gw_test']
        if lam > 0:
            gw_tr_f = row_f['gw_train'];  gw_te_f = row_f['gw_test']
            print(f'  λ={lam:.2f} s={seed}  base GW tr/te={gw_tr_b:.4f}/{gw_te_b:.4f}'
                  f'  fgw GW tr/te={gw_tr_f:.4f}/{gw_te_f:.4f}  [{elapsed:.0f}s]')
        else:
            print(f'  λ={lam:.2f} s={seed}  base GW tr/te={gw_tr_b:.4f}/{gw_te_b:.4f}'
                  f'  [{elapsed:.0f}s]')

# ── 4. Save ───────────────────────────────────────────────────────────────────
out_metrics = os.path.join(RESULTS, 'bio_metrics.json')
with open(out_metrics, 'w') as f:
    json.dump(all_metrics, f, indent=2)
print(f'\nSaved {len(all_metrics)} rows → {out_metrics}')

out_curves = os.path.join(RESULTS, 'bio_curves.json')
with open(out_curves, 'w') as f:
    json.dump(all_curves, f, indent=2)
print(f'Saved {len(all_curves)} curve entries → {out_curves}')


# ── 5. Quick summary figure ───────────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

lambdas_fgw = sorted(set(r['lam'] for r in all_metrics if r['model'] == 'fgw'))
if lambdas_fgw:
    gw_tr_fgw, gw_te_fgw = [], []
    gw_tr_base, gw_te_base = [], []

    def _mean(rows, model, lam, key):
        vs = [r[key] for r in rows if r['model'] == model and r['lam'] == lam]
        return float(np.mean(vs)) if vs else np.nan

    for lam in lambdas_fgw:
        gw_tr_fgw.append(_mean(all_metrics, 'fgw',      lam, 'gw_train'))
        gw_te_fgw.append(_mean(all_metrics, 'fgw',      lam, 'gw_test'))
        gw_tr_base.append(_mean(all_metrics, 'baseline', lam, 'gw_train'))
        gw_te_base.append(_mean(all_metrics, 'baseline', lam, 'gw_test'))

    x = np.arange(len(lambdas_fgw))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, gw_tr_fgw, w, label='FGW — train',  color='#2196F3', alpha=0.85)
    ax.bar(x + w/2, gw_te_fgw, w, label='FGW — test',   color='#2196F3', alpha=0.45)
    for i, (gtr, gte) in enumerate(zip(gw_tr_base, gw_te_base)):
        ax.hlines(gtr, i-w-0.1, i+w+0.1, colors='#444444', ls='--', lw=1.2)
        ax.hlines(gte, i-w-0.1, i+w+0.1, colors='#444444', ls=':',  lw=1.2)

    from matplotlib.lines import Line2D
    extra = [Line2D([0],[0],color='#444444',ls='--',lw=1.2,label='Baseline — train'),
             Line2D([0],[0],color='#444444',ls=':',lw=1.2,label='Baseline — test')]
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles+extra, labels_leg+['Baseline — train','Baseline — test'], fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'λ={l}' for l in lambdas_fgw])
    ax.set_ylabel(f'GW distance to V1 ({N_BIO} neurons)')
    ax.set_title('FGW → biological V1 teacher (noise split)', fontsize=11)
    plt.tight_layout()
    fig_out = os.path.join(RESULTS, 'figures', 'bio_gen_gap.png')
    plt.savefig(fig_out, dpi=150)
    plt.close()
    print(f'Saved {fig_out}')

print('\nDone.')
