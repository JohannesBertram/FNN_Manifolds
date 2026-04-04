"""Main entry point for the FGW generalization experiment.

Run with:
    cd experiments/fgw_generalization
    ../../.venv/bin/python run.py

Outputs are saved to results/ :
    metrics.json          — all scalar results indexed by (split, lambda, seed, model)
    learning_curves.json  — epoch-level losses for every run
    embeddings.npz        — PCA 2-D encoding manifold coords for inspection
    teacher_Cs.npz        — teacher structural cost matrices (train/test for both splits)
"""

import os
import sys
import json
import time
import copy

import numpy as np
import torch

# ── Locate repo root and add experiment dir to path ──────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config   import CONFIG
from data     import make_gratings, noise_split, sf_interp_split
from model    import MLP
from train    import train_teacher, train_baseline, train_fgw_student
from evaluate import (compute_encoding_manifold, compute_all_metrics,
                      pca_embedding, accuracy)

RESULTS_DIR  = os.path.join(_HERE, 'results')
MODELS_DIR   = os.path.join(RESULTS_DIR, 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

C = CONFIG


def to_tensor(X, y):
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# ============================================================================
# 1. Generate data
# ============================================================================
print('Generating gratings ...')
X_all, y_all, rep_ids, sf_ids = make_gratings(
    n_ori=C['n_ori'], n_freq=C['n_freq'], n_reps=C['n_reps_total'],
    grid=C['grid'], noise=C['noise'], freqs=C['freqs'], seed=0,
)

# Noise split  (train: reps 0..n_reps_train-1 / test: rest)
X_n_tr, y_n_tr, X_n_te, y_n_te = noise_split(
    X_all, y_all, rep_ids, sf_ids, C['n_reps_train'])

# SF interpolation split (train: even-indexed SFs / test: odd-indexed SFs)
X_s_tr, y_s_tr, X_s_te, y_s_te = sf_interp_split(X_all, y_all, rep_ids, sf_ids)

print(f'  Noise split — train: {X_n_tr.shape[0]}  test: {X_n_te.shape[0]}')
print(f'  SF interp   — train: {X_s_tr.shape[0]}  test: {X_s_te.shape[0]}')

# Convert all to tensors
Xall_t, yall_t       = to_tensor(X_all, y_all)
Xn_tr_t, yn_tr_t     = to_tensor(X_n_tr, y_n_tr)
Xn_te_t, yn_te_t     = to_tensor(X_n_te, y_n_te)
Xs_tr_t, ys_tr_t     = to_tensor(X_s_tr, y_s_tr)
Xs_te_t, ys_te_t     = to_tensor(X_s_te, y_s_te)


# ============================================================================
# 2. Train teacher (on ALL data)
# ============================================================================
print('\nTraining teacher ...')
torch.manual_seed(C['teacher_seed'])
teacher = MLP(C['input_dim'], C['h1_dim'], C['h2_dim'], C['output_dim'])
log_teacher = train_teacher(
    teacher, Xall_t, yall_t, n_ori=C['n_ori'],
    n_epochs=C['n_epochs_teacher'], lr=C['lr_teacher'],
    lam_sel=C['lam_sel'], lam_div=C['lam_div'],
)
acc_teacher = accuracy(teacher, Xall_t, yall_t)
print(f'  Teacher accuracy (all): {acc_teacher:.3f}')

# Teacher encoding manifolds on each split's train/test subsets
print('  Computing teacher manifolds ...')
C_t_n_tr, tun_t_n_tr, osi_t_n_tr, _ = compute_encoding_manifold(
    teacher, Xn_tr_t, yn_tr_t, C['n_ori'])
C_t_n_te, tun_t_n_te, osi_t_n_te, _ = compute_encoding_manifold(
    teacher, Xn_te_t, yn_te_t, C['n_ori'])
C_t_s_tr, tun_t_s_tr, osi_t_s_tr, _ = compute_encoding_manifold(
    teacher, Xs_tr_t, ys_tr_t, C['n_ori'])
C_t_s_te, tun_t_s_te, osi_t_s_te, _ = compute_encoding_manifold(
    teacher, Xs_te_t, ys_te_t, C['n_ori'])

torch.save(teacher.state_dict(), os.path.join(MODELS_DIR, 'teacher.pt'))
print(f'  Teacher OSI (noise train/test): {osi_t_n_tr.mean():.3f} / {osi_t_n_te.mean():.3f}')
print(f'  Teacher OSI (SF    train/test): {osi_t_s_tr.mean():.3f} / {osi_t_s_te.mean():.3f}')

# Save teacher manifolds for later analysis
np.savez(os.path.join(RESULTS_DIR, 'teacher_Cs.npz'),
         C_noise_train=C_t_n_tr, C_noise_test=C_t_n_te,
         C_sf_train=C_t_s_tr,    C_sf_test=C_t_s_te,
         tuning_noise_train=tun_t_n_tr, tuning_noise_test=tun_t_n_te,
         tuning_sf_train=tun_t_s_tr,    tuning_sf_test=tun_t_s_te)
print('  Teacher manifolds saved.')


# ============================================================================
# 3. Lambda sweep across both splits
# ============================================================================

all_metrics      = []   # list of dicts (one per run)
all_curves       = []   # list of dicts with learning curves
embedding_store  = {}   # key → 2D PCA embedding (numpy)

splits = [
    ('noise', Xn_tr_t, yn_tr_t, Xn_te_t, yn_te_t,
     C_t_n_tr, tun_t_n_tr, C_t_n_te),
    ('sf',    Xs_tr_t, ys_tr_t, Xs_te_t, ys_te_t,
     C_t_s_tr, tun_t_s_tr, C_t_s_te),
]

for split_name, X_tr_t, y_tr_t, X_te_t, y_te_t, C_t_tr, tun_t_tr, C_t_te in splits:
    print(f'\n══ Split: {split_name} ══')

    for lam in C['lambda_fgw_sweep']:
        for seed in range(C['n_seeds']):
            t0 = time.time()

            # ── Baseline ────────────────────────────────────────────────────
            torch.manual_seed(seed)
            base = MLP(C['input_dim'], C['h1_dim'], C['h2_dim'], C['output_dim'])
            log_b = train_baseline(base, X_tr_t, y_tr_t,
                                   n_epochs=C['n_epochs'], lr=C['lr'])
            m_b = compute_all_metrics(
                base, X_tr_t, y_tr_t, X_te_t, y_te_t,
                C_t_tr, C_t_te, C['n_ori'], C['gw_epsilon'])

            row_b = dict(split=split_name, model='baseline', lam=lam, seed=seed,
                         **{k: v for k, v in m_b.items() if not hasattr(v, '__len__')})
            all_metrics.append(row_b)
            all_curves.append(dict(split=split_name, model='baseline',
                                   lam=lam, seed=seed, **log_b))
            # Save weights (seed 0 only to save disk)
            if seed == 0:
                torch.save(base.state_dict(),
                           os.path.join(MODELS_DIR,
                                        f'{split_name}_baseline_lam{lam}_s{seed}.pt'))

            # Store embedding for one representative seed
            if seed == 0:
                emb_b_tr = pca_embedding(m_b['C_train'])
                emb_b_te = pca_embedding(m_b['C_test'])
                embedding_store[f'{split_name}_baseline_lam{lam}_train'] = emb_b_tr
                embedding_store[f'{split_name}_baseline_lam{lam}_test']  = emb_b_te

            # ── FGW student (only if lam > 0) ────────────────────────────────
            if lam > 0:
                torch.manual_seed(seed)
                fgw = MLP(C['input_dim'], C['h1_dim'], C['h2_dim'], C['output_dim'])
                log_f = train_fgw_student(
                    fgw, X_tr_t, y_tr_t,
                    C_t_tr, tun_t_tr,
                    n_ori=C['n_ori'],
                    n_epochs=C['n_epochs'], lr=C['lr'],
                    lam_fgw=lam, alpha=C['alpha_fgw'],
                    T_update_every=C['T_update_every'],
                )
                m_f = compute_all_metrics(
                    fgw, X_tr_t, y_tr_t, X_te_t, y_te_t,
                    C_t_tr, C_t_te, C['n_ori'], C['gw_epsilon'])

                row_f = dict(split=split_name, model='fgw', lam=lam, seed=seed,
                             **{k: v for k, v in m_f.items() if not hasattr(v, '__len__')})
                all_metrics.append(row_f)
                all_curves.append(dict(split=split_name, model='fgw',
                                       lam=lam, seed=seed, **log_f))

                if seed == 0:
                    torch.save(fgw.state_dict(),
                               os.path.join(MODELS_DIR,
                                            f'{split_name}_fgw_lam{lam}_s{seed}.pt'))
                    emb_f_tr = pca_embedding(m_f['C_train'])
                    emb_f_te = pca_embedding(m_f['C_test'])
                    embedding_store[f'{split_name}_fgw_lam{lam}_train'] = emb_f_tr
                    embedding_store[f'{split_name}_fgw_lam{lam}_test']  = emb_f_te

            elapsed = time.time() - t0
            gw_tr_b = row_b['gw_train']
            gw_te_b = row_b['gw_test']
            if lam > 0:
                gw_tr_f = row_f['gw_train']
                gw_te_f = row_f['gw_test']
                print(f'  {split_name}  λ={lam:.2f}  seed={seed}'
                      f'  base GW tr/te={gw_tr_b:.4f}/{gw_te_b:.4f}'
                      f'  fgw  GW tr/te={gw_tr_f:.4f}/{gw_te_f:.4f}'
                      f'  [{elapsed:.0f}s]')
            else:
                print(f'  {split_name}  λ={lam:.2f}  seed={seed}'
                      f'  base GW tr/te={gw_tr_b:.4f}/{gw_te_b:.4f}'
                      f'  [{elapsed:.0f}s]')

# ── Also store teacher embeddings ────────────────────────────────────────────
emb_t_n_tr = pca_embedding(C_t_n_tr)
emb_t_n_te = pca_embedding(C_t_n_te)
emb_t_s_tr = pca_embedding(C_t_s_tr)
emb_t_s_te = pca_embedding(C_t_s_te)
embedding_store['noise_teacher_train'] = emb_t_n_tr
embedding_store['noise_teacher_test']  = emb_t_n_te
embedding_store['sf_teacher_train']    = emb_t_s_tr
embedding_store['sf_teacher_test']     = emb_t_s_te


# ============================================================================
# 4. Save results
# ============================================================================
metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(all_metrics, f, indent=2)
print(f'\nSaved {len(all_metrics)} metric rows → {metrics_path}')

curves_path = os.path.join(RESULTS_DIR, 'learning_curves.json')
with open(curves_path, 'w') as f:
    json.dump(all_curves, f, indent=2)
print(f'Saved {len(all_curves)} curve entries → {curves_path}')

np.savez(os.path.join(RESULTS_DIR, 'embeddings.npz'),
         **{k.replace('.', 'p'): v for k, v in embedding_store.items()})
print(f'Saved {len(embedding_store)} embedding arrays → results/embeddings.npz')

print('\nDone. Run `analyze.py` to generate figures and summary tables.')
