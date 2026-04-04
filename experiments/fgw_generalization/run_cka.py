"""CKA-regularized baseline: decoding-space alignment cannot recover encoding structure.

Core claim: even when a student perfectly matches the teacher's *stimulus-space*
organization (CKA on decoding manifold), the encoding manifold (GW distance) stays
near baseline — confirming that FGW uniquely targets encoding topology.

Run with:
    cd experiments/fgw_generalization
    ../../.venv/bin/python run_cka.py

Outputs: results/cka_metrics.json, results/figures/cka_comparison.png
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config   import CONFIG
from data     import make_gratings, noise_split
from model    import MLP
from train    import train_baseline
from evaluate import compute_encoding_manifold, gw_distance, accuracy

RESULTS = os.path.join(_HERE, 'results')
os.makedirs(RESULTS, exist_ok=True)

C = CONFIG


# ── CKA loss (stimulus space) ────────────────────────────────────────────────

def _hsic(K1, K2):
    """Unbiased HSIC estimator (Gretton et al. 2012)."""
    n = K1.shape[0]
    H = torch.eye(n, device=K1.device) - 1.0 / n
    KH1 = K1 @ H
    KH2 = K2 @ H
    return (KH1 * KH2).sum() / ((n - 1) ** 2)


def decoding_cka_loss(h1, h1_teacher):
    """1 - linear CKA on stimulus kernel matrices.

    h1          : (S, N_s) student h1 activations for a stimulus batch
    h1_teacher  : (S, N_t) teacher h1 activations for the same stimuli
    Returns scalar in [0, 1]; 0 = perfect alignment.
    """
    # Stimulus kernel matrices  K = H H^T  (S×S)
    K_s = h1 @ h1.T                   # (S, S)
    K_t = h1_teacher @ h1_teacher.T   # (S, S)

    hsic_st = _hsic(K_s, K_t)
    hsic_ss = _hsic(K_s, K_s)
    hsic_tt = _hsic(K_t, K_t)
    denom   = (hsic_ss * hsic_tt).clamp(min=1e-12).sqrt()
    cka     = hsic_st / denom
    return 1.0 - cka


def train_cka_student(model, X_train, y_train, h1_teacher_all,
                      n_epochs=500, lr=1e-3, lam_cka=0.1):
    """Train MLP with CE + λ_cka * CKA_loss."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce  = nn.CrossEntropyLoss()
    log = dict(ce=[], cka=[], total=[])

    model.train()
    for ep in range(n_epochs):
        opt.zero_grad()
        logits, h1 = model(X_train)           # h1: (S, N_s)
        loss_ce  = ce(logits, y_train)
        loss_cka = decoding_cka_loss(h1, h1_teacher_all)
        loss     = loss_ce + lam_cka * loss_cka
        loss.backward()
        opt.step()

        if ep % 50 == 0:
            log['ce'].append(float(loss_ce))
            log['cka'].append(float(loss_cka))
            log['total'].append(float(loss))

    return log


def compute_decoding_cka(model, X, h1_teacher):
    """Evaluate decoding CKA on a stimulus set (lower = more aligned)."""
    model.eval()
    with torch.no_grad():
        _, h1 = model(X)
    return float(1.0 - (1.0 - decoding_cka_loss(h1, h1_teacher)))


# ── Generate data ─────────────────────────────────────────────────────────────
print('Generating gratings ...')
X_all, y_all, rep_ids, sf_ids = make_gratings(
    n_ori=C['n_ori'], n_freq=C['n_freq'], n_reps=C['n_reps_total'],
    grid=C['grid'], noise=C['noise'], freqs=C['freqs'], seed=0,
)
X_n_tr, y_n_tr, X_n_te, y_n_te = noise_split(
    X_all, y_all, rep_ids, sf_ids, C['n_reps_train'])

def to_tensor(X, y):
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

Xtr_t, ytr_t = to_tensor(X_n_tr, y_n_tr)
Xte_t, yte_t = to_tensor(X_n_te, y_n_te)
Xall_t, yall_t = to_tensor(X_all, y_all)

# Load teacher (trained during run.py)
teacher = MLP(C['input_dim'], C['h1_dim'], C['h2_dim'], C['output_dim'])
teacher_path = os.path.join(RESULTS, 'models', 'teacher.pt')
teacher.load_state_dict(torch.load(teacher_path, weights_only=True))
teacher.eval()

# Precompute teacher h1 on train and test stimuli (fixed)
with torch.no_grad():
    h1_t_tr = teacher.get_h1(Xtr_t)   # (S_tr, N_t)
    h1_t_te = teacher.get_h1(Xte_t)   # (S_te, N_t)

# Precompute teacher encoding manifolds
tCs     = np.load(os.path.join(RESULTS, 'teacher_Cs.npz'))
C_t_tr  = tCs['C_noise_train']
C_t_te  = tCs['C_noise_test']

print(f'  Train: {Xtr_t.shape[0]}  Test: {Xte_t.shape[0]}')


# ── Lambda sweep (noise split only) ──────────────────────────────────────────
all_metrics = []
all_curves  = []

# Lambdas for CKA sweep — use same values as FGW sweep (excluding 0.0)
lambda_sweep = C['lambda_fgw_sweep']

print('\n══ CKA sweep (noise split) ══')
for lam in lambda_sweep:
    for seed in range(C['n_seeds']):
        t0 = time.time()

        # ── Baseline (CE only) ────────────────────────────────────────────────
        torch.manual_seed(seed)
        base = MLP(C['input_dim'], C['h1_dim'], C['h2_dim'], C['output_dim'])
        log_b = train_baseline(base, Xtr_t, ytr_t, n_epochs=C['n_epochs'], lr=C['lr'])

        C_b_tr, _, osi_b_tr, _ = compute_encoding_manifold(base, Xtr_t, ytr_t, C['n_ori'])
        C_b_te, _, osi_b_te, _ = compute_encoding_manifold(base, Xte_t, yte_t, C['n_ori'])

        # Decoding CKA for baseline (how well does CE-only align to teacher in stim space?)
        with torch.no_grad():
            _, h1_b_tr = base(Xtr_t)
            _, h1_b_te = base(Xte_t)
        cka_b_tr = float(decoding_cka_loss(h1_b_tr, h1_t_tr))
        cka_b_te = float(decoding_cka_loss(h1_b_te, h1_t_te))

        row_b = dict(
            split='noise', model='baseline', lam=lam, seed=seed,
            acc_train        = accuracy(base, Xtr_t, ytr_t),
            acc_test         = accuracy(base, Xte_t, yte_t),
            gw_train         = gw_distance(C_b_tr, C_t_tr, epsilon=C['gw_epsilon']),
            gw_test          = gw_distance(C_b_te, C_t_te, epsilon=C['gw_epsilon']),
            decoding_cka_train = cka_b_tr,
            decoding_cka_test  = cka_b_te,
            osi_mean_train   = float(osi_b_tr.mean()),
            osi_mean_test    = float(osi_b_te.mean()),
        )
        all_metrics.append(row_b)
        all_curves.append(dict(split='noise', model='baseline', lam=lam, seed=seed, **log_b))

        # ── CKA student (only if lam > 0) ─────────────────────────────────────
        if lam > 0:
            torch.manual_seed(seed)
            cka_model = MLP(C['input_dim'], C['h1_dim'], C['h2_dim'], C['output_dim'])
            log_c = train_cka_student(
                cka_model, Xtr_t, ytr_t, h1_t_tr,
                n_epochs=C['n_epochs'], lr=C['lr'], lam_cka=lam,
            )
            C_c_tr, _, osi_c_tr, _ = compute_encoding_manifold(cka_model, Xtr_t, ytr_t, C['n_ori'])
            C_c_te, _, osi_c_te, _ = compute_encoding_manifold(cka_model, Xte_t, yte_t, C['n_ori'])

            with torch.no_grad():
                _, h1_c_tr = cka_model(Xtr_t)
                _, h1_c_te = cka_model(Xte_t)
            cka_c_tr = float(decoding_cka_loss(h1_c_tr, h1_t_tr))
            cka_c_te = float(decoding_cka_loss(h1_c_te, h1_t_te))

            row_c = dict(
                split='noise', model='cka', lam=lam, seed=seed,
                acc_train        = accuracy(cka_model, Xtr_t, ytr_t),
                acc_test         = accuracy(cka_model, Xte_t, yte_t),
                gw_train         = gw_distance(C_c_tr, C_t_tr, epsilon=C['gw_epsilon']),
                gw_test          = gw_distance(C_c_te, C_t_te, epsilon=C['gw_epsilon']),
                decoding_cka_train = cka_c_tr,
                decoding_cka_test  = cka_c_te,
                osi_mean_train   = float(osi_c_tr.mean()),
                osi_mean_test    = float(osi_c_te.mean()),
            )
            all_metrics.append(row_c)
            all_curves.append(dict(split='noise', model='cka', lam=lam, seed=seed, **log_c))

        elapsed = time.time() - t0
        if lam > 0:
            print(f'  λ={lam:.2f} s={seed}'
                  f'  base GW={row_b["gw_test"]:.4f} dCKA={cka_b_te:.4f}'
                  f'  cka  GW={row_c["gw_test"]:.4f} dCKA={cka_c_te:.4f}'
                  f'  [{elapsed:.0f}s]')
        else:
            print(f'  λ={lam:.2f} s={seed}'
                  f'  base GW={row_b["gw_test"]:.4f} dCKA={cka_b_te:.4f}'
                  f'  [{elapsed:.0f}s]')


# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(RESULTS, 'cka_metrics.json')
with open(out, 'w') as f:
    json.dump(all_metrics, f, indent=2)
print(f'\nSaved {len(all_metrics)} rows → {out}')


# ── Summary table ─────────────────────────────────────────────────────────────
print('\n' + '=' * 85)
print(f'{"λ":>5}  {"model":>8}  {"acc_te":>7}  {"GW_te":>8}  {"dCKA_te":>9}')
print('-' * 85)

def _m(rows, model, lam, key):
    vs = [r[key] for r in rows if r['model'] == model and r['lam'] == lam and key in r]
    return float(np.mean(vs)) if vs else np.nan

for lam in lambda_sweep:
    for model in ['baseline', 'cka']:
        acc = _m(all_metrics, model, lam, 'acc_test')
        gw  = _m(all_metrics, model, lam, 'gw_test')
        cka = _m(all_metrics, model, lam, 'decoding_cka_test')
        print(f'{lam:>5.2f}  {model:>8}  {acc:>7.3f}  {gw:>8.4f}  {cka:>9.4f}')
    print()
print('=' * 85)

# Also load FGW results for comparison
try:
    with open(os.path.join(RESULTS, 'metrics.json')) as f:
        fgw_rows = json.load(f)
    print('\nFGW (from run.py) for comparison:')
    for lam in C['lambda_fgw_sweep']:
        gw_fgw = _m(fgw_rows, 'fgw', lam, 'gw_test') if lam > 0 else np.nan
        if not np.isnan(gw_fgw):
            print(f'  λ={lam:.2f}  FGW GW_test={gw_fgw:.4f}')
except FileNotFoundError:
    pass


# ── Comparison figure ─────────────────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Load FGW metrics for the noise split comparison
try:
    with open(os.path.join(RESULTS, 'metrics.json')) as f:
        fgw_rows = [r for r in json.load(f) if r['split'] == 'noise']
    has_fgw = True
except FileNotFoundError:
    has_fgw = False

lambdas_nonzero = [l for l in lambda_sweep if l > 0]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('CKA vs FGW: decoding alignment ≠ encoding alignment', fontsize=12)

for ax, key, ylabel in zip(axes,
                            ['gw_test', 'decoding_cka_test'],
                            ['GW distance to teacher (encoding)', 'Decoding CKA loss (1−CKA)']):
    # Baseline (flat line — same for all λ since λ=0 baseline is what we compare)
    base_vals = [_m(all_metrics, 'baseline', l, key) for l in lambdas_nonzero]
    ax.plot(lambdas_nonzero, base_vals, color='#444444', ls='--', marker='o',
            ms=5, label='Baseline (CE only)')

    # CKA student
    cka_vals = [_m(all_metrics, 'cka', l, key) for l in lambdas_nonzero]
    ax.plot(lambdas_nonzero, cka_vals, color='#E91E63', ls='-', marker='s',
            ms=5, label='CKA student')

    # FGW student (from run.py), for encoding GW only
    if has_fgw and key == 'gw_test':
        fgw_vals = [_m(fgw_rows, 'fgw', l, 'gw_test') for l in lambdas_nonzero]
        ax.plot(lambdas_nonzero, fgw_vals, color='#2196F3', ls='-', marker='^',
                ms=5, label='FGW student')

    ax.set_xlabel('λ')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_title(ylabel, fontsize=9)

plt.tight_layout()
fig_out = os.path.join(RESULTS, 'figures', 'cka_comparison.png')
plt.savefig(fig_out, dpi=150)
plt.close()
print(f'Saved {fig_out}')

print('\nDone.')
