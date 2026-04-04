"""Diagnose the SF-split failure: is the teacher's own manifold stable across SFs?

Run with:
    cd experiments/fgw_generalization
    ../../.venv/bin/python diagnose_sf.py
"""

import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from evaluate import gw_distance
from config   import CONFIG

C = CONFIG
RESULTS_DIR = os.path.join(_HERE, 'results')
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load teacher cost matrices ────────────────────────────────────────────────
tCs     = np.load(os.path.join(RESULTS_DIR, 'teacher_Cs.npz'))
C_n_tr  = tCs['C_noise_train']
C_n_te  = tCs['C_noise_test']
C_s_tr  = tCs['C_sf_train']
C_s_te  = tCs['C_sf_test']
tun_n_tr = tCs['tuning_noise_train']
tun_n_te = tCs['tuning_noise_test']
tun_s_tr = tCs['tuning_sf_train']
tun_s_te = tCs['tuning_sf_test']

eps = C['gw_epsilon']

# ── Teacher self-consistency ──────────────────────────────────────────────────
print('Computing teacher self-consistency ...')
gw_t_noise = gw_distance(C_n_tr, C_n_te, epsilon=eps)
gw_t_sf    = gw_distance(C_s_tr, C_s_te, epsilon=eps)
print(f'  GW(teacher noise_train, noise_test) = {gw_t_noise:.4f}')
print(f'  GW(teacher SF_train,    SF_test   ) = {gw_t_sf:.4f}')

# ── Load student results for comparison ───────────────────────────────────────
with open(os.path.join(RESULTS_DIR, 'metrics.json')) as f:
    rows = json.load(f)

def _mean(rows, split, model, lam, key):
    vs = [r[key] for r in rows
          if r['split']==split and r['model']==model and r['lam']==lam]
    return float(np.mean(vs)) if vs else np.nan

lam_hi = max(C['lambda_fgw_sweep'])
results = {}
for split in ['noise', 'sf']:
    results[split] = dict(
        base_test  = _mean(rows, split, 'baseline', 0.0,    'gw_test'),
        fgw_test   = _mean(rows, split, 'fgw',      lam_hi, 'gw_test'),
        base_train = _mean(rows, split, 'baseline', 0.0,    'gw_train'),
        fgw_train  = _mean(rows, split, 'fgw',      lam_hi, 'gw_train'),
    )
results['noise']['teacher_self'] = gw_t_noise
results['sf'   ]['teacher_self'] = gw_t_sf

# ── Preferred-orientation agreement across splits ─────────────────────────────
pref_n = tun_n_tr.argmax(axis=1)
pref_s = tun_s_tr.argmax(axis=1)
agree_noise = float((pref_n == tun_n_te.argmax(axis=1)).mean())
agree_sf    = float((pref_s == tun_s_te.argmax(axis=1)).mean())
print(f'\n  Pref-orientation agreement across splits:')
print(f'    noise: {100*agree_noise:.1f}%')
print(f'    SF:    {100*agree_sf:.1f}%')

for split in ['noise', 'sf']:
    r = results[split]
    print(f'\n  {split:5s}  base={r["base_test"]:.4f}  '
          f'fgw={r["fgw_test"]:.4f}  teacher_self={r["teacher_self"]:.4f}')

# ── Figure 1: bar chart comparing GW values ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('SF-split diagnosis: is the target manifold itself stable?', fontsize=12)

for ax, split, title in zip(axes,
                              ['noise', 'sf'],
                              ['Noise split (test = held-out noise seeds)',
                               'SF split (test = held-out spatial freqs)']):
    r = results[split]
    labels = ['Baseline', f'FGW\nλ={lam_hi}', 'Teacher\nself-dist']
    vals   = [r['base_test'], r['fgw_test'], r['teacher_self']]
    colors = ['#444444', '#2196F3', '#E91E63']
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.004,
                f'{v:.3f}', ha='center', fontsize=9)
    ax.set_ylabel('GW distance to teacher (test stimuli)')
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, max(vals) * 1.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sf_diagnosis_bar.png'), dpi=150)
plt.close()

# ── Figure 2: sorted C-matrix heatmaps — teacher on train vs test SFs ─────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Teacher C matrix: SF=1,2 (train) vs SF=3,4 (test)', fontsize=11)
for ax, C_mat, tun, title in zip(
    axes,
    [C_s_tr, C_s_te],
    [tun_s_tr, tun_s_te],
    ['Teacher — SF train (1,2)', 'Teacher — SF test (3,4)'],
):
    order = np.argsort(tun.argmax(axis=1))
    Cs = C_mat[np.ix_(order, order)]
    im = ax.imshow(Cs, cmap='viridis', aspect='auto',
                   vmin=0, vmax=C_mat.max())
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.7)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sf_diagnosis_cmats.png'), dpi=150)
plt.close()

# ── Save ──────────────────────────────────────────────────────────────────────
diag = dict(
    gw_teacher_noise_self  = gw_t_noise,
    gw_teacher_sf_self     = gw_t_sf,
    pref_agree_noise       = agree_noise,
    pref_agree_sf          = agree_sf,
    noise=results['noise'],
    sf   =results['sf'],
)
with open(os.path.join(RESULTS_DIR, 'diagnosis.json'), 'w') as f:
    json.dump(diag, f, indent=2)

# ── Interpretation ────────────────────────────────────────────────────────────
print('\n── Interpretation ──')
ratio = gw_t_sf / max(results['sf']['fgw_test'], 1e-6)
print(f'  Teacher SF self-dist / FGW SF test-dist = {ratio:.2f}')
if ratio > 0.7:
    print('  → The teacher\'s own manifold shifts substantially across SF splits.')
    print('    The SF split is an ill-posed test: FGW cannot generalize to a')
    print('    target that itself changes. This is a test-design artifact.')
else:
    print('  → Teacher is SF-stable but FGW still fails.')
    print('    FGW genuinely overfits the training SF statistics.')
