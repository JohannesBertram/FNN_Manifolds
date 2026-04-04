"""Training loops: teacher, baseline (CE only), FGW-regularized student."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist

from losses import selectivity_loss, diversity_loss, linearized_fgw_loss


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------

def train_teacher(model, X, y, n_ori, n_epochs=500, lr=1e-3,
                  lam_sel=0.3, lam_div=0.1):
    """Train teacher with CE + selectivity + diversity regularizers.

    Parameters
    ----------
    model    : MLP
    X        : (N_stim, input_dim) float32 tensor — ALL stimuli
    y        : (N_stim,) int64 tensor
    n_ori    : int

    Returns
    -------
    log : dict with keys 'ce', 'sel', 'div' (lists, sampled every 20 epochs)
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = {'ce': [], 'sel': [], 'div': []}

    model.train()
    for epoch in range(n_epochs):
        out, h1 = model(X)
        l_ce  = F.cross_entropy(out, y)
        l_sel = selectivity_loss(h1, y, n_ori)
        l_div = diversity_loss(h1, y, n_ori)
        loss  = l_ce + lam_sel * l_sel + lam_div * l_div

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 20 == 0:
            log['ce'].append(l_ce.item())
            log['sel'].append(l_sel.item())
            log['div'].append(l_div.item())

    return log


# ---------------------------------------------------------------------------
# Baseline student (CE only)
# ---------------------------------------------------------------------------

def train_baseline(model, X_train, y_train, n_epochs=500, lr=1e-3):
    """Train student with cross-entropy only.

    Parameters
    ----------
    X_train : (N_train, input_dim) float32 tensor
    y_train : (N_train,) int64 tensor

    Returns
    -------
    log : dict with key 'ce'
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = {'ce': []}

    model.train()
    for epoch in range(n_epochs):
        out, _ = model(X_train)
        loss   = F.cross_entropy(out, y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 20 == 0:
            log['ce'].append(loss.item())

    return log


# ---------------------------------------------------------------------------
# FGW-regularized student
# ---------------------------------------------------------------------------

def _solve_fgw_transport(h1_np, tuning_s_np, C_t_np, tuning_t_np,
                          p_np, q_np, alpha, n_ori):
    """Solve FGW exactly (numpy, no autograd). Returns T: (N_s, N_t) float32.

    h1_np      : (S, N_s) numpy
    tuning_s_np: (N_s, n_ori) numpy
    C_t_np     : (N_t, N_t) numpy
    tuning_t_np: (N_t, n_ori) numpy
    """
    import ot

    # Student structural cost (cosine distance on response profiles)
    A_norm = h1_np.T.copy()                             # (N_s, S)
    norms  = np.linalg.norm(A_norm, axis=1, keepdims=True) + 1e-8
    A_norm /= norms
    C_s = 1.0 - A_norm @ A_norm.T
    C_s = np.clip(C_s, 0, None).astype(np.float64)

    # Cross-feature cost (student ↔ teacher tuning vectors)
    M_cross = cdist(tuning_s_np, tuning_t_np, 'euclidean').astype(np.float64)
    mx = M_cross.max()
    if mx > 0:
        M_cross /= mx

    T = ot.gromov.fused_gromov_wasserstein(
        M_cross,
        C_s,
        C_t_np.astype(np.float64),
        p_np.astype(np.float64),
        q_np.astype(np.float64),
        loss_fun='square_loss',
        alpha=alpha,
        max_iter=200,
        log=False,
    )
    return T.astype(np.float32)


def train_fgw_student(model, X_train, y_train, C_t_np, tuning_t_np,
                       n_ori, n_epochs=500, lr=1e-3,
                       lam_fgw=0.1, alpha=0.5, T_update_every=20):
    """Train student with CE + linearized FGW manifold constraint.

    FGW aligns the student's h1 encoding manifold (computed on X_train)
    toward the teacher's encoding manifold (C_t_np / tuning_t_np).

    Parameters
    ----------
    model        : MLP
    X_train      : (N_train, input_dim) float32 tensor
    y_train      : (N_train,) int64 tensor
    C_t_np       : (N_t, N_t) float32 numpy — teacher structural cost on TRAIN stimuli
    tuning_t_np  : (N_t, n_ori) float32 numpy — teacher tuning vectors on TRAIN stimuli
    n_ori        : int
    lam_fgw      : float — FGW loss weight
    alpha        : float — structural vs feature balance (0=feature, 1=GW)
    T_update_every : int — epochs between re-solving T*

    Returns
    -------
    log : dict with keys 'ce', 'fgw', 'total'
    """
    N_s = model.fc1.out_features
    N_t = C_t_np.shape[0]
    p_np = np.ones(N_s, dtype=np.float64) / N_s
    q_np = np.ones(N_t, dtype=np.float64) / N_t

    C_t_t    = torch.tensor(C_t_np, dtype=torch.float32)
    T_fixed  = torch.tensor(np.outer(p_np, q_np), dtype=torch.float32)  # uniform init

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = {'ce': [], 'fgw': [], 'total': []}

    model.train()
    for epoch in range(n_epochs):

        # ── Refresh T* ───────────────────────────────────────────────────────
        if epoch % T_update_every == 0:
            model.eval()
            with torch.no_grad():
                h1_np = model.get_h1(X_train).numpy()   # (N_train, N_s)
            y_np = y_train.numpy()
            tuning_s_np = np.zeros((N_s, n_ori), dtype=np.float32)
            for k in range(n_ori):
                mask = (y_np == k)
                if mask.sum() > 0:
                    tuning_s_np[:, k] = h1_np[mask].mean(axis=0)
            tuning_s_np = np.clip(tuning_s_np, 0, None)
            nrm = np.linalg.norm(tuning_s_np, axis=1, keepdims=True) + 1e-8
            tuning_s_np /= nrm
            try:
                T_np    = _solve_fgw_transport(h1_np, tuning_s_np,
                                               C_t_np, tuning_t_np,
                                               p_np, q_np, alpha, n_ori)
                T_fixed = torch.tensor(T_np, dtype=torch.float32)
            except Exception:
                pass   # keep previous T_fixed if solver fails
            model.train()

        # ── Gradient step ─────────────────────────────────────────────────────
        out, h1 = model(X_train)
        l_ce  = F.cross_entropy(out, y_train)
        l_fgw = linearized_fgw_loss(h1, y_train, T_fixed, C_t_t,
                                     tuning_t_np, n_ori, alpha=alpha)
        loss  = l_ce + lam_fgw * l_fgw

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 20 == 0:
            log['ce'].append(l_ce.item())
            log['fgw'].append(l_fgw.item())
            log['total'].append(loss.item())

    return log
