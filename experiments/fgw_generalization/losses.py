"""Loss functions: teacher regularizers and linearized FGW loss.

Adapted verbatim from notebook 10 cells 7 and 20.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Teacher regularizers
# ---------------------------------------------------------------------------

def _compute_tuning_matrix(h1, y_batch, n_ori):
    """(N_h1, n_ori) mean response per orientation class. Differentiable."""
    N = h1.shape[1]
    rows = []
    for k in range(n_ori):
        mask = (y_batch == k)
        if mask.sum() > 0:
            rows.append(h1[mask].mean(dim=0))
        else:
            rows.append(torch.zeros(N, device=h1.device))
    return torch.stack(rows, dim=1)   # (N, n_ori)


def selectivity_loss(h1, y_batch, n_ori):
    """Maximize mean circular resultant length (OSI proxy) across neurons.

    Resultant = 0 → flat tuning; resultant = 1 → perfectly selective.
    Minimizing this loss maximizes mean selectivity.
    """
    R = _compute_tuning_matrix(h1, y_batch, n_ori)   # (N, n_ori)
    R_pos  = F.relu(R) + 1e-8
    R_norm = R_pos / R_pos.sum(dim=1, keepdim=True)
    thetas = torch.linspace(0, 2 * torch.pi, n_ori + 1, device=h1.device)[:-1]
    cos_s  = (R_norm * torch.cos(thetas)).sum(dim=1)
    sin_s  = (R_norm * torch.sin(thetas)).sum(dim=1)
    resultant = torch.sqrt(cos_s ** 2 + sin_s ** 2)
    return -resultant.mean()   # minimize → maximize selectivity


def diversity_loss(h1, y_batch, n_ori, beta=5.0):
    """Penalize neurons sharing the same preferred orientation.

    Uses a soft argmax to get differentiable preferred orientations, then
    penalizes pairwise cosine similarity between preferred direction vectors.
    """
    R = _compute_tuning_matrix(h1, y_batch, n_ori)   # (N, n_ori)
    R_pos   = F.relu(R) + 1e-8
    weights = torch.softmax(R_pos * beta, dim=1)     # (N, n_ori)
    thetas  = torch.linspace(0, 2 * torch.pi, n_ori + 1, device=h1.device)[:-1]
    pcos = (weights * torch.cos(thetas)).sum(dim=1)  # (N,)
    psin = (weights * torch.sin(thetas)).sum(dim=1)  # (N,)
    pnorm = torch.sqrt(pcos ** 2 + psin ** 2 + 1e-8)
    pcos_n, psin_n = pcos / pnorm, psin / pnorm
    sim = (pcos_n.unsqueeze(1) * pcos_n.unsqueeze(0)
           + psin_n.unsqueeze(1) * psin_n.unsqueeze(0))
    mask = torch.triu(torch.ones(sim.shape[0], sim.shape[0],
                                 device=h1.device), diagonal=1).bool()
    return sim[mask].mean()   # minimize → spread preferred orientations


# ---------------------------------------------------------------------------
# Linearized FGW loss (differentiable w.r.t. student h1)
# ---------------------------------------------------------------------------

def compute_C_diff(h1):
    """Differentiable cosine distance matrix.

    Parameters
    ----------
    h1 : (S, N) activations across S stimuli for N neurons.

    Returns
    -------
    C : (N, N) cosine distance matrix, clamped ≥ 0.
    """
    A      = h1.T                       # (N, S)
    A_norm = F.normalize(A, dim=1)
    C      = 1 - A_norm @ A_norm.T
    return C.clamp(min=0)


def compute_M_cross_diff(h1_s, y, tuning_t_np, n_ori):
    """Differentiable cross-feature cost matrix (student → teacher).

    Student tuning is computed differentiably from h1_s; teacher tuning
    is a fixed numpy array.

    Returns
    -------
    M : (N_student, N_teacher) L2 distance on normalized tuning vectors, normalized to max=1.
    """
    N_s = h1_s.shape[1]

    rows = []
    for k in range(n_ori):
        mask = (y == k)
        rows.append(h1_s[mask].mean(dim=0) if mask.sum() > 0
                    else torch.zeros(N_s, device=h1_s.device))
    tuning_s = torch.stack(rows, dim=1)                     # (N_s, n_ori)
    tuning_s = F.normalize(F.relu(tuning_s) + 1e-8, dim=1)

    tuning_t = torch.tensor(tuning_t_np, dtype=torch.float32, device=h1_s.device)
    diff = tuning_s.unsqueeze(1) - tuning_t.unsqueeze(0)   # (N_s, N_t, n_ori)
    M    = diff.norm(dim=2)
    mx   = M.detach().max()
    return M / (mx + 1e-8)


def linearized_fgw_loss(h1_s, y, T_fixed, C_t_fixed, tuning_t_np, n_ori, alpha=0.5):
    """FGW loss linearized around fixed transport plan T_fixed.

    Differentiable with respect to h1_s.

    Parameters
    ----------
    h1_s       : (S, N_student) student hidden activations — has grad
    y          : (S,) stimulus class labels
    T_fixed    : (N_student, N_teacher) fixed transport plan — no grad
    C_t_fixed  : (N_teacher, N_teacher) teacher structural cost — constant
    tuning_t_np: (N_teacher, n_ori) numpy array — teacher tuning vectors
    n_ori      : int
    alpha      : float — weight for structural term

    Returns
    -------
    Scalar loss (differentiable through h1_s).
    """
    C_s     = compute_C_diff(h1_s)                              # (N_s, N_s)
    M_cross = compute_M_cross_diff(h1_s, y, tuning_t_np, n_ori)  # (N_s, N_t)

    p = T_fixed.sum(dim=1)                                       # (N_s,) row marginals
    H = T_fixed @ C_t_fixed @ T_fixed.T                         # (N_s, N_s)
    L_struct = (C_s * torch.outer(p, p) * C_s).sum() - 2 * (C_s * H).sum()
    L_feat   = (T_fixed * M_cross).sum()

    return alpha * L_struct + (1 - alpha) * L_feat
