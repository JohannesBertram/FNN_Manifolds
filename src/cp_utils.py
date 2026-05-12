"""Python reimplementation of MATLAB's permuted CP (PARAFAC) decomposition.

Matches the algorithm in ``matlab/run_permcp.m`` / ``matlab/perm_tt_cp_fg.m``:
non-negative CP decomposition with circular-shift alignment of the response
mode.

Default solver is HALS-ALS (Hierarchical Alternating Least Squares), which is
~10× faster than the L-BFGS-B used by MATLAB while producing comparable
quality.  A ``method='lbfgsb'`` option is available for algorithm-faithful
comparison, but it is slow (>4 min/rep for N=2000).

Output format is compatible with ``src.tensor_utils.loadPreComputedCP``
so the rest of the pipeline (getPermutedTensor, getNeuralMatrix, IAN, …)
works unchanged.

Minimal public API::

    from src.cp_utils import run_permuted_cp, run_cp_sweep, save_cp_as_mat
"""

import os
import numpy as np
import scipy.optimize
from scipy.io import savemat
from tqdm import tqdm
from joblib import Parallel, delayed


# ── Factor vector packing / unpacking ─────────────────────────────────────────

def pack_factors(factors):
    """Concatenate list of 2-D arrays into a single flat vector."""
    return np.concatenate([a.ravel() for a in factors])


def unpack_factors(flat_x, shapes):
    """Inverse of pack_factors.  shapes = [(sz0, R), (sz1, R), (sz2, R)]."""
    factors, offset = [], 0
    for (rows, cols) in shapes:
        n = rows * cols
        factors.append(flat_x[offset:offset + n].reshape(rows, cols))
        offset += n
    return factors


# ── Circular-shift search ─────────────────────────────────────────────────────

def find_optimal_shifts(tensorX, fitted_3d, NDIRS):
    """Find optimal circular shifts per (neuron, stimulus) and apply them.

    For each (neuron, stimulus) pair, choose the circular shift of the NDIRS
    axis that maximises the inner product between the current CP model and the
    data.  Matches ``perm_tt_cp_fg.m`` lines 119-139.

    Parameters
    ----------
    tensorX   : (N, S, DT) float   original preprocessed tensor
    fitted_3d : (N, S, DT) float   current CP reconstruction [[A_n,A_s,A_r]]
    NDIRS     : int                 number of shift positions (e.g. 8)

    Returns
    -------
    best_shifts : (N, S) int array
    Z_shifted   : (N, S, DT) float — tensorX with optimal shifts applied
    """
    if NDIRS <= 1:
        return np.zeros(tensorX.shape[:2], dtype=int), tensorX.copy()

    N, S, DT = tensorX.shape
    T = DT // NDIRS

    # Fortran-order reshape matches MATLAB's column-major reshape convention
    tensor4d = tensorX.reshape(N, S, NDIRS, T, order='F')   # (N, S, NDIRS, T)

    # Score each of the NDIRS possible shifts (loop over 8 is cheap)
    scores = np.empty((N, S, NDIRS))
    for d in range(NDIRS):
        shifted = np.roll(tensor4d, d, axis=2).reshape(N, S, DT, order='F')
        scores[:, :, d] = np.einsum('nsd,nsd->ns', fitted_3d, shifted)

    best_shifts = np.argmax(scores, axis=2)   # (N, S)

    # Construct Z_shifted
    Z_shifted = np.empty_like(tensorX)
    for d in range(NDIRS):
        mask = (best_shifts == d)
        if not mask.any():
            continue
        rolled_flat = np.roll(tensor4d, d, axis=2).reshape(N, S, DT, order='F')
        Z_shifted[mask] = rolled_flat[mask]

    return best_shifts, Z_shifted


# ── HALS-ALS solver (default, fast) ──────────────────────────────────────────

def _hals_column_update(mttkrp, G, A_mode, eps=1e-10):
    """In-place HALS update of A_mode.

    For each column r, solves the non-negative projection:
        A_mode[:,r] ← max(0, (mttkrp[:,r] − A_mode @ G[:,r] + A_mode[:,r]·G[r,r]) / G[r,r])
    """
    F = A_mode.shape[1]
    for r in range(F):
        numer = (mttkrp[:, r]
                 - A_mode @ G[:, r]
                 + A_mode[:, r] * G[r, r])
        A_mode[:, r] = np.maximum(0.0, numer / (G[r, r] + eps))
    return A_mode


def _run_one_rep_als(tensorX, F, NDIRS, max_iter, seed, tol=1e-6):
    """One ALS repetition with HALS non-negative updates + shift integration.

    Each iteration:
      1. Reconstruct current CP tensor.
      2. Find optimal circular shifts (matching perm_tt_cp_fg.m).
      3. HALS update each of the 3 modes on the shifted tensor.

    Returns (factors, lambdas, final_obj).
    """
    N, S, DT = tensorX.shape
    eps = 1e-10
    Znormsqr = float(np.sum(tensorX ** 2))

    rng = np.random.default_rng(seed)
    A = [rng.random((sz, F)) for sz in (N, S, DT)]
    # Warm start: unit-norm columns
    for mode in range(3):
        norms = np.linalg.norm(A[mode], axis=0) + eps
        A[mode] /= norms

    prev_obj = np.inf
    for it in range(max_iter):
        # ── Shift step ──────────────────────────────────────────────────
        fitted = np.einsum('nr,sr,dr->nsd', A[0], A[1], A[2])
        _, Z_shifted = find_optimal_shifts(tensorX, fitted, NDIRS)

        # ── HALS updates ────────────────────────────────────────────────
        AtA = [a.T @ a for a in A]   # cached Gram matrices

        for mode in range(3):
            other = [m for m in range(3) if m != mode]
            G = AtA[other[0]] * AtA[other[1]]   # (F, F)

            if mode == 0:
                mttkrp = np.einsum('nsd,sr,dr->nr', Z_shifted, A[1], A[2])
            elif mode == 1:
                mttkrp = np.einsum('nsd,nr,dr->sr', Z_shifted, A[0], A[2])
            else:
                mttkrp = np.einsum('nsd,nr,sr->dr', Z_shifted, A[0], A[1])

            _hals_column_update(mttkrp, G, A[mode], eps)
            AtA[mode] = A[mode].T @ A[mode]   # update after mode change

        # ── Convergence check ────────────────────────────────────────────
        # Recompute fitted and optimal shifts for the UPDATED A, so the
        # objective is the true permuted-CP value (not using stale shifts).
        fitted = np.einsum('nr,sr,dr->nsd', A[0], A[1], A[2])
        _, Z_fresh = find_optimal_shifts(tensorX, fitted, NDIRS)
        obj = float(0.5 * Znormsqr
                    - np.sum(fitted * Z_fresh)
                    + 0.5 * np.sum(AtA[0] * AtA[1] * AtA[2]))

        rel_change = abs(prev_obj - obj) / (abs(prev_obj) + 1.0)
        if rel_change < tol and it > 5:
            break
        prev_obj = obj

    # Normalise: unit-norm columns, norms absorbed into lambdas
    lambdas = np.ones(F, dtype=np.float64)
    for mode in range(3):
        norms = np.linalg.norm(A[mode], axis=0) + eps
        lambdas *= norms
        A[mode] /= norms

    return A, lambdas, float(obj)


# ── L-BFGS-B solver (faithful MATLAB translation, slow) ──────────────────────

def _lbfgsb_obj_grad(flat_x, tensorX, shapes, NDIRS, Znormsqr):
    """Permuted CP objective and gradient for scipy L-BFGS-B."""
    A = unpack_factors(flat_x, shapes)
    fitted = np.einsum('nr,sr,dr->nsd', A[0], A[1], A[2])
    _, Z_shifted = find_optimal_shifts(tensorX, fitted, NDIRS)

    AtA = [a.T @ a for a in A]
    f = float(0.5 * Znormsqr
              - np.sum(Z_shifted * fitted)
              + 0.5 * np.sum(AtA[0] * AtA[1] * AtA[2]))

    G0 = (-np.einsum('nsd,sr,dr->nr', Z_shifted, A[1], A[2])
          + A[0] @ (AtA[1] * AtA[2]))
    G1 = (-np.einsum('nsd,nr,dr->sr', Z_shifted, A[0], A[2])
          + A[1] @ (AtA[0] * AtA[2]))
    G2 = (-np.einsum('nsd,nr,sr->dr', Z_shifted, A[0], A[1])
          + A[2] @ (AtA[0] * AtA[1]))

    return f, pack_factors([G0, G1, G2])


def _run_one_rep_lbfgsb(tensorX, F, NDIRS, max_iter, seed):
    """One L-BFGS-B repetition. Slow but faithful to MATLAB."""
    N, S, DT = tensorX.shape
    shapes   = [(N, F), (S, F), (DT, F)]
    Znormsqr = float(np.sum(tensorX ** 2))

    rng = np.random.default_rng(seed)
    x0  = rng.random(N * F + S * F + DT * F)
    bounds = [(0.0, None)] * len(x0)

    result = scipy.optimize.minimize(
        fun=lambda x: _lbfgsb_obj_grad(x, tensorX, shapes, NDIRS, Znormsqr),
        x0=x0, jac=True, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-7},
    )
    A = unpack_factors(np.clip(result.x, 0.0, None), shapes)
    lambdas = np.ones(F, dtype=np.float64)
    for mode in range(3):
        norms = np.linalg.norm(A[mode], axis=0) + 1e-12
        lambdas *= norms
        A[mode] /= norms
    return A, lambdas, float(result.fun)


# ── Multi-rep run (public API) ─────────────────────────────────────────────────

def run_permuted_cp(tensorX, F, NDIRS=8, nreps=30, max_iter=200,
                    method='als', n_jobs=1, seed=0, verbose=True):
    """Non-negative permuted CP decomposition with multiple random restarts.

    Parameters
    ----------
    tensorX  : (N, S, DT) float   preprocessed tensor from process_tensor_data
    F        : int                 rank (number of components)
    NDIRS    : int                 number of shift positions (e.g. 8)
    nreps    : int                 random restarts (5=quick, 30=production)
    max_iter : int                 iterations per rep (ALS: 200, L-BFGS-B: 500)
    method   : 'als' | 'lbfgsb'   'als' is ~10× faster; 'lbfgsb' matches MATLAB
    n_jobs   : int                 joblib workers (-1 = all cores)
    seed     : int                 base random seed

    Returns
    -------
    dict with keys compatible with loadPreComputedCP:
        'all_factors' : {rep: [A_neural(N,F), A_stim(S,F), A_resp(DT,F)]}
        'all_lambdas' : {rep: (F,) float}
        'all_objs'    : {rep: float}
    Reps sorted best-first (index 0 = lowest objective).
    """
    _one_rep = _run_one_rep_als if method == 'als' else _run_one_rep_lbfgsb

    def _job(rep_i):
        return _one_rep(tensorX, F, NDIRS, max_iter, seed + rep_i)

    if n_jobs == 1:
        results = [_job(i)
                   for i in tqdm(range(nreps), desc=f'CP F={F}', disable=not verbose)]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_job)(i)
            for i in tqdm(range(nreps), desc=f'CP F={F}', disable=not verbose)
        )

    results.sort(key=lambda r: r[2])

    return {
        'all_factors': {i: r[0] for i, r in enumerate(results)},
        'all_lambdas': {i: r[1] for i, r in enumerate(results)},
        'all_objs':    {i: r[2] for i, r in enumerate(results)},
    }


# ── F-range sweep ──────────────────────────────────────────────────────────────

def run_cp_sweep(tensorX, F_range, NDIRS=8, nreps=30, max_iter=200,
                 method='als', n_jobs=1, seed=0,
                 basedir=None, tensorname=None, verbose=True):
    """Sweep run_permuted_cp over multiple ranks F.

    Saves .mat files (loadable by loadPreComputedCP) if basedir/tensorname given.

    Returns
    -------
    preComputed : dict keyed by F (same format as loadPreComputedCP)
    """
    Znormsqr = float(np.sum(tensorX ** 2))
    preComputed = {}
    for F in F_range:
        result = run_permuted_cp(tensorX, F, NDIRS=NDIRS, nreps=nreps,
                                 max_iter=max_iter, method=method,
                                 n_jobs=n_jobs, seed=seed, verbose=verbose)
        preComputed[F] = result
        if basedir is not None and tensorname is not None:
            save_cp_as_mat(result, F, nreps, tensorname, basedir,
                           Znormsqr=Znormsqr)
    return preComputed


# ── Save in MATLAB-compatible format ──────────────────────────────────────────

def save_cp_as_mat(result, F, nreps, tensorname, basedir, Znormsqr=None):
    """Save run_permuted_cp output as .mat loadable by loadPreComputedCP.

    File: ``{basedir}/{tensorname}_F{F:02d}_nreps{nreps}.mat``

    Parameters
    ----------
    Znormsqr : float, optional
        If provided, stores objectives as MATLAB-compatible relative % error:
        ``100 * f / (0.5 * Znormsqr)``, matching ``run_permcp.m`` output.
        If None, stores raw absolute ALS objectives.
    """
    os.makedirs(basedir, exist_ok=True)
    path = os.path.join(basedir, f'{tensorname}_F{F:02d}_nreps{nreps}.mat')

    n = len(result['all_factors'])
    factors_outer = np.empty((1, n), dtype=object)
    lams_outer    = np.empty((1, n), dtype=object)
    objs_outer    = np.empty((1, n), dtype=object)

    for r in range(n):
        inner = np.empty((1, 3), dtype=object)
        for mode in range(3):
            inner[0, mode] = np.ascontiguousarray(
                result['all_factors'][r][mode], dtype=np.float64)
        factors_outer[0, r] = inner
        lams_outer[0, r]    = result['all_lambdas'][r].reshape(1, -1).astype(np.float64)
        obj_val = result['all_objs'][r]
        if Znormsqr is not None:
            obj_val = 100.0 * obj_val / (0.5 * Znormsqr)
        objs_outer[0, r]    = np.array([[obj_val]], dtype=np.float64)

    savemat(path, {'factors': factors_outer, 'lams': lams_outer, 'objs': objs_outer})
    return path


# ── Convenience wrapper ────────────────────────────────────────────────────────

def load_or_run_cp(tensorX, tensorname, F_range, NDIRS=8, nreps=30,
                   max_iter=200, method='als', n_jobs=1, seed=0,
                   basedir=None, verbose=True):
    """Load pre-existing .mat files if available, else run Python CP.

    - tensorX: input tensor
    - tensorname: name of the tensor (used for loading/saving)
    - F_range: range of ranks to consider
    - NDIRS: number of directions for the CP decomposition
    - nreps: number of random restarts
    - max_iter: maximum number of iterations for optimizer
    - method: decomposition method (e.g., 'als')
    - n_jobs: number of parallel jobs
    - seed: random seed
    - basedir: base directory for loading/saving
    - verbose: verbosity flag

    Compatible with the existing loadPreComputedCP downstream pipeline.
    """
    from src.tensor_utils import loadPreComputedCP

    preComputed = {}
    if basedir is not None:
        existing, _ = loadPreComputedCP(tensorname, basedir,
                                        specificFs=list(F_range), verbose=verbose)
        preComputed.update(existing)

    missing = [F for F in F_range if F not in preComputed]
    if missing:
        if verbose:
            print(f'Running Python CP for F={missing}')
        new = run_cp_sweep(tensorX, missing, NDIRS=NDIRS, nreps=nreps,
                           max_iter=max_iter, method=method, n_jobs=n_jobs,
                           seed=seed, basedir=basedir, tensorname=tensorname,
                           verbose=verbose)
        preComputed.update(new)

    return preComputed
