"""Tensor loading, preprocessing, and decomposition utilities.

Covers:
  - loadPreComputedCP  — load MATLAB CP decomposition results
  - process_tensor_data — SF filtering, smoothing, normalisation
  - getPermutedTensor  — recover optimal circular shifts
  - getNeuralMatrix    — fit neural encoding matrix from permuted tensor
  - khatri_rao         — Khatri-Rao (column-wise Kronecker) product
"""

import numpy as np
from glob import glob
from scipy.io import loadmat
from scipy.optimize import lsq_linear
from scipy.ndimage import gaussian_filter1d


def khatri_rao(matrices):
    """Khatri-Rao product of a list of matrices.

    Parameters
    ----------
    matrices : list of ndarray

    Returns
    -------
    khatri_rao_product: matrix of shape ``(prod(n_i), m)``
        where ``prod(n_i) = prod([m.shape[0] for m in matrices])``
        i.e. the product of the number of rows of all the matrices in the
        product.

    Author
    ------
    Jean Kossaifi <https://github.com/tensorly>
    """
    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i + common_dim for i in target)
    operation = source + '->' + target + common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))


def from0to1(arr):
    arr = np.asanyarray(arr)
    arr[np.isclose(arr, 0)] = 1
    return arr


# Stimulus indices for the two spatial-frequency groups in the 11-stim grating layout.
# Only valid when the raw tensor has exactly 11 stimuli (FNN/FlyVis/CORnet/MouseNet/R(2+1)D).
SF_MED_IDXS = [0, 2, 5, 6, 9, 10]   # medium SF indices used by process_tensor_data (encoding path)
SF_HI_IDXS  = [0, 1, 3, 4, 7, 8]    # high SF indices used by process_tensor_data (encoding path)

# Decoding-path stimulus groups: coherent 6-stim sets matching notebook 04's LOW_SF / HIGH_SF split.
# LOW_SF matches the 6 bio-compatible stimuli in Retina/V1 data exactly.
# HIGH_SF keeps LF-grat (stim 0) and replaces the other 5 with the higher-SF variants (6–10).
SF_LOW_IDXS  = list(range(6))         # [0,1,2,3,4,5]   — bio-compatible set
SF_HIGH_IDXS = [0, 6, 7, 8, 9, 10]   # LF-grat + high-SF variants


def select_sf_stims_for_decoding(tensor4d_nsdt):
    """Select 6 stimuli from an 11-stim tensor using a population-level SF majority vote.

    For 11-stim grating datasets (FNN, FlyVis, CORnet, MouseNet, R(2+1)D), compares
    the population's mean response to the LOW_SF set (stims 0–5, bio-compatible) versus
    the HIGH_SF set (stim 0 + stims 6–10, higher-SF variants) and returns the better-
    responding 6-stim subset.  Mirrors the per-neuron majority-vote logic of notebook 03
    applied at the population level, but uses the coherent LOW/HIGH split of notebook 04
    so the decoding manifold is computed over a consistent set of 6 stimuli.

    Parameters
    ----------
    tensor4d_nsdt : ndarray, shape (N, S, D, T)
        Raw tensor (neurons × stimuli × directions × time).

    Returns
    -------
    chosen_idx : list of int  (length 6, or S if S != 11)
    tensor_sub : ndarray, shape (N, 6, D, T)  (or original if S != 11)
    """
    N, S, D, T = tensor4d_nsdt.shape
    if S != 11:
        return list(range(S)), tensor4d_nsdt

    # Per-neuron mean FR per stim (nanmean handles any residual NaN)
    stim_means = np.nanmean(tensor4d_nsdt, axis=(2, 3))   # (N, S)

    # Majority vote: does the population prefer the low-SF or high-SF stimulus set?
    low_score  = stim_means[:, SF_LOW_IDXS].mean(axis=1)   # (N,)
    high_score = stim_means[:, SF_HIGH_IDXS].mean(axis=1)  # (N,)

    if (low_score > high_score).sum() > (high_score >= low_score).sum():
        return list(SF_LOW_IDXS), tensor4d_nsdt[:, SF_LOW_IDXS]
    else:
        return list(SF_HIGH_IDXS), tensor4d_nsdt[:, SF_HIGH_IDXS]


def process_tensor_data(tensor4d, optSF, smooth_sig, method):
    """Process tensor data by applying SF optimization, smoothing, and normalisation.

    Args:
        tensor4d: 4D tensor (Neurons, Stimuli, Directions, Trial_len)
        optSF: bool, whether to use optimal spatial frequency
        smooth_sig: int, smoothing sigma (frames)
        method: str, normalisation method ('relFR', 'Norm', or 'relNorm')

    Returns:
        tuple: (tensorX, relFRs, optStims)
    """
    N, NSTIMS, NDIRS, TRIAL_LEN = tensor4d.shape

    # SF filtering only valid for the 11-stim FNN layout
    _do_optSF = optSF and NSTIMS == 11

    optStims = []
    if _do_optSF:
        tensorX = np.zeros((N, NSTIMS // 2 + 1, NDIRS * TRIAL_LEN))
        relFRs = np.zeros((N, NSTIMS // 2 + 1))
    else:
        tensorX = np.zeros((N, NSTIMS, NDIRS * TRIAL_LEN))
        relFRs = np.zeros((N, NSTIMS))

    for nii in range(tensor4d.shape[0]):
        relMeanPosFRs = []
        all_psts = np.zeros((NSTIMS, NDIRS * TRIAL_LEN))
        for stimi in range(NSTIMS):
            pst = tensor4d[nii, stimi].copy()   # (NDIRS, TRIAL_LEN)
            _nan_mask = ~np.isfinite(pst)        # NaN mask for padded frames
            if smooth_sig > 0:
                pst[_nan_mask] = 0.0             # fill before smoothing to avoid NaN spread
                pst = gaussian_filter1d(pst, smooth_sig, axis=1)
                pst[_nan_mask] = np.nan          # restore after smoothing
            relMeanPosFRs.append(np.nanmax(np.nanmean(pst, axis=1)))  # valid frames only
            all_psts[stimi] = np.where(np.isfinite(pst), pst, 0.0).ravel(order='F')

        relMeanPosFRs = np.array(relMeanPosFRs)

        if _do_optSF:
            med_FRs = relMeanPosFRs[SF_MED_IDXS]
            hi_FRs = relMeanPosFRs[SF_HI_IDXS]
            if (med_FRs > hi_FRs).sum() > (hi_FRs > med_FRs).sum():
                relMeanPosFRs = med_FRs
                tensorX[nii] = all_psts[SF_MED_IDXS]
                optStims.append('med')
            else:
                relMeanPosFRs = hi_FRs
                tensorX[nii] = all_psts[SF_HI_IDXS]
                optStims.append('hi')
        else:
            tensorX[nii] = all_psts

        relFRs[nii] = relMeanPosFRs / np.nanmax(relMeanPosFRs)

        if method == 'relFR':
            tensorX[nii] /= tensorX[nii].max()
        if method == 'Norm':
            tensorX[nii] /= tensorX[nii].mean()
        elif method == 'relNorm':
            stim_norms = from0to1(np.linalg.norm(tensorX[nii], axis=1, keepdims=1))
            tensorX[nii] /= from0to1(stim_norms)
            tensorX[nii] *= relFRs[nii][:, None]

    return tensorX, relFRs, optStims


def getPermutedTensor(factors, lambdas, tensorX, NDIRS):
    """Find the optimal circular shifts used by the permuted decomposition and apply them.

    Returns:
        shifted_tensor, fittensor
    """
    fittensor = np.reshape(
        (lambdas * factors[0]) @ khatri_rao(factors[1:]).T,
        tensorX.shape,
    )

    if NDIRS == 1:
        return tensorX, fittensor

    N = tensorX.shape[0]
    NSTIMS = tensorX.shape[1]
    RLEN = tensorX.shape[2]

    shape4d = (N, NSTIMS, NDIRS, RLEN // NDIRS)
    shapeDot = (N, RLEN)
    tensor4d = np.reshape(tensorX, shape4d, order='F')

    objs = np.empty((NSTIMS, N, NDIRS))
    obj_shifts = np.empty((NSTIMS, N))
    for si in range(NSTIMS):
        for shifti in range(NDIRS):
            objs[si, :, shifti] = -np.sum(
                fittensor[:, si, :]
                * np.reshape(np.roll(tensor4d[:, si], shifti, 1), shapeDot, order='F'),
                1,
            )
        obj_shifts[si] = np.argmin(objs[si], axis=1)

    shifted_tensor = np.zeros_like(tensorX)
    for shifti in range(NDIRS):
        rolledX = np.reshape(np.roll(tensor4d, shifti, 2), tensorX.shape, order='F')
        for si in range(NSTIMS):
            shifted_tensor[(obj_shifts[si] == shifti), si, :] = rolledX[
                (obj_shifts[si] == shifti), si, :
            ]

    return shifted_tensor, fittensor


def getNeuralMatrix(
    scld_permT, factors, lambdas, NDIRS, all_zeroed_stims=None, order='F', verbose=True
):
    """Fit the neural encoding matrix from the permuted tensor and NTF factors.

    Args:
        scld_permT: ndarray, permuted tensor scaled by relative stimulus FRs
        factors: list [neural_factors, stimulus_factors, response_factors] (normalised)
        lambdas: ndarray, shape (R,)
        NDIRS: int, number of stimulus directions
        all_zeroed_stims: dict {cell: (tuple of zeroed stim idxs)}, default None
        order: str, flatten order, default 'F'
        verbose: bool

    Returns:
        X: ndarray, shape (Ncells, R) — neural encoding matrix
        new_scld_permT: ndarray — tensor with previously zeroed responses filled
    """
    R = lambdas.size

    stim_factors = factors[1].copy()
    stim_scls = stim_factors.max(0, keepdims=1)
    stim_factors /= stim_scls

    neural_factors = factors[0].copy()
    neural_factors *= lambdas * stim_scls

    new_coords = np.stack(
        [
            khatri_rao([stim_factors[:, r][:, None], factors[2][:, r][:, None]]).ravel()
            for r in range(R)
        ],
        axis=1,
    )

    Ncells = scld_permT.shape[0]
    NSTIMS = scld_permT.shape[1]

    X = np.zeros((Ncells, R))
    new_scld_permT = scld_permT.copy()

    for c in range(Ncells):
        if verbose and (c + 1) % 50 == 0:
            print(c + 1, end=' ')

        if all_zeroed_stims is not None and c in all_zeroed_stims:
            lowest_cost = np.inf
            for shifti in range(NDIRS):
                shifted_cell_data = scld_permT[c].copy()
                for si in all_zeroed_stims[c]:
                    si_2d = shifted_cell_data[si].reshape((NDIRS, -1), order=order)
                    shifted_cell_data[si] = np.roll(si_2d, shifti, axis=0).ravel(order=order)
                res = lsq_linear(new_coords, shifted_cell_data.ravel(), bounds=(0, np.inf))
                coeffs, cost = res['x'], res['cost']
                if cost < lowest_cost:
                    lowest_cost = cost
                    best_coeffs = coeffs
                    new_scld_permT[c] = shifted_cell_data
            new_coeffs = best_coeffs
        else:
            new_coeffs = lsq_linear(new_coords, scld_permT[c].ravel(), bounds=(0, np.inf))['x']

        X[c] = np.sqrt(new_coeffs)

    return X, new_scld_permT


def loadPreComputedCP(tensorname, basedir, specificFs=[], NMODES=3, verbose=True):
    """Load MATLAB CP decomposition results and aggregate across F choices and reps.

    Converts the .mat files produced by ``matlab/run_permcp.m`` into a Python dict.

    Args:
        tensorname: base filename stem (without path or extension)
        basedir: directory containing the .mat files
        specificFs: list of F values to load (empty = load all)
        NMODES: expected number of tensor modes
        verbose: bool or int

    Returns:
        preComputed: dict {F: {'all_factors': ..., 'all_lambdas': ..., 'all_objs': ...}}
        Fs: sorted list of F values
    """
    preComputed = {}

    def parseF(s):
        assert s[-4:] == '.mat'
        F_str = s[:-4].split('_F')[-1]
        if '_nreps' in F_str:
            F_str = F_str.split('_nreps')[0]
        return int(F_str)

    query = '%s/%s*_F*.mat' % (basedir, tensorname)
    print(query)
    queryfiles = glob(query)

    F_ = None
    counted_reps = 0
    for r in sorted(queryfiles):
        F = parseF(r)

        if specificFs and F not in specificFs:
            continue

        if F != F_:
            if int(verbose) > 1:
                print()
            elif int(verbose) == 1 and counted_reps > 0:
                print(f'({counted_reps})', end=' ')
            counted_reps = 0
            if verbose:
                print(f'F{F}:', end=' ')
            F_ = F

        matfile = loadmat(r)
        nreps = len(matfile['factors'][0])
        factors = {counted_reps + rep: matfile['factors'][0][rep].squeeze() for rep in range(nreps)}
        lambdas = {counted_reps + rep: matfile['lams'][0][rep].squeeze() for rep in range(nreps)}
        objs = {counted_reps + rep: matfile['objs'][0][rep].squeeze() for rep in range(nreps)}

        F_precomp = {'all_factors': factors, 'all_lambdas': lambdas, 'all_objs': objs}
        counted_reps += nreps

        if F not in preComputed:
            preComputed[F] = F_precomp.copy()
        else:
            for dkey in F_precomp.keys():
                preComputed[F][dkey].update(F_precomp[dkey])

    if int(verbose) == 1 and counted_reps > 0:
        print(f'({counted_reps})')

    Fs = sorted(preComputed.keys())
    return preComputed, Fs
