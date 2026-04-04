"""Cache utilities for the encoding-decoding explorer.

The main entry point is `load_for_explorer(PREFIX)`, which loads all data
needed by ManifoldExplorer from on-disk caches (tensor .npy, decomposition .mat,
IAN graph .npz), runs the fast post-processing steps (PCA, outlier removal,
diffusion map, MDS), and returns a ready-to-use dict.

Pre-requisite: notebook 03_encoding_manifolds.ipynb must have been run to
generate the IAN graph cache at {basedir_wG}/{PREFIX}_{R}.npz.
"""

import os
import glob as _glob
import re as _re
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

from .tensor_utils import (
    process_tensor_data,
    loadPreComputedCP,
    getPermutedTensor,
    getNeuralMatrix,
)
from .graph_utils import compute_graph_statistics, handle_disconnected_points
from .manifold_utils import compute_mds_embedding
from .metrics import (
    compute_osi_and_pref_stim,
    compute_temporal_variance,
    compute_mean_activation,
)

_KNOWN_STIM_LABELS = {
    6:  ['LF-grat', 'HF-grat', '-1dot', '-3dot', '+1dot', '+3dot'],
    11: ['LF-grat', 'HF-grat', '-1dot', '-3dot', '+1dot', '+3dot',
         'HF-grat-2', '-1dot-2', '-3dot-2', '+1dot-2', '+3dot-2'],
    # Natural movie scenes (VISp_natural_movie_one_scenes_all156: all 14 hand-labelled scenes)
    14: ['linear mvmt', 'stable', 'man→left', 'stable 2',
         'man→right+pan', 'pan right', 'shadow+pan', 'pan right 2',
         'zoom', 'man→right 2', 'pan car', 'couple→car',
         'stable 3', 'car+pan'],
    # Natural movie scenes (VISp_natural_movie_one_scenes20: 12 long scenes, 20-frame clips)
    12: ['stable', 'man→left', 'stable 2', 'man→right+pan',
         'pan right', 'shadow+pan', 'pan right 2', 'zoom',
         'pan car', 'couple→car', 'stable 3', 'car+pan'],
}

MIN_EXPL_VAR_RATIO = 0.8


def load_for_explorer(
    PREFIX,
    basedir_data='../data/sampled',
    basedir_mat='../data/decompositions',
    basedir_wG='../data/graphs',
    optSF=True,
    method='relNorm',
    smooth_sig=3,
    n_far=2,
    n_close=5,
):
    """Load all data needed by ManifoldExplorer from disk cache.

    Parameters
    ----------
    PREFIX : str
        Dataset prefix (e.g. 'flyvis_Medulla_i3_n550_model000').
    basedir_data : str
        Directory containing tensor4d_{PREFIX}.npy files.
    basedir_mat : str
        Directory containing MATLAB CP decomposition .mat files.
    basedir_wG : str
        Directory containing IAN graph .npz files.
    optSF : bool
        Use per-neuron optimal spatial frequency (only applies to 11-stim data).
    method : str
        Normalisation method (default: 'relNorm').
    smooth_sig : int
        Smoothing window in frames.
    n_far : int
        Number of most-isolated neurons to remove as outliers.
    n_close : int
        Number of most-similar neurons to remove as outliers.

    Returns
    -------
    dict with keys:
        embedding_, tensor4d, nonoutliers, neurons_used, my_stims,
        NDIRS, NSTIMS, extra_colorings, nPCs, PREFIX2, tensorname, R
    """
    from ian.embed_utils import diffusionMapSparseK
    from ian.utils import pwdists

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    tensor4d = np.load(f'{basedir_data}/tensor4d_{PREFIX}.npy')
    # Save the raw tensor (with NaN at padded frames) before any modification.
    # This is used for decoding so it matches notebook 04 exactly.
    tensor4d_raw = tensor4d.copy()               # (N, S, D, T), NaN at padded positions
    # Convert NaN → 0 to match notebook 03's preprocessing exactly:
    #   tensor4d = np.nan_to_num(np.load(...)); tensor4d -= np.min(tensor4d)
    # The IAN graph and CP decomposition were built from this zero-padded version,
    # so we must use the same tensorX for the encoding path.
    tensor4d = np.nan_to_num(tensor4d)           # NaN → 0  (matches notebook 03)
    tensor4d = tensor4d - np.min(tensor4d)
    neurons_used = np.load(f'{basedir_data}/neurons_used_{PREFIX}.npy')

    N, _N_orig_stims, NDIRS, TRIAL_LEN = tensor4d.shape

    # ── 2. Process tensor data ────────────────────────────────────────────────
    tensorX, relFRs, optStims = process_tensor_data(tensor4d, optSF, smooth_sig, method)

    NSTIMS = tensorX.shape[1]
    PREFIX2 = f'{method}_sig{smooth_sig}_n{N}'
    if optSF and optStims:
        PREFIX2 += '_SF'

    CPMETHOD = 'shift'
    AREA = 'deepnet'
    tensorname = f'{PREFIX}-{PREFIX2}-{AREA}-{CPMETHOD}'

    # ── 3. Auto-detect R ──────────────────────────────────────────────────────
    wG_pattern = os.path.join(basedir_wG, f'{PREFIX}_*.npz')
    wG_files = _glob.glob(wG_pattern)
    if not wG_files:
        raise FileNotFoundError(
            f'No IAN graph cache found matching {wG_pattern}. '
            'Run notebook 03_encoding_manifolds.ipynb first to generate the cache.'
        )
    R_candidates = []
    for f in wG_files:
        m = _re.search(r'_(\d+)\.npz$', f)
        if m:
            R_candidates.append(int(m.group(1)))
    if not R_candidates:
        raise FileNotFoundError(
            f'Could not parse R from files matching {wG_pattern}. '
            'Expected filenames like {PREFIX}_{R}.npz.'
        )
    R = max(R_candidates)
    wG_path = os.path.join(basedir_wG, f'{PREFIX}_{R}.npz')

    # ── 4. Load decomposition ─────────────────────────────────────────────────
    preComputed, Fs = loadPreComputedCP(tensorname, basedir_mat, specificFs=[R])
    if R not in preComputed and Fs:
        R = Fs[0]
        wG_path = os.path.join(basedir_wG, f'{PREFIX}_{R}.npz')
    if not preComputed:
        # Fallback: mat file may have been built with a different method name
        # (e.g. 'Norm' instead of 'relNorm'). Search by PREFIX alone.
        preComputed, Fs = loadPreComputedCP(PREFIX, basedir_mat, specificFs=[R])
        if not preComputed and Fs:
            R = Fs[0]
            preComputed, Fs = loadPreComputedCP(PREFIX, basedir_mat, specificFs=[R])
            wG_path = os.path.join(basedir_wG, f'{PREFIX}_{R}.npz')

    # ── 5. Extract factors ────────────────────────────────────────────────────
    best_rep, _ = min(preComputed[R]['all_objs'].items(), key=lambda x: x[1])
    best_factors = preComputed[R]['all_factors'][best_rep]
    best_lambdas = preComputed[R]['all_lambdas'][best_rep]

    posnorms = ~np.isclose(best_lambdas, 0)
    lambdas = best_lambdas[posnorms]
    factors = [
        f[:, posnorms] / np.linalg.norm(f[:, posnorms], axis=0, keepdims=1)
        for f in best_factors
    ]

    # ── 6. Build neural matrix ────────────────────────────────────────────────
    permT, _fitT = getPermutedTensor(factors, lambdas, tensorX, NDIRS)
    X, _ = getNeuralMatrix(permT, factors, lambdas, NDIRS, None, order='F', verbose=False)

    # ── 7. PCA ────────────────────────────────────────────────────────────────
    pca = PCA(len(lambdas))
    pcaX = pca.fit_transform(X)
    nPCs = (
        np.flatnonzero(np.cumsum(pca.explained_variance_ratio_) > MIN_EXPL_VAR_RATIO)[0] + 1
    )
    X = pcaX[:, :nPCs]
    D2 = pwdists(X, sqdists=True)
    N_pts = D2.shape[0]

    # ── 8. Outlier removal ────────────────────────────────────────────────────
    D1 = np.sqrt(D2)
    mindists = np.min(D1 + np.eye(N_pts) * D1.max(), axis=0)
    outls_far   = np.argsort(mindists)[::-1][:n_far]
    outls_close = np.argsort(mindists)[:n_close]
    outliers_list = np.append(outls_far, outls_close)

    myX = X[[c for c in range(N_pts) if c not in outliers_list]]
    nonoutliers = np.array([i for i in range(N_pts) if i not in outliers_list])
    D2 = pwdists(myX, sqdists=True)

    # ── 9. Load IAN graph from cache ──────────────────────────────────────────
    if not os.path.exists(wG_path):
        raise FileNotFoundError(
            f'IAN cache not found at {wG_path}. '
            'Run notebook 03_encoding_manifolds.ipynb first.'
        )
    _cache = np.load(wG_path, allow_pickle=False)
    wG_dense  = _cache['wG']
    wG        = csr_matrix(wG_dense)
    G         = (wG_dense > 0).astype(float)
    optScales = _cache['optScales']
    disc_pts  = [[i] for i in _cache['disc_pts_indices']]

    assert wG.shape[0] == D2.shape[0], (
        f'IAN graph size {wG.shape[0]} ≠ distance matrix size {D2.shape[0]}. '
        f'The cached graph was built with different n_far/n_close values '
        f'(current: n_far={n_far}, n_close={n_close}). '
        'Adjust n_far/n_close to match how notebook 03 was run, or re-run notebook 03.'
    )

    # ── 10. Handle disconnected points ────────────────────────────────────────
    wG, G, outliers_list, nonoutliers, nonout_mask = handle_disconnected_points(
        disc_pts, optScales, G, D2, outliers_list, neurons_used, X
    )

    # ── 11. Diffusion map ─────────────────────────────────────────────────────
    diffmap_y, _diffmap_evals = diffusionMapSparseK(csr_matrix(wG), 20, 1, t=1)

    # ── 12. MDS embedding ─────────────────────────────────────────────────────
    embedding_ = compute_mds_embedding(diffmap_y, nPCs, n_components=10)

    # ── 13. Reshape tensor to nonoutlier space ────────────────────────────────
    _N_all = len(tensorX)
    tensorX_4d = np.reshape(tensorX, (_N_all, NSTIMS, TRIAL_LEN, NDIRS))
    tensorX_4d_nonout = tensorX_4d[nonoutliers]   # (N_nonout, N_STIM, T, N_DIR)

    # Raw tensor for decoding: transpose from (N, S, D, T) → (N, S, T, D) explorer
    # format, then subset to nonoutliers. NaN at padded frames is preserved.
    tensor4d_raw_nonout = tensor4d_raw.transpose(0, 1, 3, 2)[nonoutliers]  # (N_nonout, S, T, D)

    # ── 14. Stimulus labels ───────────────────────────────────────────────────
    my_stims = _KNOWN_STIM_LABELS.get(NSTIMS, [f'stim {i}' for i in range(NSTIMS)])

    # ── 15. Extra colorings ───────────────────────────────────────────────────
    extra_colorings = {}
    extra_colorings['temporal var.'] = compute_temporal_variance(tensorX)
    extra_colorings['mean act.']     = compute_mean_activation(tensorX)

    if NDIRS > 1:
        _osi, _pref = compute_osi_and_pref_stim(tensorX, n_stim=NSTIMS, n_dir=NDIRS)
        extra_colorings['OSI']        = _osi
        extra_colorings['pref. stim'] = _pref

    try:
        graph_stats = compute_graph_statistics(wG, neurons_used, nonoutliers)
        for _key, _gkey in [('degree', 'degrees'),
                             ('wt. degree', 'weighted_degrees'),
                             ('same-fmap edges', 'same_fmaps')]:
            _full = np.full(_N_all, np.nan)
            _full[nonoutliers] = graph_stats[_gkey]
            extra_colorings[_key] = _full
    except Exception:
        pass

    try:
        _nf = factors[0]   # (N_all, R) neural factors
        for _fi in range(min(5, _nf.shape[1])):
            extra_colorings[f'factor {_fi}'] = _nf[:, _fi]
    except Exception:
        pass

    # ── 15b. VISp external manifold DCs ──────────────────────────────────────
    if PREFIX.startswith('VISp_natural_movie_one'):
        import pickle
        from .metrics import build_index_maps

        _mfd_path = os.path.join(os.path.dirname(basedir_data), 'natural_video', 'VISp-manifold.npy')
        _ids_path = os.path.join(os.path.dirname(basedir_data), 'natural_video', 'cell_ids_to_use_VISp_dg.pkl')
        _info_dir = os.path.join(os.path.dirname(basedir_data), 'natural_video', 'VISp')

        _MY_SESSION_IDS = [
            719161530, 737581020, 746083955, 798911424, 743475441, 744228101, 756029989, 758798717,
            715093703, 762602078, 797828357, 754312389, 760345702, 739448407, 763673393, 799864342,
            721123822, 761418226, 732592105, 742951821, 762120172, 750749662, 791319847, 760693773,
            755434585, 759883607, 750332458, 757216464, 754829445, 757970808, 773418906, 751348571,
        ]

        try:
            _session_uids_used = []
            for _sess_id in _MY_SESSION_IDS:
                _info_path = os.path.join(
                    _info_dir, f's{_sess_id}_VISp_natural_movie_one_trialFRs_trial_info.pkl'
                )
                with open(_info_path, 'rb') as _f:
                    _info = pickle.load(_f)
                for _uid in _info['uis']:
                    _session_uids_used.append((_sess_id, int(_uid)))

            _mfd = np.load(_mfd_path)  # (1261, 6)
            with open(_ids_path, 'rb') as _f:
                _cell_ids = [(int(a), int(b)) for a, b in pickle.load(_f)]

            _, _m2nm = build_index_maps(_session_uids_used, _cell_ids)

            _dc_arrays = [np.full(_N_all, np.nan) for _ in range(_mfd.shape[1])]
            for _mfd_row, _tensor_row in _m2nm.items():
                for _d in range(_mfd.shape[1]):
                    _dc_arrays[_d][_tensor_row] = _mfd[_mfd_row, _d]

            for _d, _arr in enumerate(_dc_arrays):
                extra_colorings[f'ext. DC{_d + 1}'] = _arr
            print(f'VISp ext. manifold: {len(_m2nm)}/{len(_cell_ids)} neurons matched')
        except Exception as e:
            print(f'Warning: could not load VISp ext. manifold colorings: {e}')

    # ── 16. Return ────────────────────────────────────────────────────────────
    return {
        'embedding_':      embedding_,
        'tensor4d':        tensorX_4d_nonout,
        'tensor4d_raw':    tensor4d_raw_nonout,
        'nonoutliers':     nonoutliers,
        'neurons_used':    neurons_used,
        'my_stims':        my_stims,
        'NDIRS':           NDIRS,
        'NSTIMS':          NSTIMS,
        'extra_colorings': extra_colorings,
        'nPCs':            nPCs,
        'PREFIX2':         PREFIX2,
        'tensorname':      tensorname,
        'R':               R,
    }
