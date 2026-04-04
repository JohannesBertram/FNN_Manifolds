#!/usr/bin/env python3
"""
build.py — Generate website/index.html from pre-computed data.

Usage (from repo root):
    python website/build.py [--prefix flyvis_Medulla_i3_n550_model000]
    python website/build.py --prefix fnn07_act_i3_n2000_SCL0_7_TL37_blocks2_maxFr_maxNr_seed1

The script:
  1. Loads all explorer data via src.cache_utils.load_for_explorer
  2. Pre-computes full-population decoding manifold + trajectories
  3. Serializes everything to a single JSON payload
  4. Injects the payload into website/template.html → website/index.html
"""

import sys
import os
import json
import base64
import gzip
import argparse
import struct
import warnings

import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Make sure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache_utils import load_for_explorer
from src.subpop_utils import compute_decoding_manifold, compute_decoding_trajectories, compute_dynamic_metrics


def round_sig(x, sig=5):
    """Round a float to `sig` significant figures. Returns None for NaN/inf."""
    if not np.isfinite(x):
        return None
    if x == 0:
        return 0.0
    from math import floor, log10
    d = sig - 1 - int(floor(log10(abs(x))))
    return round(x, d)


def to_json_list(arr, sig=5):
    """Convert ndarray to nested Python list with sig-fig rounding."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return [round_sig(float(v), sig) for v in arr]
    elif arr.ndim == 2:
        return [[round_sig(float(v), sig) for v in row] for row in arr]
    elif arr.ndim == 3:
        return [[[round_sig(float(v), sig) for v in col] for col in row] for row in arr]
    else:
        raise ValueError(f"Unsupported ndim={arr.ndim}")


def encode_tensor4d(tensor4d):
    """Encode tensor4d as gzip-compressed base64 float32 bytes (C order).

    Parameters
    ----------
    tensor4d : ndarray, shape (N, S, T, D)

    Returns
    -------
    b64 : str — base64-encoded gzip-compressed raw bytes of float32 array
    shape : list of int — [N, S, T, D]
    """
    arr_f32 = np.ascontiguousarray(tensor4d, dtype=np.float32)
    raw = arr_f32.tobytes()
    compressed = gzip.compress(raw, compresslevel=6)
    b64 = base64.b64encode(compressed).decode('ascii')
    ratio = len(raw) / len(compressed)
    print(f"  gzip compression ratio: {ratio:.1f}x  ({len(raw)/1e6:.1f} MB → {len(compressed)/1e6:.1f} MB)")
    return b64, list(tensor4d.shape)


def build_extra_colorings(extra_colorings, coloring_types, nonoutliers):
    """Extract extra colorings subset to nonoutlier indices."""
    out_vals = {}
    out_types = {}
    for name, arr in extra_colorings.items():
        arr = np.asarray(arr)
        sub = arr[nonoutliers]
        # Replace NaN with None for JSON
        if np.issubdtype(sub.dtype, np.floating):
            vals = [None if np.isnan(v) else round_sig(float(v), 5) for v in sub]
        else:
            vals = [int(v) for v in sub]
        out_vals[name] = vals
        out_types[name] = coloring_types.get(name, 'continuous')
    return out_vals, out_types


def compute_sweep_for_website(tensor_sdt, dyn_metrics):
    """Pre-compute acc and r2 sweep for all 5 properties × hi/lo + random.

    Parameters
    ----------
    tensor_sdt : (N, S, D, T) — already transposed from explorer format

    Returns
    -------
    fracs : list of float  (10 values)
    sweep_acc, sweep_r2 : dict[metric_name -> {hi, lo, rand_mean, rand_std}]
                          each value is a list of floats aligned to fracs
    """
    from src.subpop_utils import (
        compute_decoding_manifold, knn_decoding_accuracy,
        procrustes_r2, select_top_k_by_metric,
    )
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    rng = np.random.default_rng(0)
    N_SEEDS = 5
    FRACS = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.0])
    METRIC_NAMES = ['speed', 'stability', 'curvature', 'classifiability', 'pc_contrib']

    N = tensor_sdt.shape[0]
    NSTIMS, NDIRS = tensor_sdt.shape[1], tensor_sdt.shape[2]
    stim_labels = np.repeat(np.arange(NSTIMS), NDIRS)

    coords_full, _ = compute_decoding_manifold(tensor_sdt, n_components=3)

    def _pad(cs, ref):
        if cs.shape[1] < ref.shape[1]:
            return np.hstack([cs, np.zeros((cs.shape[0], ref.shape[1] - cs.shape[1]))])
        return cs

    sweep_acc = {}
    sweep_r2  = {}

    # Pre-compute random baseline (shared across all metrics)
    rand_acc_list, rand_r2_list = [], []
    for f in FRACS:
        k = max(1, int(round(f * N)))
        n_comp = min(3, k)
        _accs, _r2s = [], []
        for _ in range(N_SEEDS):
            idx = rng.choice(N, k, replace=False)
            t_sub = tensor_sdt[idx]
            cs, _ = compute_decoding_manifold(t_sub, n_components=n_comp)
            _accs.append(knn_decoding_accuracy(cs, stim_labels))
            _r2s.append(procrustes_r2(coords_full, _pad(cs, coords_full)))
        rand_acc_list.append((float(np.nanmean(_accs)), float(np.nanstd(_accs))))
        rand_r2_list.append((float(np.nanmean(_r2s)),  float(np.nanstd(_r2s))))

    for mname in METRIC_NAMES:
        hi_acc, lo_acc, hi_r2, lo_r2 = [], [], [], []
        for f in FRACS:
            k = max(1, int(round(f * N)))
            n_comp = min(3, k)
            for high, acc_list, r2_list in [
                (True,  hi_acc, hi_r2),
                (False, lo_acc, lo_r2),
            ]:
                idx = select_top_k_by_metric(dyn_metrics, mname, k=k, high=high)
                if len(idx) < 1:
                    acc_list.append(None)
                    r2_list.append(None)
                    continue
                t_sub = tensor_sdt[idx]
                cs, _ = compute_decoding_manifold(t_sub, n_components=min(3, len(idx)))
                acc_list.append(round_sig(float(knn_decoding_accuracy(cs, stim_labels)), 4))
                r2_list.append(round_sig(float(procrustes_r2(coords_full, _pad(cs, coords_full))), 4))

        sweep_acc[mname] = {
            'hi':        hi_acc,
            'lo':        lo_acc,
            'rand_mean': [round_sig(v[0], 4) for v in rand_acc_list],
            'rand_std':  [round_sig(v[1], 4) for v in rand_acc_list],
        }
        sweep_r2[mname] = {
            'hi':        hi_r2,
            'lo':        lo_r2,
            'rand_mean': [round_sig(v[0], 4) for v in rand_r2_list],
            'rand_std':  [round_sig(v[1], 4) for v in rand_r2_list],
        }

    return FRACS.tolist(), sweep_acc, sweep_r2


def detect_coloring_types(extra_colorings, nonoutliers):
    """Detect whether each extra coloring is categorical or continuous."""
    types = {}
    for name, arr in extra_colorings.items():
        arr = np.asarray(arr)
        sub = arr[nonoutliers]
        finite_vals = sub[np.isfinite(sub.astype(float))]
        if np.issubdtype(arr.dtype, np.integer) or len(np.unique(finite_vals)) <= 20:
            types[name] = 'categorical'
        else:
            types[name] = 'continuous'
    return types


def main():
    parser = argparse.ArgumentParser(description='Build static manifold explorer HTML')
    parser.add_argument('--prefix', default='flyvis_Medulla_i3_n550_model000',
                        help='Dataset prefix (default: flyvis_Medulla_i3_n550_model000)')
    parser.add_argument('--basedir-data', default='data/sampled',
                        help='Directory with tensor4d_*.npy files')
    parser.add_argument('--basedir-mat', default='data/decompositions',
                        help='Directory with .mat decomposition files')
    parser.add_argument('--basedir-wg', default='data/graphs',
                        help='Directory with IAN graph .npz files')
    parser.add_argument('--output-dir', default='website/site',
                        help='Output directory for the HTML file (default: website/site)')
    parser.add_argument('--output-name', default=None,
                        help='Output filename (default: <prefix>.html)')
    args = parser.parse_args()

    PREFIX = args.prefix
    if args.output_name is None:
        args.output_name = f'{PREFIX}.html'
    print(f"Loading data for prefix: {PREFIX}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    data = load_for_explorer(
        PREFIX,
        basedir_data=args.basedir_data,
        basedir_mat=args.basedir_mat,
        basedir_wG=args.basedir_wg,
    )

    embedding_      = data['embedding_']            # (N, 10)
    tensor4d        = data['tensor4d']              # (N, S, T, D) processed, for display
    tensor4d_raw    = data.get('tensor4d_raw', tensor4d)  # (N, S, T, D) raw, for decoding
    nonoutliers     = data['nonoutliers']           # (N,) int
    neurons_used    = data['neurons_used']          # (N_all, k)
    my_stims        = data['my_stims']
    NDIRS           = data['NDIRS']
    NSTIMS          = data['NSTIMS']
    extra_colorings = data['extra_colorings']

    N, S, T, D = tensor4d.shape
    print(f"  Neurons (nonoutliers): {N}, stims: {S}, time: {T}, dirs: {D}")
    print(f"  Embedding shape: {embedding_.shape}")
    print(f"  Stim labels: {my_stims}")

    # ── 2. Pre-compute full-population decoding (from raw tensor) ─────────────
    print("Computing full-population decoding manifold...")
    # subpop_utils expects (N, S, D, T); explorer stores (N, S, T, D)
    sub_raw = tensor4d_raw.transpose(0, 1, 3, 2)   # (N, S, D, T)
    decoding_coords, _ = compute_decoding_manifold(sub_raw, n_components=3)   # (S*D, 3)
    print("Computing full-population decoding trajectories...")
    decoding_trajs, _  = compute_decoding_trajectories(sub_raw, n_components=3)  # (S*D, T, 3)
    print(f"  decoding_coords: {decoding_coords.shape}")
    print(f"  decoding_trajs:  {decoding_trajs.shape}")

    # ── 3. Pre-compute dynamic metrics (speed, stability, curvature, etc.) ────
    # Uses processed tensor (smoothed, normalised) — these are encoding-side metrics.
    print("Computing dynamic metrics...")
    sub_sdt = tensor4d.transpose(0, 1, 3, 2)   # (N, S, D, T)
    dyn_metrics = compute_dynamic_metrics(sub_sdt)
    print(f"  metrics computed: {list(dyn_metrics.keys())}")

    # ── 3b. Pre-compute fraction sweep (from raw tensor, consistent with decoding) ─
    print("Computing fraction sweep for all 5 properties...")
    sweep_fracs, sweep_acc, sweep_r2 = compute_sweep_for_website(sub_raw, dyn_metrics)
    print(f"  sweep_fracs: {len(sweep_fracs)} points")

    # ── 4. Detect coloring types ──────────────────────────────────────────────
    coloring_types = detect_coloring_types(extra_colorings, nonoutliers)

    # ── 5. Extra colorings for neurons (subset to nonoutlier space) ───────────
    extra_vals, extra_types = build_extra_colorings(
        extra_colorings, coloring_types, nonoutliers)

    # ── 6. Encode tensor4d_raw as base64 float32 ─────────────────────────────
    # Only the raw tensor is stored — used for both subpop decoding and PSTH display.
    # The processed (smoothed/normalized) tensor is no longer embedded.
    print("Encoding tensor4d_raw as base64 float32...")
    tensor4d_raw_b64, tensor4d_shape = encode_tensor4d(tensor4d_raw)
    print(f"  raw base64 size: {len(tensor4d_raw_b64)/1e6:.1f} MB")

    # ── 7. Build neurons_fmap (column 1 of neurons_used, nonoutlier rows) ─────
    if neurons_used is not None and neurons_used.ndim >= 2 and neurons_used.shape[1] > 1:
        neurons_fmap = [int(neurons_used[i, 1]) for i in nonoutliers]
    else:
        neurons_fmap = list(range(N))

    # ── 8. Assemble JSON payload ──────────────────────────────────────────────
    print("Assembling JSON payload...")
    payload = {
        "prefix":        PREFIX,
        "nstims":        int(NSTIMS),
        "ndirs":         int(NDIRS),
        "trial_len":     int(T),
        "categories":    list(my_stims),
        "embedding":     to_json_list(embedding_, sig=5),
        "tensor4d_raw_b64": tensor4d_raw_b64,
        "tensor4d_shape":   tensor4d_shape,
        "neurons_fmap":  neurons_fmap,
        "extra_colorings":      extra_vals,
        "extra_coloring_types": extra_types,
        "decoding_coords": to_json_list(decoding_coords, sig=5),
        "decoding_trajs":  to_json_list(decoding_trajs, sig=5),
        "nonoutliers":     [int(i) for i in nonoutliers],
        "metrics":         {k: [round_sig(float(v), 5) for v in arr]
                            for k, arr in dyn_metrics.items()},
        "sweep_fracs": sweep_fracs,
        "sweep_acc":   sweep_acc,
        "sweep_r2":    sweep_r2,
    }

    json_str = json.dumps(payload, separators=(',', ':'))
    json_mb  = len(json_str.encode('utf-8')) / 1e6
    print(f"  JSON size: {json_mb:.1f} MB")

    # ── 8. Render template ────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, 'template.html')
    out_dir   = args.output_dir if args.output_dir is not None else script_dir
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, args.output_name)

    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    html = template.replace('{{DATA_JSON}}', json_str)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    html_mb = os.path.getsize(output_path) / 1e6
    print(f"\nWrote: {output_path}")
    print(f"  File size: {html_mb:.1f} MB")


if __name__ == '__main__':
    main()
