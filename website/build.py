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
import argparse
import struct

import numpy as np

# Make sure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache_utils import load_for_explorer
from src.subpop_utils import compute_decoding_manifold, compute_decoding_trajectories


def round_sig(x, sig=5):
    """Round a float to `sig` significant figures."""
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
    """Encode tensor4d as base64 float32 bytes (C order).

    Parameters
    ----------
    tensor4d : ndarray, shape (N, S, T, D)

    Returns
    -------
    b64 : str — base64-encoded raw bytes of float32 array
    shape : list of int — [N, S, T, D]
    """
    arr_f32 = np.ascontiguousarray(tensor4d, dtype=np.float32)
    raw = arr_f32.tobytes()
    b64 = base64.b64encode(raw).decode('ascii')
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
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for the HTML file (default: same as template.html)')
    parser.add_argument('--output-name', default='index.html',
                        help='Output filename (default: index.html)')
    args = parser.parse_args()

    PREFIX = args.prefix
    print(f"Loading data for prefix: {PREFIX}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    data = load_for_explorer(
        PREFIX,
        basedir_data=args.basedir_data,
        basedir_mat=args.basedir_mat,
        basedir_wG=args.basedir_wg,
    )

    embedding_     = data['embedding_']     # (N, 10)
    tensor4d       = data['tensor4d']       # (N, S, T, D)  explorer format
    nonoutliers    = data['nonoutliers']    # (N,) int
    neurons_used   = data['neurons_used']   # (N_all, k)
    my_stims       = data['my_stims']
    NDIRS          = data['NDIRS']
    NSTIMS         = data['NSTIMS']
    extra_colorings = data['extra_colorings']

    N, S, T, D = tensor4d.shape
    print(f"  Neurons (nonoutliers): {N}, stims: {S}, time: {T}, dirs: {D}")
    print(f"  Embedding shape: {embedding_.shape}")
    print(f"  Stim labels: {my_stims}")

    # ── 2. Pre-compute full-population decoding ───────────────────────────────
    print("Computing full-population decoding manifold...")
    # subpop_utils expects (N, S, D, T); explorer stores (N, S, T, D)
    sub = tensor4d.transpose(0, 1, 3, 2)   # (N, S, D, T)
    decoding_coords, _ = compute_decoding_manifold(sub, n_components=3)   # (S*D, 3)
    print("Computing full-population decoding trajectories...")
    decoding_trajs, _  = compute_decoding_trajectories(sub, n_components=3)  # (S*D, T, 3)
    print(f"  decoding_coords: {decoding_coords.shape}")
    print(f"  decoding_trajs:  {decoding_trajs.shape}")

    # ── 3. Detect coloring types ──────────────────────────────────────────────
    coloring_types = detect_coloring_types(extra_colorings, nonoutliers)

    # ── 4. Extra colorings for neurons (subset to nonoutlier space) ───────────
    extra_vals, extra_types = build_extra_colorings(
        extra_colorings, coloring_types, nonoutliers)

    # ── 5. Encode tensor4d as base64 float32 ─────────────────────────────────
    print("Encoding tensor4d as base64 float32...")
    tensor4d_b64, tensor4d_shape = encode_tensor4d(tensor4d)
    b64_mb = len(tensor4d_b64) / 1e6
    print(f"  base64 size: {b64_mb:.1f} MB")

    # ── 6. Build neurons_fmap (column 1 of neurons_used, nonoutlier rows) ─────
    if neurons_used is not None and neurons_used.ndim >= 2 and neurons_used.shape[1] > 1:
        neurons_fmap = [int(neurons_used[i, 1]) for i in nonoutliers]
    else:
        neurons_fmap = list(range(N))

    # ── 7. Assemble JSON payload ──────────────────────────────────────────────
    print("Assembling JSON payload...")
    payload = {
        "prefix":        PREFIX,
        "nstims":        int(NSTIMS),
        "ndirs":         int(NDIRS),
        "trial_len":     int(T),
        "categories":    list(my_stims),
        "embedding":     to_json_list(embedding_, sig=5),
        "tensor4d_b64":  tensor4d_b64,
        "tensor4d_shape": tensor4d_shape,
        "neurons_fmap":  neurons_fmap,
        "extra_colorings":      extra_vals,
        "extra_coloring_types": extra_types,
        "decoding_coords": to_json_list(decoding_coords, sig=5),
        "decoding_trajs":  to_json_list(decoding_trajs, sig=5),
        "nonoutliers":     [int(i) for i in nonoutliers],
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
