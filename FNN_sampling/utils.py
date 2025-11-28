import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
from scipy.io import loadmat

def createFlowDataset(categories, topdir, mydirs, orig_shape, input_shape, scl_factor, N_INSTANCES, trial_len, stride):
    scld_shape = tuple((np.array(orig_shape)*scl_factor).astype('int'))
    NDIRS = len(mydirs)
    frames_per_stim = int(np.ceil(trial_len/stride))
    
    shift_foos = {'0':lambda im,step: np.roll(im,step,1),
                  '45':lambda im,step: np.roll(np.roll(im,step,1),-step,0),
                  '90':lambda im,step: np.roll(im,-step,0),
                  '135':lambda im,step: np.roll(np.roll(im,-step,1),-step,0),
                  '180':lambda im,step: np.roll(im,-step,1),
                  '225':lambda im,step: np.roll(np.roll(im,-step,1),step,0),
                  '270':lambda im,step: np.roll(im,step,0),
                  '315':lambda im,step: np.roll(np.roll(im,step,0),step,1),
                 }
    
    flow_datasets = {}

    for inst_i in range(N_INSTANCES):   
        print('*INSTANCE',inst_i,end=' ',flush=True)
        for cat in categories: 
            print('.',end='',flush=True)
            stim_arrays = None

            for di,d in enumerate(mydirs):

                image_path = f'{topdir}/{cat}_inst{inst_i}/{d}/0.png'
                img = Image.open(image_path)

                assert orig_shape == img.size

                if scl_factor != 1:
                    img = img.resize(scld_shape, Image.Resampling.LANCZOS)

                #cropping idxs
                w,h = img.size
                assert w == scld_shape[0] and h == scld_shape[1]
                i0, j0 = h//2-input_shape[0]//2, w//2-input_shape[1]//2
                i1, j1 = i0 + input_shape[0], j0 + input_shape[1]

                img_array = np.array(img)[:,:,0] #since grayscale, use only one channel

                for fii,fi in enumerate(range(0,trial_len,stride)):
                    #shift full img
                    shifted_img = shift_foos[d](img_array,fi)
                    #crop from center
                    shifted_img = shifted_img[i0:i1,j0:j1]
                    #save
                    if stim_arrays is None:
                        stim_arrays = np.zeros((NDIRS*frames_per_stim,shifted_img.size))
                    stim_arrays[di*frames_per_stim+fii] = shifted_img.ravel()


            if inst_i not in flow_datasets:
                flow_datasets[inst_i] = stim_arrays
            else:
                flow_datasets[inst_i] = np.concatenate([flow_datasets[inst_i],stim_arrays])

        print()
    return flow_datasets

def from0to1(arr):
    arr = np.asanyarray(arr)
    arr[np.isclose(arr,0)] = 1
    return arr

def subps(nrows,ncols,rowsz=3,colsz=4,d3=False,axlist=False):
    if d3:
        f = plt.figure(figsize=(ncols*colsz,nrows*rowsz))
        axes = [[f.add_subplot(nrows,ncols,ri*ncols+ci+1, projection='3d') for ci in range(ncols)] \
                for ri in range(nrows)]
        if nrows == 1:
            axes = axes[0]
            if ncols == 1:
                axes = axes[0]
    else:
        f,axes = plt.subplots(nrows,ncols,figsize=(ncols*colsz,nrows*rowsz))
    if axlist and ncols*nrows == 1:
        
        axes = [axes]
    return f,axes

def twx():
    ax = plt.subplot(111)
    return ax, ax.twinx()
    
def npprint(a,precision=3):
    with np.printoptions(precision=precision, suppress=True):
        print(a)
    return

def plot_image(orig_image, fig_sz, ax=None, vmin=None, vmax=None,
              axis_off=True):


    image = orig_image.copy()
    
    assert image.min() >= 0
    if image.max() <= 1:
        image = (image*255).astype('int32')
    else:
        image = image.astype('int32')

    imsiz = image.shape[1]
    
    if ax is None:
        plt.figure(figsize=(fig_sz,fig_sz))
        plt.imshow(image, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(image, vmin=vmin, vmax=vmax)
        if axis_off:
            ax.axis('off')

def plot_images(images_, fig_sz=2, nrows=None, labels=None, vmin=None, vmax=None):

    images = images_.copy()
    if nrows is not None:
        ncols = int(np.ceil(len(images)/nrows))
    else:
        nrows = int(np.floor(np.sqrt(len(images))))
        ncols = int(np.ceil(np.sqrt(len(images))))
        
    if ncols*nrows < len(images):
        nrows += 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_sz*ncols,fig_sz*nrows))


    for i, ax in enumerate(axes.ravel()):
        if i >= len(images):
            ax.axis('off')
            continue

        img = images[i]
        

        ax.set_title('%d' % i,size=11)
        
        if labels is not None:
            plot_image(img, fig_sz, ax, vmin=vmin, vmax=vmax, axis_off=False)
            ax.set_xlabel(labels[i],size=8)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            plot_image(img, fig_sz, ax, vmin=vmin, vmax=vmax)
            
    fig.tight_layout()
    plt.show() 



def predict_images(images_data, fig_sz=None, nrows=None):
    imgs = []
    top1s = []
    for image_data in images_data:
        if type(image_data) == str:
            # Load and resize the image using PIL.
            img = Image.open(image_data)
            img_resized = img.resize(input_shape, Image.Resampling.LANCZOS)
            # Convert the PIL image to a numpy-array with the proper shape.
            img_array = np.array(img_resized)
        else:
            assert image_data.shape == (input_shape[0],input_shape[1],3)
            img_array = image_data.copy()
            
        imgs.append(img_array.astype('uint8'))
        
        img_array = preprocess_input(img_array)

        pred = model.predict(np.expand_dims(img_array, axis=0),verbose=0)

        pred_decoded = decode_predictions(pred)[0]

        code, name, score = pred_decoded[0]
        top1s.append("{0:>6.2%} : {1}".format(score, name))
        
    plot_images(np.array(imgs), fig_sz, nrows, labels=top1s, vmin=0, vmax=255)

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
    source = ','.join(i+common_dim for i in target)
    operation = source+'->'+target+common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))





"""def loadPreComputedCP(tensorname,basedir,specificFs=[],NMODES=3,verbose=True):
    
    Converts the .mat files produced by tensor decomposition (see `matlab/run_permcp.m`)
    and aggregates the results from multiple choices of F (number of components) and
    initializations into a single Python dict.
    
    
    preComputed = {}

    #parse the F (# of factors) from the file name
    def parseF(s):
        assert s[-4:] == '.mat'
        extn = 4
        F_str = s[:-extn].split('_F')[-1]
        return int(F_str)
    
    query = '%s/%s*_F*.mat' % (basedir,tensorname)
 
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
                print(f'({counted_reps})',end=' ')
            
            counted_reps = 0 #reset
            
            if verbose: print(f'F{F}:',end=' ')
            F_ = F
        #if another file from the same F, keep updating the number of reps
        


        matfile = loadmat(r)
        assert matfile['factors'][0,0].shape[1] == NMODES

        nreps = len(matfile['factors'][0])

        factors = {counted_reps+rep:matfile['factors'][0][rep].squeeze() for rep in range(nreps)}
        lambdas = {counted_reps+rep:matfile['lams'][0][rep].squeeze() for rep in range(nreps)}
        objs = {counted_reps+rep:matfile['objs'][0][rep].squeeze() for rep in range(nreps)}


        F_precomp = {'all_factors':factors, 'all_lambdas':lambdas, 'all_objs':objs}


        counted_reps += nreps
        
        if F not in preComputed:
            preComputed[F] = F_precomp.copy()

        else:#merge results
            for dkey in F_precomp.keys():
                preComputed[F][dkey].update(F_precomp[dkey])

    if int(verbose) == 1 and counted_reps > 0:
        print(f'({counted_reps})')
        
    Fs = sorted(preComputed.keys())
    return preComputed,Fs
"""


"""
metrics.py
-----------
Implements Gromov-Wasserstein, Gromov-Hausdorff approximation,
and single-linkage ultrametric-based GH approximation.
"""

import numpy as np
import ot
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import linkage
from collections import defaultdict

##########################################################
# 1) Gromov-Wasserstein
##########################################################

def compute_gromov_wasserstein(C1, C2, p, q, loss_fun='square_loss',
                               max_iter=10000, tol=1e-4):
    """
    Use POT library for Gromov-Wasserstein cost between cost/distance matrices C1, C2.
    p, q are distributions over the two sets.
    Returns the GW cost (NOT the sqrt).
    """
    gw_cost = ot.gromov.gromov_wasserstein2(
        C1, C2, p, q,
        loss_fun=loss_fun,
        max_iter=max_iter,
        tol=tol
    )
    return gw_cost

def compute_gromov_hausdorff_approx(X, Y, metric='euclidean'):
    """
    Approx GH distance ~ sqrt(GW). Return raw GW cost from NxD data X, Y.
    If you want the GH distance, do sqrt() of the returned cost.
    """
    distX = pdist(X, metric=metric)
    distY = pdist(Y, metric=metric)
    Cx = squareform(distX)
    Cy = squareform(distY)
    N, M = len(X), len(Y)
    p = np.ones(N)/N
    q = np.ones(M)/M
    gw_cost = ot.gromov.gromov_wasserstein2(
        Cx, Cy, p, q, loss_fun='square_loss',
        max_iter=10000, tol=1e-4
    )
    return gw_cost

##########################################################
# 2) Single-Linkage Ultrametric & GH on Ultrametrics
##########################################################

def compute_single_linkage_ultrametric(points, metric='euclidean'):
    """
    NxD => custom single-linkage => NxN ultrametric matrix of merge heights.
    Implements the algorithm from the specified paper with alpha = sqrt(2) and k = d * log(n).
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from math import log, ceil

    n, d = points.shape
    if n < 2:
        return np.zeros((n, n), dtype=float)

    alpha = np.sqrt(2)
    k = max(2, ceil(d * log(n)))  # Ensure k is at least 2

    # Step 1: Compute rk(xi) for each point xi
    # Using cKDTree for efficient k-nearest neighbor search
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k+1, p=2)  # k+1 because the first neighbor is the point itself
    rk = distances[:, -1]  # distance to k-th nearest neighbor

    # Step 2: Prepare all possible edges with distance <= alpha * max(rk)
    max_rk = np.max(rk)
    cutoff_distance = alpha * max_rk

    # Compute all pairs within cutoff_distance using cKDTree
    pairs = tree.query_pairs(r=cutoff_distance, p=2)

    # Convert set of pairs to a sorted list based on distance using vectorized operations
    if pairs:
        pair_list = np.array(list(pairs))
        # Vectorized computation of Euclidean distances
        diffs = points[pair_list[:, 0]] - points[pair_list[:, 1]]
        pair_distances = np.linalg.norm(diffs, axis=1)
        sorted_indices = np.argsort(pair_distances)
        sorted_pairs = pair_list[sorted_indices]
        sorted_distances = pair_distances[sorted_indices]
    else:
        sorted_pairs = np.empty((0, 2), dtype=int)
        sorted_distances = np.array([])

    # Initialize Union-Find structure
    parent = np.arange(n)
    rank_union = np.zeros(n, dtype=int)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu == pv:
            return False  # Already in the same set
        # Union by rank
        if rank_union[pu] < rank_union[pv]:
            parent[pu] = pv
        else:
            parent[pv] = pu
            if rank_union[pu] == rank_union[pv]:
                rank_union[pu] += 1
        return True

    # Function to perform full path compression for accurate member retrieval
    def compress_paths(parent):
        for u in range(len(parent)):
            find(u)

    # Initialize ultrametric matrix with zeros on the diagonal and infinities elsewhere
    U = np.full((n, n), np.inf)
    np.fill_diagonal(U, 0)

    # Initialize a list to keep track of when clusters are merged
    # We will iterate through the sorted pairs and merge clusters accordingly
    for (i, j), dist in zip(sorted_pairs, sorted_distances):
        # Determine the current r as the maximum of rk[i] and rk[j]
        current_r = max(rk[i], rk[j])
        # The condition to include the edge is dist <= alpha * r
        if dist > alpha * current_r:
            continue  # Do not include this edge

        # Attempt to union the clusters
        if union(i, j):
            # Perform full path compression to ensure accurate parent pointers
            compress_paths(parent)
            # Find the root of the merged cluster
            root = find(i)
            # Retrieve all members of the merged cluster
            members = np.where(parent == root)[0]
            # Update the ultrametric distances for all pairs within the merged cluster
            for m1 in members:
                for m2 in members:
                    if m1 < m2:
                        U[m1, m2] = min(U[m1, m2], current_r)
                        U[m2, m1] = U[m1, m2]

    # After processing all pairs, some pairs might still be infinity if they were never connected
    # To handle this, we can set their ultrametric distance to the maximum rk
    U[U == np.inf] = max_rk

    return U

# def compute_single_linkage_ultrametric(points, metric='euclidean'):
#     """
#     NxD => single-link => NxN ultrametric matrix of merge heights.
#     """
#     N = points.shape[0]
#     if N < 2:
#         return np.zeros((N,N), dtype=float)
#     condensed = pdist(points, metric=metric)
#     Z = linkage(condensed, 'single')
#     U = np.zeros((N, N), dtype=float)
#     cluster_members = {i: [i] for i in range(N)}
#     next_id = N
#     for i in range(Z.shape[0]):
#         c1, c2, dist, sample_count = Z[i]
#         c1, c2 = int(c1), int(c2)
#         mem1 = cluster_members.pop(c1, [c1])
#         mem2 = cluster_members.pop(c2, [c2])
#         merged = mem1 + mem2
#         for m1 in merged:
#             for m2 in merged:
#                 if m1 < m2:
#                     U[m1, m2] = dist
#                 elif m2 < m1:
#                     U[m2, m1] = dist
#         cluster_members[next_id] = merged
#         next_id += 1
#     return U

def approximate_gh_on_ultrametrics(U1, U2, loss_fun='square_loss', max_iter=10000, tol=1e-4):
    """
    Approx GH distance = sqrt( Gromov-Wasserstein(U1, U2) ) with uniform weights.
    U1, U2: NxN and MxM ultrametric distance matrices (not necessarily same size).
    """
    import ot
    N = U1.shape[0]
    M = U2.shape[0]
    p = np.ones(N)/N
    q = np.ones(M)/M
    # Normalize them so max=1
    U1n = normalize_distance_matrix(U1)
    U2n = normalize_distance_matrix(U2)

    cost = ot.gromov.gromov_wasserstein2(
        U1n, U2n, p, q, loss_fun=loss_fun, max_iter=max_iter, tol=tol
    )
    return np.sqrt(abs(cost))


# utils.py

import numpy as np
from typing import Optional, Tuple
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist

import logging

logger = logging.getLogger("utils")

DISCONNECTED_BRIDGING_FACTOR = 10.0

def connected_comp_helper(A: Optional[np.ndarray],
                          X: np.ndarray,
                          connect: bool = True) -> Optional[np.ndarray]:
    """
    Ensures graph connectivity by bridging disconnected components if connect=True.
    We replace 'inf' or 0 edges between components with bridging_val = largest_edge * DISCONNECTED_BRIDGING_FACTOR.
    """

    if A is None:
        logger.warning("Adjacency is None => skipping connectivity.")
        return A

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square. shape={A.shape}")

    n_components, comp_labels = connected_components(A, directed=False, return_labels=True)
    if n_components > 1:
        if connect:
            logger.info(f"Graph has {n_components} disconnected components => bridging them.")
            finite_mask = np.isfinite(A) & (A>0)
            if not np.any(finite_mask):
                bridging_val = 1e6
            else:
                largest_edge = np.max(A[finite_mask])
                bridging_val = largest_edge * DISCONNECTED_BRIDGING_FACTOR

            # For each adjacent pair of components c, c+1, we connect them
            # in the minimal-dist pair.
            for c in range(n_components - 1):
                comp_i = np.where(comp_labels == c)[0]
                comp_j = np.where(comp_labels == c+1)[0]

                dist_ij = cdist(X[comp_i], X[comp_j], metric='euclidean')
                # dist_ij = squareform(pdist(X[np.ix_(comp_i)], X[np.ix_(comp_j)]))
                
                min_idx = np.unravel_index(np.argmin(dist_ij), dist_ij.shape)
                vi = comp_i[min_idx[0]]
                vj = comp_j[min_idx[1]]
                A[vi, vj] = bridging_val
                A[vj, vi] = bridging_val
        else:
            logger.info(f"Graph has {n_components} disconnected parts, not bridging.")

    return A

def remove_duplicates(X: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes nearly-duplicate points based on a tolerance, returning the unique subset and the indices.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, shape={X.shape}")
    # naive approach: sort, find diffs
    sorted_idx = np.lexsort(np.argsort(X, axis=1))
    sorted_X = X[sorted_idx]
    diffs = np.diff(sorted_X, axis=0)
    dist = np.linalg.norm(diffs, axis=1)
    keep_mask = np.insert(dist > tol, 0, True)
    X_unique = sorted_X[keep_mask]
    # Re-map to original indices
    unique_indices = np.where(keep_mask)[0]
    return X_unique, unique_indices

def preprocess_distance_matrix(D, large_value_multiplier=20.0):
    """
    Replaces inf with largest_finite*large_value_multiplier if any inf appear in D.
    """
    if not np.isfinite(D).all():
        finite_mask = np.isfinite(D)
        if not np.any(finite_mask):
            raise ValueError("All distances are infinite => cannot proceed.")
        max_finite = np.max(D[finite_mask])
        large_val = max_finite * large_value_multiplier
        n_infs = np.sum(~finite_mask)
        logging.getLogger("experiment_logger").info(f"Replaced {n_infs} inf distances with {large_val}.")
        D = np.where(np.isinf(D), large_val, D)
    return D

def normalize_distance_matrix(D):
    """
    Scale matrix so that max=1. If max=0 => return D unchanged.
    """
    dmax = np.max(D)
    if dmax > 0:
        return D / dmax
    return D

def measure_from_potential(X, potential_name, potential_params, min_sum_threshold=1e-14):
    """
    Evaluate measure ~ exp(- potential(x)), then normalize.
    """
    from src.mesh_sampling import get_potential_func
    logger = logging.getLogger("experiment_logger")
    pot_func = get_potential_func(potential_name, potential_params)
    pot_vals = np.apply_along_axis(pot_func, 1, X)
    log_w = -pot_vals
    mx = np.max(log_w)
    log_w -= mx
    w = np.exp(log_w)
    s = w.sum()
    if s < min_sum_threshold:
        logger.warning(f"Potential measure sum < {min_sum_threshold} => fallback to uniform.")
        measure = np.ones(len(X)) / len(X)
    else:
        measure = w / s
    return measure