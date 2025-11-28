import numpy as np
from scipy.optimize import lsq_linear
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
from scipy import sparse
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering, dendrogram
from sklearn.decomposition import KernelPCA
from hdbscan.hdbscan_ import _tree_to_labels
from hdbscan.plots import CondensedTree

def process_tensor_data(tensor4d, optSF, smooth_sig, method):
    """
    Process tensor data by applying spatial frequency optimization, smoothing, 
    and normalization methods.
    
    Args:
        tensor4d: 4D tensor (Neurons, Stimuli, Directions, Trial_len)
        optSF: bool, whether to use optimal spatial frequency
        smooth_sig: int, smoothing sigma (frames)
        method: str, normalization method ('relFR' or 'relNorm')
    
    Returns:
        tuple: (tensorX, relFRs, optStims, PREFIX2, tensorname, CPMETHOD, AREA)
    """
    N, NSTIMS, NDIRS, TRIAL_LEN = tensor4d.shape

    SF_med_idxs = [0,2,5,6,9,10] # idxs of medium SF stims plus LF Gratings (0)
    SF_hi_idxs = [0,1,3,4,7,8] #high SF plus LF Gratings (0)
    
    if optSF:
        optStims = []
        tensorX = np.zeros((N,NSTIMS//2+1,NDIRS * TRIAL_LEN))
        relFRs = np.zeros((N,NSTIMS//2+1))
    else:
        tensorX = np.zeros((N,NSTIMS,NDIRS * TRIAL_LEN))
        relFRs = np.zeros((N,NSTIMS))

    # Collect psts for those sampled neurons
    for nii in range(tensor4d.shape[0]):
        relMeanPosFRs = []
        all_psts = np.zeros((NSTIMS,NDIRS * TRIAL_LEN))
        for stimi in range(NSTIMS):
            
            pst = tensor4d[nii,stimi]
            
            if smooth_sig > 0:
                pst = gaussian_filter1d(pst,smooth_sig,axis=1)
                    
            relMeanPosFRs.append(max(pst.mean(1)))
            all_psts[stimi] = pst.ravel(order='F')
            
        relMeanPosFRs = np.array(relMeanPosFRs)

        if optSF:
            #choose optimal SF
            med_FRs = relMeanPosFRs[SF_med_idxs]
            hi_FRs = relMeanPosFRs[SF_hi_idxs]
            if (med_FRs > hi_FRs).sum() > (hi_FRs > med_FRs).sum():
                relMeanPosFRs = med_FRs
                tensorX[nii] = all_psts[SF_med_idxs]
                optStims.append('med')
            else:
                relMeanPosFRs = hi_FRs
                tensorX[nii] = all_psts[SF_hi_idxs]
                optStims.append('hi')
                
        relFRs[nii] = relMeanPosFRs/max(relMeanPosFRs)

        if method == 'relFR':
            tensorX[nii] /= tensorX[nii].max()

        if method == 'Norm':
            tensorX[nii] /= tensorX[nii].mean()
            
        elif method == 'relNorm':#make each pst unit norm, then rescale according to relative FR
            stim_norms = from0to1(np.linalg.norm(tensorX[nii],axis=1,keepdims=1))
            tensorX[nii] /= from0to1(stim_norms)
            tensorX[nii] *= relFRs[nii][:,None]
    
    return tensorX, relFRs, optStims

def getPermutedTensor(factors, lambdas, tensorX, NDIRS):
    
    """Find the optimal circular-shifts used by the permuted decomposition to produce
    the tensor components, and apply it to the original tensor."""
    
    # Compute reconstructed tensor by scaling the first mode by the lambdas and 
    # multiplying by the kathri rao product of the other modes
    fittensor = np.reshape((lambdas * factors[0]) @ khatri_rao(factors[1:]).T, tensorX.shape)

    if NDIRS == 1: #no shifting possible, so simply return original tensor
        return tensorX, fittensor

    N = tensorX.shape[0]
    NSTIMS = tensorX.shape[1]
    RLEN = tensorX.shape[2]

    shape4d = (N,NSTIMS,NDIRS,RLEN//NDIRS)
    shapeDot = (N,RLEN)
    tensor4d = np.reshape(tensorX,shape4d,order='F')

    objs = np.empty((NSTIMS,N,NDIRS))
    obj_shifts = np.empty((NSTIMS,N))
    #find best shift (argmin) per stim for all cells at once
    for si in range(NSTIMS):
        for shifti in range(NDIRS):
            # cf. matlab code in `permuted-decomposition/matlab/my_tt_cp_fg.m`
            objs[si,:,shifti] = -np.sum(fittensor[:,si,:] * np.reshape(np.roll(tensor4d[:,si],shifti,1),shapeDot,order='F'), 1)
        obj_shifts[si] = np.argmin(objs[si],axis=1)

    #apply shifts
    shifted_tensor = np.zeros_like(tensorX)
    for shifti in range(NDIRS):
        rolledX = np.reshape(np.roll(tensor4d,shifti,2), tensorX.shape, order='F')
        for si in range(NSTIMS):
            shifted_tensor[(obj_shifts[si] == shifti),si,:] = rolledX[(obj_shifts[si] == shifti),si,:]

    #check that we get the same fit -- OK
    # normsqX = np.square(norm(tensorX.ravel()))
    # print((np.square(norm(shifted_tensor.ravel() - fittensor.ravel())))/( normsqX))
    # print('rec. error',preComputed[best_nfactors]['all_objs'][best_rep])
    return shifted_tensor, fittensor

def getNeuralMatrix(scld_permT, factors, lambdas, NDIRS, all_zeroed_stims=None,
                    order='F', verbose=True):
    """Computes the final neural matrix, X, by fitting the permuted tensor scaled by
    relative stimulus magnitudes using the factors obtained from NTF.
    
    Any previously zeroed out responses are now also permuted by the circular-shift
    producing the best fit.
    
    Additionally, a rebalancing of the factor magnitudes is applied to attribute
    a meaningful interpretation to the final coefficients.
    
    -------------------
    Arguments:
    
    scld_permT: ndarray, permuted tensor scaled by relative stimulus FRs
    
    factors: list, [neural_factors, stimulus_factors, response_factors] (normalized)
    
    lambdas: ndarray, shape (R,), where R is the number of components being used
    
    NDIRS: int, number of stimulus directions (rows in original 2D response maps)
    
    all_zeroed_stims: dict, {cell: (tuple of zeroed stim idxs)}, default None
    
    order: str, order used to flatten the original 2D response maps, default 'F' 
    
    -------------------
    Returns:
    X: ndarray, shape (Ncells, R), neural encoding matrix
    
    new_scld_permT: ndarray, tensor including previously zeroed out responses (if any)
    
    """

    R = lambdas.size
    
    #rebalance factor loadings based on relative stimulus contributions + scale by lambdas
    stim_factors = factors[1].copy()
    stim_scls = stim_factors.max(0,keepdims=1)
    stim_factors /= stim_scls

    neural_factors = factors[0].copy()
    neural_factors *= lambdas * stim_scls
    
    # rescaled stim x response coords
    new_coords = np.stack([khatri_rao([stim_factors[:,r][:,None],factors[2][:,r][:,None]]).ravel() for r in range(R)],axis=1)

    
    Ncells = scld_permT.shape[0]
    NSTIMS = scld_permT.shape[1]
    
    X = np.zeros((Ncells,R))
    
    new_scld_permT = scld_permT.copy()

    for c in range(Ncells):
        
        if verbose and (c+1) % 50 == 0: print(c+1,end=' ')

        if all_zeroed_stims is not None and c in all_zeroed_stims:
            # Any previously zeroed out responses are now also permuted by the circular-shift
            # producing the best fit.
            
            lowest_cost = np.inf
            #for each shift of all zeroed-stims together
            for shifti in range(NDIRS):
                shifted_cell_data = scld_permT[c].copy()

                for si in all_zeroed_stims[c]:
                    #rotate orig_data
                    si_2d = shifted_cell_data[si].reshape((NDIRS,-1),order=order)
                    shifted_cell_data[si] = np.roll(si_2d,shifti,axis=0).ravel(order=order)

                #compute fit cost
                res = lsq_linear(new_coords,shifted_cell_data.ravel(),bounds=(0,np.inf))
                coeffs, cost = res['x'], res['cost']

                #if lower reconstruction cost, update best shift combo
                if cost < lowest_cost:
                    lowest_cost = cost
                    best_shift = shifti
                    best_coeffs = coeffs
                    best_partial = True
                    new_scld_permT[c] = shifted_cell_data

            new_coeffs = best_coeffs

                
        else:#if no zeroed stims
            # update coefficients to fit our stimulus-rescaled tensor
            new_coeffs = lsq_linear(new_coords,scld_permT[c].ravel(),bounds=(0,np.inf))['x']

        # sqrt so that, for each stimulus, the magnitude of a vector of coeffs for factors
        # representing that stimulus can be at most 1, even if that stimulus response 
        # is split across multiple factors. This ultimately leads to better distances
        # between neurons
        X[c] = np.sqrt(new_coeffs)

    return X, new_scld_permT

def from0to1(arr):
    arr = np.asanyarray(arr)
    arr[np.isclose(arr,0)] = 1
    return arr


def loadPreComputedCP(tensorname,basedir,specificFs=[],NMODES=3,verbose=True):
    
    """Converts the .mat files produced by tensor decomposition (see `matlab/run_permcp.m`)
    and aggregates the results from multiple choices of F (number of components) and
    initializations into a single Python dict."""
    
    
    preComputed = {}

    #parse the F (# of factors) from the file name
    def parseF(s):
        assert s[-4:] == '.mat'
        extn = 4
        F_str = s[:-extn].split('_F')[-1]
        if '_nreps' in F_str:
            F_str = F_str.split('_nreps')[0]
        return int(F_str)
    
    query = '%s/%s*_F*.mat' % (basedir,tensorname)
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
                print(f'({counted_reps})',end=' ')
            
            counted_reps = 0 #reset
            
            if verbose: print(f'F{F}:',end=' ')
            F_ = F
        #if another file from the same F, keep updating the number of reps
        


        matfile = loadmat(r)
        #assert matfile['factors'][0,0].shape[1] == NMODES

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

def compute_graph_statistics(wG, neurons_used, nonoutliers):
    """
    Compute graph connectivity statistics including degrees and feature map connections.
    
    Args:
        wG: sparse weighted graph
        neurons_used: array of neuron metadata 
        nonoutliers: indices of non-outlier neurons
    
    Returns:
        dict: Statistics including degrees, weighted degrees, same/other fmap connections
    """
    dense_wG = wG.toarray()

    c = neurons_used[nonoutliers,1] # labels for the different feature maps
    ulbls, labels = np.unique(c,return_inverse=True)
    same_fmaps = []
    other_fmaps = []
    weighted_degrees = []
    degrees = []
    
    for i in range(len(dense_wG)):
        weighted_degrees.append(np.sum(dense_wG[i]) - 1)

        degrees.append(np.sum(dense_wG[i] > 0) - 1)
        num_same_fmap = 0
        num_other_fmap = 0
        for j in range(len(dense_wG)):
            if dense_wG[i, j] > 0 and i != j:
                if labels[i] == labels[j]:
                    num_same_fmap += 1
                else:
                    num_other_fmap += 1
        same_fmaps.append(num_same_fmap)
        other_fmaps.append(num_other_fmap)

    stats = {
        'degree_mean': np.mean(degrees),
        'degree_min': np.min(degrees),
        'degree_max': np.max(degrees),
        'degree_std': np.std(degrees),
        'weighted_degree_mean': np.mean(weighted_degrees),
        'weighted_degree_min': np.min(weighted_degrees),
        'weighted_degree_max': np.max(weighted_degrees),
        'weighted_degree_std': np.std(weighted_degrees),
        'same_fmap_mean': np.mean(same_fmaps),
        'same_fmap_min': np.min(same_fmaps),
        'same_fmap_max': np.max(same_fmaps),
        'same_fmap_std': np.std(same_fmaps),
        'other_fmap_mean': np.mean(other_fmaps),
        'other_fmap_min': np.min(other_fmaps),
        'other_fmap_max': np.max(other_fmaps),
        'other_fmap_std': np.std(other_fmaps),
        'degrees': degrees,
        'weighted_degrees': weighted_degrees,
        'same_fmaps': same_fmaps,
        'other_fmaps': other_fmaps
    }
    
    return stats

def handle_disconnected_points(disc_pts, optScales, G, D2, outliers_list, neurons_used, X):
    """
    Handle disconnected points from IAN optimization by updating graphs and outlier lists.
    
    Args:
        disc_pts: list of disconnected points from IAN
        optScales: optimal scales from IAN
        G: graph matrix
        D2: distance matrix
        outliers_list: current list of outliers
        neurons_used: neuron metadata
        X: data matrix
    
    Returns:
        tuple: (wG, G, outliers_list, nonoutliers) updated versions
    """
    from ian.ian import getSparseMultiScaleK
    
    new_outliers = [disc_pts[di][0] for di in range(len(disc_pts))]
    nonout_mask = np.ones(optScales.size, dtype=bool)
    nonout_mask[new_outliers] = False
    
    if new_outliers:
        wG = getSparseMultiScaleK(D2[nonout_mask][:,nonout_mask],optScales[nonout_mask])
        G = G[nonout_mask][:,nonout_mask]
        
        # Update list of outliers and nonoutliers
        index_map = np.ones(len(neurons_used), dtype=bool)
        index_map[outliers_list] = False
        original_indices = np.where(index_map)[0]

        # Convert second round indices to original indices
        mapped_second_outliers = original_indices[new_outliers]

        outliers_list = np.append(outliers_list, mapped_second_outliers)
        nonoutliers = np.array([i for i in range(X.shape[0]) if i not in outliers_list])
        
        return wG, G, outliers_list, nonoutliers, nonout_mask
    else:
        # Return original values if no new outliers
        from ian.ian import getSparseMultiScaleK
        wG = getSparseMultiScaleK(D2, optScales)
        nonoutliers = np.array([i for i in range(X.shape[0]) if i not in outliers_list])
        return wG, G, outliers_list, nonoutliers, nonout_mask

def compute_mds_embedding(diffmap_y, nPCs, n_components=10):
    """
    Compute MDS embedding using metric MDS to find low-dim projection.
    
    Args:
        diffmap_y: diffusion map coordinates
        nPCs: number of principal components to use
        n_components: number of MDS components
    
    Returns:
        numpy.ndarray: MDS embedding
    """
    ndcs = nPCs
    embedding_D2 = squareform(pdist(diffmap_y[:,:ndcs],'sqeuclidean'))
    G_new = -.5 * embedding_D2
    kernel_pca_ = KernelPCA(
        n_components=n_components,
        kernel="precomputed",
    )
    embedding_ = kernel_pca_.fit_transform(G_new)
    return embedding_

def run_hdbscan_clustering(diffmap_y, nPCs, G, min_cluster_size=10):
    """
    Run HDBSCAN clustering to estimate number of clusters in embedded data.
    
    Args:
        diffmap_y: diffusion map coordinates
        nPCs: number of principal components to use
        G: sparse graph matrix
        min_cluster_size: minimum cluster size for HDBSCAN
    
    Returns:
        tuple: (cluster_labels, num_clusters, cond_tree)
    """
    ndcs = nPCs 
    embedding_D2 = squareform(pdist(diffmap_y[:,:ndcs],'sqeuclidean'))
    N = embedding_D2.shape[0]
    D1 = np.sqrt(embedding_D2)

    if not sparse.issparse(G):
        G = sparse.csr_matrix(G)
    elif not isinstance(G, sparse.csr_matrix):
        G = G.tocsr()

    # Compute "core distances" using the IAN discrete graph G
    nbrs_idxs = np.split(G.indices, G.indptr)[1:-1]
    core_dists = np.array([max(D1[xi,nbrs_idxs[xi]]) for xi in range(N)])

    # Compute "mutual reachability" from core distances
    mutreach = D1.copy()
    for xi in range(N):
        mutreach[xi] = np.max([core_dists,core_dists[xi]*np.ones(N), D1[xi]],axis=0)
    np.fill_diagonal(mutreach,0)

    # Compute the dendrogram + condensed tree
    flat_dist_mat = squareform(mutreach)
    Z = linkage(flat_dist_mat, method='single')
    leaves = leaves_list(optimal_leaf_ordering(Z, flat_dist_mat))

    cluster_labels, probabilities, stabilities, condensed_tree, single_linkage_tree = _tree_to_labels(None,Z,
                                                                                              min_cluster_size=min_cluster_size)
    cond_tree = CondensedTree(condensed_tree)
    num_clusters = np.unique(cluster_labels).size
    
    return cluster_labels, num_clusters, cond_tree, leaves

def compute_osi_and_pref_stim(tensorX):
    """
    Compute Orientation Selectivity Index (OSI) and preferred stimulus for each neuron.
    
    Args:
        tensorX: processed tensor data
    
    Returns:
        tuple: (OSI_final, pref_stim)
    """
    # Reshape tensor assuming 6 stimuli and 8 directions
    tensor_reshaped = tensorX.reshape((len(tensorX), 6, 8, -1))
    data_avg = tensor_reshaped.mean(axis=3)  # Shape: (N_neurons, 6, 8)

    OSI_per_stimulus = np.zeros((len(tensor_reshaped), 6))  # (Neurons, Stimuli)

    for stim_idx in range(6):
        tuning_curves = data_avg[:, stim_idx, :] 
        pref_ori = tuning_curves.argmax(axis=1) 
        orth_ori_pos = (pref_ori + 2) % 8  
        orth_ori_neg = (pref_ori - 2) % 8
        
        pref_responses = tuning_curves[np.arange(len(tensor_reshaped)), pref_ori]
        orth_responses = (tuning_curves[np.arange(len(tensor_reshaped)), orth_ori_pos] + 
                         tuning_curves[np.arange(len(tensor_reshaped)), orth_ori_neg]) / 2
        
        OSI_per_stimulus[:, stim_idx] = (pref_responses - orth_responses) / (pref_responses + orth_responses + 1e-12)

    OSI_final = OSI_per_stimulus.max(axis=1) 

    # Pref stim calculation
    stim_responses = data_avg.mean(axis=2) 
    pref_stim = stim_responses.argmax(axis=1) 
    
    return OSI_final, pref_stim


import scipy
def tsp_linearize(data, niter=1000, metric='euclidean', **kwargs):
    """Sorts a matrix dataset to (approximately) solve the traveling salesperson problem."""
    N = data.shape[0]
    D = scipy.spatial.distance.pdist(data, metric=metric, **kwargs)
    dist = np.zeros((N+1, N+1))
    dist[:N, :N] = scipy.spatial.distance.squareform(D)
    perm, _ = _solve_tsp(dist, niter)
    i = np.argwhere(perm == N).ravel()[0]
    perm = np.hstack((perm[(i+1):], perm[:i]))
    return perm

def reverse_segment(path, n1, n2):
    q = path.copy()
    if n2 > n1:
        q[n1:(n2+1)] = path[n1:(n2+1)][::-1]
        return q
    else:
        seg = np.hstack((path[n1:], path[:(n2+1)]))[::-1]
        brk = len(q) - n1
        q[n1:] = seg[:brk]
        q[:(n2+1)] = seg[brk:]
        return q

def _solve_tsp(dist, niter=1000):
    N = dist.shape[0] - 1
    path = np.arange(N+1)
    path[-1] = N
    cost_hist = []
    for _ in range(niter):
        n1, n2 = np.sort(np.random.choice(N, 2, replace=False))
        if n1 == n2:
            continue
        new_path = reverse_segment(path, n1, n2)
        old_cost = dist[path[n1-1], path[n1]] + dist[path[n2], path[(n2+1)%len(path)]]
        new_cost = dist[new_path[n1-1], new_path[n1]] + dist[new_path[n2], new_path[(n2+1)%len(path)]]
        if new_cost < old_cost:
            path = new_path
        cost_hist.append(np.sum([dist[path[i], path[i+1]] for i in range(len(path)-1)]))
    return path, cost_hist