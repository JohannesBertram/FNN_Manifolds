"""Manifold learning utilities.

Covers:
  - compute_mds_embedding  — MDS via kernel PCA on diffusion map coordinates
  - run_hdbscan_clustering — HDBSCAN on IAN-graph mutual-reachability distances
"""

import numpy as np
from scipy import sparse
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import KernelPCA
from hdbscan.hdbscan_ import _tree_to_labels
from hdbscan.plots import CondensedTree


def compute_mds_embedding(diffmap_y, nPCs, n_components=10):
    """Compute MDS embedding using metric MDS (kernel PCA).

    Args:
        diffmap_y: diffusion map coordinates, shape (N, D)
        nPCs: number of leading diffusion-map dimensions to use
        n_components: number of output MDS components

    Returns:
        numpy.ndarray: MDS embedding, shape (N, n_components)
    """
    embedding_D2 = squareform(pdist(diffmap_y[:, :nPCs], 'sqeuclidean'))
    G_new = -0.5 * embedding_D2
    kernel_pca_ = KernelPCA(n_components=n_components, kernel="precomputed", random_state=0)
    return kernel_pca_.fit_transform(G_new)


def run_hdbscan_clustering(diffmap_y, nPCs, G, min_cluster_size=10):
    """Run HDBSCAN clustering using IAN graph core distances.

    Args:
        diffmap_y: diffusion map coordinates, shape (N, D)
        nPCs: number of leading dimensions to use
        G: sparse IAN graph (N × N)
        min_cluster_size: HDBSCAN min_cluster_size

    Returns:
        tuple: (cluster_labels, num_clusters, cond_tree, leaves)
    """
    embedding_D2 = squareform(pdist(diffmap_y[:, :nPCs], 'sqeuclidean'))
    N = embedding_D2.shape[0]
    D1 = np.sqrt(embedding_D2)

    if not sparse.issparse(G):
        G = sparse.csr_matrix(G)
    elif not isinstance(G, sparse.csr_matrix):
        G = G.tocsr()

    nbrs_idxs = np.split(G.indices, G.indptr)[1:-1]
    core_dists = np.array([max(D1[xi, nbrs_idxs[xi]]) for xi in range(N)])

    mutreach = D1.copy()
    for xi in range(N):
        mutreach[xi] = np.max([core_dists, core_dists[xi] * np.ones(N), D1[xi]], axis=0)
    np.fill_diagonal(mutreach, 0)

    flat_dist_mat = squareform(mutreach)
    Z = linkage(flat_dist_mat, method='single')
    leaves = leaves_list(optimal_leaf_ordering(Z, flat_dist_mat))

    cluster_labels, probabilities, stabilities, condensed_tree, single_linkage_tree = _tree_to_labels(
        None, Z, min_cluster_size=min_cluster_size
    )
    cond_tree = CondensedTree(condensed_tree, cluster_labels)
    num_clusters = np.unique(cluster_labels).size

    return cluster_labels, num_clusters, cond_tree, leaves
