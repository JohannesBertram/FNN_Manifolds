"""FNN_Manifolds shared analysis package.

Import everything from submodules so notebooks can do::

    import sys; sys.path.insert(0, '..')
    from src import *

Or import individual modules::

    from src.tensor_utils import loadPreComputedCP, process_tensor_data
    from src.manifold_utils import compute_mds_embedding
    from src.graph_utils import compute_graph_statistics
    from src.metrics import compute_osi_and_pref_stim
    from src.plot_utils import subps
"""

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from .tensor_utils import (
    khatri_rao,
    from0to1,
    process_tensor_data,
    select_sf_stims_for_decoding,
    SF_MED_IDXS,
    SF_HI_IDXS,
    SF_LOW_IDXS,
    SF_HIGH_IDXS,
    getPermutedTensor,
    getNeuralMatrix,
    loadPreComputedCP,
)

from .manifold_utils import (
    compute_mds_embedding,
    run_hdbscan_clustering,
)

from .graph_utils import (
    compute_graph_statistics,
    handle_disconnected_points,
    connected_comp_helper,
)

from .metrics import (
    compute_osi_and_pref_stim,
    compute_temporal_variance,
    compute_mean_activation,
    compute_dsi,
    normalize_distance_matrix,
    compute_gromov_wasserstein,
    compute_entropic_gw,
    compute_fgw,
    compute_gromov_hausdorff_approx,
    compute_single_linkage_ultrametric,
    approximate_gh_on_ultrametrics,
    build_index_maps,
    natmovie_to_manifold_indices,
    manifold_to_natmovie_indices,
)

from .plot_utils import (
    createFlowDataset,
    subps,
    twx,
    npprint,
    plot_image,
    plot_images,
)

from .cache_utils import load_for_explorer
from .explorer_utils import ManifoldExplorer
from .cp_utils import load_or_run_cp
from .subpop_utils import variance_reproduced_from_tensor, compute_dsa

__all__ = [
    # tensor_utils
    'khatri_rao', 'from0to1', 'process_tensor_data',
    'select_sf_stims_for_decoding', 'SF_MED_IDXS', 'SF_HI_IDXS', 'SF_LOW_IDXS', 'SF_HIGH_IDXS',
    'getPermutedTensor', 'getNeuralMatrix', 'loadPreComputedCP',
    # manifold_utils
    'compute_mds_embedding', 'run_hdbscan_clustering',
    # graph_utils
    'compute_graph_statistics', 'handle_disconnected_points', 'connected_comp_helper',
    # metrics
    'compute_osi_and_pref_stim', 'compute_temporal_variance', 'compute_mean_activation',
    'compute_dsi',
    'normalize_distance_matrix', 'compute_gromov_wasserstein',
    'compute_entropic_gw', 'compute_fgw',
    'compute_gromov_hausdorff_approx', 'compute_single_linkage_ultrametric',
    'approximate_gh_on_ultrametrics', 'build_index_maps',
    'natmovie_to_manifold_indices', 'manifold_to_natmovie_indices',
    # plot_utils
    'createFlowDataset', 'subps', 'twx', 'npprint', 'plot_image', 'plot_images',
    # cache_utils
    'load_for_explorer',
    # explorer_utils
    'ManifoldExplorer',
    # cp_utils
    'load_or_run_cp',
    # subpop_utils (tensor-based variants)
    'variance_reproduced_from_tensor', 'compute_dsa',
]
