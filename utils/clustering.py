"""
Clustering wrappers for BERTopic pipeline.

Provides factory functions for UMAP (dimensionality reduction) and HDBSCAN
(density-based clustering) with locked hyperparameters per methodology.
Only the grid-search variables (n_neighbors, min_cluster_size) are exposed
as arguments; all other settings are fixed to ensure reproducibility across
sensitivity analysis runs.

Methodology reference: bertopic_session_handoff_v2.md
"""

from umap import UMAP
from hdbscan import HDBSCAN


# Fixed UMAP settings (locked per methodology)
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

# Fixed HDBSCAN settings (locked per methodology)
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"
HDBSCAN_PREDICTION_DATA = True


def build_umap_model(n_neighbors: int) -> UMAP:
    """
    Build UMAP instance with locked hyperparameters.

    Only n_neighbors is variable (grid: [10, 15, 25]). All other parameters
    are fixed to ensure reproducibility across sensitivity analysis runs.

    Parameters
    ----------
    n_neighbors : int
        Number of neighboring points used in local approximations of manifold
        structure. Larger values capture more global structure; smaller values
        preserve more local detail.

    Returns
    -------
    UMAP
        Configured UMAP instance, ready to be passed to BERTopic.

    Examples
    --------
    >>> umap_model = build_umap_model(n_neighbors=15)
    >>> reduced = umap_model.fit_transform(embeddings)  # (n_docs, 5)
    """
    return UMAP(
        n_neighbors=n_neighbors,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )


def build_hdbscan_model(min_cluster_size: int) -> HDBSCAN:
    """
    Build HDBSCAN instance with locked hyperparameters.

    Only min_cluster_size is variable (grid: [30, 60, 100]). min_samples is
    derived as min_cluster_size // 2 per methodology. All other parameters
    are fixed.

    Parameters
    ----------
    min_cluster_size : int
        Minimum number of documents required to form a cluster (topic).
        Smaller values yield more, finer-grained topics; larger values yield
        fewer, broader topics.

    Returns
    -------
    HDBSCAN
        Configured HDBSCAN instance, ready to be passed to BERTopic.

    Examples
    --------
    >>> hdbscan_model = build_hdbscan_model(min_cluster_size=60)
    >>> # min_samples will be 30 (= 60 // 2)
    """
    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_cluster_size // 2,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
        prediction_data=HDBSCAN_PREDICTION_DATA,
    )
