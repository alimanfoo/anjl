import numpy as np
from numpy.typing import NDArray
from numba import njit, int64, float32, bool_, void
from . import params
from numpydoc_decorator import doc


INT64_MIN = np.int64(np.iinfo(np.int64).min)
FLOAT32_INF = np.float32(np.inf)
NOGIL = True
FASTMATH = False  # setting True actually seems to slow things down
ERROR_MODEL = "numpy"
BOUNDSCHECK = False


@doc(
    summary="""Perform neighbour-joining using the canonical algorithm.""",
    extended_summary="""
        This implementation performs a full scan of the distance matrix in each
        iteration of the algorithm to find the pair of nearest neighbours. It is
        therefore slower and scales with the cube of the number of original observations
        in the distance matrix, i.e., O(n^3).
    """,
)
def canonical_nj(
    D: params.D,
    disallow_negative_distances: params.disallow_negative_distances = True,
    progress: params.progress = None,
    progress_options: params.progress_options = {},
) -> params.Z:
    """TODO"""

    # Make a copy of distance matrix D because we will overwrite it during the
    # algorithm.
    D_copy: NDArray[np.float32] = np.array(D, copy=True, order="C", dtype=np.float32)
    del D

    # Number of original observations.
    n_original = D_copy.shape[0]

    # Expected number of new (internal) nodes that will be created.
    n_internal = n_original - 1

    # Map row indices to node IDs.
    index_to_id: NDArray[np.int64] = np.arange(n_original, dtype=np.int64)

    # Initialise output. This is similar to the output that scipy hierarchical
    # clustering functions return, where each row contains data for one internal node
    # in the tree, except that each row here contains:
    # - left child node ID
    # - right child node ID
    # - distance to left child node
    # - distance to right child node
    # - total number of leaves
    Z: NDArray[np.float32] = np.zeros(shape=(n_internal, 5), dtype=np.float32)

    # Initialize the "divergence" array, containing sum of distances to other nodes.
    U: NDArray[np.float32] = np.sum(D_copy, axis=1)

    # Keep track of which rows correspond to nodes that have been clustered.
    obsolete: NDArray[np.bool_] = np.zeros(shape=n_original, dtype=np.bool_)

    # Support wrapping the iterator in a progress bar.
    iterator = range(n_internal)
    if progress:
        iterator = progress(iterator, **progress_options)

    # Begin iterating.
    for iteration in iterator:
        # Perform one iteration of the neighbour-joining algorithm.
        canonical_iteration(
            iteration=iteration,
            D=D_copy,
            U=U,
            index_to_id=index_to_id,
            obsolete=obsolete,
            Z=Z,
            n_original=n_original,
            disallow_negative_distances=disallow_negative_distances,
        )

    return Z


@njit(
    (
        float32[:, :],  # D
        float32[:],  # U
        bool_[:],  # obsolete
        int64,  # n_remaining
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def canonical_search(
    D: NDArray[np.float32],
    U: NDArray[np.float32],
    obsolete: NDArray[np.bool_],
    n_remaining: int,
) -> tuple[np.int64, np.int64]:
    # Search for the closest pair of neighbouring nodes to join.
    q_min = FLOAT32_INF
    i_min = INT64_MIN
    j_min = INT64_MIN
    coefficient = float32(n_remaining - 2)
    m = D.shape[0]
    for i in range(m):
        if obsolete[i]:
            continue
        u_i = U[i]
        for j in range(i):
            if obsolete[j]:
                continue
            u_j = U[j]
            d = D[i, j]
            q = coefficient * d - u_i - u_j
            if q < q_min:
                q_min = q
                i_min = int64(i)
                j_min = int64(j)
    return i_min, j_min


@njit(
    void(
        float32[:, :],  # D
        float32[:],  # U
        int64[:],  # index_to_id
        bool_[:],  # obsolete
        int64,  # parent
        int64,  # i_min
        int64,  # j_min
        float32,  # d_ij
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def canonical_update(
    D: NDArray[np.float32],
    U: NDArray[np.float32],
    index_to_id: NDArray[np.int64],
    obsolete: NDArray[np.bool_],
    parent: np.int64,
    i_min: np.int64,
    j_min: np.int64,
    d_ij: np.float32,
) -> None:
    # Here we obsolete the row and column corresponding to the node at j_min, and we
    # reuse the row and column at i_min for the new node.
    obsolete[j_min] = True
    index_to_id[i_min] = parent

    # Initialize divergence for the new node.
    u_new = float32(0)

    # Update distances and divergence.
    for k in range(D.shape[0]):
        if obsolete[k] or k == i_min or k == j_min:
            continue

        # Calculate distance from k to the new node.
        d_ki = D[k, i_min]
        d_kj = D[k, j_min]
        d_k_new = 0.5 * (d_ki + d_kj - d_ij)
        D[i_min, k] = d_k_new
        D[k, i_min] = d_k_new

        # Subtract out the distances for the nodes that have just been joined and add
        # in distance for the new node.
        u_k = U[k] - d_ki - d_kj + d_k_new
        U[k] = u_k

        # Accumulate divergence for the new node.
        u_new += d_k_new

    # Assign divergence for the new node.
    U[i_min] = u_new


@njit(
    void(
        int64,  # iteration
        float32[:, :],  # D
        float32[:],  # U
        int64[:],  # index_to_id
        bool_[:],  # obsolete
        float32[:, :],  # Z
        int64,  # n_original
        bool_,  # disallow_negative_distances
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def canonical_iteration(
    iteration: int,
    D: NDArray[np.float32],
    U: NDArray[np.float32],
    index_to_id: NDArray[np.int64],
    obsolete: NDArray[np.bool_],
    Z: NDArray[np.float32],
    n_original: int,
    disallow_negative_distances: bool,
) -> None:
    # This will be the identifier for the new node to be created in this iteration.
    parent = iteration + n_original

    # Number of nodes remaining in this iteration.
    n_remaining = n_original - iteration

    if n_remaining > 2:
        # Search for the closest pair of nodes to join.
        i_min, j_min = canonical_search(
            D=D, U=U, obsolete=obsolete, n_remaining=n_remaining
        )

        # Calculate distances to the new internal node.
        d_ij = D[i_min, j_min]
        d_i = 0.5 * (d_ij + (1 / (n_remaining - 2)) * (U[i_min] - U[j_min]))
        d_j = 0.5 * (d_ij + (1 / (n_remaining - 2)) * (U[j_min] - U[i_min]))

    else:
        # Termination. Join the two remaining nodes, placing the final node at the
        # midpoint.
        i_min, j_min = np.nonzero(~obsolete)[0]
        d_ij = D[i_min, j_min]
        d_i = d_ij / 2
        d_j = d_ij / 2

    # Handle possibility of negative distances.
    if disallow_negative_distances:
        d_i = max(0, d_i)
        d_j = max(0, d_j)

    # Get IDs for the nodes to be joined.
    child_i = index_to_id[i_min]
    child_j = index_to_id[j_min]

    # Sanity checks.
    assert i_min >= 0
    assert j_min >= 0
    assert i_min != j_min
    assert child_i >= 0
    assert child_j >= 0
    assert child_i != child_j

    # Stabilise ordering for easier comparisons.
    if child_i > child_j:
        child_i, child_j = child_j, child_i
        i_min, j_min = j_min, i_min
        d_i, d_j = d_j, d_i

    # Get number of leaves.
    if child_i < n_original:
        leaves_i = 1
    else:
        leaves_i = Z[child_i - n_original, 4]
    if child_j < n_original:
        leaves_j = 1
    else:
        leaves_j = Z[child_j - n_original, 4]

    # Store new node data.
    Z[iteration, 0] = child_i
    Z[iteration, 1] = child_j
    Z[iteration, 2] = d_i
    Z[iteration, 3] = d_j
    Z[iteration, 4] = leaves_i + leaves_j

    if n_remaining > 2:
        # Update data structures.
        canonical_update(
            D=D,
            U=U,
            index_to_id=index_to_id,
            obsolete=obsolete,
            parent=parent,
            i_min=i_min,
            j_min=j_min,
            d_ij=d_ij,
        )
