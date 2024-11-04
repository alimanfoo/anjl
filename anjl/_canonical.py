import numpy as np
from numpy.typing import NDArray
from numba import njit, uintp, float32, bool_, void
from numpydoc_decorator import doc
from . import params
from ._util import NOGIL, FASTMATH, ERROR_MODEL, BOUNDSCHECK, FLOAT32_INF, UINTP_MAX


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
    copy: params.copy = True,
) -> params.Z:
    # Make a copy of distance matrix D because we will overwrite it during the
    # algorithm.
    D_copy: NDArray[np.float32] = np.array(D, copy=copy, order="C", dtype=np.float32)
    del D

    # Number of original observations.
    n_original = D_copy.shape[0]

    # Expected number of new (internal) nodes that will be created.
    n_internal = n_original - 1

    # Map row indices to node IDs.
    index_to_id: NDArray[np.uintp] = np.arange(n_original, dtype=np.uintp)

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
    R: NDArray[np.float32] = np.sum(D_copy, axis=1)

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
            R=R,
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
        float32[:],  # R
        bool_[:],  # obsolete
        uintp,  # n_remaining
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def canonical_search(
    D: NDArray[np.float32],
    R: NDArray[np.float32],
    obsolete: NDArray[np.bool_],
    n_remaining: np.uintp,
) -> tuple[np.uintp, np.uintp]:
    """Search for the closest pair of neighbouring nodes to join."""

    # Size of the distance matrix.
    m = np.uintp(D.shape[0])

    # Global minimum join criterion.
    q_xy = FLOAT32_INF

    # Indices of the pair of nodes with the global minimum, to be joined.
    x = UINTP_MAX
    y = UINTP_MAX

    # Partially compute outside loop.
    coefficient = float32(n_remaining - 2)

    # Iterate over rows of the distance matrix.
    for _i in range(m):
        i = np.uintp(_i)  # row index

        if obsolete[i]:
            continue

        r_i = R[i]

        # Iterate over columns of the distance matrix.
        for _j in range(i):
            j = uintp(_j)  # column index

            if obsolete[j]:
                continue

            r_j = R[j]
            d = D[i, j]
            q = coefficient * d - r_i - r_j

            if q < q_xy:
                # Found new global minimum.
                q_xy = q
                x = i
                y = j

    return x, y


@njit(
    void(
        float32[:, :],  # D
        float32[:],  # R
        uintp[:],  # index_to_id
        bool_[:],  # obsolete
        uintp,  # parent
        uintp,  # x
        uintp,  # y
        float32,  # d_xy
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def canonical_update(
    D: NDArray[np.float32],
    R: NDArray[np.float32],
    index_to_id: NDArray[np.uintp],
    obsolete: NDArray[np.bool_],
    parent: np.uintp,
    x: np.uintp,
    y: np.uintp,
    d_xy: np.float32,
) -> None:
    # Here we obsolete the row and column corresponding to the node at y, and we
    # reuse the row and column at x for the new node.
    obsolete[y] = True

    # Row index to be used for the new node.
    z = x

    # Node identifier.
    index_to_id[z] = parent

    # Initialize divergence for the new node.
    r_z = float32(0)

    # Update distances and divergence.
    for _k in range(D.shape[0]):
        k = uintp(_k)

        if obsolete[k] or k == x or k == y:
            continue

        # Calculate distance from k to the new node.
        d_kx = D[k, x]
        d_ky = D[k, y]
        d_kz = float32(0.5) * (d_kx + d_ky - d_xy)
        D[z, k] = d_kz
        D[k, z] = d_kz

        # Subtract out the distances for the nodes that have just been joined and add
        # in distance for the new node.
        r_k = R[k] - d_kx - d_ky + d_kz
        R[k] = r_k

        # Accumulate divergence for the new node.
        r_z += d_kz

    # Assign divergence for the new node.
    R[z] = r_z


@njit(
    void(
        uintp,  # iteration
        float32[:, :],  # D
        float32[:],  # R
        uintp[:],  # index_to_id
        bool_[:],  # obsolete
        float32[:, :],  # Z
        uintp,  # n_original
        bool_,  # disallow_negative_distances
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def canonical_iteration(
    iteration: np.uintp,
    D: NDArray[np.float32],
    R: NDArray[np.float32],
    index_to_id: NDArray[np.uintp],
    obsolete: NDArray[np.bool_],
    Z: NDArray[np.float32],
    n_original: np.uintp,
    disallow_negative_distances: bool,
) -> None:
    # This will be the identifier for the new node to be created in this iteration.
    parent = iteration + n_original

    # Number of nodes remaining in this iteration.
    n_remaining = n_original - iteration

    if n_remaining > 2:
        # Search for the closest pair of nodes to join.
        x, y = canonical_search(D=D, R=R, obsolete=obsolete, n_remaining=n_remaining)

        # Calculate distances to the new internal node.
        d_xy = D[x, y]
        d_xz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (R[x] - R[y]))
        d_yz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (R[y] - R[x]))

    else:
        # Termination. Join the two remaining nodes, placing the final node at the
        # midpoint.
        _x, _y = np.nonzero(~obsolete)[0]
        x = uintp(_x)
        y = uintp(_y)
        d_xy = D[x, y]
        d_xz = d_xy / 2
        d_yz = d_xy / 2

    # Handle possibility of negative distances.
    if disallow_negative_distances:
        d_xz = max(float32(0), d_xz)
        d_yz = max(float32(0), d_yz)

    # Get IDs for the nodes to be joined.
    child_x = index_to_id[x]
    child_y = index_to_id[y]

    # Sanity checks.
    assert x >= 0
    assert y >= 0
    assert x < D.shape[0]
    assert y < D.shape[0]
    assert x != y
    assert child_x >= 0
    assert child_y >= 0
    assert child_x != child_y

    # Stabilise ordering for easier comparisons.
    if child_x > child_y:
        child_x, child_y = child_y, child_x
        x, y = y, x
        d_xz, d_yz = d_yz, d_xz

    # Get number of leaves.
    if child_x < n_original:
        leaves_x = float32(1)
    else:
        leaves_x = Z[child_x - n_original, 4]
    if child_y < n_original:
        leaves_y = float32(1)
    else:
        leaves_y = Z[child_y - n_original, 4]

    # Store new node data.
    Z[iteration, 0] = child_x
    Z[iteration, 1] = child_y
    Z[iteration, 2] = d_xz
    Z[iteration, 3] = d_yz
    Z[iteration, 4] = leaves_x + leaves_y

    if n_remaining > 2:
        # Update data structures.
        canonical_update(
            D=D,
            R=R,
            index_to_id=index_to_id,
            obsolete=obsolete,
            parent=parent,
            x=x,
            y=y,
            d_xy=d_xy,
        )
