import numpy as np
from numpy.typing import NDArray
from numba import njit, uintp, float32, bool_, void
from numpydoc_decorator import doc
from . import params
from ._util import NOGIL, FASTMATH, ERROR_MODEL, BOUNDSCHECK, FLOAT32_INF, UINTP_MAX


@njit(
    (
        float32[:, :],  # D
        float32[:],  # U
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def heuristic_init(
    D,
    U,
    Z,
    obsolete,
    index_to_id,
    disallow_negative_distances,
):
    # Here we take a first pass through the distance matrix to locate the first pair
    # of nodes to join, and initialise the data structures needed for the heuristic
    # algotithm.

    # Size of the distance matrix.
    n = uintp(D.shape[0])

    # Distance between pair of nodes with global minimum.
    d_xy = FLOAT32_INF

    # Global minimum join criterion.
    q_xy = FLOAT32_INF

    # Indices of the pair of nodes with the global minimum, to be joined.
    x = UINTP_MAX
    y = UINTP_MAX

    # Partially compute outside loop.
    coefficient = float32(n - 2)

    # Minimum join criterion per row.
    Q = np.empty(shape=n, dtype=float32)

    # Index of node where minimum join criterion per node, i.e., nearest neighbour
    # within each row.
    J = np.empty(shape=n, dtype=uintp)

    # Scan the lower triangle of the distance matrix.
    for _i in range(n):
        i = uintp(_i)
        row_q_min = FLOAT32_INF
        row_j_min = UINTP_MAX
        u_i = U[i]
        for _j in range(i):
            j = uintp(_j)
            u_j = U[j]
            d = D[i, j]
            q = coefficient * d - u_i - u_j
            if q < row_q_min:
                # Found new minimum within this row.
                row_q_min = q
                row_j_min = j
            if q < q_xy:
                # Found new global minimum.
                q_xy = q
                d_xy = d
                x = i
                y = j
        # Store minimum for this row.
        Q[i] = row_q_min
        J[i] = row_j_min

    # Sanity checks.
    assert x < n
    assert y < n
    assert x != y

    # Calculate distances to the new internal node.
    d_xz = 0.5 * (d_xy + (1 / (n - 2)) * (U[x] - U[y]))
    d_yz = 0.5 * (d_xy + (1 / (n - 2)) * (U[y] - U[x]))

    # Handle possibility of negative distances.
    if disallow_negative_distances:
        d_xz = max(float32(0), d_xz)
        d_yz = max(float32(0), d_yz)

    # Store new node data.
    Z[0, 0] = x
    Z[0, 1] = y
    Z[0, 2] = d_xz
    Z[0, 3] = d_yz
    Z[0, 4] = 2

    # Identifier for the new node.
    parent = n

    # Row index to be used for the new node.
    z = x

    # Update data structures.
    obsolete[y] = True
    index_to_id[z] = parent

    # Initialize divergence for the new node.
    u_z = float32(0)

    # Update distances and divergence.
    for _k in range(D.shape[0]):
        k = uintp(_k)

        if k == x or k == y:
            continue

        # Calculate distance from k to the new node.
        d_kx = D[k, x]
        d_ky = D[k, y]
        d_kz = float32(0.5) * (d_kx + d_ky - d_xy)
        D[z, k] = d_kz
        D[k, z] = d_kz

        # Subtract out the distances for the nodes that have just been joined and add
        # in distance for the new node.
        u_k = U[k] - d_kx - d_ky + d_kz
        U[k] = u_k

        # Accumulate divergence for the new node.
        u_z += d_kz

    # Assign divergence for the new node.
    U[z] = u_z

    return Q, J, z


@doc(
    summary="""@@TODO.""",
    extended_summary="""
        @@TODO
    """,
)
def heuristic_nj(
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
    U: NDArray[np.float32] = np.sum(D_copy, axis=1)

    # Keep track of which rows correspond to nodes that have been clustered.
    obsolete: NDArray[np.bool_] = np.zeros(shape=n_original, dtype=np.bool_)

    # Initialise the heuristic algorithm.
    Q, J, z = heuristic_init(
        D=D_copy,
        U=U,
        Z=Z,
        obsolete=obsolete,
        index_to_id=index_to_id,
        disallow_negative_distances=disallow_negative_distances,
    )

    # Support wrapping the iterator in a progress bar.
    iterator = range(1, n_internal)
    if progress:
        iterator = progress(iterator, **progress_options)

    # Begin iterating.
    for iteration in iterator:
        # Perform one iteration of the neighbour-joining algorithm.
        heuristic_iteration(
            iteration=iteration,
            D=D_copy,
            U=U,
            Q=Q,
            J=J,
            z=z,
            index_to_id=index_to_id,
            obsolete=obsolete,
            Z=Z,
            n_original=n_original,
            disallow_negative_distances=disallow_negative_distances,
        )

    return Z


@njit(
    void(
        float32[:, :],  # D
        float32[:],  # U
        uintp[:],  # index_to_id
        bool_[:],  # obsolete
        uintp,  # parent
        uintp,  # i_min
        uintp,  # j_min
        float32,  # d_ij
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def heuristic_update(
    D: NDArray[np.float32],
    U: NDArray[np.float32],
    index_to_id: NDArray[np.uintp],
    obsolete: NDArray[np.bool_],
    parent: np.uintp,
    i_min: np.uintp,
    j_min: np.uintp,
    d_ij: np.float32,
) -> None:
    # Here we obsolete the row and column corresponding to the node at j_min, and we
    # reuse the row and column at i_min for the new node.
    obsolete[j_min] = True
    index_to_id[i_min] = parent

    # Initialize divergence for the new node.
    u_new = float32(0)

    # Update distances and divergence.
    for _k in range(D.shape[0]):
        k = uintp(_k)

        if obsolete[k] or k == i_min or k == j_min:
            continue

        # Calculate distance from k to the new node.
        d_ki = D[k, i_min]
        d_kj = D[k, j_min]
        d_k_new = float32(0.5) * (d_ki + d_kj - d_ij)
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
        uintp,  # iteration
        float32[:, :],  # D
        float32[:],  # U
        float32[:],  # Q
        uintp[:],  # J
        uintp,  # z
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
def heuristic_iteration(
    iteration: np.uintp,
    D: NDArray[np.float32],
    U: NDArray[np.float32],
    Q,
    J,
    z,
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
        x, y, d_xy = heuristic_search(
            D=D, U=U, obsolete=obsolete, n_remaining=n_remaining
        )

        # Calculate distances to the new internal node.
        d_xz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (U[x] - U[y]))
        d_yz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (U[y] - U[x]))

    else:
        # Termination. Join the two remaining nodes, placing the final node at the
        # midpoint.
        _i_min, _j_min = np.nonzero(~obsolete)[0]
        x = uintp(_i_min)
        y = uintp(_j_min)
        d_xy = D[x, y]
        d_xz = d_xy / 2
        d_yz = d_xy / 2

    # Handle possibility of negative distances.
    if disallow_negative_distances:
        d_xz = max(float32(0), d_xz)
        d_yz = max(float32(0), d_yz)

    # Get IDs for the nodes to be joined.
    child_i = index_to_id[x]
    child_j = index_to_id[y]

    # Sanity checks.
    assert x >= 0
    assert y >= 0
    assert x != y
    assert child_i >= 0
    assert child_j >= 0
    assert child_i != child_j

    # Stabilise ordering for easier comparisons.
    if child_i > child_j:
        child_i, child_j = child_j, child_i
        x, y = y, x
        d_xz, d_yz = d_yz, d_xz

    # Get number of leaves.
    if child_i < n_original:
        leaves_i = float32(1)
    else:
        leaves_i = Z[child_i - n_original, 4]
    if child_j < n_original:
        leaves_j = float32(1)
    else:
        leaves_j = Z[child_j - n_original, 4]

    # Store new node data.
    Z[iteration, 0] = child_i
    Z[iteration, 1] = child_j
    Z[iteration, 2] = d_xz
    Z[iteration, 3] = d_yz
    Z[iteration, 4] = leaves_i + leaves_j

    if n_remaining > 2:
        # Update data structures.
        heuristic_update(
            D=D,
            U=U,
            index_to_id=index_to_id,
            obsolete=obsolete,
            parent=parent,
            i_min=x,
            j_min=y,
            d_ij=d_xy,
        )


def heuristic_search():
    pass
