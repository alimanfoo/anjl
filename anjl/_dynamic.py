import numpy as np
from numpy.typing import NDArray

from numba import njit, uintp, float32, bool_
from numpydoc_decorator import doc
from . import params
from ._util import NOGIL, FASTMATH, ERROR_MODEL, BOUNDSCHECK, FLOAT32_INF, UINTP_MAX


@doc(
    summary="""Perform neighbour-joining using the dynamic algorithm of Clausen [1]_.""",
    extended_summary="""
        This is the fastest and most scalable implementation currently available. The
        dynamic algorithm exploits the fact that the neighbour-joining criterion Q is
        gradually weakened with each iteration, and therefore the minimum value of Q
        found initially within a given row provides a lower bound for all values within
        the same row in subsequent iterations. This allows many rows of the distance
        matrix to be skipped in each iteration.
    """,
    references={
        "1": "https://doi.org/10.1093/bioinformatics/btac774",
    },
)
def dynamic_nj(
    D: params.D,
    disallow_negative_distances: params.disallow_negative_distances = True,
    progress: params.progress = None,
    progress_options: params.progress_options = {},
) -> params.Z:
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
    S: NDArray[np.float32] = np.sum(D_copy, axis=1)

    # Keep track of which rows correspond to nodes that have been clustered.
    obsolete: NDArray[np.bool_] = np.zeros(shape=n_original, dtype=np.bool_)

    # Initialise the dynamic algorithm.
    Q, z = dynamic_init(
        D=D_copy,
        S=S,
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
        z = dynamic_iteration(
            iteration=np.uintp(iteration),
            D=D_copy,
            S=S,
            Q=Q,
            previous_z=z,
            index_to_id=index_to_id,
            obsolete=obsolete,
            Z=Z,
            n_original=np.uintp(n_original),
            disallow_negative_distances=disallow_negative_distances,
        )

    return Z


@njit(
    (
        float32[:, :],  # D
        float32[:],  # S
        float32[:, :],  # Z
        bool_[:],  # obsolete
        uintp[:],  # index_to_id
        bool_,  # disallow_negative_distances
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def dynamic_init(
    D: NDArray[np.float32],
    S: NDArray[np.float32],
    Z: NDArray[np.float32],
    obsolete: NDArray[np.bool_],
    index_to_id: NDArray[np.uintp],
    disallow_negative_distances: bool,
):
    # Here we take a first pass through the distance matrix to locate the first pair
    # of nodes to join, and initialise the data structures needed for the dynamic
    # algorithm.

    # Size of the distance matrix.
    n = np.uintp(D.shape[0])

    # Distance between pair of nodes with global minimum.
    d_xy = FLOAT32_INF

    # Global minimum join criterion.
    q_xy = FLOAT32_INF

    # Indices of the pair of nodes with the global minimum, to be joined.
    x = UINTP_MAX
    y = UINTP_MAX

    # Partially compute outside loop.
    coefficient = np.float32(n - 2)

    # Minimum join criterion per row.
    Q = np.empty(shape=n, dtype=np.float32)

    # Scan the distance matrix.
    for _i in range(n):
        i = np.uintp(_i)  # row index
        j = UINTP_MAX  # column index of row q minimum
        q_ij = FLOAT32_INF  # row q minimum
        d_ij = FLOAT32_INF  # distance at row q minimum
        s_i = S[i]  # divergence for node at row i
        # Search the lower triangle of the distance matrix.
        for _k in range(i):
            k = np.uintp(_k)
            if i == k:
                continue
            s_k = S[k]
            d = D[i, k]
            q = coefficient * d - s_i - s_k
            if q < q_ij:
                # Found new minimum within this row.
                q_ij = q
                d_ij = d
                j = k
        # Store minimum for this row.
        Q[i] = q_ij
        if q_ij < q_xy:
            # Found new global minimum.
            q_xy = q_ij
            d_xy = d_ij
            x = i
            y = j

    # Sanity checks.
    assert x < n
    assert y < n
    assert x != y

    # Stabilise ordering for easier comparisons.
    if x > y:
        x, y = y, x

    # Calculate distances to the new internal node.
    d_xz = 0.5 * (d_xy + (1 / (n - 2)) * (S[x] - S[y]))
    d_yz = 0.5 * (d_xy + (1 / (n - 2)) * (S[y] - S[x]))

    # Handle possibility of negative distances.
    if disallow_negative_distances:
        d_xz = max(np.float32(0), d_xz)
        d_yz = max(np.float32(0), d_yz)

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
    s_z = np.float32(0)

    # Update distances and divergence.
    for _k in range(D.shape[0]):
        k = np.uintp(_k)

        if k == x or k == y:
            continue

        # Calculate distance from k to the new node.
        d_kx = D[k, x]
        d_ky = D[k, y]
        d_kz = np.float32(0.5) * (d_kx + d_ky - d_xy)
        D[z, k] = d_kz
        D[k, z] = d_kz

        # Subtract out the distances for the nodes that have just been joined and add
        # in distance for the new node.
        s_k = S[k] - d_kx - d_ky + d_kz
        S[k] = s_k

        # Accumulate divergence for the new node.
        s_z += d_kz

    # Assign divergence for the new node.
    S[z] = s_z

    return Q, z


@njit(
    (
        float32[:, :],  # D
        float32[:],  # S
        float32[:],  # Q
        bool_[:],  # obsolete
        uintp,  # i
        float32,  # coefficient
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def search_row(
    D: NDArray[np.float32],
    S: NDArray[np.float32],
    Q: NDArray[np.float32],
    obsolete: NDArray[np.bool_],
    i: np.uintp,
    coefficient: np.float32,
):
    # Search a single row of the distance matrix to find the row minimum join criterion.
    q_ij = FLOAT32_INF  # row minimum q
    d_ij = FLOAT32_INF  # distance at row minimum q
    j = UINTP_MAX  # column index at row minimum q
    s_i = S[i]  # divergence for node at row i
    for _k in range(i):
        k = np.uintp(_k)
        if i == k or obsolete[k]:
            continue
        s_k = S[k]
        d = D[i, k]
        q = coefficient * d - s_i - s_k
        if q < q_ij:
            # Found new row minimum.
            q_ij = q
            d_ij = d
            j = k
    # Remember best match.
    Q[i] = q_ij
    return j, q_ij, d_ij


@njit(
    (
        float32[:, :],  # D
        float32[:],  # S
        float32[:],  # Q
        uintp,  # z
        bool_[:],  # obsolete
        uintp,  # n_remaining
    ),
    nogil=NOGIL,
    fastmath=FASTMATH,
    error_model=ERROR_MODEL,
    boundscheck=BOUNDSCHECK,
)
def dynamic_search(
    D: NDArray[np.float32],
    S: NDArray[np.float32],
    Q: NDArray[np.float32],
    z: np.uintp,  # index of new node created in previous iteration
    obsolete: NDArray[np.bool_],
    n_remaining: np.uintp,
):
    # Size of the distance matrix.
    n = np.uintp(D.shape[0])

    # Distance between pair of nodes with global minimum.
    d_xy = FLOAT32_INF

    # Global minimum join criterion.
    q_xy = FLOAT32_INF

    # Indices of the pair of nodes with the global minimum, to be joined.
    x = UINTP_MAX
    y = UINTP_MAX

    # Partially compute outside loop.
    coefficient = np.float32(n_remaining - 2)

    # First scan the new row at index z and use as starting point for search.
    y, q_xy, d_xy = search_row(
        D=D, S=S, Q=Q, obsolete=obsolete, i=z, coefficient=coefficient
    )
    x = z

    for _i in range(n):
        i = np.uintp(_i)  # row index

        if obsolete[i]:
            continue

        if i == z:
            continue

        if i > z:
            # Calculate join criterion for the new node, and update Q if necessary.
            s_i = S[i]
            s_z = S[z]
            d_iz = D[i, z]
            q_iz = coefficient * d_iz - s_i - s_z
            if q_iz < Q[i]:
                Q[i] = q_iz

        if Q[i] > q_xy:
            # We can skip this row. The previous row optimum join criterion is greater
            # than the current global optimum, and so there is now way that this row
            # can contain a better match. This is the core optimisation of the dynamic
            # algorithm.
            continue

        # Join criterion could be lower than the current global minimum. Fully search
        # the row.
        j, q_ij, d_ij = search_row(
            D=D, S=S, Q=Q, obsolete=obsolete, i=i, coefficient=coefficient
        )

        if q_ij < q_xy:
            # Found new global minimum.
            q_xy = q_ij
            d_xy = d_ij
            x = i
            y = j

    return x, y, d_xy


@njit(
    (
        float32[:, :],  # D
        float32[:],  # S
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
def dynamic_update(
    D: NDArray[np.float32],
    S: NDArray[np.float32],
    index_to_id: NDArray[np.uintp],
    obsolete: NDArray[np.bool_],
    parent: np.uintp,
    x: np.uintp,
    y: np.uintp,
    d_xy: np.float32,
):
    # Here we obsolete the row and column corresponding to the node at y, and we
    # reuse the row and column at x for the new node.
    obsolete[y] = True

    # Row index to be used for the new node.
    z = x

    # Node identifier.
    index_to_id[z] = parent

    # Initialize divergence for the new node.
    s_z = np.float32(0)

    # Update distances and divergence.
    for _k in range(D.shape[0]):
        k = np.uintp(_k)

        if obsolete[k] or k == x or k == y:
            continue

        # Calculate distance from k to the new node.
        d_kx = D[k, x]
        d_ky = D[k, y]
        d_kz = np.float32(0.5) * (d_kx + d_ky - d_xy)
        D[z, k] = d_kz
        D[k, z] = d_kz

        # Subtract out the distances for the nodes that have just been joined and add
        # in distance for the new node.
        s_k = S[k] - d_kx - d_ky + d_kz
        S[k] = s_k

        # Accumulate divergence for the new node.
        s_z += d_kz

    # Assign divergence for the new node.
    S[z] = s_z

    return z


@njit(
    (
        uintp,  # iteration
        float32[:, :],  # D
        float32[:],  # S
        float32[:],  # Q
        uintp,  # previous_z
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
def dynamic_iteration(
    iteration: np.uintp,
    D: NDArray[np.float32],
    S: NDArray[np.float32],
    Q,
    previous_z,
    index_to_id: NDArray[np.uintp],
    obsolete: NDArray[np.bool_],
    Z: NDArray[np.float32],
    n_original: np.uintp,
    disallow_negative_distances: bool,
):
    # This will be the identifier for the new node to be created in this iteration.
    parent = iteration + n_original

    # Number of nodes remaining in this iteration.
    n_remaining = n_original - iteration

    if n_remaining > 2:
        # Search for the closest pair of nodes to join.
        x, y, d_xy = dynamic_search(
            D=D, S=S, Q=Q, z=previous_z, obsolete=obsolete, n_remaining=n_remaining
        )
        assert x < D.shape[0], x
        assert y < D.shape[0], y
        assert not np.isinf(d_xy), d_xy

        # Calculate distances to the new internal node.
        d_xz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (S[x] - S[y]))
        d_yz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (S[y] - S[x]))

    else:
        # Termination. Join the two remaining nodes, placing the final node at the
        # midpoint.
        _x, _y = np.nonzero(~obsolete)[0]
        x = np.uintp(_x)
        y = np.uintp(_y)
        d_xy = D[x, y]
        d_xz = d_xy / 2
        d_yz = d_xy / 2

    # Handle possibility of negative distances.
    if disallow_negative_distances:
        d_xz = max(np.float32(0), d_xz)
        d_yz = max(np.float32(0), d_yz)

    # Get IDs for the nodes to be joined.
    child_x = index_to_id[x]
    child_y = index_to_id[y]

    # Sanity checks.
    assert x < D.shape[0]
    assert y < D.shape[0]
    assert x != y
    assert child_x != child_y

    # Stabilise ordering for easier comparisons.
    if child_x > child_y:
        child_x, child_y = child_y, child_x
        x, y = y, x
        d_xz, d_yz = d_yz, d_xz

    # Get number of leaves.
    if child_x < n_original:
        leaves_i = np.float32(1)
    else:
        leaves_i = Z[child_x - n_original, 4]
    if child_y < n_original:
        leaves_j = np.float32(1)
    else:
        leaves_j = Z[child_y - n_original, 4]

    # Store new node data.
    Z[iteration, 0] = child_x
    Z[iteration, 1] = child_y
    Z[iteration, 2] = d_xz
    Z[iteration, 3] = d_yz
    Z[iteration, 4] = leaves_i + leaves_j

    if n_remaining > 2:
        # Update data structures.
        new_z = dynamic_update(
            D=D,
            S=S,
            index_to_id=index_to_id,
            obsolete=obsolete,
            parent=parent,
            x=x,
            y=y,
            d_xy=d_xy,
        )

    else:
        new_z = UINTP_MAX

    return new_z
