import numpy as np
from numpy.typing import NDArray
from numba import njit, uintp, float32, bool_, void
from numpydoc_decorator import doc
from . import params
from ._util import (
    NUMBA_NOGIL,
    NUMBA_FASTMATH,
    NUMBA_ERROR_MODEL,
    NUMBA_BOUNDSCHECK,
    NUMBA_CACHE,
    FLOAT32_INF,
    UINTP_MAX,
    ensure_square_distance,
)


@doc(
    summary="""Perform neighbour-joining using an implementation based on the rapid
    algorithm of Simonsen et al. [1]_""",
    extended_summary="""
        This implementation builds and maintains a sorted copy of the distance matrix
        and uses heuristics to avoid searching pairs that cannot possibly be neighbours
        in each iteration.
    """,
    notes="""
        The ordering of the internal nodes may be different between the canonical and
        the rapid algorithms, because these algorithms search the distance matrix in a
        different order. However, the resulting trees will be topologically equivalent.
    """,
    references={
        "1": "https://pure.au.dk/ws/files/19821675/rapidNJ.pdf",
    },
)
def rapid_nj(
    D: params.D,
    disallow_negative_distances: params.disallow_negative_distances = True,
    progress: params.progress = None,
    progress_options: params.progress_options = {},
    copy: params.copy = True,
    gc: params.gc = 100,
) -> params.Z:
    # Set up the distance matrix, ensure it is in square form.
    D_copy = ensure_square_distance(D, copy=copy)
    del D

    # Ensure zeros on diagonal for the initial sum.
    np.fill_diagonal(D_copy, 0)

    # Initialize the "divergence" array, containing sum of distances to other nodes.
    R: NDArray[np.float32] = np.sum(D_copy, axis=1, dtype=np.float32)

    # Set up a sorted version of the distance array.
    D_sorted, nodes_sorted = rapid_setup_distance(D_copy)

    # Number of original observations.
    n_original = D_copy.shape[0]

    # Expected number of new (internal) nodes that will be created.
    n_internal = n_original - 1

    # Total number of nodes in the tree, including internal nodes.
    n_nodes = n_original + n_internal

    # Map row indices to node IDs.
    index_to_id: NDArray[np.uintp] = np.arange(n_original, dtype=np.uintp)

    # Map node IDs to row indices.
    id_to_index: NDArray[np.uintp] = np.full(
        shape=n_nodes, dtype=np.uintp, fill_value=UINTP_MAX
    )
    id_to_index[:n_original] = np.arange(n_original)

    # Initialise output. This is similar to the output that scipy hierarchical
    # clustering functions return, where each row contains data for one internal node
    # in the tree, except that each row here contains:
    # - left child node ID
    # - right child node ID
    # - distance to left child node
    # - distance to right child node
    # - total number of leaves
    Z: NDArray[np.float32] = np.zeros(shape=(n_internal, 5), dtype=np.float32)

    # Keep track of which nodes have been clustered and are now "obsolete". N.B., this
    # is different from canonical implementation because we index here by node ID.
    clustered: NDArray[np.bool_] = np.zeros(shape=n_nodes - 1, dtype=np.bool_)

    # Convenience to also keep track of which rows are no longer in use.
    obsolete: NDArray[np.bool_] = np.zeros(shape=n_original, dtype=np.bool_)

    # Initialise max divergence values.
    R_max = np.zeros(shape=R.shape, dtype=np.float32)
    rapid_update_r_max(
        parent=n_original - 1,
        R=R,
        R_max=R_max,
        id_to_index=id_to_index,
        clustered=clustered,
    )

    # Support wrapping the iterator in a progress bar like tqdm.
    iterator = range(n_internal)
    if progress:
        iterator = progress(iterator, **progress_options)

    # Begin iterating.
    for iteration in iterator:
        # Number of nodes remaining in this iteration.
        n_remaining = n_original - iteration

        # Garbage collection.
        if gc and iteration > 0 and iteration % gc == 0:
            nodes_sorted, D_sorted = rapid_gc(
                nodes_sorted=nodes_sorted,
                D_sorted=D_sorted,
                clustered=clustered,
                obsolete=obsolete,
                n_remaining=n_remaining,
            )

        # Perform one iteration of the neighbour-joining algorithm.
        rapid_iteration(
            iteration=iteration,
            D=D_copy,
            D_sorted=D_sorted,
            R=R,
            nodes_sorted=nodes_sorted,
            index_to_id=index_to_id,
            id_to_index=id_to_index,
            clustered=clustered,
            obsolete=obsolete,
            Z=Z,
            n_original=n_original,
            disallow_negative_distances=disallow_negative_distances,
            R_max=R_max,
        )

    return Z


@njit(
    (float32[:, :],),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    cache=NUMBA_CACHE,
)
def rapid_setup_distance(D: NDArray[np.float32]):
    # Set the diagonal and upper triangle to inf so we can skip self-comparisons and
    # avoid double-comparison between leaf nodes.
    D_sorted = np.full(shape=D.shape, dtype=float32, fill_value=FLOAT32_INF)
    nodes_sorted = np.full(shape=D.shape, dtype=uintp, fill_value=UINTP_MAX)
    for _i in range(D.shape[0]):
        i = uintp(_i)
        D[i, i] = FLOAT32_INF  # avoid self comparisons in all iterations
        d = D[i, :i]
        nx = np.argsort(d)
        dx = d[nx]
        D_sorted[i, :i] = dx
        nodes_sorted[i, :i] = nx
    return D_sorted, nodes_sorted


@njit(
    void(
        uintp,  # iteration
        float32[:],  # R
        float32[:],  # R_max
        uintp[:],  # id_to_index
        bool_[:],  # clustered
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    cache=NUMBA_CACHE,
)
def rapid_update_r_max(
    parent: np.uintp,
    R: NDArray[np.float32],
    R_max: NDArray[np.float32],
    id_to_index: NDArray[np.uintp],
    clustered: NDArray[np.bool_],
) -> None:
    # Here we exploit the fact that comparisons are always between a node and other
    # nodes with lower identifiers, so we can obtain a max divergence for each row.
    r_max = float32(0)
    for _node in range(parent + 1):
        node = uintp(_node)
        if not clustered[node]:
            i = id_to_index[node]
            R_max[i] = r_max
            r_i = R[i]
            r_max = max(r_i, r_max)


@njit(
    (
        float32[:, :],  # D_sorted
        uintp[:, :],  # nodes_sorted
        bool_[:],  # clustered
        bool_[:],  # obsolete
        uintp,  # n_remaining
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    cache=NUMBA_CACHE,
)
def rapid_gc(
    D_sorted: NDArray[np.float32],
    nodes_sorted: NDArray[np.uintp],
    clustered: NDArray[np.bool_],
    obsolete: NDArray[np.bool_],
    n_remaining: int,
) -> tuple[NDArray[np.uintp], NDArray[np.float32]]:
    for _i in range(nodes_sorted.shape[0]):
        i = uintp(_i)
        if obsolete[i]:
            continue
        j_new = uintp(0)
        for _j in range(nodes_sorted.shape[1]):
            j = uintp(_j)
            node_j = nodes_sorted[i, j]
            if node_j == UINTP_MAX:
                break
            if clustered[node_j]:
                continue
            nodes_sorted[i, j_new] = node_j
            D_sorted[i, j_new] = D_sorted[i, j]
            j_new += uintp(1)
    nodes_sorted = nodes_sorted[:, :n_remaining]
    D_sorted = D_sorted[:, :n_remaining]
    return nodes_sorted, D_sorted


@njit(
    (
        float32[:, :],  # D_sorted
        float32[:],  # R
        uintp[:, :],  # nodes_sorted
        bool_[:],  # clustered
        bool_[:],  # obsolete
        uintp[:],  # id_to_index
        # uintp[:],  # index_to_id
        uintp,  # n_remaining
        float32[:],  # R_max
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    cache=NUMBA_CACHE,
)
def rapid_search(
    D_sorted: NDArray[np.float32],
    R: NDArray[np.float32],
    nodes_sorted: NDArray[np.uintp],
    clustered: NDArray[np.bool_],
    obsolete: NDArray[np.bool_],
    id_to_index: NDArray[np.uintp],
    # index_to_id: NDArray[np.uintp],
    n_remaining: int,
    R_max: NDArray[np.float32],
) -> tuple[np.uintp, np.uintp]:
    # Initialize working variables.
    q_xy = FLOAT32_INF
    x = UINTP_MAX
    y = UINTP_MAX
    coefficient = np.float32(n_remaining - 2)
    m = nodes_sorted.shape[0]
    n = nodes_sorted.shape[1]
    assert m == D_sorted.shape[0]
    assert n == D_sorted.shape[1]

    # Search all values up to threshold.
    for _i in range(m):
        i = uintp(_i)

        # Skip if row is no longer in use.
        if obsolete[i]:
            continue

        # Obtain divergence for node corresponding to this row.
        r_i = R[i]

        # Obtain max divergence for comparisons for this node.
        r_j_max = R_max[i]
        r_i_j_max = r_i + r_j_max
        threshold = q_xy + r_i_j_max

        # Search the row up to threshold.
        for _s in range(n):
            s = uintp(_s)

            # Obtain node identifier for the current item.
            node_j = nodes_sorted[i, s]

            # Break at end of active nodes.
            if node_j == UINTP_MAX:
                break

            # Skip if this node is already clustered.
            if clustered[node_j]:
                continue

            # # Ensure we are always looking backwards.
            # assert node_j < node_i, (node_i, node_j)

            # Access distance.
            d = D_sorted[i, s]

            # Partially calculate q.
            q_partial = coefficient * d

            # Limit search. Because the row is sorted, if we are already above this
            # threshold then we know there is no need to search remaining nodes in the
            # row.
            if q_partial >= threshold:
                break

            # Fully calculate q.
            j = id_to_index[node_j]
            r_j = R[j]
            q = q_partial - r_i - r_j

            if q < q_xy:
                q_xy = q
                threshold = q_xy + r_i_j_max
                x = i
                y = j

    return x, y


@njit(
    void(
        float32[:, :],  # D
        float32[:, :],  # D_sorted
        float32[:],  # R
        uintp[:, :],  # nodes_sorted
        uintp[:],  # index_to_id
        uintp[:],  # id_to_index
        bool_[:],  # clustered
        bool_[:],  # obsolete
        uintp,  # parent
        uintp,  # child_x
        uintp,  # child_y
        uintp,  # x
        uintp,  # y
        float32,  # d_xy
        float32[:],  # R_max
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    cache=NUMBA_CACHE,
)
def rapid_update(
    D: NDArray[np.float32],
    D_sorted: NDArray[np.float32],
    R: NDArray[np.float32],
    nodes_sorted: NDArray[np.uintp],
    index_to_id: NDArray[np.uintp],
    id_to_index: NDArray[np.uintp],
    clustered: NDArray[np.bool_],
    obsolete: NDArray[np.bool_],
    parent: np.uintp,
    child_x: np.uintp,
    child_y: np.uintp,
    x: np.uintp,
    y: np.uintp,
    d_xy: np.float32,
    R_max: NDArray[np.float32],
) -> None:
    # Update data structures. Here we obsolete the row corresponding to the node at
    # y, and we reuse the row at x for the new node.
    obsolete[y] = True
    clustered[child_x] = True
    clustered[child_y] = True

    # Row index to be used for the new node.
    z = x

    # Node identifier.
    index_to_id[z] = parent
    id_to_index[parent] = z

    # Initialize divergence for the new node.
    r_z = np.float32(0)

    # Update distances and divergence.
    for _k in range(D.shape[0]):
        k = uintp(_k)

        if obsolete[k] or k == x or k == y:
            continue

        # Calculate distance from k to the new node.
        d_kx = D[k, x]
        d_ky = D[k, y]
        d_kz = 0.5 * (d_kx + d_ky - d_xy)
        D[z, k] = d_kz
        D[k, z] = d_kz

        # Subtract out the distances for the nodes that have just been joined and add
        # in distance for the new node.
        r_k = R[k] - d_kx - d_ky + d_kz
        R[k] = r_k

        # Accumulate divergence for the new node.
        r_z += d_kz

    # Store divergence for the new node.
    R[x] = r_z

    # First cut down to just the active nodes.
    active = ~obsolete
    active[z] = False  # exclude self
    distances_active = D[z, active]
    nodes_active = index_to_id[active]

    # Now sort the new distances.
    loc_sorted = np.argsort(distances_active)
    nodes_active_sorted = nodes_active[loc_sorted]
    distances_active_sorted = distances_active[loc_sorted]

    # Now update sorted nodes and distances.
    p = nodes_active_sorted.shape[0]
    nodes_sorted[z, :p] = nodes_active_sorted
    D_sorted[z, :p] = distances_active_sorted
    # Mark the end of active nodes.
    nodes_sorted[z, p] = UINTP_MAX
    D_sorted[z, p] = FLOAT32_INF

    # Update max divergences.
    rapid_update_r_max(
        parent=parent,
        R=R,
        R_max=R_max,
        id_to_index=id_to_index,
        clustered=clustered,
    )


@njit(
    void(
        uintp,  # iteration
        float32[:, :],  # D
        float32[:, :],  # D_sorted
        float32[:],  # R
        uintp[:, :],  # nodes_sorted
        uintp[:],  # index_to_id
        uintp[:],  # id_to_index
        bool_[:],  # clustered
        bool_[:],  # obsolete
        float32[:, :],  # Z
        uintp,  # n_original
        bool_,  # disallow_negative_distances
        float32[:],  # R_max
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    cache=NUMBA_CACHE,
)
def rapid_iteration(
    iteration: int,
    D: NDArray[np.float32],
    D_sorted: NDArray[np.float32],
    R: NDArray[np.float32],
    nodes_sorted: NDArray[np.uintp],
    index_to_id: NDArray[np.uintp],
    id_to_index: NDArray[np.uintp],
    clustered: NDArray[np.bool_],
    obsolete: NDArray[np.bool_],
    Z: NDArray[np.float32],
    n_original: int,
    disallow_negative_distances: bool,
    R_max: NDArray[np.float32],
) -> None:
    # This will be the identifier for the new node to be created in this iteration.
    parent = iteration + n_original

    # Number of nodes remaining in this iteration.
    n_remaining = n_original - iteration

    if n_remaining > 2:
        # Search for the closest pair of nodes to join.
        x, y = rapid_search(
            D_sorted=D_sorted,
            R=R,
            nodes_sorted=nodes_sorted,
            clustered=clustered,
            obsolete=obsolete,
            id_to_index=id_to_index,
            # index_to_id=index_to_id,
            n_remaining=n_remaining,
            R_max=R_max,
        )

        # Get IDs for the nodes to be joined.
        child_x = index_to_id[x]
        child_y = index_to_id[y]

        # Calculate distances to the new internal node.
        d_xy = D[x, y]
        d_xz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (R[x] - R[y]))
        d_yz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (R[y] - R[x]))

    else:
        # Termination. Join the two remaining nodes, placing the final node at the
        # midpoint.
        _child_x, _child_y = np.nonzero(~clustered)[0]
        child_x = uintp(_child_x)
        child_y = uintp(_child_y)
        x = id_to_index[child_x]
        y = id_to_index[child_y]
        d_xy = D[x, y]
        d_xz = d_xy / 2
        d_yz = d_xy / 2

    # Sanity checks.
    assert x >= 0
    assert y >= 0
    assert x != y
    assert child_x >= 0
    assert child_y >= 0
    assert child_x != child_y

    # Handle possibility of negative distances.
    if disallow_negative_distances:
        d_xz = max(0, d_xz)
        d_yz = max(0, d_yz)

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
        rapid_update(
            D=D,
            D_sorted=D_sorted,
            R=R,
            nodes_sorted=nodes_sorted,
            index_to_id=index_to_id,
            id_to_index=id_to_index,
            clustered=clustered,
            obsolete=obsolete,
            parent=parent,
            child_x=child_x,
            child_y=child_y,
            x=x,
            y=y,
            d_xy=d_xy,
            R_max=R_max,
        )
