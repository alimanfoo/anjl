import numpy as np
from numpy.typing import NDArray

from numba import njit, uintp, float32, bool_, get_num_threads, prange
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
    ensure_condensed_distance,
    setup_divergence,
    condensed_index,
    condensed_offset,
)


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
    copy: params.copy = True,
    parallel: params.parallel = True,
) -> params.Z:
    # Set up the distance matrix, ensure it is in condensed form.
    distance, n_original = ensure_condensed_distance(D=D, copy=copy)
    del D

    # Expected number of new (internal) nodes that will be created.
    n_internal = n_original - 1

    # Map row indices to node IDs.
    index_to_id = np.arange(n_original, dtype=np.uintp)

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
    R = setup_divergence(distance=distance, n_original=n_original)

    # Keep track of which rows correspond to nodes that have been clustered.
    obsolete = np.zeros(shape=n_original, dtype=np.bool_)

    # Initialise the dynamic algorithm.
    Q, z = dynamic_init(
        distance=distance,
        R=R,
        Z=Z,
        obsolete=obsolete,
        index_to_id=index_to_id,
        disallow_negative_distances=disallow_negative_distances,
        n_original=n_original,
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
            distance=distance,
            R=R,
            Q=Q,
            previous_z=z,
            index_to_id=index_to_id,
            obsolete=obsolete,
            Z=Z,
            n_original=np.uintp(n_original),
            disallow_negative_distances=disallow_negative_distances,
            parallel=parallel,
        )

    return Z


@njit(
    (
        float32[::1],  # distance
        float32[::1],  # R
        float32[:, ::1],  # Z
        bool_[::1],  # obsolete
        uintp[::1],  # index_to_id
        bool_,  # disallow_negative_distances
        uintp,  # n_original
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    cache=NUMBA_CACHE,
)
def dynamic_init(
    distance: NDArray[np.float32],
    R: NDArray[np.float32],
    Z: NDArray[np.float32],
    obsolete: NDArray[np.bool_],
    index_to_id: NDArray[np.uintp],
    disallow_negative_distances: bool,
    n_original: np.uintp,
):
    # Here we take a first pass through the distance matrix to locate the first pair
    # of nodes to join, and initialise the data structures needed for the dynamic
    # algorithm.

    # Distance between pair of nodes with global minimum.
    d_xy = FLOAT32_INF

    # Global minimum join criterion.
    q_xy = FLOAT32_INF

    # Indices of the pair of nodes with the global minimum, to be joined.
    x = UINTP_MAX
    y = UINTP_MAX

    # Partially compute outside loop.
    coefficient = np.float32(n_original - 2)

    # Minimum join criterion per row.
    Q = np.empty(shape=n_original, dtype=np.float32)

    # Full scan of the distance matrix.
    c = np.uintp(0)  # condensed index
    for _i in range(n_original):
        i = np.uintp(_i)  # row index

        j = UINTP_MAX  # column index of row q minimum
        q_ij = FLOAT32_INF  # row q minimum
        d_ij = FLOAT32_INF  # distance at row q minimum
        r_i = R[i]  # divergence for node at row i

        # Search the upper triangle of the distance matrix.
        for _k in range(i + 1, n_original):
            k = np.uintp(_k)  # column index

            r_k = R[k]  # divergence for node at row k
            d = distance[c]
            q = coefficient * d - r_i - r_k
            if q < q_ij:
                # Found new minimum within this row.
                q_ij = q
                d_ij = d
                j = k
            c += np.uintp(1)

        # Store minimum for this row.
        Q[i] = q_ij

        if q_ij < q_xy:
            # Found new global minimum.
            q_xy = q_ij
            d_xy = d_ij
            x = i
            y = j

    # Sanity checks.
    assert x < n_original
    assert y < n_original
    assert x != y

    # Stabilise ordering for easier comparisons.
    if x > y:
        x, y = y, x

    # Calculate distances to the new internal node.
    d_xz = 0.5 * (d_xy + (1 / (n_original - 2)) * (R[x] - R[y]))
    d_yz = 0.5 * (d_xy + (1 / (n_original - 2)) * (R[y] - R[x]))

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
    parent = n_original

    # Row index to be used for the new node.
    z = x

    # Update data structures.
    obsolete[y] = True
    index_to_id[z] = parent

    # Initialize divergence for the new node.
    r_z = np.float32(0)

    # Update distances and divergence.
    for _k in range(n_original):
        k = np.uintp(_k)

        if k == x or k == y:
            continue

        # Calculate and store distance from k to the new node.
        c_kx = condensed_index(k, x, n_original)
        d_kx = distance[c_kx]
        c_ky = condensed_index(k, y, n_original)
        d_ky = distance[c_ky]
        d_kz = float32(0.5) * (d_kx + d_ky - d_xy)
        c_kz = c_kx
        distance[c_kz] = d_kz

        # Subtract out the distances for the nodes that have just been joined and add
        # in distance for the new node.
        r_k = R[k] - d_kx - d_ky + d_kz
        R[k] = r_k

        # Accumulate divergence for the new node.
        r_z += d_kz

    # Assign divergence for the new node.
    R[z] = r_z

    return Q, z


@njit(
    (
        float32[::1],  # distance
        float32[::1],  # R
        float32[::1],  # Q
        bool_[::1],  # obsolete
        uintp,  # i
        float32,  # coefficient
        uintp,  # n_original
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    cache=NUMBA_CACHE,
)
def search_row(
    distance: NDArray[np.float32],
    R: NDArray[np.float32],
    Q: NDArray[np.float32],
    obsolete: NDArray[np.bool_],
    i: np.uintp,
    coefficient: np.float32,
    n_original: np.uintp,
):
    # Search a single row of the distance matrix to find the row minimum join criterion.

    q_ij = FLOAT32_INF  # row minimum q
    d_ij = FLOAT32_INF  # distance at row minimum q
    j = UINTP_MAX  # column index at row minimum q
    r_i = R[i]  # divergence for node at row i

    # Compute offset into condensed distance matrix.
    _offset = condensed_offset(i, n_original)

    for _k in range(i + 1, n_original):
        k = np.uintp(_k)
        if obsolete[k]:
            continue

        # Access divergence for the current column.
        r_k = R[k]

        # Compute index into condensed distance matrix.
        c = np.uintp(_offset + _k)

        # Compute join criterion.
        d = distance[c]
        q = coefficient * d - r_i - r_k

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
        float32[::1],  # distance
        float32[::1],  # R
        float32[::1],  # Q
        uintp,  # z
        bool_[::1],  # obsolete
        uintp,  # n_remaining
        uintp,  # n_original
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    cache=NUMBA_CACHE,
)
def dynamic_search(
    distance: NDArray[np.float32],
    R: NDArray[np.float32],
    Q: NDArray[np.float32],
    z: np.uintp,  # index of new node created in previous iteration
    obsolete: NDArray[np.bool_],
    n_remaining: np.uintp,
    n_original: np.uintp,
):
    """Search for the closest pair of neighbouring nodes to join."""

    # Partially compute outside loop.
    coefficient = np.float32(n_remaining - 2)

    # x, y - Row/column indices of the pair of nodes with the global minimum, to be
    # joined.
    # q_xy - Global minimum join criterion.
    # d_xy - Distance between pair of nodes with global minimum.

    # First scan the new row at index z and use as starting point for search.
    x = z
    y, q_xy, d_xy = search_row(
        distance=distance,
        R=R,
        Q=Q,
        obsolete=obsolete,
        i=x,
        coefficient=coefficient,
        n_original=n_original,
    )

    # Iterate over all rows of the distance matrix.
    for _i in range(n_original):
        i = np.uintp(_i)  # row index

        if obsolete[i]:
            continue

        if i == z:
            # Already searched.
            continue

        if i < z:
            # Calculate join criterion for the new node, and update Q if necessary.
            r_i = R[i]
            r_z = R[z]
            c_iz = condensed_index(i, z, n_original)
            d_iz = distance[c_iz]
            q_iz = coefficient * d_iz - r_i - r_z
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
            distance=distance,
            R=R,
            Q=Q,
            obsolete=obsolete,
            i=i,
            coefficient=coefficient,
            n_original=n_original,
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
        float32[::1],  # distance
        float32[::1],  # R
        float32[::1],  # Q
        uintp,  # z
        bool_[::1],  # obsolete
        uintp,  # n_remaining
        uintp,  # n_original
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    # cache=NUMBA_CACHE,  # warning that cannot cache, though not clear why
    parallel=True,
)
def dynamic_search_parallel(
    distance: NDArray[np.float32],
    R: NDArray[np.float32],
    Q: NDArray[np.float32],
    z: np.uintp,  # index of new node created in previous iteration
    obsolete: NDArray[np.bool_],
    n_remaining: np.uintp,
    n_original: np.uintp,
):
    """Search for the closest pair of neighbouring nodes to join."""

    # Partially compute outside loop.
    coefficient = np.float32(n_remaining - 2)

    # First scan the new row at index z and use as starting point for search.
    global_x = z
    global_y, global_q_xy, global_d_xy = search_row(
        distance=distance,
        R=R,
        Q=Q,
        obsolete=obsolete,
        i=z,
        coefficient=coefficient,
        n_original=n_original,
    )

    # Prepare for parallel search.
    n_threads = get_num_threads()
    results_q_xy = np.empty(n_threads, dtype=np.float32)
    results_d_xy = np.empty(n_threads, dtype=np.float32)
    results_xy = np.empty((n_threads, 2), dtype=np.uintp)

    # Set up parallel threads.
    for t in prange(n_threads):
        # Thread local variables.
        local_q_xy = global_q_xy
        local_d_xy = global_d_xy
        local_x = global_x
        local_y = global_y

        # Iterate over rows of the distance matrix.
        # Striped work distribution.
        for _i in range(t, n_original, n_threads):
            i = np.uintp(_i)  # row index

            if obsolete[i]:
                continue

            if i == z:
                # Already searched.
                continue

            # Previous row minimum.
            q_i = Q[i]

            if i < z:
                # Calculate join criterion for the new node, and update Q if necessary.
                r_i = R[i]
                r_z = R[z]
                c_iz = condensed_index(i, z, n_original)
                d_iz = distance[c_iz]
                q_iz = coefficient * d_iz - r_i - r_z
                if q_iz < q_i:
                    Q[i] = q_iz

            if q_i > global_q_xy:
                # We can skip this row. The previous row optimum join criterion is greater
                # than the current global optimum, and so there is now way that this row
                # can contain a better match. This is the core optimisation of the dynamic
                # algorithm.
                continue

            # Join criterion could be lower than the current global minimum. Fully search
            # the row.
            j, q_ij, d_ij = search_row(
                distance=distance,
                R=R,
                Q=Q,
                obsolete=obsolete,
                i=i,
                coefficient=coefficient,
                n_original=n_original,
            )

            if q_ij < local_q_xy:
                # Found new minimum.
                local_q_xy = q_ij
                local_d_xy = d_ij
                local_x = i
                local_y = j

                # Share update between threads, in case it helps other threads skip
                # more rows. This is a supported parallel reduction.
                global_q_xy = min(global_q_xy, local_q_xy)

        # Store results for this thread.
        results_q_xy[t] = local_q_xy
        results_d_xy[t] = local_d_xy
        results_xy[t, 0] = local_x
        results_xy[t, 1] = local_y

    # Final reduction across thread results.
    t_best = np.argmin(results_q_xy)
    global_d_xy = results_d_xy[t_best]
    global_x = results_xy[t_best, 0]
    global_y = results_xy[t_best, 1]

    return global_x, global_y, global_d_xy


@njit(
    uintp(
        float32[::1],  # distance
        float32[::1],  # R
        uintp[::1],  # index_to_id
        bool_[::1],  # obsolete
        uintp,  # parent
        uintp,  # x
        uintp,  # y
        float32,  # d_xy
        uintp,  # n_original
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    cache=NUMBA_CACHE,
)
def dynamic_update(
    distance: NDArray[np.float32],
    R: NDArray[np.float32],
    index_to_id: NDArray[np.uintp],
    obsolete: NDArray[np.bool_],
    parent: np.uintp,
    x: np.uintp,
    y: np.uintp,
    d_xy: np.float32,
    n_original: np.uintp,
) -> np.uintp:
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
    for _k in range(n_original):
        k = np.uintp(_k)

        if obsolete[k] or k == x or k == y:
            continue

        # Calculate and store distance from k to the new node.
        c_kx = condensed_index(k, x, n_original)
        d_kx = distance[c_kx]
        c_ky = condensed_index(k, y, n_original)
        d_ky = distance[c_ky]
        d_kz = float32(0.5) * (d_kx + d_ky - d_xy)
        c_kz = c_kx
        distance[c_kz] = d_kz

        # Subtract out the distances for the nodes that have just been joined and add
        # in distance for the new node.
        r_k = R[k] - d_kx - d_ky + d_kz
        R[k] = r_k

        # Accumulate divergence for the new node.
        r_z += d_kz

    # Assign divergence for the new node.
    R[z] = r_z

    return z


@njit(
    uintp(
        float32[::1],  # distance
        float32[::1],  # R
        uintp[::1],  # index_to_id
        bool_[::1],  # obsolete
        uintp,  # parent
        uintp,  # x
        uintp,  # y
        float32,  # d_xy
        uintp,  # n_original
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    cache=NUMBA_CACHE,
    parallel=True,
)
def dynamic_update_parallel(
    distance: NDArray[np.float32],
    R: NDArray[np.float32],
    index_to_id: NDArray[np.uintp],
    obsolete: NDArray[np.bool_],
    parent: np.uintp,
    x: np.uintp,
    y: np.uintp,
    d_xy: np.float32,
    n_original: np.uintp,
) -> np.uintp:
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
    for _k in prange(n_original):
        k = np.uintp(_k)

        if obsolete[k] or k == x or k == y:
            continue

        # Calculate and store distance from k to the new node.
        c_kx = condensed_index(k, x, n_original)
        d_kx = distance[c_kx]
        c_ky = condensed_index(k, y, n_original)
        d_ky = distance[c_ky]
        d_kz = float32(0.5) * (d_kx + d_ky - d_xy)
        c_kz = c_kx
        distance[c_kz] = d_kz

        # Subtract out the distances for the nodes that have just been joined and add
        # in distance for the new node.
        r_k = R[k] - d_kx - d_ky + d_kz
        R[k] = r_k

        # Accumulate divergence for the new node.
        r_z += d_kz

    # Assign divergence for the new node.
    R[z] = r_z

    return z


@njit(
    uintp(
        uintp,  # iteration
        float32[::1],  # distance
        float32[::1],  # R
        float32[::1],  # Q
        uintp,  # previous_z
        uintp[::1],  # index_to_id
        bool_[::1],  # obsolete
        float32[:, ::1],  # Z
        uintp,  # n_original
        bool_,  # disallow_negative_distances
        bool_,  # parallel
    ),
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
    error_model=NUMBA_ERROR_MODEL,
    boundscheck=NUMBA_BOUNDSCHECK,
    # cache=NUMBA_CACHE,  # warning that cannot cache, though not clear why
)
def dynamic_iteration(
    iteration: np.uintp,
    distance: NDArray[np.float32],
    R: NDArray[np.float32],
    Q,
    previous_z,
    index_to_id: NDArray[np.uintp],
    obsolete: NDArray[np.bool_],
    Z: NDArray[np.float32],
    n_original: np.uintp,
    disallow_negative_distances: bool,
    parallel: bool,
) -> np.uintp:
    # This will be the identifier for the new node to be created in this iteration.
    parent = iteration + n_original

    # Number of nodes remaining in this iteration.
    n_remaining = n_original - iteration

    if n_remaining > 2:
        # Search for the closest pair of nodes to join.
        if parallel:
            x, y, d_xy = dynamic_search_parallel(
                distance=distance,
                R=R,
                Q=Q,
                z=previous_z,
                obsolete=obsolete,
                n_remaining=n_remaining,
                n_original=n_original,
            )
        else:
            x, y, d_xy = dynamic_search(
                distance=distance,
                R=R,
                Q=Q,
                z=previous_z,
                obsolete=obsolete,
                n_remaining=n_remaining,
                n_original=n_original,
            )

        # Calculate distances to the new internal node.
        d_xz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (R[x] - R[y]))
        d_yz = 0.5 * (d_xy + (1 / (n_remaining - 2)) * (R[y] - R[x]))

    else:
        # Termination. Join the two remaining nodes, placing the final node at the
        # midpoint.
        _x, _y = np.nonzero(~obsolete)[0]
        x = np.uintp(_x)
        y = np.uintp(_y)
        c_xy = condensed_index(x, y, n_original)
        d_xy = distance[c_xy]
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
    assert x >= 0
    assert y >= 0
    assert x < n_original
    assert y < n_original
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
        leaves_x = np.float32(1)
    else:
        leaves_x = Z[child_x - n_original, 4]
    if child_y < n_original:
        leaves_y = np.float32(1)
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
        if parallel:
            new_z: np.uintp = dynamic_update_parallel(
                distance=distance,
                R=R,
                index_to_id=index_to_id,
                obsolete=obsolete,
                parent=parent,
                x=x,
                y=y,
                d_xy=d_xy,
                n_original=n_original,
            )
        else:
            new_z = dynamic_update(
                distance=distance,
                R=R,
                index_to_id=index_to_id,
                obsolete=obsolete,
                parent=parent,
                x=x,
                y=y,
                d_xy=d_xy,
                n_original=n_original,
            )

    else:
        new_z = UINTP_MAX

    return new_z
